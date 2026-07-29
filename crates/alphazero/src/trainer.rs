use super::network::{Network, RawValueOutput};
use super::replay::ReplayBuffer;
use super::representation::AlphaZeroRepresentation;
use engine_core::GameState;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::Serialize;
use std::sync::mpsc::sync_channel;
use std::time::{Duration, Instant};
use tch::nn::{Optimizer, OptimizerConfig};
use tch::{nn, Device, Kind, Tensor};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OptimizerSpec {
    Adam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
}

impl Default for OptimizerSpec {
    fn default() -> Self {
        Self::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LearningRateSchedule {
    #[default]
    Constant,
    LinearWarmupThenCosine {
        warmup_steps: u64,
        total_steps: u64,
        minimum_lr: f64,
    },
    Piecewise {
        boundaries: Vec<(u64, f64)>,
    },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TrainConfig {
    pub micro_batch_size: usize,
    pub batch_size: usize,
    pub train_steps: usize,
    /// Upper bound on sampled replay positions per newly generated position.
    ///
    /// This paces optimizer work to self-play so a small generation cannot be
    /// repeatedly fitted before the next network produces fresh data.
    #[serde(default)]
    pub max_replay_reuse_per_iteration: Option<f64>,
    pub progress_every: usize,
    pub lr: f64,
    pub weight_decay: f64,
    /// Relative contribution of the value objective to the combined training loss.
    pub value_loss_weight: f64,
    #[serde(default)]
    pub optimizer: OptimizerSpec,
    #[serde(default)]
    pub learning_rate_schedule: LearningRateSchedule,
    /// Number of CPU-encoded replay batches queued ahead of the GPU step.
    #[serde(default = "default_prefetch_depth")]
    pub prefetch_depth: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            micro_batch_size: 256,
            batch_size: 4096,
            train_steps: 80,
            max_replay_reuse_per_iteration: None,
            progress_every: 10,
            lr: 1e-3,
            weight_decay: 1e-4,
            value_loss_weight: 1.0,
            optimizer: OptimizerSpec::default(),
            learning_rate_schedule: LearningRateSchedule::default(),
            prefetch_depth: 1,
        }
    }
}

impl TrainConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.micro_batch_size > 0,
            "micro_batch_size must be positive"
        );
        anyhow::ensure!(self.prefetch_depth > 0, "prefetch depth must be positive");
        anyhow::ensure!(
            self.prefetch_depth <= 2,
            "prefetch depth must be at most two"
        );
        anyhow::ensure!(self.batch_size > 0, "batch_size must be positive");
        if let Some(max_replay_reuse) = self.max_replay_reuse_per_iteration {
            anyhow::ensure!(
                max_replay_reuse.is_finite() && max_replay_reuse > 0.0,
                "max replay reuse per iteration must be finite and positive"
            );
        }
        anyhow::ensure!(
            self.lr.is_finite() && self.lr > 0.0,
            "lr must be finite and positive"
        );
        anyhow::ensure!(
            self.weight_decay.is_finite() && self.weight_decay >= 0.0,
            "weight_decay must be finite and non-negative"
        );
        anyhow::ensure!(
            self.value_loss_weight.is_finite() && self.value_loss_weight >= 0.0,
            "value_loss_weight must be finite and non-negative"
        );
        validate_optimizer(&self.optimizer)?;
        validate_learning_rate_schedule(&self.learning_rate_schedule)?;
        Ok(())
    }

    /// Returns the number of optimizer steps permitted after one self-play
    /// generation. A non-empty generation always receives one step so small
    /// games do not starve training.
    pub fn train_steps_for_fresh_replay_samples(&self, fresh_replay_samples: usize) -> usize {
        if self.train_steps == 0 {
            return 0;
        }

        let Some(max_replay_reuse) = self.max_replay_reuse_per_iteration
        else {
            return self.train_steps;
        };

        if fresh_replay_samples == 0 {
            return 0;
        }

        let permitted_samples = fresh_replay_samples as f64 * max_replay_reuse;
        let permitted_steps = (permitted_samples / self.batch_size as f64).floor() as usize;

        self.train_steps.min(permitted_steps.max(1))
    }

    pub fn learning_rate_at(&self, step: u64) -> f64 {
        match &self.learning_rate_schedule {
            LearningRateSchedule::Constant => self.lr,
            LearningRateSchedule::LinearWarmupThenCosine {
                warmup_steps,
                total_steps,
                minimum_lr,
            } if step < *warmup_steps => self.lr * (step + 1) as f64 / *warmup_steps as f64,
            LearningRateSchedule::LinearWarmupThenCosine {
                warmup_steps,
                total_steps,
                minimum_lr,
            } => {
                let span = total_steps - warmup_steps;
                let progress = (step.saturating_sub(*warmup_steps) as f64 / span as f64).min(1.0);
                let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                minimum_lr + (self.lr - minimum_lr) * cosine
            }
            LearningRateSchedule::Piecewise { boundaries } => boundaries
                .iter()
                .take_while(|(boundary, _)| *boundary <= step)
                .last()
                .map_or(self.lr, |(_, lr)| *lr),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainMetrics {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub train_steps: usize,
    pub configured_train_steps: usize,
    pub fresh_replay_samples: usize,
    pub replay_reuse: f64,
    pub replay_sampling_seconds: f64,
    pub host_to_device_seconds: f64,
    pub forward_backward_seconds: f64,
    pub optimizer_seconds: f64,
    pub samples_per_second: f64,
    pub learning_rate: f64,
}

/// Reproducibility inputs for one training invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingSeed {
    pub experiment_seed: u64,
    pub global_step: u64,
}

/// Owns the optimizer together with the immutable training configuration.
pub struct Trainer {
    config: TrainConfig,
    optimizer: Optimizer,
}

impl Trainer {
    pub fn new(vs: &nn::VarStore, config: TrainConfig) -> anyhow::Result<Self> {
        let optimizer = build_optimizer(vs, &config)?;

        Ok(Self { config, optimizer })
    }

    pub fn train<S, R>(
        &mut self,
        network: &Network,
        replay: &ReplayBuffer<S>,
        representation: &R,
        device: Device,
        seed: TrainingSeed,
        fresh_replay_samples: usize,
    ) -> Option<TrainMetrics>
    where
        S: GameState + Clone + Send + Sync,
        R: AlphaZeroRepresentation<S>,
    {
        let effective_train_steps = self
            .config
            .train_steps_for_fresh_replay_samples(fresh_replay_samples);
        let mut effective_config = self.config.clone();
        effective_config.train_steps = effective_train_steps;

        let mut metrics = train_with_optimizer(
            network,
            &mut self.optimizer,
            replay,
            representation,
            device,
            &effective_config,
            seed,
        )?;
        metrics.configured_train_steps = self.config.train_steps;
        metrics.fresh_replay_samples = fresh_replay_samples;
        metrics.replay_reuse = metrics.train_steps as f64 * self.config.batch_size as f64
            / fresh_replay_samples.max(1) as f64;

        Some(metrics)
    }
}

const REPLAY_SAMPLING_PURPOSE: u64 = 0x7265_706c_6179_7361;

const fn default_prefetch_depth() -> usize {
    1
}

pub fn build_optimizer(vs: &nn::VarStore, cfg: &TrainConfig) -> anyhow::Result<Optimizer> {
    cfg.validate()?;
    let OptimizerSpec::Adam {
        beta1,
        beta2,
        epsilon,
    } = cfg.optimizer;
    Ok(nn::Adam {
        beta1,
        beta2,
        eps: epsilon,
        wd: cfg.weight_decay,
        ..Default::default()
    }
    .build(vs, cfg.learning_rate_at(0))?)
}

/// Trains either scalar or WDL models from one compact replay format.
#[allow(clippy::too_many_arguments)] // Typed model, optimizer, replay, and reproducibility inputs meet at this boundary.
pub fn train<S, R>(
    network: &Network,
    optimizer: &mut Optimizer,
    replay: &ReplayBuffer<S>,
    representation: &R,
    device: Device,
    cfg: &TrainConfig,
    experiment_seed: u64,
    global_step: u64,
) -> Option<TrainMetrics>
where
    S: GameState + Clone + Send + Sync,
    R: AlphaZeroRepresentation<S>,
{
    train_with_optimizer(
        network,
        optimizer,
        replay,
        representation,
        device,
        cfg,
        TrainingSeed {
            experiment_seed,
            global_step,
        },
    )
}

fn train_with_optimizer<S, R>(
    network: &Network,
    optimizer: &mut Optimizer,
    replay: &ReplayBuffer<S>,
    representation: &R,
    device: Device,
    cfg: &TrainConfig,
    seed: TrainingSeed,
) -> Option<TrainMetrics>
where
    S: GameState + Clone + Send + Sync,
    R: AlphaZeroRepresentation<S>,
{
    cfg.validate()
        .expect("Trainer only accepts validated training configuration");

    if cfg.train_steps == 0 {
        return None;
    }

    let started = Instant::now();
    let (sender, receiver) = sync_channel(cfg.prefetch_depth.max(1));
    let mut sampling_time = Duration::ZERO;
    let mut transfer_time = Duration::ZERO;
    let mut forward_backward_time = Duration::ZERO;
    let mut optimizer_time = Duration::ZERO;
    let mut policy_total = 0.0;
    let mut value_total = 0.0;
    let mut completed = 0;

    std::thread::scope(|scope| {
        scope.spawn(move || {
            for offset in 0..cfg.train_steps {
                let sampling_started = Instant::now();
                let mut rng = SmallRng::seed_from_u64(replay_seed(
                    seed.experiment_seed,
                    seed.global_step + offset as u64,
                ));
                let batch = replay.sample(cfg.batch_size, representation, &mut rng);
                let elapsed = sampling_started.elapsed();
                let replay_was_empty = batch.is_none();
                if sender.send((batch, elapsed)).is_err() {
                    break;
                }
                if replay_was_empty {
                    break;
                }
            }
        });

        while let Ok((Some(batch), sampled_for)) = receiver.recv() {
            sampling_time += sampled_for;
            let transfer_started = Instant::now();
            let batch = DeviceReplayBatch::from_cpu(batch, device);
            transfer_time += transfer_started.elapsed();
            let learning_rate = cfg.learning_rate_at(seed.global_step + completed as u64);
            optimizer.set_lr(learning_rate);
            let metrics = train_device_batch(
                network,
                optimizer,
                &batch,
                cfg.micro_batch_size,
                cfg.value_loss_weight,
                representation_shape::<S, R>(),
            );
            forward_backward_time += metrics.forward_backward;
            optimizer_time += metrics.optimizer;
            policy_total += metrics.policy_loss;
            value_total += metrics.value_loss;
            completed += 1;
        }
    });
    (completed > 0).then_some(TrainMetrics {
        policy_loss: policy_total / completed as f64,
        value_loss: value_total / completed as f64,
        train_steps: completed,
        configured_train_steps: cfg.train_steps,
        fresh_replay_samples: 0,
        replay_reuse: 0.0,
        replay_sampling_seconds: sampling_time.as_secs_f64(),
        host_to_device_seconds: transfer_time.as_secs_f64(),
        forward_backward_seconds: forward_backward_time.as_secs_f64(),
        optimizer_seconds: optimizer_time.as_secs_f64(),
        samples_per_second: completed as f64 * cfg.batch_size as f64
            / started.elapsed().as_secs_f64().max(f64::EPSILON),
        learning_rate: cfg.learning_rate_at(seed.global_step + completed.saturating_sub(1) as u64),
    })
}

fn validate_optimizer(optimizer: &OptimizerSpec) -> anyhow::Result<()> {
    match optimizer {
        OptimizerSpec::Adam {
            beta1,
            beta2,
            epsilon,
        } => {
            anyhow::ensure!(
                beta1.is_finite() && (0.0..1.0).contains(beta1),
                "Adam beta1 must be in [0, 1)"
            );
            anyhow::ensure!(
                beta2.is_finite() && (0.0..1.0).contains(beta2),
                "Adam beta2 must be in [0, 1)"
            );
            anyhow::ensure!(
                epsilon.is_finite() && *epsilon > 0.0,
                "Adam epsilon must be finite and positive"
            );
        }
    }
    Ok(())
}

fn validate_learning_rate_schedule(schedule: &LearningRateSchedule) -> anyhow::Result<()> {
    match schedule {
        LearningRateSchedule::Constant => {}
        LearningRateSchedule::LinearWarmupThenCosine {
            warmup_steps,
            total_steps,
            minimum_lr,
        } => {
            anyhow::ensure!(
                *total_steps > *warmup_steps,
                "cosine total_steps must exceed warmup_steps"
            );
            anyhow::ensure!(
                minimum_lr.is_finite() && *minimum_lr >= 0.0,
                "cosine minimum_lr must be finite and non-negative"
            );
        }
        LearningRateSchedule::Piecewise { boundaries } => {
            let mut previous = None;
            for &(step, lr) in boundaries {
                anyhow::ensure!(
                    lr.is_finite() && lr > 0.0,
                    "piecewise learning rates must be finite and positive"
                );
                anyhow::ensure!(
                    previous.is_none_or(|earlier| step > earlier),
                    "piecewise boundaries must increase"
                );
                previous = Some(step);
            }
        }
    }
    Ok(())
}

fn representation_shape<S: GameState, R: AlphaZeroRepresentation<S>>() -> [i64; 4] {
    let [channels, height, width] = R::STATE_SHAPE.map(|dimension| dimension as i64);
    [-1, channels, height, width]
}

struct DeviceReplayBatch {
    states: Tensor,
    actions: Tensor,
    probabilities: Tensor,
    rows: Tensor,
    outcomes: Tensor,
    policy_weights: Tensor,
    policy_weight_sum: f32,
    value_weights: Tensor,
    value_weight_sum: f32,
    offsets: Vec<i64>,
}

impl DeviceReplayBatch {
    fn from_cpu(batch: super::replay::ReplayBatch, device: Device) -> Self {
        Self {
            states: batch.states.to_device(device),
            actions: batch.policies.actions.to_device(device),
            probabilities: batch.policies.probabilities.to_device(device),
            rows: batch.policies.rows.to_device(device),
            outcomes: batch.outcomes.to_device(device),
            policy_weights: batch.policy_weights.to_device(device),
            policy_weight_sum: batch.policy_weight_sum,
            value_weights: batch.value_weights.to_device(device),
            value_weight_sum: batch.value_weight_sum,
            offsets: batch.policies.offsets,
        }
    }

    fn policy_rows(&self, start: usize, count: usize) -> DevicePolicyBatch {
        let first = self.offsets[start];
        let end = self.offsets[start + count];
        let entries = end - first;
        DevicePolicyBatch {
            actions: self.actions.narrow(0, first, entries),
            probabilities: self.probabilities.narrow(0, first, entries),
            rows: self.rows.narrow(0, first, entries) - start as i64,
        }
    }
}

struct DevicePolicyBatch {
    actions: Tensor,
    probabilities: Tensor,
    rows: Tensor,
}

struct StepMetrics {
    policy_loss: f64,
    value_loss: f64,
    forward_backward: Duration,
    optimizer: Duration,
}

fn train_device_batch(
    network: &Network,
    optimizer: &mut Optimizer,
    batch: &DeviceReplayBatch,
    micro_batch_size: usize,
    value_loss_weight: f64,
    shape: [i64; 4],
) -> StepMetrics {
    let rows = batch.outcomes.size()[0] as usize;
    let has_policy = batch.policy_weight_sum > 0.0;
    let has_value = batch.value_weight_sum > 0.0;
    let policy_denominator = has_policy.then(|| batch.policy_weights.sum(Kind::Float));
    let value_denominator = has_value.then(|| batch.value_weights.sum(Kind::Float));
    let device = batch.states.device();
    let mut policy_metric = Tensor::zeros([], (Kind::Float, device));
    let mut value_metric = Tensor::zeros([], (Kind::Float, device));
    optimizer.zero_grad();
    let forward_started = Instant::now();
    for start in (0..rows).step_by(micro_batch_size) {
        let count = (rows - start).min(micro_batch_size);
        let states = batch
            .states
            .narrow(0, start as i64, count as i64)
            .view(shape);
        let outcomes = batch.outcomes.narrow(0, start as i64, count as i64);
        let policy_weights = batch.policy_weights.narrow(0, start as i64, count as i64);
        let value_weights = batch.value_weights.narrow(0, start as i64, count as i64);
        let output = network.forward_t(&states, true);
        let mut loss = None;
        if has_policy {
            let policies = batch.policy_rows(start, count);
            let numerator = sparse_policy_numerator(
                &output.policy_logits,
                &policies.actions,
                &policies.probabilities,
                &policies.rows,
                &policy_weights,
            );
            policy_metric += numerator.detach();
            loss = Some(numerator / policy_denominator.as_ref().unwrap());
        }
        if has_value {
            let numerator = match output.value {
                RawValueOutput::Scalar(value) => {
                    scalar_mse_numerator(&value, &outcomes, &value_weights)
                }
                RawValueOutput::WdlLogits(logits) => {
                    wdl_cross_entropy_numerator(&logits, &outcomes, &value_weights)
                }
            };
            value_metric += numerator.detach();
            let scaled = numerator / value_denominator.as_ref().unwrap() * value_loss_weight;
            loss = Some(loss.map_or(scaled.shallow_clone(), |policy| policy + scaled));
        }
        if let Some(loss) = loss {
            loss.backward();
        }
    }
    let forward_backward = forward_started.elapsed();
    let optimizer_started = Instant::now();
    optimizer.step();
    let optimizer_elapsed = optimizer_started.elapsed();
    StepMetrics {
        policy_loss: if has_policy {
            policy_metric.double_value(&[]) / f64::from(batch.policy_weight_sum)
        }
        else {
            0.0
        },
        value_loss: if has_value {
            value_metric.double_value(&[]) / f64::from(batch.value_weight_sum)
        }
        else {
            0.0
        },
        forward_backward,
        optimizer: optimizer_elapsed,
    }
}

fn replay_seed(experiment_seed: u64, global_step: u64) -> u64 {
    let mut value =
        experiment_seed ^ REPLAY_SAMPLING_PURPOSE ^ global_step.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value.wrapping_mul(0x94d0_49bb_1331_11eb) ^ (value >> 31)
}

fn sparse_policy_numerator(
    logits: &Tensor,
    actions: &Tensor,
    probabilities: &Tensor,
    rows: &Tensor,
    weights: &Tensor,
) -> Tensor {
    let action_size = logits.size()[1];
    let indices = rows * action_size + actions;
    let selected = logits
        .log_softmax(1, Kind::Float)
        .flatten(0, -1)
        .index_select(0, &indices);
    let entry_weights = weights.index_select(0, rows);
    -(probabilities * selected * entry_weights).sum(Kind::Float)
}

fn scalar_mse_numerator(value: &Tensor, outcomes: &Tensor, weights: &Tensor) -> Tensor {
    let targets = outcomes.eq(0).to_kind(Kind::Float) - outcomes.eq(2).to_kind(Kind::Float);
    let errors = (value.squeeze_dim(-1) - targets).pow_tensor_scalar(2.0);
    (&errors * weights).sum(Kind::Float)
}

/// WDL cross entropy consumes categorical outcome targets directly; no float
/// equality is used to reconstruct classes.
pub fn wdl_cross_entropy(logits: &Tensor, outcomes: &Tensor, weights: &Tensor) -> Tensor {
    weighted_mean(
        wdl_cross_entropy_numerator(logits, outcomes, weights),
        weights,
        logits,
    )
}

fn wdl_cross_entropy_numerator(logits: &Tensor, outcomes: &Tensor, weights: &Tensor) -> Tensor {
    let logp = logits.log_softmax(1, Kind::Float);
    let selected = logp
        .gather(1, &outcomes.view([-1, 1]), false)
        .squeeze_dim(-1);
    -(&selected * weights).sum(Kind::Float)
}

fn weighted_mean(numerator: Tensor, weights: &Tensor, reference: &Tensor) -> Tensor {
    let denominator = weights.sum(Kind::Float);
    if denominator.double_value(&[]) > 0.0 {
        numerator / denominator
    }
    else {
        reference.sum(Kind::Float) * 0.0
    }
}

#[cfg(test)]
mod tests;

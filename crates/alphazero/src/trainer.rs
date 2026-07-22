use super::network::{Network, RawValueOutput};
use super::replay::{ReplayBuffer, SparsePolicyBatch};
use super::representation::AlphaZeroRepresentation;
use engine_core::GameState;
use serde::Serialize;
use tch::nn::{Optimizer, OptimizerConfig};
use tch::{nn, Device, Kind, Tensor};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainConfig {
    pub micro_batch_size: usize,
    pub batch_size: usize,
    pub train_steps: usize,
    pub progress_every: usize,
    pub lr: f64,
    pub weight_decay: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            micro_batch_size: 256,
            batch_size: 4096,
            train_steps: 80,
            progress_every: 10,
            lr: 1e-3,
            weight_decay: 1e-4,
        }
    }
}

impl TrainConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.micro_batch_size > 0,
            "micro_batch_size must be positive"
        );
        anyhow::ensure!(self.batch_size > 0, "batch_size must be positive");
        anyhow::ensure!(self.train_steps > 0, "train_steps must be positive");
        anyhow::ensure!(
            self.lr.is_finite() && self.lr > 0.0,
            "lr must be finite and positive"
        );
        anyhow::ensure!(
            self.weight_decay.is_finite() && self.weight_decay >= 0.0,
            "weight_decay must be finite and non-negative"
        );
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct TrainMetrics {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub train_steps: usize,
}

pub fn build_optimizer(vs: &nn::VarStore, cfg: &TrainConfig) -> anyhow::Result<Optimizer> {
    cfg.validate()?;
    Ok(nn::Adam {
        wd: cfg.weight_decay,
        ..Default::default()
    }
    .build(vs, cfg.lr)?)
}

/// Trains either scalar or WDL models from one compact replay format.
pub fn train<S, R>(
    network: &Network,
    optimizer: &mut Optimizer,
    replay: &ReplayBuffer<S>,
    representation: &R,
    device: Device,
    cfg: &TrainConfig,
) -> Option<TrainMetrics>
where
    S: GameState + Clone,
    R: AlphaZeroRepresentation<S>,
{
    cfg.validate().expect("invalid training configuration");
    let mut policy_total = 0.0;
    let mut value_total = 0.0;
    let mut completed = 0;
    let shape = representation_shape::<S, R>();
    for _ in 0..cfg.train_steps {
        let batch = replay.sample(cfg.batch_size, representation)?;
        let rows = batch.outcomes.size()[0] as usize;
        let policy_denominator = batch.policy_weights.sum(Kind::Float).double_value(&[]);
        let value_denominator = batch.value_weights.sum(Kind::Float).double_value(&[]);
        let mut policy_numerator = 0.0;
        let mut value_numerator = 0.0;
        optimizer.zero_grad();
        for start in (0..rows).step_by(cfg.micro_batch_size) {
            let count = (rows - start).min(cfg.micro_batch_size);
            let states = batch
                .states
                .narrow(0, start as i64, count as i64)
                .view(shape)
                .to_device(device);
            let outcomes = batch
                .outcomes
                .narrow(0, start as i64, count as i64)
                .to_device(device);
            let policy_weights = batch
                .policy_weights
                .narrow(0, start as i64, count as i64)
                .to_device(device);
            let value_weights = batch
                .value_weights
                .narrow(0, start as i64, count as i64)
                .to_device(device);
            let output = network.forward_t(&states, true);
            let mut loss = None;
            if policy_denominator > 0.0 {
                let policies = DevicePolicyBatch::from_rows(&batch.policies, start, count, device);
                let numerator = sparse_policy_numerator(
                    &output.policy_logits,
                    &policies.actions,
                    &policies.probabilities,
                    &policies.rows,
                    &policy_weights,
                );
                policy_numerator += numerator.double_value(&[]);
                loss = Some(numerator / policy_denominator);
            }
            if value_denominator > 0.0 {
                let numerator = match output.value {
                    RawValueOutput::Scalar(value) => {
                        scalar_mse_numerator(&value, &outcomes, &value_weights)
                    }
                    RawValueOutput::WdlLogits(logits) => {
                        wdl_cross_entropy_numerator(&logits, &outcomes, &value_weights)
                    }
                };
                value_numerator += numerator.double_value(&[]);
                let scaled = numerator / value_denominator;
                loss = Some(match loss {
                    Some(policy) => policy + scaled,
                    None => scaled,
                });
            }
            if let Some(loss) = loss {
                loss.backward();
            }
        }
        optimizer.step();
        policy_total += if policy_denominator > 0.0 {
            policy_numerator / policy_denominator
        } else {
            0.0
        };
        value_total += if value_denominator > 0.0 {
            value_numerator / value_denominator
        } else {
            0.0
        };
        completed += 1;
    }
    Some(TrainMetrics {
        policy_loss: policy_total / completed as f64,
        value_loss: value_total / completed as f64,
        train_steps: completed,
    })
}

fn representation_shape<S: GameState, R: AlphaZeroRepresentation<S>>() -> [i64; 4] {
    let [channels, height, width] = R::STATE_SHAPE.map(|dimension| dimension as i64);
    [-1, channels, height, width]
}

struct DevicePolicyBatch {
    actions: Tensor,
    probabilities: Tensor,
    rows: Tensor,
}

impl DevicePolicyBatch {
    fn from_rows(batch: &SparsePolicyBatch, start: usize, rows: usize, device: Device) -> Self {
        let first = batch.offsets[start];
        let end = batch.offsets[start + rows];
        let count = end - first;
        Self {
            actions: batch.actions.narrow(0, first, count).to_device(device),
            probabilities: batch
                .probabilities
                .narrow(0, first, count)
                .to_device(device),
            rows: (batch.rows.narrow(0, first, count) - start as i64).to_device(device),
        }
    }
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
    } else {
        reference.sum(Kind::Float) * 0.0
    }
}

#[cfg(test)]
mod tests;

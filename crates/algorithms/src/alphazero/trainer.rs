use super::network::{AlphaZeroNet, ChessAzV2Config, ChessAzV2Net, NetConfig};
use super::replay::{ReplayBatch, ReplayBuffer, SparsePolicyBatch};
use serde::Serialize;
use std::sync::mpsc;
use std::thread;
use tch::nn::{Optimizer, OptimizerConfig};
use tch::{nn, Device, Kind, Tensor};

#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// Micro-batch size that fits on the GPU.
    pub micro_batch_size: usize,
    /// Requested batch size per optimizer step; trains on fewer samples if the
    /// replay buffer is not full yet.
    pub batch_size: usize,
    /// Optimizer steps per call to `train`.
    pub train_steps: usize,
    /// Print training progress every N optimizer steps. Set 0 to disable.
    pub progress_every: usize,
    pub lr: f64,
    pub weight_decay: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
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
    /// Checks invariants required by the training loop.
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

#[cfg(test)]
/// Dense reference retained only for sparse-loss equivalence tests.
fn dense_policy_cross_entropy(logits: &Tensor, target: &Tensor) -> Tensor {
    let logp = logits.log_softmax(1, Kind::Float);
    -(target * logp)
        .sum_dim_intlist(1, false, Kind::Float)
        .mean(Kind::Float)
}

/// Cross-entropy between packed MCTS targets and network logits. Since each
/// target row is a probability distribution, summing packed entries and
/// dividing by the number of positions is identical to the dense formulation.
fn sparse_policy_cross_entropy(
    logits: &Tensor,
    actions: &Tensor,
    probabilities: &Tensor,
    rows: &Tensor,
    positions: i64,
) -> Tensor {
    let action_size = logits.size()[1];
    let flat_indices = rows * action_size + actions;
    let selected_logp = logits
        .log_softmax(1, Kind::Float)
        .flatten(0, -1)
        .index_select(0, &flat_indices);
    -(probabilities * selected_logp).sum(Kind::Float) / positions
}

/// Cross-entropy for a WDL target represented by the existing self-play
/// reward convention: `+1` win, `0` draw, `-1` loss.
pub fn wdl_cross_entropy(logits: &Tensor, reward: &Tensor) -> Tensor {
    let reward = reward.view([-1]);
    let target = Tensor::stack(
        &[
            reward.eq(1).to_kind(Kind::Float),
            reward.eq(0).to_kind(Kind::Float),
            reward.eq(-1).to_kind(Kind::Float),
        ],
        1,
    );
    let logp = logits.log_softmax(1, Kind::Float);
    -(target * logp)
        .sum_dim_intlist(1, false, Kind::Float)
        .mean(Kind::Float)
}

#[derive(Clone, Copy)]
enum ValueLoss {
    ScalarMse,
    Wdl,
}

struct DevicePolicyBatch {
    actions: Tensor,
    probabilities: Tensor,
    rows: Tensor,
    offsets: Vec<i64>,
}

impl SparsePolicyBatch {
    fn into_device(self, device: Device) -> DevicePolicyBatch {
        DevicePolicyBatch {
            actions: self.actions.to_device(device),
            probabilities: self.probabilities.to_device(device),
            rows: self.rows.to_device(device),
            offsets: self.offsets,
        }
    }
}

struct PipelineSettings<'a> {
    device: Device,
    train: &'a TrainConfig,
    state_shape: &'a [i64],
    progress_label: &'a str,
    value_loss: ValueLoss,
}

/// Shared legacy/v2 pipeline. Sampling step n+1 overlaps device work for step
/// n, and detached metric tensors remain on device until a progress boundary.
fn train_pipeline<F>(
    opt: &mut Optimizer,
    replay: &ReplayBuffer,
    settings: PipelineSettings<'_>,
    mut forward: F,
) -> Option<TrainMetrics>
where
    F: FnMut(&Tensor) -> (Tensor, Tensor),
{
    let PipelineSettings {
        device,
        train: cfg,
        state_shape,
        progress_label,
        value_loss: value_loss_kind,
    } = settings;
    let (req_tx, req_rx) = mpsc::sync_channel::<()>(1);
    let (batch_tx, batch_rx) = mpsc::sync_channel::<Option<ReplayBatch>>(1);

    thread::scope(|s| {
        s.spawn(move || {
            while req_rx.recv().is_ok() {
                let batch = replay.sample(cfg.batch_size);
                if batch_tx.send(batch).is_err() {
                    break;
                }
            }
        });

        let _ = req_tx.send(());

        let mut policy_loss_sum = Tensor::zeros([], (Kind::Float, device));
        let mut value_loss_sum = Tensor::zeros([], (Kind::Float, device));
        let mut samples = 0usize;
        let mut optimizer_steps = 0usize;

        for step in 0..cfg.train_steps {
            let ReplayBatch {
                states,
                policies,
                rewards,
            } = match batch_rx.recv() {
                Ok(Some(b)) => b,
                Ok(None) | Err(_) => break,
            };

            // Pipeline: request the next batch while the GPU works on this one.
            if step + 1 < cfg.train_steps {
                let _ = req_tx.send(());
            }

            let batch = states.size()[0] as usize;
            let states = states.view(state_shape).to_device(device);
            let policies = policies.into_device(device);
            let values = rewards.to_device(device);
            let done = step + 1;
            let report_step = cfg.progress_every > 0
                && done != cfg.train_steps
                && done.is_multiple_of(cfg.progress_every);
            let mut step_policy_loss_sum =
                report_step.then(|| Tensor::zeros([], (Kind::Float, device)));
            let mut step_value_loss_sum =
                report_step.then(|| Tensor::zeros([], (Kind::Float, device)));

            for i in (0..batch).step_by(cfg.micro_batch_size) {
                let chunk = (batch - i).min(cfg.micro_batch_size);
                let row_start = i as i64;
                let chunk = chunk as i64;
                let s = states.narrow(0, row_start, chunk);
                let v = values.narrow(0, row_start, chunk);
                let entry_start = policies.offsets[i];
                let entry_end = policies.offsets[i + chunk as usize];
                let entry_count = entry_end - entry_start;
                let actions = policies.actions.narrow(0, entry_start, entry_count);
                let probabilities = policies.probabilities.narrow(0, entry_start, entry_count);
                let rows = policies.rows.narrow(0, entry_start, entry_count) - row_start;

                let (logits, value_prediction) = forward(&s);
                let policy_loss = sparse_policy_cross_entropy(
                    &logits.flatten(1, -1),
                    &actions,
                    &probabilities,
                    &rows,
                    chunk,
                );
                let value_loss = match value_loss_kind {
                    ValueLoss::ScalarMse => value_prediction
                        .squeeze_dim(-1)
                        .mse_loss(&v, tch::Reduction::Mean),
                    ValueLoss::Wdl => wdl_cross_entropy(&value_prediction, &v),
                };

                let weight = chunk as f64 / batch as f64;
                let loss = (&policy_loss + &value_loss) * weight;
                loss.backward();

                // Losses are means over each microbatch. Weight detached values
                // by sample count so a short final microbatch is not over-represented.
                let weighted_policy_loss = policy_loss.detach() * chunk;
                let weighted_value_loss = value_loss.detach() * chunk;
                policy_loss_sum += &weighted_policy_loss;
                value_loss_sum += &weighted_value_loss;
                samples += chunk as usize;
                if let Some(sum) = &mut step_policy_loss_sum {
                    *sum += &weighted_policy_loss;
                }
                if let Some(sum) = &mut step_value_loss_sum {
                    *sum += &weighted_value_loss;
                }
            }

            opt.step();
            opt.zero_grad();
            optimizer_steps += 1;

            if let (Some(policy), Some(value)) = (step_policy_loss_sum, step_value_loss_sum) {
                let n = batch as f64;
                println!(
                    "{progress_label}: {done}/{} steps, policy_loss={:.4}, value_loss={:.4}",
                    cfg.train_steps,
                    (&policy / n).double_value(&[]),
                    (&value / n).double_value(&[])
                );
            }
        }

        // Drop the sender so the worker's recv() unblocks and the thread exits.
        drop(req_tx);

        if samples == 0 {
            return None;
        }
        let n = samples as f64;
        let policy_loss = (&policy_loss_sum / n).double_value(&[]);
        let value_loss = (&value_loss_sum / n).double_value(&[]);
        if cfg.progress_every > 0 {
            println!(
                "{progress_label}: {optimizer_steps}/{} steps, policy_loss={policy_loss:.4}, value_loss={value_loss:.4}",
                cfg.train_steps
            );
        }
        Some(TrainMetrics {
            policy_loss,
            value_loss,
            train_steps: optimizer_steps,
        })
    })
}

/// Runs legacy policy/value training with gradient accumulation and replay
/// prefetch. Returns `None` if replay is empty.
pub fn train(
    net: &AlphaZeroNet,
    opt: &mut Optimizer,
    replay: &ReplayBuffer,
    device: Device,
    net_cfg: &NetConfig,
    cfg: &TrainConfig,
) -> Option<TrainMetrics> {
    cfg.validate().expect("invalid training configuration");
    let state_shape = [-1, net_cfg.input_channels, net_cfg.height, net_cfg.width];
    train_pipeline(
        opt,
        replay,
        PipelineSettings {
            device,
            train: cfg,
            state_shape: &state_shape,
            progress_label: "train progress",
            value_loss: ValueLoss::ScalarMse,
        },
        |states| net.forward_t(states, true),
    )
}

/// Trains a chess-v2 network. Replay keeps scalar rewards, which are converted
/// to WDL targets at the loss boundary so no separate replay representation is
/// needed for the new value head.
pub fn train_chess_az_v2(
    net: &ChessAzV2Net,
    opt: &mut Optimizer,
    replay: &ReplayBuffer,
    device: Device,
    net_cfg: ChessAzV2Config,
    cfg: &TrainConfig,
) -> Option<TrainMetrics> {
    cfg.validate().expect("invalid training configuration");
    net_cfg
        .validate()
        .expect("invalid chess az v2 configuration");
    let state_shape = [-1, net_cfg.input_channels(), 8, 8];
    train_pipeline(
        opt,
        replay,
        PipelineSettings {
            device,
            train: cfg,
            state_shape: &state_shape,
            progress_label: "train v2 progress",
            value_loss: ValueLoss::Wdl,
        },
        |states| net.forward_t(states, true),
    )
}

#[cfg(test)]
mod tests;

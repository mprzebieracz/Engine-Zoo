use super::network::{AlphaZeroNet, ChessAzV2Config, ChessAzV2Net, NetConfig};
use super::replay::ReplayBuffer;
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

/// Cross-entropy between MCTS target distribution `target` and network logits.
fn policy_cross_entropy(logits: &Tensor, target: &Tensor) -> Tensor {
    let logp = logits.log_softmax(1, Kind::Float);
    -(target * logp)
        .sum_dim_intlist(1, false, Kind::Float)
        .mean(Kind::Float)
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

/// Runs `cfg.train_steps` optimizer steps of policy cross-entropy + value MSE
/// with gradient accumulation. Prefetches the next batch from the replay buffer
/// on a background thread while the GPU runs the forward/backward pass.
/// Returns None if the replay buffer is empty.
pub fn train(
    net: &AlphaZeroNet,
    opt: &mut Optimizer,
    replay: &ReplayBuffer,
    device: Device,
    net_cfg: &NetConfig,
    cfg: &TrainConfig,
) -> Option<TrainMetrics> {
    cfg.validate().expect("invalid training configuration");

    // req_tx signals the worker to sample; batch_rx delivers the result.
    let (req_tx, req_rx) = mpsc::sync_channel::<()>(1);
    let (batch_tx, batch_rx) = mpsc::sync_channel::<Option<(Tensor, Tensor, Tensor)>>(1);

    thread::scope(|s| {
        s.spawn(move || {
            while req_rx.recv().is_ok() {
                let batch = replay.sample(cfg.batch_size);
                if batch_tx.send(batch).is_err() {
                    break;
                }
            }
        });

        // Kick off the first sample before we start the GPU loop.
        let _ = req_tx.send(());

        let mut policy_loss_sum = Tensor::zeros([], (Kind::Float, device));
        let mut value_loss_sum = Tensor::zeros([], (Kind::Float, device));
        let mut samples = 0usize;
        let mut optimizer_steps = 0usize;

        for step in 0..cfg.train_steps {
            let (states, policies, values) = match batch_rx.recv() {
                Ok(Some(b)) => b,
                Ok(None) | Err(_) => break,
            };

            // Pipeline: request the next batch while the GPU works on this one.
            if step + 1 < cfg.train_steps {
                let _ = req_tx.send(());
            }

            let batch = states.size()[0] as usize;
            let states = states
                .view([-1, net_cfg.input_channels, net_cfg.height, net_cfg.width])
                .to_device(device);
            let policies = policies.to_device(device);
            let values = values.to_device(device);
            let mut step_policy_loss_sum = Tensor::zeros([], (Kind::Float, device));
            let mut step_value_loss_sum = Tensor::zeros([], (Kind::Float, device));
            let mut step_micro_batches = 0usize;

            for i in (0..batch).step_by(cfg.micro_batch_size) {
                let chunk = (batch - i).min(cfg.micro_batch_size);
                let i = i as i64;
                let chunk = chunk as i64;
                let s = states.narrow(0, i, chunk);
                let pi = policies.narrow(0, i, chunk);
                let v = values.narrow(0, i, chunk);

                let (logits, v_pred) = net.forward_t(&s, true);
                let v_pred = v_pred.squeeze_dim(-1);

                let policy_loss = policy_cross_entropy(&logits, &pi);
                let value_loss = v_pred.mse_loss(&v, tch::Reduction::Mean);

                let weight = chunk as f64 / batch as f64;
                let loss = (&policy_loss + &value_loss) * weight;
                loss.backward();

                // Losses are means over each microbatch. Weight detached values
                // by sample count so a short final microbatch is not over-represented.
                policy_loss_sum += policy_loss.detach() * chunk;
                value_loss_sum += value_loss.detach() * chunk;
                samples += chunk as usize;
                step_policy_loss_sum += policy_loss.detach() * chunk;
                step_value_loss_sum += value_loss.detach() * chunk;
                step_micro_batches += chunk as usize;
            }

            opt.step();
            opt.zero_grad();
            optimizer_steps += 1;

            let done = step + 1;
            if cfg.progress_every > 0
                && (done == cfg.train_steps || done.is_multiple_of(cfg.progress_every))
            {
                let n = step_micro_batches as f64;
                println!(
                    "train progress: {done}/{} steps, policy_loss={:.4}, value_loss={:.4}",
                    cfg.train_steps,
                    (&step_policy_loss_sum / n).double_value(&[]),
                    (&step_value_loss_sum / n).double_value(&[])
                );
            }
        }

        // Drop the sender so the worker's recv() unblocks and the thread exits.
        drop(req_tx);

        if samples == 0 {
            return None;
        }
        let n = samples as f64;
        Some(TrainMetrics {
            policy_loss: (&policy_loss_sum / n).double_value(&[]),
            value_loss: (&value_loss_sum / n).double_value(&[]),
            train_steps: optimizer_steps,
        })
    })
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
    let mut policy_loss_sum = 0.0;
    let mut value_loss_sum = 0.0;
    let mut samples = 0usize;

    for step in 0..cfg.train_steps {
        let (states, policies, values) = replay.sample(cfg.batch_size)?;
        let batch = states.size()[0] as usize;
        let states = states
            .view([-1, net_cfg.input_channels(), 8, 8])
            .to_device(device);
        let policies = policies.to_device(device);
        let values = values.to_device(device);
        let mut step_policy_loss = 0.0;
        let mut step_value_loss = 0.0;
        let mut step_samples = 0usize;

        for i in (0..batch).step_by(cfg.micro_batch_size) {
            let chunk = (batch - i).min(cfg.micro_batch_size) as i64;
            let i = i as i64;
            let (policy, wdl) = net.forward_t(&states.narrow(0, i, chunk), true);
            let policy_loss =
                policy_cross_entropy(&policy.flatten(1, -1), &policies.narrow(0, i, chunk));
            let value_loss = wdl_cross_entropy(&wdl, &values.narrow(0, i, chunk));
            let weight = chunk as f64 / batch as f64;
            let loss = (&policy_loss + &value_loss) * weight;
            loss.backward();
            let policy_loss = policy_loss.double_value(&[]);
            let value_loss = value_loss.double_value(&[]);
            policy_loss_sum += policy_loss * chunk as f64;
            value_loss_sum += value_loss * chunk as f64;
            step_policy_loss += policy_loss * chunk as f64;
            step_value_loss += value_loss * chunk as f64;
            step_samples += chunk as usize;
            samples += chunk as usize;
        }
        opt.step();
        opt.zero_grad();

        let completed = step + 1;
        if cfg.progress_every > 0
            && (completed == cfg.train_steps || completed.is_multiple_of(cfg.progress_every))
        {
            println!(
                "train v2 progress: {completed}/{} steps, policy_loss={:.4}, value_loss={:.4}",
                cfg.train_steps,
                step_policy_loss / step_samples as f64,
                step_value_loss / step_samples as f64,
            );
        }
    }

    (samples > 0).then(|| TrainMetrics {
        policy_loss: policy_loss_sum / samples as f64,
        value_loss: value_loss_sum / samples as f64,
        train_steps: cfg.train_steps,
    })
}

#[cfg(test)]
mod tests;

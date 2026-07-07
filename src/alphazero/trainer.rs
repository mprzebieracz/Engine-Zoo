use super::network::{AlphaZeroNet, NetConfig};
use super::replay::ReplayBuffer;
use serde::Serialize;
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
    pub lr: f64,
    pub weight_decay: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            micro_batch_size: 256,
            batch_size: 4096,
            train_steps: 20,
            lr: 1e-3,
            weight_decay: 1e-4,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct TrainMetrics {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub train_steps: usize,
}

pub fn build_optimizer(vs: &nn::VarStore, cfg: &TrainConfig) -> anyhow::Result<Optimizer> {
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

/// Runs `cfg.train_steps` optimizer steps of policy cross-entropy + value MSE
/// with gradient accumulation. Returns None if the replay buffer is empty.
pub fn train(
    net: &AlphaZeroNet,
    opt: &mut Optimizer,
    replay: &ReplayBuffer,
    device: Device,
    net_cfg: &NetConfig,
    cfg: &TrainConfig,
) -> Option<TrainMetrics> {
    debug_assert!(cfg.micro_batch_size > 0);

    let mut policy_loss_sum = Tensor::zeros([], (Kind::Float, device));
    let mut value_loss_sum = Tensor::zeros([], (Kind::Float, device));
    let mut micro_batches = 0usize;

    for _ in 0..cfg.train_steps {
        let (states, policies, values) = replay.sample(cfg.batch_size)?;
        let batch = states.size()[0] as usize;

        let states = states
            .view([-1, net_cfg.input_channels, net_cfg.height, net_cfg.width])
            .to_device(device);
        let policies = policies.to_device(device);
        let values = values.to_device(device);

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

            policy_loss_sum += &policy_loss;
            value_loss_sum += &value_loss;
            micro_batches += 1;
        }

        opt.step();
        opt.zero_grad();
    }

    if micro_batches == 0 {
        return None;
    }
    let n = micro_batches as f64;
    Some(TrainMetrics {
        policy_loss: (&policy_loss_sum / n).double_value(&[]),
        value_loss: (&value_loss_sum / n).double_value(&[]),
        train_steps: cfg.train_steps,
    })
}

use super::{ModelSpec, RawNetworkOutput, RawValueOutput, ValueHeadSpec};
use tch::{nn, Tensor};

struct ResBlock {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
}

impl ResBlock {
    fn new(p: &nn::Path, channels: i64) -> Self {
        let cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        Self {
            conv1: nn::conv2d(p / "conv1", channels, channels, 3, cfg),
            bn1: nn::batch_norm2d(p / "bn1", channels, Default::default()),
            conv2: nn::conv2d(p / "conv2", channels, channels, 3, cfg),
            bn2: nn::batch_norm2d(p / "bn2", channels, Default::default()),
        }
    }

    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let ys = xs
            .apply(&self.conv1)
            .apply_t(&self.bn1, train)
            .relu()
            .apply(&self.conv2)
            .apply_t(&self.bn2, train);
        (xs + ys).relu()
    }
}

/// Basic residual trunk with dense policy and value heads.
pub struct ClassicResidualNet {
    conv_in: nn::Conv2D,
    bn_in: nn::BatchNorm,
    blocks: Vec<ResBlock>,
    policy_conv: nn::Conv2D,
    policy_bn: nn::BatchNorm,
    policy_fc: nn::Linear,
    value_conv: nn::Conv2D,
    value_bn: nn::BatchNorm,
    value_fc1: nn::Linear,
    value_fc2: nn::Linear,
}

impl ClassicResidualNet {
    pub fn new(p: &nn::Path, spec: &ModelSpec) -> Self {
        let [input_channels, height, width] = spec.state_shape();
        let (blocks, channels) = match spec {
            ModelSpec::Connect4Residual {
                blocks, channels, ..
            }
            | ModelSpec::ChessClassic { blocks, channels } => (*blocks, *channels),
            ModelSpec::ChessSe { .. } => {
                unreachable!("classic residual network requires a basic model")
            }
        };
        let value_head = spec.value_head();
        let value_hidden = match value_head {
            ValueHeadSpec::Scalar { hidden } | ValueHeadSpec::Wdl { hidden } => hidden,
        };
        let value_outputs = match value_head {
            ValueHeadSpec::Scalar { .. } => 1,
            ValueHeadSpec::Wdl { .. } => 3,
        };
        let conv_cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let hw = height * width;
        Self {
            conv_in: nn::conv2d(p / "conv_in", input_channels, channels, 3, conv_cfg),
            bn_in: nn::batch_norm2d(p / "bn_in", channels, Default::default()),
            blocks: (0..blocks)
                .map(|i| ResBlock::new(&(p / "blocks" / i), channels))
                .collect(),
            policy_conv: nn::conv2d(p / "policy_conv", channels, 2, 1, Default::default()),
            policy_bn: nn::batch_norm2d(p / "policy_bn", 2, Default::default()),
            policy_fc: nn::linear(
                p / "policy_fc",
                2 * hw,
                spec.action_size() as i64,
                Default::default(),
            ),
            value_conv: nn::conv2d(p / "value_conv", channels, 1, 1, Default::default()),
            value_bn: nn::batch_norm2d(p / "value_bn", 1, Default::default()),
            value_fc1: nn::linear(p / "value_fc1", hw, value_hidden, Default::default()),
            value_fc2: nn::linear(
                p / "value_fc2",
                value_hidden,
                value_outputs,
                Default::default(),
            ),
        }
    }

    pub fn forward_t(&self, xs: &Tensor, train: bool) -> RawNetworkOutput {
        let mut x = xs.apply(&self.conv_in).apply_t(&self.bn_in, train).relu();
        for block in &self.blocks {
            x = block.forward_t(&x, train);
        }
        let policy = x
            .apply(&self.policy_conv)
            .apply_t(&self.policy_bn, train)
            .relu()
            .flatten(1, -1)
            .apply(&self.policy_fc);
        let value_logits = x
            .apply(&self.value_conv)
            .apply_t(&self.value_bn, train)
            .relu()
            .flatten(1, -1)
            .apply(&self.value_fc1)
            .relu()
            .apply(&self.value_fc2);
        let value = if value_logits.size()[1] == 1 {
            RawValueOutput::Scalar(value_logits.tanh())
        }
        else {
            RawValueOutput::WdlLogits(value_logits)
        };
        RawNetworkOutput {
            policy_logits: policy,
            value,
        }
    }
}

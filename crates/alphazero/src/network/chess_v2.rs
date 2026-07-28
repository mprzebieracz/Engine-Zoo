use super::{ModelSpec, RawNetworkOutput, RawValueOutput, ValueHeadSpec};
use tch::{nn, Tensor};

struct SeResBlock {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
    se_reduce: nn::Linear,
    se_expand: nn::Linear,
}

impl SeResBlock {
    fn new(p: &nn::Path, channels: i64, se_hidden: i64) -> Self {
        let conv = nn::ConvConfig {
            padding: 1,
            bias: false,
            ..Default::default()
        };
        Self {
            conv1: nn::conv2d(p / "conv1", channels, channels, 3, conv),
            bn1: nn::batch_norm2d(p / "bn1", channels, Default::default()),
            conv2: nn::conv2d(p / "conv2", channels, channels, 3, conv),
            bn2: nn::batch_norm2d(p / "bn2", channels, Default::default()),
            se_reduce: nn::linear(p / "se_reduce", channels, se_hidden, Default::default()),
            se_expand: nn::linear(p / "se_expand", se_hidden, 2 * channels, Default::default()),
        }
    }

    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let y = xs
            .apply(&self.conv1)
            .apply_t(&self.bn1, train)
            .relu()
            .apply(&self.conv2)
            .apply_t(&self.bn2, train);
        let se = y
            .mean_dim(&[2_i64, 3][..], false, y.kind())
            .apply(&self.se_reduce)
            .relu()
            .apply(&self.se_expand);
        let scale = se
            .narrow(1, 0, xs.size()[1])
            .sigmoid()
            .unsqueeze(-1)
            .unsqueeze(-1);
        let bias = se
            .narrow(1, xs.size()[1], xs.size()[1])
            .unsqueeze(-1)
            .unsqueeze(-1);
        (xs + scale * y + bias).relu()
    }
}

/// Squeeze-excitation residual trunk with a convolutional policy head.
pub struct SeResidualNet {
    stem_conv: nn::Conv2D,
    stem_bn: nn::BatchNorm,
    blocks: Vec<SeResBlock>,
    policy_conv: nn::Conv2D,
    policy_bn: nn::BatchNorm,
    policy_out: nn::Conv2D,
    value_conv: nn::Conv2D,
    value_bn: nn::BatchNorm,
    value_fc1: nn::Linear,
    value_fc2: nn::Linear,
}

impl SeResidualNet {
    pub fn new(p: &nn::Path, spec: &ModelSpec) -> Self {
        let [input_channels, _, _] = spec.state_shape();
        let ModelSpec::ChessSe {
            blocks,
            channels,
            se_hidden,
            value_head,
            ..
        } = spec
        else {
            unreachable!("SE residual network requires a chess SE model")
        };
        let value_hidden = match value_head {
            ValueHeadSpec::Scalar { hidden } | ValueHeadSpec::Wdl { hidden } => *hidden,
        };
        let value_outputs = match value_head {
            ValueHeadSpec::Scalar { .. } => 1,
            ValueHeadSpec::Wdl { .. } => 3,
        };
        let no_bias_3x3 = nn::ConvConfig {
            padding: 1,
            bias: false,
            ..Default::default()
        };
        Self {
            stem_conv: nn::conv2d(p / "stem_conv", input_channels, *channels, 3, no_bias_3x3),
            stem_bn: nn::batch_norm2d(p / "stem_bn", *channels, Default::default()),
            blocks: (0..*blocks)
                .map(|i| SeResBlock::new(&(p / "blocks" / i), *channels, *se_hidden))
                .collect(),
            policy_conv: nn::conv2d(p / "policy_conv", *channels, *channels, 3, no_bias_3x3),
            policy_bn: nn::batch_norm2d(p / "policy_bn", *channels, Default::default()),
            policy_out: nn::conv2d(
                p / "policy_out",
                *channels,
                73,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ),
            value_conv: nn::conv2d(
                p / "value_conv",
                *channels,
                32,
                1,
                nn::ConvConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            value_bn: nn::batch_norm2d(p / "value_bn", 32, Default::default()),
            value_fc1: nn::linear(
                p / "value_fc1",
                32 * 8 * 8,
                value_hidden,
                Default::default(),
            ),
            value_fc2: nn::linear(
                p / "value_fc2",
                value_hidden,
                value_outputs,
                Default::default(),
            ),
        }
    }

    pub fn forward_t(&self, xs: &Tensor, train: bool) -> RawNetworkOutput {
        let mut x = xs
            .apply(&self.stem_conv)
            .apply_t(&self.stem_bn, train)
            .relu();
        for block in &self.blocks {
            x = block.forward_t(&x, train);
        }
        let policy = x
            .apply(&self.policy_conv)
            .apply_t(&self.policy_bn, train)
            .relu()
            .apply(&self.policy_out)
            .flatten(1, -1);
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

use super::ChessAzV2Config;
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
    fn new(p: &nn::Path) -> Self {
        let conv = nn::ConvConfig {
            padding: 1,
            bias: false,
            ..Default::default()
        };
        Self {
            conv1: nn::conv2d(
                p / "conv1",
                ChessAzV2Config::CHANNELS,
                ChessAzV2Config::CHANNELS,
                3,
                conv,
            ),
            bn1: nn::batch_norm2d(p / "bn1", ChessAzV2Config::CHANNELS, Default::default()),
            conv2: nn::conv2d(
                p / "conv2",
                ChessAzV2Config::CHANNELS,
                ChessAzV2Config::CHANNELS,
                3,
                conv,
            ),
            bn2: nn::batch_norm2d(p / "bn2", ChessAzV2Config::CHANNELS, Default::default()),
            se_reduce: nn::linear(
                p / "se_reduce",
                ChessAzV2Config::CHANNELS,
                ChessAzV2Config::SE_HIDDEN,
                Default::default(),
            ),
            se_expand: nn::linear(
                p / "se_expand",
                ChessAzV2Config::SE_HIDDEN,
                2 * ChessAzV2Config::CHANNELS,
                Default::default(),
            ),
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
            .narrow(1, 0, ChessAzV2Config::CHANNELS)
            .sigmoid()
            .unsqueeze(-1)
            .unsqueeze(-1);
        let bias = se
            .narrow(1, ChessAzV2Config::CHANNELS, ChessAzV2Config::CHANNELS)
            .unsqueeze(-1)
            .unsqueeze(-1);
        (xs + scale * y + bias).relu()
    }
}

/// Chess-only AlphaZero v2 network: 12 128-channel SE residual blocks, a
/// spatial 73-plane policy, and a WDL value head.
pub struct ChessAzV2Net {
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

impl ChessAzV2Net {
    pub fn new(p: &nn::Path, cfg: ChessAzV2Config) -> Self {
        cfg.validate().expect("invalid chess az v2 configuration");
        let no_bias_3x3 = nn::ConvConfig {
            padding: 1,
            bias: false,
            ..Default::default()
        };
        Self {
            stem_conv: nn::conv2d(
                p / "stem_conv",
                cfg.input_channels(),
                ChessAzV2Config::CHANNELS,
                3,
                no_bias_3x3,
            ),
            stem_bn: nn::batch_norm2d(p / "stem_bn", ChessAzV2Config::CHANNELS, Default::default()),
            blocks: (0..ChessAzV2Config::RESIDUAL_BLOCKS)
                .map(|i| SeResBlock::new(&(p / "blocks" / i)))
                .collect(),
            policy_conv: nn::conv2d(
                p / "policy_conv",
                ChessAzV2Config::CHANNELS,
                ChessAzV2Config::CHANNELS,
                3,
                no_bias_3x3,
            ),
            policy_bn: nn::batch_norm2d(
                p / "policy_bn",
                ChessAzV2Config::CHANNELS,
                Default::default(),
            ),
            policy_out: nn::conv2d(
                p / "policy_out",
                ChessAzV2Config::CHANNELS,
                ChessAzV2Config::POLICY_PLANES,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ),
            value_conv: nn::conv2d(
                p / "value_conv",
                ChessAzV2Config::CHANNELS,
                32,
                1,
                nn::ConvConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            value_bn: nn::batch_norm2d(p / "value_bn", 32, Default::default()),
            value_fc1: nn::linear(p / "value_fc1", 32 * 8 * 8, 128, Default::default()),
            value_fc2: nn::linear(p / "value_fc2", 128, 3, Default::default()),
        }
    }

    /// Returns spatial policy logits `[B, 73, 8, 8]` and WDL logits `[B, 3]`.
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
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
            .apply(&self.policy_out);
        let value = x
            .apply(&self.value_conv)
            .apply_t(&self.value_bn, train)
            .relu()
            .flatten(1, -1)
            .apply(&self.value_fc1)
            .relu()
            .apply(&self.value_fc2);
        (policy, value)
    }
}

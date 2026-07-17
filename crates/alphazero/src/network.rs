use super::representation::AlphaZeroRepresentation;
use engine_core::GameState;
use serde::{Deserialize, Serialize};
use tch::{nn, Kind, Tensor};

mod chess_v2;
mod legacy;

pub use chess_v2::ChessAzV2Net;
pub use legacy::LegacyAlphaZeroNet;

/// Backwards-compatible name for the legacy network.
pub type AlphaZeroNet = LegacyAlphaZeroNet;

/// Configuration for the original scalar-value AlphaZero network.
///
/// This format is retained so existing chess and Connect4 checkpoints remain
/// usable. New chess-v2 runs use [`ChessAzV2Config`] instead.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetConfig {
    pub input_channels: i64,
    pub height: i64,
    pub width: i64,
    pub num_res_blocks: i64,
    pub num_filters: i64,
    pub action_size: i64,
}

impl NetConfig {
    pub fn for_representation<G: GameState, R: AlphaZeroRepresentation<G>>(
        num_res_blocks: i64,
        num_filters: i64,
    ) -> Self {
        let [input_channels, height, width] = R::STATE_SHAPE.map(|size| size as i64);
        Self {
            input_channels,
            height,
            width,
            num_res_blocks,
            num_filters,
            action_size: R::ACTION_SIZE as i64,
        }
    }

    pub fn state_size(&self) -> usize {
        (self.input_channels * self.height * self.width) as usize
    }
}

/// Fixed configuration of the chess AlphaZero v2 network.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChessAzV2Config {
    #[serde(default = "default_history")]
    pub history: usize,
}

const fn default_history() -> usize {
    4
}

impl Default for ChessAzV2Config {
    fn default() -> Self {
        Self {
            history: default_history(),
        }
    }
}

impl ChessAzV2Config {
    pub const CHANNELS: i64 = 128;
    pub const RESIDUAL_BLOCKS: usize = 12;
    pub const SE_HIDDEN: i64 = 16;
    pub const POLICY_PLANES: i64 = 73;
    pub const ACTION_SIZE: i64 = 8 * 8 * Self::POLICY_PLANES;

    pub fn validate(self) -> anyhow::Result<()> {
        anyhow::ensure!(
            matches!(self.history, 1 | 4 | 8),
            "chess az v2 history must be one of 1, 4, or 8, got {}",
            self.history
        );
        Ok(())
    }

    pub fn input_channels(self) -> i64 {
        (14 * self.history + 7) as i64
    }
    pub fn state_size(self) -> usize {
        (self.input_channels() * 8 * 8) as usize
    }
}

/// The network format selected by a run. It is deliberately explicit: tensor
/// shapes are never used to guess which checkpoint format is being loaded.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "architecture", content = "config", rename_all = "kebab-case")]
pub enum NetworkConfig {
    Legacy(NetConfig),
    ChessAzV2(ChessAzV2Config),
}

impl NetworkConfig {
    pub fn state_size(&self) -> usize {
        match self {
            Self::Legacy(cfg) => cfg.state_size(),
            Self::ChessAzV2(cfg) => cfg.state_size(),
        }
    }

    pub fn state_shape(&self) -> [i64; 3] {
        match self {
            Self::Legacy(cfg) => [cfg.input_channels, cfg.height, cfg.width],
            Self::ChessAzV2(cfg) => [cfg.input_channels(), 8, 8],
        }
    }
}

pub enum NetworkOutput {
    Legacy { policy: Tensor, value: Tensor },
    ChessAzV2 { policy: Tensor, wdl: Tensor },
}

pub enum Network {
    Legacy(LegacyAlphaZeroNet),
    ChessAzV2(ChessAzV2Net),
}

impl Network {
    pub fn new(p: &nn::Path, cfg: &NetworkConfig) -> Self {
        match cfg {
            NetworkConfig::Legacy(cfg) => Self::Legacy(LegacyAlphaZeroNet::new(p, cfg)),
            NetworkConfig::ChessAzV2(cfg) => Self::ChessAzV2(ChessAzV2Net::new(p, *cfg)),
        }
    }

    pub fn forward_t(&self, xs: &Tensor, train: bool) -> NetworkOutput {
        match self {
            Self::Legacy(net) => {
                let (policy, value) = net.forward_t(xs, train);
                NetworkOutput::Legacy { policy, value }
            }
            Self::ChessAzV2(net) => {
                let (policy, wdl) = net.forward_t(xs, train);
                NetworkOutput::ChessAzV2 { policy, wdl }
            }
        }
    }
}

/// Converts `[win, draw, loss]` logits to the scalar value consumed by MCTS.
pub fn wdl_scalar(wdl_logits: &Tensor) -> Tensor {
    let probabilities = wdl_logits.softmax(1, Kind::Float);
    probabilities.narrow(1, 0, 1) - probabilities.narrow(1, 2, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Kind};

    #[test]
    fn chess_v2_shapes_and_wdl_value_are_valid() {
        let vs = nn::VarStore::new(Device::Cpu);
        let cfg = ChessAzV2Config::default();
        let net = ChessAzV2Net::new(&vs.root(), cfg);
        let input = Tensor::zeros([2, cfg.input_channels(), 8, 8], (Kind::Float, Device::Cpu));
        let (policy, wdl) = net.forward_t(&input, false);
        assert_eq!(policy.size(), [2, 73, 8, 8]);
        assert_eq!(wdl.size(), [2, 3]);
        let value = wdl_scalar(&wdl);
        assert_eq!(value.size(), [2, 1]);
        assert!(value.isfinite().all().int64_value(&[]) != 0);
    }

    #[test]
    fn chess_v2_accepts_supported_history_lengths() {
        for history in [1, 4, 8] {
            let cfg = ChessAzV2Config { history };
            cfg.validate().unwrap();
            assert_eq!(cfg.input_channels(), (14 * history + 7) as i64);
        }
        assert!(ChessAzV2Config { history: 2 }.validate().is_err());
    }
}

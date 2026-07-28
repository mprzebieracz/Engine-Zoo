use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tch::{nn, Kind, Tensor};

#[path = "network/legacy.rs"]
mod classic_residual;
#[path = "network/chess_v2.rs"]
mod se_residual;

pub use classic_residual::ClassicResidualNet;
pub use se_residual::SeResidualNet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GameKind {
    Connect4,
    Chess,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChessHistory {
    One,
    Four,
    Eight,
}

/// Stable identity for tensor-shape-compatible model data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelFingerprint(pub [u8; 32]);

impl ChessHistory {
    pub const fn as_usize(self) -> usize {
        match self {
            Self::One => 1,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "architecture", rename_all = "kebab-case")]
pub enum ModelSpec {
    Connect4Residual {
        blocks: usize,
        channels: i64,
        value_head: ValueHeadSpec,
    },
    ChessClassic {
        blocks: usize,
        channels: i64,
    },
    ChessSe {
        history: ChessHistory,
        blocks: usize,
        channels: i64,
        se_hidden: i64,
        value_head: ValueHeadSpec,
    },
}

impl ModelSpec {
    pub fn connect4_basic(blocks: usize, channels: i64) -> Self {
        Self::Connect4Residual {
            blocks,
            channels,
            value_head: ValueHeadSpec::Scalar { hidden: channels },
        }
    }

    pub fn chess_se(history: ChessHistory, value_head: ValueHeadSpec) -> Self {
        Self::ChessSe {
            history,
            blocks: 12,
            channels: 128,
            se_hidden: 16,
            value_head,
        }
    }

    pub fn chess_se_with_trunk(
        history: ChessHistory,
        blocks: usize,
        channels: i64,
        se_hidden: i64,
        value_head: ValueHeadSpec,
    ) -> Self {
        Self::ChessSe {
            history,
            blocks,
            channels,
            se_hidden,
            value_head,
        }
    }

    pub fn chess_classic(blocks: usize, channels: i64) -> Self {
        Self::ChessClassic { blocks, channels }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        match self {
            Self::Connect4Residual {
                blocks,
                channels,
                value_head,
            }
            | Self::ChessSe {
                blocks,
                channels,
                value_head,
                ..
            } => {
                positive("blocks", *blocks as i64)?;
                positive("channels", *channels)?;
                value_head.validate()?;
            }
            Self::ChessClassic { blocks, channels } => {
                positive("blocks", *blocks as i64)?;
                positive("channels", *channels)?;
            }
        }
        if let Self::ChessSe { se_hidden, .. } = self {
            positive("se hidden", *se_hidden)?;
        }
        Ok(())
    }

    pub const fn game(&self) -> GameKind {
        match self {
            Self::Connect4Residual { .. } => GameKind::Connect4,
            Self::ChessClassic { .. } | Self::ChessSe { .. } => GameKind::Chess,
        }
    }

    pub fn state_shape(&self) -> [i64; 3] {
        match self {
            Self::Connect4Residual { .. } => [1, 6, 7],
            Self::ChessClassic { .. } => [19, 8, 8],
            Self::ChessSe { history, .. } => [(14 * history.as_usize() + 7) as i64, 8, 8],
        }
    }

    pub const fn action_size(&self) -> usize {
        match self {
            Self::Connect4Residual { .. } => 7,
            Self::ChessClassic { .. } => 64 * 64 * 5,
            Self::ChessSe { .. } => 4672,
        }
    }

    pub fn value_head(&self) -> ValueHeadSpec {
        match self {
            Self::Connect4Residual { value_head, .. } | Self::ChessSe { value_head, .. } => {
                value_head.clone()
            }
            Self::ChessClassic { channels, .. } => ValueHeadSpec::Scalar { hidden: *channels },
        }
    }

    pub const fn chess_history(&self) -> Option<ChessHistory> {
        match self {
            Self::ChessSe { history, .. } => Some(*history),
            Self::Connect4Residual { .. } | Self::ChessClassic { .. } => None,
        }
    }

    pub const fn is_chess_classic(&self) -> bool {
        matches!(self, Self::ChessClassic { .. })
    }

    pub fn fingerprint(&self) -> ModelFingerprint {
        let bytes = serde_json::to_vec(self).expect("model specifications serialize");
        ModelFingerprint(Sha256::digest(bytes).into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ValueHeadSpec {
    Scalar { hidden: i64 },
    Wdl { hidden: i64 },
}

impl ValueHeadSpec {
    fn validate(&self) -> anyhow::Result<()> {
        positive(
            "value head hidden",
            match self {
                Self::Scalar { hidden } | Self::Wdl { hidden } => *hidden,
            },
        )
    }
}

fn positive(name: &str, value: i64) -> anyhow::Result<()> {
    anyhow::ensure!(value > 0, "{name} must be positive");
    Ok(())
}

#[derive(Debug)]
pub struct RawNetworkOutput {
    pub policy_logits: Tensor,
    pub value: RawValueOutput,
}

#[derive(Debug)]
pub enum RawValueOutput {
    Scalar(Tensor),
    WdlLogits(Tensor),
}

impl RawValueOutput {
    pub fn expected_value(&self) -> Tensor {
        match self {
            Self::Scalar(value) => value.shallow_clone(),
            Self::WdlLogits(logits) => {
                let probabilities = logits.softmax(1, Kind::Float);
                probabilities.narrow(1, 0, 1) - probabilities.narrow(1, 2, 1)
            }
        }
    }
}

pub enum Network {
    ClassicResidual(ClassicResidualNet),
    SeResidual(SeResidualNet),
}

impl Network {
    pub fn new(path: &nn::Path, spec: &ModelSpec) -> anyhow::Result<Self> {
        spec.validate()?;
        Ok(match spec {
            ModelSpec::Connect4Residual { .. } | ModelSpec::ChessClassic { .. } => {
                Self::ClassicResidual(ClassicResidualNet::new(path, spec))
            }
            ModelSpec::ChessSe { .. } => Self::SeResidual(SeResidualNet::new(path, spec)),
        })
    }

    pub fn forward_t(&self, states: &Tensor, train: bool) -> RawNetworkOutput {
        match self {
            Self::ClassicResidual(net) => net.forward_t(states, train),
            Self::SeResidual(net) => net.forward_t(states, train),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Kind};

    fn chess(value_head: ValueHeadSpec) -> ModelSpec {
        ModelSpec::chess_se_with_trunk(ChessHistory::Four, 1, 8, 2, value_head)
    }

    #[test]
    fn chess_outputs_are_flat_and_semantic() {
        for value_head in [
            ValueHeadSpec::Scalar { hidden: 8 },
            ValueHeadSpec::Wdl { hidden: 8 },
        ] {
            let spec = chess(value_head);
            let vs = nn::VarStore::new(Device::Cpu);
            let net = Network::new(&vs.root(), &spec).unwrap();
            let output = net.forward_t(
                &Tensor::zeros([2, 63, 8, 8], (Kind::Float, Device::Cpu)),
                false,
            );
            assert_eq!(output.policy_logits.size(), [2, 4672]);
            assert_eq!(output.value.expected_value().size(), [2, 1]);
        }
    }

    #[test]
    fn chess_classic_preserves_the_scalar_checkpoint_shape() {
        let spec = ModelSpec::chess_classic(1, 8);
        let vs = nn::VarStore::new(Device::Cpu);
        let net = Network::new(&vs.root(), &spec).unwrap();
        let output = net.forward_t(
            &Tensor::zeros([2, 19, 8, 8], (Kind::Float, Device::Cpu)),
            false,
        );
        assert_eq!(output.policy_logits.size(), [2, 20_480]);
        assert_eq!(output.value.expected_value().size(), [2, 1]);
        let names = vs.variables();
        for name in [
            "conv_in.weight",
            "blocks.0.conv1.weight",
            "policy_fc.weight",
            "value_fc2.weight",
        ] {
            assert!(
                names.contains_key(name),
                "missing classic checkpoint key {name}"
            );
        }
    }

    #[test]
    fn chess_se_preserves_the_canonical_checkpoint_tensor_names() {
        let spec = ModelSpec::chess_se_with_trunk(
            ChessHistory::Four,
            1,
            8,
            2,
            ValueHeadSpec::Wdl { hidden: 8 },
        );
        let vs = nn::VarStore::new(Device::Cpu);
        let _network = Network::new(&vs.root(), &spec).unwrap();
        let names = vs.variables();

        for name in [
            "stem_conv.weight",
            "blocks.0.conv1.weight",
            "blocks.0.se_reduce.weight",
            "policy_out.weight",
            "value_fc2.weight",
        ] {
            assert!(
                names.contains_key(name),
                "missing canonical checkpoint key {name}"
            );
        }
    }

    #[test]
    fn fingerprints_include_every_checkpoint_shape_field() {
        let a = ModelSpec::chess_se_with_trunk(
            ChessHistory::Four,
            4,
            32,
            8,
            ValueHeadSpec::Wdl { hidden: 16 },
        );
        let b = ModelSpec::chess_se_with_trunk(
            ChessHistory::Four,
            5,
            32,
            8,
            ValueHeadSpec::Wdl { hidden: 16 },
        );
        assert_eq!(a.fingerprint(), a.fingerprint());
        assert_ne!(a.fingerprint(), b.fingerprint());
    }
}

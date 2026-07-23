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
pub enum GameSpec {
    Connect4,
    Chess,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChessHistoryLength {
    One,
    Four,
    Eight,
}

/// Complete shape of the canonical chess squeeze-excitation trunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeTrunkSpec {
    pub blocks: usize,
    pub channels: i64,
    pub se_hidden: i64,
}

impl Default for SeTrunkSpec {
    fn default() -> Self {
        Self {
            blocks: 12,
            channels: 128,
            se_hidden: 16,
        }
    }
}

/// Stable identity for tensor-shape-compatible model data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelFingerprint(pub [u8; 32]);

impl ChessHistoryLength {
    pub const fn as_usize(self) -> usize {
        match self {
            Self::One => 1,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RepresentationSpec {
    Connect4Canonical,
    ChessClassic,
    ChessCanonical { history: ChessHistoryLength },
}

impl RepresentationSpec {
    pub fn state_shape(&self) -> [i64; 3] {
        match self {
            Self::Connect4Canonical => [1, 6, 7],
            Self::ChessClassic => [19, 8, 8],
            Self::ChessCanonical { history } => [(14 * history.as_usize() + 7) as i64, 8, 8],
        }
    }

    pub const fn action_size(&self) -> usize {
        match self {
            Self::Connect4Canonical => 7,
            Self::ChessClassic => 64 * 64 * 5,
            Self::ChessCanonical { .. } => 4672,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub game: GameSpec,
    pub representation: RepresentationSpec,
    pub network: NetworkSpec,
}

impl ModelSpec {
    pub fn connect4_basic(blocks: usize, channels: i64) -> Self {
        Self {
            game: GameSpec::Connect4,
            representation: RepresentationSpec::Connect4Canonical,
            network: NetworkSpec::Residual(ResidualNetworkConfig {
                trunk: ResidualTrunkConfig::Basic { blocks, channels },
                policy_head: PolicyHeadConfig::Dense { channels: 2 },
                value_head: ValueHeadConfig::Scalar { hidden: channels },
            }),
        }
    }

    pub fn chess_se(history: ChessHistoryLength, value_head: ValueHeadConfig) -> Self {
        Self::chess_se_with_trunk(history, SeTrunkSpec::default(), value_head)
    }

    pub fn chess_se_with_trunk(
        history: ChessHistoryLength,
        trunk: SeTrunkSpec,
        value_head: ValueHeadConfig,
    ) -> Self {
        Self {
            game: GameSpec::Chess,
            representation: RepresentationSpec::ChessCanonical { history },
            network: NetworkSpec::Residual(ResidualNetworkConfig {
                trunk: ResidualTrunkConfig::SqueezeExcitation {
                    blocks: trunk.blocks,
                    channels: trunk.channels,
                    se_hidden: trunk.se_hidden,
                },
                policy_head: PolicyHeadConfig::ConvolutionalPlanes { planes: 73 },
                value_head,
            }),
        }
    }

    pub fn chess_classic(blocks: usize, channels: i64) -> Self {
        Self {
            game: GameSpec::Chess,
            representation: RepresentationSpec::ChessClassic,
            network: NetworkSpec::Residual(ResidualNetworkConfig {
                trunk: ResidualTrunkConfig::Basic { blocks, channels },
                policy_head: PolicyHeadConfig::Dense { channels: 2 },
                value_head: ValueHeadConfig::Scalar { hidden: channels },
            }),
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        match (&self.game, &self.representation) {
            (GameSpec::Connect4, RepresentationSpec::Connect4Canonical)
            | (GameSpec::Chess, RepresentationSpec::ChessClassic)
            | (GameSpec::Chess, RepresentationSpec::ChessCanonical { .. }) => {}
            _ => anyhow::bail!("game and representation do not match"),
        }
        self.network.validate(&self.representation)
    }

    pub fn state_shape(&self) -> [i64; 3] {
        self.representation.state_shape()
    }

    pub const fn action_size(&self) -> usize {
        self.representation.action_size()
    }

    pub fn chess_history(&self) -> Option<ChessHistoryLength> {
        match self.representation {
            RepresentationSpec::ChessCanonical { history } => Some(history),
            RepresentationSpec::Connect4Canonical | RepresentationSpec::ChessClassic => None,
        }
    }

    pub const fn is_chess_classic(&self) -> bool {
        matches!(self.representation, RepresentationSpec::ChessClassic)
    }

    pub fn fingerprint(&self) -> ModelFingerprint {
        let bytes = serde_json::to_vec(self).expect("model specifications serialize");
        ModelFingerprint(Sha256::digest(bytes).into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "architecture", content = "config", rename_all = "kebab-case")]
pub enum NetworkSpec {
    Residual(ResidualNetworkConfig),
}

impl NetworkSpec {
    fn validate(&self, representation: &RepresentationSpec) -> anyhow::Result<()> {
        match self {
            Self::Residual(config) => config.validate(representation),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidualNetworkConfig {
    pub trunk: ResidualTrunkConfig,
    pub policy_head: PolicyHeadConfig,
    pub value_head: ValueHeadConfig,
}

impl ResidualNetworkConfig {
    fn validate(&self, representation: &RepresentationSpec) -> anyhow::Result<()> {
        self.trunk.validate()?;
        self.policy_head.validate()?;
        self.value_head.validate()?;
        match (representation, &self.trunk, &self.policy_head) {
            (
                RepresentationSpec::Connect4Canonical,
                ResidualTrunkConfig::Basic { .. },
                PolicyHeadConfig::Dense { .. },
            ) => {}
            (
                RepresentationSpec::ChessCanonical { .. },
                ResidualTrunkConfig::SqueezeExcitation { .. },
                PolicyHeadConfig::ConvolutionalPlanes { planes: 73 },
            ) => {}
            (
                RepresentationSpec::ChessClassic,
                ResidualTrunkConfig::Basic { channels, .. },
                PolicyHeadConfig::Dense { channels: 2 },
            ) if matches!(&self.value_head, ValueHeadConfig::Scalar { hidden } if *hidden == *channels) => {}
            (
                RepresentationSpec::ChessCanonical { .. },
                _,
                PolicyHeadConfig::ConvolutionalPlanes { planes },
            ) => anyhow::bail!("chess canonical policy requires exactly 73 planes, got {planes}"),
            (RepresentationSpec::ChessCanonical { .. }, _, _) => anyhow::bail!(
                "chess canonical representation requires an SE trunk and convolutional policy"
            ),
            (RepresentationSpec::ChessClassic, _, _) => anyhow::bail!(
                "chess classic representation requires the basic trunk, a two-channel dense policy, and a scalar value head matching the trunk width"
            ),
            (RepresentationSpec::Connect4Canonical, _, _) => anyhow::bail!(
                "connect4 canonical representation requires a basic trunk and dense policy"
            ),
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResidualTrunkConfig {
    Basic {
        blocks: usize,
        channels: i64,
    },
    SqueezeExcitation {
        blocks: usize,
        channels: i64,
        se_hidden: i64,
    },
}

impl ResidualTrunkConfig {
    fn validate(&self) -> anyhow::Result<()> {
        match self {
            Self::Basic { blocks, channels } => {
                positive("blocks", *blocks as i64).and(positive("channels", *channels))
            }
            Self::SqueezeExcitation {
                blocks,
                channels,
                se_hidden,
            } => positive("blocks", *blocks as i64)
                .and(positive("channels", *channels))
                .and(positive("se_hidden", *se_hidden)),
        }
    }

    pub const fn channels(&self) -> i64 {
        match self {
            Self::Basic { channels, .. } | Self::SqueezeExcitation { channels, .. } => *channels,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PolicyHeadConfig {
    Dense { channels: i64 },
    ConvolutionalPlanes { planes: i64 },
}

impl PolicyHeadConfig {
    fn validate(&self) -> anyhow::Result<()> {
        positive(
            "policy head channels",
            match self {
                Self::Dense { channels } => *channels,
                Self::ConvolutionalPlanes { planes } => *planes,
            },
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ValueHeadConfig {
    Scalar { hidden: i64 },
    Wdl { hidden: i64 },
}

impl ValueHeadConfig {
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
        let NetworkSpec::Residual(config) = &spec.network;
        Ok(match config.trunk {
            ResidualTrunkConfig::Basic { .. } => {
                Self::ClassicResidual(ClassicResidualNet::new(path, spec, config))
            }
            ResidualTrunkConfig::SqueezeExcitation { .. } => {
                Self::SeResidual(SeResidualNet::new(path, spec, config))
            }
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

    fn chess(value_head: ValueHeadConfig) -> ModelSpec {
        ModelSpec {
            game: GameSpec::Chess,
            representation: RepresentationSpec::ChessCanonical {
                history: ChessHistoryLength::Four,
            },
            network: NetworkSpec::Residual(ResidualNetworkConfig {
                trunk: ResidualTrunkConfig::SqueezeExcitation {
                    blocks: 1,
                    channels: 8,
                    se_hidden: 2,
                },
                policy_head: PolicyHeadConfig::ConvolutionalPlanes { planes: 73 },
                value_head,
            }),
        }
    }

    #[test]
    fn chess_outputs_are_flat_and_semantic() {
        for value_head in [
            ValueHeadConfig::Scalar { hidden: 8 },
            ValueHeadConfig::Wdl { hidden: 8 },
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
    fn rejects_invalid_representation_network_pair() {
        let mut spec = chess(ValueHeadConfig::Wdl { hidden: 8 });
        let NetworkSpec::Residual(network) = &mut spec.network;
        network.policy_head = PolicyHeadConfig::ConvolutionalPlanes { planes: 72 };
        assert!(spec.validate().is_err());
    }

    #[test]
    fn fingerprints_include_every_checkpoint_shape_field() {
        let a = ModelSpec::chess_se_with_trunk(
            ChessHistoryLength::Four,
            SeTrunkSpec {
                blocks: 4,
                channels: 32,
                se_hidden: 8,
            },
            ValueHeadConfig::Wdl { hidden: 16 },
        );
        let b = ModelSpec::chess_se_with_trunk(
            ChessHistoryLength::Four,
            SeTrunkSpec {
                blocks: 5,
                channels: 32,
                se_hidden: 8,
            },
            ValueHeadConfig::Wdl { hidden: 16 },
        );
        assert_eq!(a.fingerprint(), a.fingerprint());
        assert_ne!(a.fingerprint(), b.fingerprint());
    }
}

use crate::{BatcherConfig, InferencePrecision, ModelSpec, SelfPlayConfig, TrainConfig};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const EXPERIMENT_FORMAT_VERSION: u32 = 2;

/// Serializable duration used by immutable experiment files.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurationConfig {
    pub milliseconds: u64,
}

/// Selects the implementation used for self-play inference.
///
/// TensorRT modules are opt-in because they require the matching Torch-TensorRT
/// runtime to be available to LibTorch at process startup.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InferenceEngine {
    #[default]
    Native,
    TensorRtTorchScript,
}

impl DurationConfig {
    pub const fn as_duration(self) -> Duration {
        Duration::from_millis(self.milliseconds)
    }
}

/// Batching choices that affect inference throughput but not model semantics.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    pub engine: InferenceEngine,
    /// A TorchScript module compiled with Torch-TensorRT. Its `forward` method
    /// returns the policy logits and scalar value packed along dimension one.
    pub tensor_rt_module: Option<PathBuf>,
    pub precision: InferencePrecision,
    pub preferred_batch_size: usize,
    pub max_batch_size: usize,
    pub max_wait: DurationConfig,
    pub max_queued_states: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            engine: InferenceEngine::Native,
            tensor_rt_module: None,
            precision: InferencePrecision::Fp32,
            preferred_batch_size: 32,
            max_batch_size: 256,
            max_wait: DurationConfig { milliseconds: 2 },
            max_queued_states: 4096,
        }
    }
}

impl InferenceConfig {
    fn validate(&self) -> Result<()> {
        if self.engine == InferenceEngine::TensorRtTorchScript {
            ensure!(
                self.tensor_rt_module.is_some(),
                "tensor-rt-torch-script inference requires tensor_rt_module"
            );
        }
        BatcherConfig {
            preferred_batch_size: self.preferred_batch_size,
            max_batch_size: self.max_batch_size,
            max_wait: self.max_wait.as_duration(),
            max_queued_states: self.max_queued_states,
        }
        .validate()?;
        Ok(())
    }

    pub fn batcher_config(&self) -> BatcherConfig {
        BatcherConfig {
            preferred_batch_size: self.preferred_batch_size,
            max_batch_size: self.max_batch_size,
            max_wait: self.max_wait.as_duration(),
            max_queued_states: self.max_queued_states,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ReplayConfig {
    pub capacity: usize,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self { capacity: 500_000 }
    }
}

impl ReplayConfig {
    fn validate(&self) -> Result<()> {
        ensure!(self.capacity > 0, "replay capacity must be positive");
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub format_version: u32,
    pub model: ModelSpec,
    #[serde(default)]
    pub self_play: SelfPlayConfig,
    #[serde(default)]
    pub replay: ReplayConfig,
    #[serde(default)]
    pub training: TrainConfig,
    #[serde(default)]
    pub inference: InferenceConfig,
    pub seed: u64,
}

impl ExperimentConfig {
    pub fn read_toml(path: &Path) -> Result<Self> {
        let config: Self = toml::from_str(&std::fs::read_to_string(path)?)?;
        config.validate()?;
        Ok(config)
    }

    pub fn to_toml(&self) -> Result<String> {
        self.validate()?;
        Ok(toml::to_string_pretty(self)?)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.format_version == EXPERIMENT_FORMAT_VERSION,
            "unsupported experiment format version {}",
            self.format_version
        );
        self.model.validate()?;
        self.self_play.validate()?;
        self.replay.validate()?;
        self.training.validate()?;
        self.inference.validate()
    }

    pub fn fingerprint(&self) -> crate::ModelFingerprint {
        self.model.fingerprint()
    }

    pub(crate) fn upgrade_from_v1(mut self) -> Result<Self> {
        ensure!(
            self.format_version == 1,
            "expected experiment format version 1, got {}",
            self.format_version
        );
        self.format_version = EXPERIMENT_FORMAT_VERSION;
        Ok(self)
    }
}

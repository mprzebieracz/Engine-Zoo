use crate::{ModelSpec, SelfPlayConfig, TrainConfig};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

pub const EXPERIMENT_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    pub self_play: SelfPlayConfig,
    pub replay: ReplayConfig,
    pub training: TrainConfig,
    pub seed: u64,
}

impl ExperimentConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.format_version == EXPERIMENT_FORMAT_VERSION,
            "unsupported experiment format version {}",
            self.format_version
        );
        self.model.validate()?;
        self.self_play.validate()?;
        self.replay.validate()?;
        self.training.validate()
    }
}

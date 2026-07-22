use serde::{Deserialize, Serialize};

pub const STATE_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunState {
    pub format_version: u32,
    pub iteration: u64,
    pub model_generation: u64,
    pub global_step: u64,
    pub latest_checkpoint: Option<String>,
    pub replay_sample_count: usize,
    pub optimizer_moments_restored: bool,
}

impl Default for RunState {
    fn default() -> Self {
        Self {
            format_version: STATE_FORMAT_VERSION,
            iteration: 0,
            model_generation: 0,
            global_step: 0,
            latest_checkpoint: None,
            replay_sample_count: 0,
            // tch does not expose portable optimizer serialization.
            optimizer_moments_restored: false,
        }
    }
}

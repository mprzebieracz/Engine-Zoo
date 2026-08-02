use serde::{Deserialize, Serialize};

pub const STATE_FORMAT_VERSION: u32 = 3;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointIdentity {
    pub generation: u64,
    pub relative_path: String,
    pub sha256: String,
}

/// How much mutable training state was restored when this process started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResumeKind {
    #[default]
    Fresh,
    WeightsOnly,
    Full,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunState {
    pub format_version: u32,
    pub iteration: u64,
    pub model_generation: u64,
    pub global_step: u64,
    /// Total games emitted by this run. Unlike an iteration-local index, this
    /// is never reused as a replay game ID.
    pub total_games_generated: u64,
    pub checkpoint_identity: Option<CheckpointIdentity>,
    pub replay_sample_count: usize,
    pub optimizer_moments_restored: bool,
    pub resume_kind: ResumeKind,
}

impl Default for RunState {
    fn default() -> Self {
        Self {
            format_version: STATE_FORMAT_VERSION,
            iteration: 0,
            model_generation: 0,
            global_step: 0,
            total_games_generated: 0,
            checkpoint_identity: None,
            replay_sample_count: 0,
            // tch does not expose portable optimizer serialization.
            optimizer_moments_restored: false,
            resume_kind: ResumeKind::Fresh,
        }
    }
}

impl RunState {
    /// Replay and optimizer state are not persisted yet, so a checkpoint can
    /// only restore network weights.
    pub fn mark_weights_only_resume(&mut self) {
        self.replay_sample_count = 0;
        self.optimizer_moments_restored = false;
        self.resume_kind = ResumeKind::WeightsOnly;
    }

    pub(crate) fn from_v1(previous: RunStateV1) -> (Self, Option<String>) {
        // Version 1 reused game IDs on each iteration, so their historical
        // total cannot be recovered. Start the new global sequence at zero.
        let resume_kind = if previous.latest_checkpoint.is_some() {
            ResumeKind::WeightsOnly
        }
        else {
            ResumeKind::Fresh
        };
        let legacy_checkpoint = previous.latest_checkpoint;

        (
            Self {
                format_version: STATE_FORMAT_VERSION,
                iteration: previous.iteration,
                model_generation: previous.model_generation,
                global_step: previous.global_step,
                total_games_generated: 0,
                checkpoint_identity: None,
                // Neither payload existed in the old format, regardless of its
                // counter values.
                replay_sample_count: 0,
                optimizer_moments_restored: false,
                resume_kind,
            },
            legacy_checkpoint,
        )
    }

    pub(crate) fn from_v2(previous: RunStateV2) -> (Self, Option<String>) {
        let legacy_checkpoint = previous.latest_checkpoint;

        (
            Self {
                format_version: STATE_FORMAT_VERSION,
                iteration: previous.iteration,
                model_generation: previous.model_generation,
                global_step: previous.global_step,
                total_games_generated: previous.total_games_generated,
                checkpoint_identity: None,
                replay_sample_count: previous.replay_sample_count,
                optimizer_moments_restored: previous.optimizer_moments_restored,
                resume_kind: previous.resume_kind,
            },
            legacy_checkpoint,
        )
    }
}

#[derive(Deserialize)]
pub(crate) struct RunStateV2 {
    pub format_version: u32,
    pub iteration: u64,
    pub model_generation: u64,
    pub global_step: u64,
    pub total_games_generated: u64,
    pub latest_checkpoint: Option<String>,
    pub replay_sample_count: usize,
    pub optimizer_moments_restored: bool,
    pub resume_kind: ResumeKind,
}

#[derive(Deserialize)]
pub(crate) struct RunStateV1 {
    pub format_version: u32,
    pub iteration: u64,
    pub model_generation: u64,
    pub global_step: u64,
    pub latest_checkpoint: Option<String>,
    #[serde(rename = "replay_sample_count")]
    pub _replay_sample_count: usize,
    #[serde(rename = "optimizer_moments_restored")]
    pub _optimizer_moments_restored: bool,
}

//! Immutable experiment specifications and crash-safe run-directory state.

mod config;
mod migration;
mod run_dir;
mod state;

pub use config::{
    DurationConfig, ExperimentConfig, InferenceConfig, ReplayConfig, EXPERIMENT_FORMAT_VERSION,
};
pub use run_dir::RunDir;
pub use state::{ResumeKind, RunState, STATE_FORMAT_VERSION};

#[cfg(test)]
mod tests;

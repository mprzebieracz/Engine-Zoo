use super::*;
use crate::{ChessHistoryLength, ModelSpec, ValueHeadConfig};
use std::fs;

fn config() -> ExperimentConfig {
    ExperimentConfig {
        format_version: EXPERIMENT_FORMAT_VERSION,
        model: ModelSpec::chess_se(
            ChessHistoryLength::One,
            ValueHeadConfig::Scalar { hidden: 8 },
        ),
        self_play: crate::SelfPlayConfig::default(),
        replay: ReplayConfig { capacity: 8 },
        training: crate::TrainConfig {
            train_steps: 1,
            ..Default::default()
        },
        seed: 3,
    }
}

#[test]
fn run_writes_immutable_experiment_and_atomic_latest_state() {
    let root = std::env::temp_dir().join(format!("engine-zoo-experiment-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let (run, _, mut state) = RunDir::open_or_create(&root, config).unwrap();
    run.write_latest(&mut state, |path| {
        fs::write(path, b"weights")?;
        Ok(())
    })
    .unwrap();
    assert_eq!(fs::read(run.latest_path()).unwrap(), b"weights");
    assert_eq!(run.latest_checkpoint(&state), Some(run.latest_path()));
    assert!(!run.latest_path().with_extension("safetensors.tmp").exists());
    let _ = fs::remove_dir_all(root);
}

use super::*;
use crate::{ChessHistoryLength, ModelSpec, ValueHeadConfig};
use std::fs;
use tch::{nn, Device};

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
        inference: InferenceConfig::default(),
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
    assert!(run.experiment_path().is_file());
    assert_eq!(run.latest_checkpoint(&state), Some(run.latest_path()));
    assert!(!run
        .latest_path()
        .with_file_name("latest.tmp.safetensors")
        .exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn experiment_toml_round_trips_and_carries_inference_settings() {
    let expected = config();
    let parsed: ExperimentConfig = toml::from_str(&expected.to_toml().unwrap()).unwrap();
    assert_eq!(parsed.inference, expected.inference);
    assert_eq!(parsed.fingerprint(), expected.fingerprint());
}

#[test]
fn initialize_refuses_a_nonempty_directory() {
    let root = std::env::temp_dir().join(format!("engine-zoo-init-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("unrelated.txt"), "keep").unwrap();
    assert!(RunDir::initialize(&root, config()).is_err());
    assert!(root.join("unrelated.txt").is_file());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn latest_checkpoint_round_trips_a_real_cpu_model_as_safetensors() {
    let root = std::env::temp_dir().join(format!(
        "engine-zoo-checkpoint-roundtrip-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    let (run, _, mut state) = RunDir::open_or_create(&root, config).unwrap();
    let spec = ModelSpec::connect4_basic(1, 4);
    let original = nn::VarStore::new(Device::Cpu);
    let _network = crate::Network::new(&original.root(), &spec).unwrap();
    let expected = original.variables();

    run.write_latest(&mut state, |path| Ok(original.save(path)?))
        .unwrap();
    assert!(run.latest_path().is_file());
    assert!(!run
        .latest_path()
        .with_file_name("latest.tmp.safetensors")
        .exists());

    let mut restored = nn::VarStore::new(Device::Cpu);
    let _network = crate::Network::new(&restored.root(), &spec).unwrap();
    restored.load(run.latest_path()).unwrap();
    let actual = restored.variables();
    assert_eq!(expected.len(), actual.len());
    for (name, expected) in expected {
        let actual = actual.get(&name).unwrap();
        let difference = (expected - actual).abs().max().double_value(&[]);
        assert!(difference <= 1e-7, "tensor {name} changed by {difference}");
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn version_one_state_migrates_with_a_new_global_game_sequence() {
    let root =
        std::env::temp_dir().join(format!("engine-zoo-state-migration-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let (run, _, _) = RunDir::open_or_create(&root, config).unwrap();
    fs::write(
        run.state_path(),
        r#"{
            "format_version": 1,
            "iteration": 4,
            "model_generation": 4,
            "global_step": 12,
            "latest_checkpoint": "checkpoints/latest.safetensors",
            "replay_sample_count": 99,
            "optimizer_moments_restored": false
        }"#,
    )
    .unwrap();

    let (_, _, state) = RunDir::open_or_create(&root, config).unwrap();
    assert_eq!(state.format_version, STATE_FORMAT_VERSION);
    assert_eq!(state.total_games_generated, 0);
    assert_eq!(state.resume_kind, ResumeKind::WeightsOnly);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn weights_only_resume_never_claims_replay_or_optimizer_state() {
    let mut state = RunState {
        replay_sample_count: 42,
        optimizer_moments_restored: true,
        ..RunState::default()
    };
    state.mark_weights_only_resume();
    assert_eq!(state.resume_kind, ResumeKind::WeightsOnly);
    assert_eq!(state.replay_sample_count, 0);
    assert!(!state.optimizer_moments_restored);
}

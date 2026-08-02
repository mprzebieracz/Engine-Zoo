use super::*;
use crate::{ChessHistory, InferenceEngine, ModelSpec, ValueHeadSpec};
use std::fs;
use std::path::Path;
use tch::{nn, Device};

fn config() -> ExperimentConfig {
    ExperimentConfig {
        format_version: EXPERIMENT_FORMAT_VERSION,
        model: ModelSpec::chess_se(ChessHistory::One, ValueHeadSpec::Scalar { hidden: 8 }),
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
    assert_eq!(
        run.latest_checkpoint(&state),
        Some(run.archived_checkpoint_path(0))
    );
    assert!(!run
        .latest_path()
        .with_file_name("latest.tmp.safetensors")
        .exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn run_writes_periodic_archive_atomically_without_changing_latest_state() {
    let root = std::env::temp_dir().join(format!("engine-zoo-archive-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let (run, _, mut state) = RunDir::open_or_create(&root, config).unwrap();
    run.write_latest(&mut state, |path| {
        fs::write(path, b"latest")?;
        Ok(())
    })
    .unwrap();
    let latest_state = state.checkpoint_identity.clone();

    run.write_archived(25, |path| {
        fs::write(path, b"archive")?;
        Ok(())
    })
    .unwrap();

    assert_eq!(
        fs::read(run.archived_checkpoint_path(25)).unwrap(),
        b"archive"
    );
    assert_eq!(state.checkpoint_identity, latest_state);
    assert!(!run
        .archived_checkpoint_path(25)
        .with_file_name("generation-000025.tmp.safetensors")
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
fn version_three_experiment_aliases_are_read_without_overwriting() {
    let root = std::env::temp_dir().join(format!(
        "engine-zoo-config-migration-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let path = root.join("experiment.toml");
    let mut old = config();
    old.format_version = 3;
    old.inference.engine = InferenceEngine::TensorRtRaw;
    old.inference.compiled_artifact = Some("artifact.engine".into());
    let old = toml::to_string_pretty(&old)
        .unwrap()
        .replace("compiled_artifact", "tensor_rt_module");
    fs::write(&path, &old).unwrap();

    let upgraded = ExperimentConfig::read_toml(&path).unwrap();

    assert_eq!(upgraded.format_version, EXPERIMENT_FORMAT_VERSION);
    assert_eq!(
        upgraded.inference.compiled_artifact.as_deref(),
        Some(Path::new("artifact.engine"))
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), old);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn checked_in_experiments_use_the_closed_model_specification() {
    let experiments = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../experiments");
    for path in fs::read_dir(experiments).unwrap() {
        let path = path.unwrap().path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("toml") {
            ExperimentConfig::read_toml(&path)
                .unwrap_or_else(|error| panic!("{} does not parse: {error:#}", path.display()));
        }
    }
}

#[test]
fn tensor_rt_inference_requires_a_module_path() {
    let mut config = config();
    config.inference.engine = InferenceEngine::TensorRtTorchScript;
    assert!(config.validate().is_err());
}

#[test]
fn fp16_host_staging_requires_native_fp16_inference() {
    let mut config = config();
    config.inference.fp16_host_staging = true;
    assert!(config.validate().is_err());

    config.inference.precision = crate::InferencePrecision::Fp16;
    assert!(config.validate().is_ok());

    config.inference.engine = InferenceEngine::TensorRtTorchScript;
    config.inference.compiled_artifact = Some("model.ts".into());
    assert!(config.validate().is_err());
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
    drop(run);

    let (_, _, state) = RunDir::open_or_create(&root, config).unwrap();
    assert_eq!(state.format_version, STATE_FORMAT_VERSION);
    assert_eq!(state.total_games_generated, 0);
    assert_eq!(state.resume_kind, ResumeKind::WeightsOnly);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn version_two_checkpoint_is_migrated_to_an_immutable_generation() {
    let root = std::env::temp_dir().join(format!(
        "engine-zoo-state-v2-migration-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    let (run, _, _) = RunDir::open_or_create(&root, config).unwrap();
    fs::write(run.latest_path(), b"legacy-weights").unwrap();
    fs::write(
        run.state_path(),
        r#"{
            "format_version": 2,
            "iteration": 4,
            "model_generation": 4,
            "global_step": 12,
            "total_games_generated": 8,
            "latest_checkpoint": "checkpoints/latest.safetensors",
            "replay_sample_count": 0,
            "optimizer_moments_restored": false,
            "resume_kind": "weights-only"
        }"#,
    )
    .unwrap();
    drop(run);

    let (run, _, state) = RunDir::open_writer(&root).unwrap();
    let identity = state.checkpoint_identity.unwrap();

    assert_eq!(identity.generation, 4);
    assert_eq!(
        identity.relative_path,
        "checkpoints/generation-000004.safetensors"
    );
    assert_eq!(
        fs::read(run.archived_checkpoint_path(4)).unwrap(),
        b"legacy-weights"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn second_run_writer_is_rejected() {
    let root = std::env::temp_dir().join(format!("engine-zoo-writer-lock-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let (writer, _, _) = RunDir::open_or_create(&root, config).unwrap();

    let error = match RunDir::open_writer(&root) {
        Ok(_) => panic!("second writer unexpectedly acquired the run lock"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("run is already locked by another writer"));

    drop(writer);
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

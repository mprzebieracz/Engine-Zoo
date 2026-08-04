use super::*;
use std::fs;

fn temporary(name: &str) -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("engine-model-runtime-{name}-{}", unique_suffix()));
    fs::create_dir_all(&path).unwrap();
    path
}

fn config(root: &Path) -> RepositoryConfig {
    RepositoryConfig {
        root: root.to_path_buf(),
        ..RepositoryConfig::default()
    }
}

fn write_experiment(path: &Path) {
    fs::write(
        path,
        r#"format_version = 4
seed = 1
[model]
architecture = "chess-classic"
blocks = 1
channels = 8
"#,
    )
    .unwrap();
}

#[test]
fn configuration_discovers_local_file_and_resolves_relative_cuda_paths() {
    let root = temporary("config");
    fs::write(
        root.join(REPOSITORY_CONFIG_FILE),
        "[paths]\npermanent_models = 'saved-models'\n",
    )
    .unwrap();
    let nested = root.join("nested");
    fs::create_dir(&nested).unwrap();
    let found = RepositoryConfig::discover(&nested).unwrap();
    assert_eq!(found.permanent_models_dir(), root.join("saved-models"));
    assert_eq!(
        found.cuda_tensorrt_artifact_cache_dir(),
        root.join("artifacts/cuda-tensorrt")
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn configuration_reads_cuda_tensorrt_batch_profile_keys() {
    let runtime: CudaTorchRuntime = toml::from_str(
        "cuda_tensorrt_batch_min = 2\ncuda_tensorrt_batch_optimal = 8\ncuda_tensorrt_batch_max = 16\n",
    )
    .unwrap();
    assert_eq!(runtime.cuda_tensorrt_batch_min, 2);
    assert_eq!(runtime.cuda_tensorrt_batch_optimal, 8);
    assert_eq!(runtime.cuda_tensorrt_batch_max, 16);
}

#[test]
fn promotion_is_immutable_and_validates_checksum() {
    let root = temporary("promotion");
    let checkpoint = root.join("candidate.safetensors");
    fs::write(&checkpoint, b"weights").unwrap();
    let store = ModelStore::new(config(&root));
    let promoted = store
        .promote(
            "candidate",
            &checkpoint,
            ModelSpec::chess_classic(1, 8),
            None,
        )
        .unwrap();
    assert_eq!(
        promoted.checkpoint_sha256,
        sha256_file(&checkpoint).unwrap()
    );
    assert!(store
        .promote(
            "candidate",
            &checkpoint,
            ModelSpec::chess_classic(1, 8),
            None
        )
        .is_err());
    fs::write(&promoted.checkpoint, b"changed").unwrap();
    assert!(store
        .resolve(&ModelSelector::Permanent("candidate".into()))
        .is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn checkpoint_infers_training_run_experiment_and_run_aliases() {
    let root = temporary("selector");
    let run = root.join("runs/chess");
    let checkpoints = run.join("checkpoints");
    fs::create_dir_all(&checkpoints).unwrap();
    write_experiment(&run.join("experiment.toml"));
    let checkpoint = checkpoints.join("latest.safetensors");
    fs::write(&checkpoint, b"weights").unwrap();
    let store = ModelStore::new(config(&root));
    let direct = store
        .resolve(&ModelSelector::Checkpoint {
            checkpoint: checkpoint.clone(),
            experiment: None,
        })
        .unwrap();
    assert_eq!(direct.experiment, Some(run.join("experiment.toml")));
    let alias = store
        .resolve(&ModelSelector::RunAlias {
            run_dir: run.clone(),
            alias: "latest".into(),
        })
        .unwrap();
    assert_eq!(alias.checkpoint, checkpoint);

    let generation = checkpoints.join("generation-000123.safetensors");
    fs::write(&generation, b"older-weights").unwrap();
    assert_eq!(
        resolve_run_alias(&run, "generation-000123.safetensors").unwrap(),
        generation
    );
    assert!(resolve_run_alias(&run, "generation-../escape.safetensors").is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn backend_preferences_only_fall_back_for_unavailable_backends() {
    assert_eq!(
        BackendPreference::Auto.candidates(),
        &[
            RuntimeBackend::RawTensorrt,
            RuntimeBackend::TorchTensorrt,
            RuntimeBackend::Native
        ]
    );
    assert_eq!(
        BackendPreference::Tensorrt.candidates(),
        &[RuntimeBackend::RawTensorrt, RuntimeBackend::TorchTensorrt]
    );
    assert_eq!(
        BackendPreference::Auto
            .select_available(|backend| backend == RuntimeBackend::Native)
            .unwrap()
            .actual,
        RuntimeBackend::Native
    );
    assert!(BackendPreference::RawTensorrt
        .select_available(|_| false)
        .is_err());
}

#[test]
fn selector_prefers_an_existing_checkpoint_path_over_a_model_name() {
    let root = temporary("selector-precedence");
    let checkpoint = root.join("same-name");
    fs::write(&checkpoint, b"weights").unwrap();
    assert!(matches!(
        ModelSelector::from_input(checkpoint.display().to_string(), None),
        ModelSelector::Checkpoint { .. }
    ));
    assert!(matches!(
        ModelSelector::from_input("same-name", None),
        ModelSelector::Permanent(_)
    ));
    fs::remove_dir_all(root).unwrap();
}

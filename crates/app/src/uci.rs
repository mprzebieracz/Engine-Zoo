use alphazero::{
    artifact::{CompiledBackendKind, TensorRtBuildPrecision},
    ChessAlphaZeroEngine, ExperimentConfig, GameKind, InferenceEngine, InferencePrecision,
    InferenceSource, RunDir,
};
use anyhow::{Context, Result};
use engine_core::game::GameState;
use engine_core::notation::GameNotation;
use engine_model_runtime::tensor_rt::{TensorRtBatchShapes, TensorRtCompiler};
use engine_model_runtime::{
    BackendPreference, ModelSelector, ModelStore, RepositoryConfig, ResolvedModel, RuntimeBackend,
};
use games::ChessGame;
use search::{PuctConfig, SearchBudget, SearchConfig, SearchRequest};
use std::path::{Path, PathBuf};
use std::process::Command;
use tch::Device;

mod protocol;

pub use protocol::{parse_command, UciCommand, STARTPOS};

#[derive(Clone, Debug)]
pub struct Settings {
    pub model: String,
    /// Experiment metadata for a standalone checkpoint outside a training run.
    pub experiment: Option<PathBuf>,
    pub run_dir: PathBuf,
    pub simulations: usize,
    pub device: Device,
    pub temperature: f32,
    pub threads: usize,
    pub opening_plies: usize,
    pub backend: BackendPreference,
    /// Optional precompiled Torch-TensorRT TorchScript module. When set, the
    /// engine uses TensorRT inference regardless of the run's configured
    /// engine; when None, the engine forces native inference so it can serve
    /// checkpoints from TensorRT training runs without their compiled module.
    pub tensor_rt_module: Option<PathBuf>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model: "best".into(),
            experiment: None,
            run_dir: "data/runs/chess".into(),
            simulations: 800,
            device: Device::cuda_if_available(),
            temperature: 0.0,
            threads: 1,
            opening_plies: 0,
            backend: BackendPreference::Auto,
            tensor_rt_module: None,
        }
    }
}

pub struct ChessUciEngine {
    pub settings: Settings,
    pub game: ChessGame,
    engine: Option<ChessAlphaZeroEngine>,
    position_moves: Vec<String>,
}

impl Default for ChessUciEngine {
    fn default() -> Self {
        Self::new(Settings::default())
    }
}

impl ChessUciEngine {
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            game: ChessGame::default(),
            engine: None,
            position_moves: Vec::new(),
        }
    }

    pub fn set_position(&mut self, fen: Option<&str>, moves: &[String]) -> Result<()> {
        let mut game = match fen {
            Some(fen) => ChessGame::from_fen(fen)?,
            None => ChessGame::default(),
        };
        for text in moves {
            let mv = games::chess::ChessUciNotation
                .parse_move(&game.position(), text)
                .ok_or_else(|| anyhow::anyhow!("illegal chess move {text}"))?;
            game.play(mv);
        }
        self.game = game;
        self.position_moves = moves.to_vec();
        Ok(())
    }

    pub fn invalidate_model(&mut self) {
        self.engine = None;
    }

    /// Starts a fresh game without reloading unchanged network weights.
    /// MCTS clears its per-search tree before every search.
    pub fn new_game(&mut self) {
        self.game = ChessGame::default();
        self.position_moves.clear();
    }

    fn ensure_model(&mut self) -> Result<()> {
        if self.engine.is_some() {
            return Ok(());
        }
        let (model, mut cfg) = load_config_and_model(&self.settings)?;
        let engine = match &self.settings.tensor_rt_module {
            Some(module) => {
                anyhow::ensure!(
                    module.is_file(),
                    "TensorRtModule path is not a file: {}",
                    module.display()
                );
                validate_compiled_checkpoint(module, &model.checkpoint_sha256)?;

                cfg.inference.engine = InferenceEngine::TensorRtTorchScript;
                ChessAlphaZeroEngine::open(
                    model.model,
                    InferenceSource::TensorRtTorchScript(module),
                    self.settings.device,
                    &cfg.inference,
                    application_search(),
                )?
            }
            None => open_preferred_model(&model, &mut cfg, &self.settings)?,
        };
        self.engine = Some(engine);
        Ok(())
    }

    pub fn bestmove(&mut self, simulations: Option<usize>) -> Result<String> {
        if self.game.is_terminal() {
            return Ok("0000".into());
        }
        self.ensure_model()?;
        let simulations = simulations.unwrap_or(self.settings.simulations).max(1);
        let sampled = self.position_moves.len() < self.settings.opening_plies;
        let request = if sampled {
            SearchRequest {
                mode: engine_core::agent::PolicyMode::Explore,
                budget: SearchBudget::Puct { simulations },
            }
        }
        else {
            SearchRequest::deterministic_puct(simulations)
        };
        let mv = self
            .engine
            .as_mut()
            .expect("engine was loaded before move selection")
            .select_move(&self.game, request)?;

        Ok(games::chess::ChessUciNotation.format_move(&self.game.position(), mv))
    }
}

fn application_search() -> SearchConfig {
    SearchConfig::Puct(PuctConfig::analysis_default(1))
}

fn load_config_and_model(settings: &Settings) -> Result<(ResolvedModel, ExperimentConfig)> {
    let run_dir = &settings.run_dir;
    let repository = RepositoryConfig::discover(std::env::current_dir()?)?;
    let store = ModelStore::new(repository);
    let path = PathBuf::from(&settings.model);
    let (model, mut config) = if path.is_file() {
        let model = store.resolve(&ModelSelector::Checkpoint {
            checkpoint: path,
            experiment: settings.experiment.clone(),
        })?;
        let config = model
            .experiment
            .as_deref()
            .map(ExperimentConfig::read_toml)
            .transpose()?
            .unwrap_or_else(default_experiment);
        (model, config)
    }
    else if store.model_dir(&settings.model)?.is_dir() {
        (
            store.resolve(&ModelSelector::Permanent(settings.model.clone()))?,
            default_experiment(),
        )
    }
    else {
        let (run, config, state) = RunDir::open(run_dir)
            .with_context(|| format!("no experiment found at {}", run_dir.display()))?;
        anyhow::ensure!(
            config.model.game() == GameKind::Chess,
            "run is not a chess model"
        );
        let model =
            if settings.model.starts_with("ckpt_") || settings.model.starts_with("generation-") {
                store.resolve(&ModelSelector::RunAlias {
                    run_dir: run_dir.clone(),
                    alias: settings.model.clone(),
                })?
            }
            else {
                let checkpoint = match settings.model.as_str() {
                    "best" | "candidate" => checkpoint_path(&run, &settings.model),
                    "latest" => run
                        .latest_checkpoint(&state)
                        .or_else(|| existing_classic_checkpoint(run.root(), "best"))
                        .context("no latest checkpoint found")?,
                    name => run_dir.join(name),
                };
                store.resolve(&ModelSelector::Checkpoint {
                    checkpoint,
                    experiment: Some(run.experiment_path()),
                })?
            };
        (model, config)
    };
    anyhow::ensure!(
        model.model.game() == GameKind::Chess,
        "model is not a chess model"
    );
    config.model = model.model.clone();
    Ok((model, config))
}

fn default_experiment() -> ExperimentConfig {
    // Only `model` and application inference are used by UCI; permanent
    // models intentionally do not inherit mutable training-run settings.
    ExperimentConfig {
        format_version: alphazero::EXPERIMENT_FORMAT_VERSION,
        model: alphazero::ModelSpec::chess_classic(1, 8),
        self_play: Default::default(),
        replay: Default::default(),
        training: Default::default(),
        inference: Default::default(),
        seed: 0,
    }
}

fn open_preferred_model(
    model: &ResolvedModel,
    cfg: &mut ExperimentConfig,
    settings: &Settings,
) -> Result<ChessAlphaZeroEngine> {
    let repository = RepositoryConfig::discover(std::env::current_dir()?)?;
    for backend in settings.backend.candidates() {
        match backend {
            RuntimeBackend::Native => {
                cfg.inference.engine = InferenceEngine::Native;
                cfg.inference.compiled_artifact = None;
                cfg.inference.precision =
                    if matches!(settings.device, Device::Cuda(_)) && repository.runtime.fp16 {
                        InferencePrecision::Fp16
                    }
                    else {
                        InferencePrecision::Fp32
                    };
                return ChessAlphaZeroEngine::open(
                    model.model.clone(),
                    InferenceSource::Checkpoint(&model.checkpoint),
                    settings.device,
                    &cfg.inference,
                    application_search(),
                );
            }
            RuntimeBackend::RawTensorrt | RuntimeBackend::TorchTensorrt => {
                let Some((kind, artifact)) =
                    prepare_tensorrt(&repository, model, *backend, settings.device)?
                else {
                    continue;
                };
                cfg.inference.preferred_batch_size = repository.runtime.cuda_tensorrt_batch_optimal;
                cfg.inference.max_batch_size = repository.runtime.cuda_tensorrt_batch_max;
                cfg.inference.precision = if repository.runtime.fp16 {
                    InferencePrecision::Fp16
                }
                else {
                    InferencePrecision::Fp32
                };
                cfg.inference.tensor_rt_precision = Some(if repository.runtime.fp16 {
                    TensorRtBuildPrecision::Fp16
                }
                else {
                    TensorRtBuildPrecision::Fp32
                });
                cfg.inference.compiled_artifact = Some(artifact.clone());
                cfg.inference.engine = match kind {
                    CompiledBackendKind::TensorRtRaw => InferenceEngine::TensorRtRaw,
                    CompiledBackendKind::TensorRtTorchScript => {
                        InferenceEngine::TensorRtTorchScript
                    }
                };
                let source = match kind {
                    CompiledBackendKind::TensorRtRaw => InferenceSource::TensorRtEngine(&artifact),
                    CompiledBackendKind::TensorRtTorchScript => {
                        InferenceSource::TensorRtTorchScript(&artifact)
                    }
                };
                return ChessAlphaZeroEngine::open(
                    model.model.clone(),
                    source,
                    settings.device,
                    &cfg.inference,
                    application_search(),
                );
            }
        }
    }
    anyhow::bail!(
        "no usable backend for requested preference {:?}",
        settings.backend
    )
}

fn prepare_tensorrt(
    repository: &RepositoryConfig,
    model: &ResolvedModel,
    backend: RuntimeBackend,
    device: Device,
) -> Result<Option<(CompiledBackendKind, PathBuf)>> {
    if !matches!(device, Device::Cuda(_))
        || !tch::Cuda::is_available()
        || !matches!(
            repository.runtime.device.as_str(),
            "auto" | "cuda" | "cuda:0"
        )
    {
        return Ok(None);
    }
    let (kind, python, export_python, script, label) = match compiler_for(repository, backend) {
        Some(compiler) => compiler,
        None => return Ok(None),
    };
    let precision = if repository.runtime.fp16 {
        TensorRtBuildPrecision::Fp16
    }
    else {
        TensorRtBuildPrecision::Fp32
    };
    let compiler = TensorRtCompiler::application(
        python,
        export_python,
        script,
        TensorRtBatchShapes {
            min: repository.runtime.cuda_tensorrt_batch_min,
            optimal: repository.runtime.cuda_tensorrt_batch_optimal,
            max: repository.runtime.cuda_tensorrt_batch_max,
        },
        precision,
    );
    let output = repository
        .cuda_tensorrt_artifact_cache_dir()
        .join(&model.checkpoint_sha256)
        .join(format!("{label}.engine"));
    compiler.ensure_artifact(
        &model.checkpoint,
        &model.model,
        kind,
        &output,
        Device::Cuda(0),
    )?;
    Ok(Some((kind, output)))
}

fn compiler_for(
    repository: &RepositoryConfig,
    backend: RuntimeBackend,
) -> Option<(
    CompiledBackendKind,
    PathBuf,
    Option<PathBuf>,
    PathBuf,
    &'static str,
)> {
    let configured = |path: Option<&Path>| {
        path.map(|path| {
            if path.is_absolute() {
                path.to_path_buf()
            }
            else {
                repository.root().join(path)
            }
        })
        .filter(|path| path.is_file())
    };
    let imports = |python: &Path, modules: &str| {
        Command::new(python)
            .arg("-c")
            .arg(format!("import {modules}"))
            .output()
            .is_ok_and(|output| output.status.success())
    };
    match backend {
        RuntimeBackend::RawTensorrt => {
            #[cfg(feature = "raw-tensorrt")]
            {
                let python = configured(repository.toolchain.tensorrt_python.as_deref())?;
                let export_python = configured(repository.toolchain.cuda_torch_python.as_deref())?;
                let script = repository.root().join("scripts/compile_tensorrt_raw.py");
                (script.is_file()
                    && imports(&python, "tensorrt")
                    && imports(&export_python, "torch"))
                .then_some((
                    CompiledBackendKind::TensorRtRaw,
                    python,
                    Some(export_python),
                    script,
                    "raw-tensorrt",
                ))
            }
            #[cfg(not(feature = "raw-tensorrt"))]
            {
                None
            }
        }
        RuntimeBackend::TorchTensorrt => {
            let python = configured(repository.toolchain.cuda_torch_python.as_deref())?;
            let script = repository.root().join("scripts/compile_tensorrt.py");
            (script.is_file() && imports(&python, "torch, torch_tensorrt")).then_some((
                CompiledBackendKind::TensorRtTorchScript,
                python,
                None,
                script,
                "torch-tensorrt",
            ))
        }
        RuntimeBackend::Native => None,
    }
}

fn validate_compiled_checkpoint(artifact: &Path, checkpoint_sha256: &str) -> Result<()> {
    let manifest = alphazero::artifact::read_manifest(artifact)?;

    validate_checkpoint_digest(&manifest.checkpoint_sha256, checkpoint_sha256)?;

    Ok(())
}

fn validate_checkpoint_digest(
    compiled: &str,
    current: &str,
) -> std::result::Result<(), alphazero::RecompileRequired> {
    if compiled != current {
        return Err(alphazero::RecompileRequired {
            reason: alphazero::RecompileReason::StaleIdentity("checkpoint_sha256"),
        });
    }

    Ok(())
}

fn checkpoint_path(run: &RunDir, name: &str) -> PathBuf {
    let current = match name {
        "best" => run.best_path(),
        "candidate" => run.candidate_path(),
        _ => unreachable!("only named checkpoint aliases are supported"),
    };
    if current.is_file() {
        current
    }
    else {
        run.root().join(format!("{name}.safetensors"))
    }
}

fn existing_classic_checkpoint(run_dir: &Path, name: &str) -> Option<PathBuf> {
    let path = run_dir.join(format!("{name}.safetensors"));
    path.is_file().then_some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_game_resets_the_position() {
        let mut engine = ChessUciEngine::default();
        engine
            .set_position(None, &["e2e4".into(), "e7e5".into()])
            .unwrap();
        engine.new_game();
        assert_eq!(engine.game.to_string(), ChessGame::default().to_string());
    }

    #[test]
    fn set_position_is_transactional() {
        let mut engine = ChessUciEngine::default();
        engine.set_position(None, &["e2e4".into()]).unwrap();
        let before = engine.game.to_string();

        let error = engine
            .set_position(None, &["d2d4".into(), "d2d3".into()])
            .unwrap_err();

        assert!(error.to_string().contains("illegal chess move d2d3"));
        assert_eq!(engine.game.to_string(), before);
        assert_eq!(engine.position_moves, ["e2e4"]);
    }

    #[test]
    fn set_position_replays_from_fen_and_rejects_invalid_fen() {
        let mut engine = ChessUciEngine::default();
        engine
            .set_position(Some("8/8/8/8/8/8/4K3/7k w - - 0 1"), &["e2f3".into()])
            .unwrap();
        assert!(games::chess::ChessUciNotation
            .parse_move(&engine.game.position(), "h1g1")
            .is_some());

        let before = engine.game.to_string();
        assert!(engine.set_position(Some("not a fen"), &[]).is_err());
        assert_eq!(engine.game.to_string(), before);
    }

    #[test]
    fn compiled_checkpoint_mismatch_requires_recompilation() {
        let error = validate_checkpoint_digest("compiled", "current").unwrap_err();

        assert_eq!(
            error.reason,
            alphazero::RecompileReason::StaleIdentity("checkpoint_sha256")
        );
        assert!(validate_checkpoint_digest("current", "current").is_ok());
    }
}

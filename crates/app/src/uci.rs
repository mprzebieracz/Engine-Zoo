use alphazero::{
    ChessAlphaZeroEngine, ExperimentConfig, GameKind, InferenceEngine, InferenceSource, RunDir,
};
use anyhow::{Context, Result};
use engine_core::game::GameState;
use engine_core::notation::GameNotation;
use games::ChessGame;
use search::{PuctConfig, SearchBudget, SearchConfig, SearchRequest};
use std::path::{Path, PathBuf};
use tch::Device;

mod protocol;

pub use protocol::{parse_command, UciCommand, STARTPOS};

#[derive(Clone, Debug)]
pub struct Settings {
    pub model: String,
    pub run_dir: PathBuf,
    pub simulations: usize,
    pub device: Device,
    pub temperature: f32,
    pub threads: usize,
    pub opening_plies: usize,
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
            run_dir: "data/runs/chess".into(),
            simulations: 800,
            device: Device::cuda_if_available(),
            temperature: 0.0,
            threads: 1,
            opening_plies: 0,
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
        let (weights, mut cfg, state) =
            load_config_and_model(&self.settings.run_dir, &self.settings.model)?;
        let engine = match &self.settings.tensor_rt_module {
            Some(module) => {
                anyhow::ensure!(
                    module.is_file(),
                    "TensorRtModule path is not a file: {}",
                    module.display()
                );
                validate_compiled_checkpoint(module, &state)?;

                cfg.inference.engine = InferenceEngine::TensorRtTorchScript;
                ChessAlphaZeroEngine::open(
                    cfg.model,
                    InferenceSource::TensorRtTorchScript(module),
                    self.settings.device,
                    &cfg.inference,
                    application_search(),
                )?
            }
            None => {
                // A run configured for TensorRT stores tensor_rt_module in its
                // experiment; force native inference so arenas can serve those
                // checkpoints from a plain safetensors file without the
                // compiled module.
                cfg.inference.engine = InferenceEngine::Native;
                cfg.inference.compiled_artifact = None;
                ChessAlphaZeroEngine::open(
                    cfg.model,
                    InferenceSource::Checkpoint(&weights),
                    self.settings.device,
                    &cfg.inference,
                    application_search(),
                )?
            }
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

fn load_config_and_model(
    run_dir: &Path,
    model: &str,
) -> Result<(PathBuf, ExperimentConfig, alphazero::RunState)> {
    let (run, config, state) = RunDir::open(run_dir)
        .with_context(|| format!("no experiment found at {}", run_dir.display()))?;
    anyhow::ensure!(
        config.model.game() == GameKind::Chess,
        "run is not a chess model"
    );
    let path = PathBuf::from(model);
    let weights = if path.exists() {
        path
    }
    else {
        match model {
            "best" => checkpoint_path(&run, "best"),
            "candidate" => checkpoint_path(&run, "candidate"),
            "latest" => run
                .latest_checkpoint(&state)
                .or_else(|| existing_classic_checkpoint(run.root(), "best"))
                .context("no latest checkpoint found")?,
            name if name.starts_with("ckpt_") => run_dir.join("checkpoints").join(name),
            name => run_dir.join(name),
        }
    };
    anyhow::ensure!(
        weights.is_file(),
        "checkpoint not found: {}",
        weights.display()
    );
    Ok((weights, config, state))
}

fn validate_compiled_checkpoint(artifact: &Path, state: &alphazero::RunState) -> Result<()> {
    let checkpoint = state
        .checkpoint_identity
        .as_ref()
        .context("run has no current checkpoint identity")?;
    let manifest = alphazero::artifact::read_manifest(artifact)?;

    validate_checkpoint_digest(&manifest.checkpoint_sha256, &checkpoint.sha256)?;

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

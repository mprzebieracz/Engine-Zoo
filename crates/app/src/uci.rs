use alphazero::representation::{ChessAzRepresentation, ChessClassicRepresentation};
use alphazero::ChessRepetitionRules;
use alphazero::{
    Batcher, BatcherConfig, ChessHistoryLength, ExperimentConfig, GameSpec, Mcts, ModelSpec,
    RepresentedEvaluator, RunDir, SearchConfig,
};
use anyhow::{Context, Result};
use engine_core::agent::PolicyMode;
use engine_core::game::GameState;
use engine_core::notation::GameNotation;
use games::ChessGame;
use search::NoExtraRules;
use std::path::{Path, PathBuf};
use std::time::Duration;
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
        }
    }
}

pub struct ChessUciEngine {
    pub settings: Settings,
    pub game: ChessGame,
    batcher: Option<Batcher>,
    loaded: Option<LoadedModel>,
    position_moves: Vec<String>,
}

struct LoadedModel {
    model: ModelSpec,
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
            batcher: None,
            loaded: None,
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
        self.batcher = None;
        self.loaded = None;
    }

    /// Starts a fresh game without reloading unchanged network weights.
    /// MCTS clears its per-search tree before every search.
    pub fn new_game(&mut self) {
        self.game = ChessGame::default();
        self.position_moves.clear();
    }

    fn ensure_model(&mut self) -> Result<()> {
        if self.batcher.is_some() {
            return Ok(());
        }
        let (weights, cfg) = load_config_and_model(&self.settings.run_dir, &self.settings.model)?;
        let batcher = Batcher::new_with_model(
            cfg.model.clone(),
            &weights,
            self.settings.device,
            batcher_config(),
        )?;
        self.batcher = Some(batcher);
        self.loaded = Some(LoadedModel { model: cfg.model });
        Ok(())
    }

    pub fn bestmove(&mut self, simulations: Option<usize>) -> Result<String> {
        if self.game.is_terminal() {
            return Ok("0000".into());
        }
        self.ensure_model()?;
        let simulations = simulations.unwrap_or(self.settings.simulations).max(1);
        let evaluator = self.batcher.as_ref().expect("model loaded").client();
        let model = &self.loaded.as_ref().expect("model loaded").model;
        let sampled = self.position_moves.len() < self.settings.opening_plies;
        let bestmove = match model.chess_history() {
            Some(ChessHistoryLength::One) => {
                chess_action::<1>(evaluator, &self.game, simulations, sampled)
            }
            Some(ChessHistoryLength::Four) => {
                chess_action::<4>(evaluator, &self.game, simulations, sampled)
            }
            Some(ChessHistoryLength::Eight) => {
                chess_action::<8>(evaluator, &self.game, simulations, sampled)
            }
            None if model.is_chess_classic() => {
                classic_chess_action(evaluator, &self.game, simulations, sampled)
            }
            None => anyhow::bail!("run is not a chess model"),
        }?;
        Ok(bestmove)
    }
}

fn classic_chess_action(
    evaluator: alphazero::BatcherClient,
    game: &ChessGame,
    simulations: usize,
    sampled: bool,
) -> Result<String> {
    let mut mcts = Mcts::<_, _, _>::new(
        RepresentedEvaluator::new(ChessClassicRepresentation, evaluator),
        puct_search(simulations),
        NoExtraRules,
    );
    let result = mcts.search(
        &game.position(),
        (),
        if sampled {
            PolicyMode::Explore
        }
        else {
            PolicyMode::Deterministic
        },
    )?;
    let mv = if sampled {
        result.sample_move(&mut rand::rng())
    }
    else {
        result.best_move()
    };
    Ok(games::chess::ChessUciNotation.format_move(&game.position(), mv))
}

fn chess_action<const HISTORY: usize>(
    evaluator: alphazero::BatcherClient,
    game: &ChessGame,
    simulations: usize,
    sampled: bool,
) -> Result<String> {
    let mut mcts = Mcts::<_, _, _>::new(
        RepresentedEvaluator::new(ChessAzRepresentation::<HISTORY>, evaluator),
        puct_search(simulations),
        ChessRepetitionRules,
    );
    let result = mcts.search(
        &alphazero::representation::ChessAzState::from_game(game),
        game.repetition_context(),
        if sampled {
            PolicyMode::Explore
        }
        else {
            PolicyMode::Deterministic
        },
    )?;
    let mv = if sampled {
        result.sample_move(&mut rand::rng())
    }
    else {
        result.best_move()
    };
    Ok(games::chess::ChessUciNotation.format_move(&game.position(), mv))
}

fn load_config_and_model(run_dir: &Path, model: &str) -> Result<(PathBuf, ExperimentConfig)> {
    let (run, config, state) = RunDir::open_or_create(run_dir, || {
        panic!("no experiment found at {}", run_dir.display())
    })?;
    anyhow::ensure!(
        config.model.game == GameSpec::Chess,
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
    Ok((weights, config))
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

fn puct_search(simulations: usize) -> SearchConfig {
    let mut search = search::PuctConfig::default();
    search.common.simulations = simulations.max(1);
    search.common.leaf_batch_size = 1;
    search.root_noise = None;
    SearchConfig::Puct(search)
}

fn batcher_config() -> BatcherConfig {
    BatcherConfig {
        preferred_batch_size: 1,
        max_batch_size: 256,
        max_wait: Duration::from_millis(1),
        max_queued_states: 4096,
    }
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
}

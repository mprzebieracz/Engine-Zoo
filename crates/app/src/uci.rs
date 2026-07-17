use alphazero::representation::{ChessAzRepresentation, ChessV1Representation};
use alphazero::{
    Batcher, Mcts, MctsConfig, MctsVariant, ModelConfig, RepresentedEvaluator, RunDir,
};
use alphazero::ChessRepetitionRules;
use anyhow::{Context, Result};
use engine_core::agent::PolicyMode;
use engine_core::game::GameState;
use engine_core::notation::GameNotation;
use games::{ChessGame, ChessPosition};
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
    model: ModelConfig,
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
        let batcher = Batcher::new_with_network(
            &cfg.network_config(),
            &weights,
            self.settings.device,
            1,
            Duration::from_millis(1),
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
        match &self.loaded.as_ref().expect("model loaded").model {
            ModelConfig::ChessScalarAzV1(_) => {
                let mut mcts = Mcts::<ChessPosition, _, _>::new(
                    RepresentedEvaluator::new(ChessV1Representation, evaluator),
                    MctsConfig {
                        simulations,
                        eps: 0.0,
                        ..Default::default()
                    },
                    ChessRepetitionRules,
                );
                let sampled = self.position_moves.len() < self.settings.opening_plies;
                let result = mcts.search(
                    &self.game.position_state(),
                    self.game.repetition_context(),
                    if sampled {
                        PolicyMode::Explore
                    } else {
                        PolicyMode::Deterministic
                    },
                );
                let mv = if sampled {
                    result.sample_move(&mut rand::rng())
                } else {
                    result.best_move()
                };
                Ok(games::chess::ChessUciNotation.format_move(&self.game.position(), mv))
            }
            ModelConfig::ChessAzV2(v2) => {
                let sampled = self.position_moves.len() < self.settings.opening_plies;
                let bestmove = match v2.history {
                    1 => v2_action::<1>(evaluator, &self.game, simulations, sampled),
                    4 => v2_action::<4>(evaluator, &self.game, simulations, sampled),
                    8 => v2_action::<8>(evaluator, &self.game, simulations, sampled),
                    history => anyhow::bail!("unsupported chess history {history}"),
                }?;
                Ok(bestmove)
            }
            ModelConfig::Connect4ScalarAz(_) => anyhow::bail!("run is not a chess model"),
        }
    }
}

fn v2_action<const HISTORY: usize>(
    evaluator: alphazero::BatcherClient,
    game: &ChessGame,
    simulations: usize,
    sampled: bool,
) -> Result<String> {
    let mut mcts = Mcts::<_, _, _>::new(
        RepresentedEvaluator::new(ChessAzRepresentation::<HISTORY>, evaluator),
        MctsConfig {
            simulations,
            variant: MctsVariant::Puct,
            eps: 0.0,
            ..Default::default()
        },
        ChessRepetitionRules,
    );
    let result = mcts.search(
        &game.history_state::<HISTORY>(),
        game.repetition_context(),
        if sampled {
            PolicyMode::Explore
        } else {
            PolicyMode::Deterministic
        },
    );
    let mv = if sampled {
        result.sample_move(&mut rand::rng())
    } else {
        result.best_move()
    };
    Ok(games::chess::ChessUciNotation.format_move(&game.position(), mv))
}

fn load_config_and_model(
    run_dir: &Path,
    model: &str,
) -> Result<(PathBuf, alphazero::RunConfig)> {
    let config_path = run_dir.join("config.json");
    let config = alphazero::RunConfig::parse_json(
        &std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?,
    )
    .with_context(|| format!("parsing {}", config_path.display()))?;
    config.validate()?;
    anyhow::ensure!(
        config.model.game_name() == "chess",
        "run is for {}, not chess",
        config.model.game_name()
    );
    let path = PathBuf::from(model);
    let weights = if path.exists() {
        path
    } else {
        match model {
            "best" => run_dir.join("best.safetensors"),
            "candidate" => run_dir.join("candidate.safetensors"),
            "latest" => RunDir::open_or_create(run_dir, || config.clone())?
                .0
                .latest_checkpoint()
                .map(|(_, p)| p)
                .context("no numbered checkpoint found")?,
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

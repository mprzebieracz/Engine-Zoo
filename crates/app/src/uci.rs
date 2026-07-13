use algorithms::alphazero::{Batcher, Mcts, MctsConfig, MctsVariant, RunArchitecture, RunDir};
use anyhow::{Context, Result};
use engine_core::agent::PolicyMode;
use engine_core::game::Game;
use games::{ChessAzGame, ChessGame};
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
    v2_game: Option<ChessAzGameKind>,
    position_base: ChessGame,
    position_moves: Vec<String>,
}

struct LoadedModel {
    architecture: RunArchitecture,
}

/// The v2 feature history is part of the search state, whereas the UCI
/// protocol's `position` command is kept in the legacy `ChessGame` format.
enum ChessAzGameKind {
    H1(Box<ChessAzGame<1>>),
    H4(Box<ChessAzGame<4>>),
    H8(Box<ChessAzGame<8>>),
}

impl ChessAzGameKind {
    fn new(history: usize, game: &ChessGame) -> Result<Self> {
        match history {
            1 => Ok(Self::H1(Box::new(ChessAzGame::from_game(game)))),
            4 => Ok(Self::H4(Box::new(ChessAzGame::from_game(game)))),
            8 => Ok(Self::H8(Box::new(ChessAzGame::from_game(game)))),
            _ => anyhow::bail!("chess-az-v2 history must be 1, 4, or 8, got {history}"),
        }
    }

    fn parse_move(&self, mv: &str) -> Option<engine_core::game::Action> {
        match self {
            Self::H1(state) => state.parse_move(mv),
            Self::H4(state) => state.parse_move(mv),
            Self::H8(state) => state.parse_move(mv),
        }
    }

    fn step(&mut self, action: engine_core::game::Action) {
        match self {
            Self::H1(state) => state.step(action),
            Self::H4(state) => state.step(action),
            Self::H8(state) => state.step(action),
        }
    }

    fn format_action(&self, action: engine_core::game::Action) -> String {
        match self {
            Self::H1(state) => state.format_action(action),
            Self::H4(state) => state.format_action(action),
            Self::H8(state) => state.format_action(action),
        }
    }
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
            v2_game: None,
            position_base: ChessGame::default(),
            position_moves: Vec::new(),
        }
    }

    pub fn set_position(&mut self, fen: Option<&str>, moves: &[String]) -> Result<()> {
        let mut game = match fen {
            Some(fen) => ChessGame::from_fen(fen)?,
            None => ChessGame::default(),
        };
        let position_base = game.clone();
        for mv in moves {
            let action = game
                .parse_move(mv)
                .ok_or_else(|| anyhow::anyhow!("illegal chess move {mv}"))?;
            game.step(action);
        }
        self.game = game;
        self.position_base = position_base;
        self.position_moves = moves.to_vec();
        self.v2_game = None;
        Ok(())
    }

    pub fn invalidate_model(&mut self) {
        self.batcher = None;
        self.loaded = None;
        self.v2_game = None;
    }

    /// Starts a fresh game without reloading unchanged network weights.
    /// MCTS clears its per-search tree before every search.
    pub fn new_game(&mut self) {
        self.game = ChessGame::default();
        self.position_base = ChessGame::default();
        self.position_moves.clear();
        self.v2_game = None;
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
        self.loaded = Some(LoadedModel {
            architecture: cfg.architecture,
        });
        Ok(())
    }

    pub fn bestmove(&mut self, simulations: Option<usize>) -> Result<String> {
        if self.game.is_terminal() {
            return Ok("0000".into());
        }
        self.ensure_model()?;
        let simulations = simulations.unwrap_or(self.settings.simulations).max(1);
        let evaluator = self.batcher.as_ref().expect("model loaded").client();
        match &self.loaded.as_ref().expect("model loaded").architecture {
            RunArchitecture::Legacy => {
                let result = Mcts::new(
                    evaluator,
                    MctsConfig {
                        simulations,
                        eps: 0.0,
                        ..Default::default()
                    },
                )
                .search_with_mode(
                    &self.game,
                    if self.position_moves.len() < self.settings.opening_plies {
                        PolicyMode::Explore
                    }
                    else {
                        PolicyMode::Deterministic
                    },
                );
                let action = if self.position_moves.len() < self.settings.opening_plies {
                    result.sample_action(&mut rand::rng())
                }
                else {
                    result.best_action()
                };
                Ok(self.game.format_action(action))
            }
            RunArchitecture::ChessAzV2(v2) => {
                if self.v2_game.is_none() {
                    let mut game = ChessAzGameKind::new(v2.history, &self.position_base)?;
                    for mv in &self.position_moves {
                        let action = game
                            .parse_move(mv)
                            .ok_or_else(|| anyhow::anyhow!("illegal v2 chess move {mv}"))?;
                        game.step(action);
                    }
                    self.v2_game = Some(game);
                }
                let game = self.v2_game.as_mut().expect("v2 game initialized");
                let action = match game {
                    ChessAzGameKind::H1(game) => v2_action(
                        evaluator,
                        game,
                        simulations,
                        self.position_moves.len() < self.settings.opening_plies,
                    ),
                    ChessAzGameKind::H4(game) => v2_action(
                        evaluator,
                        game,
                        simulations,
                        self.position_moves.len() < self.settings.opening_plies,
                    ),
                    ChessAzGameKind::H8(game) => v2_action(
                        evaluator,
                        game,
                        simulations,
                        self.position_moves.len() < self.settings.opening_plies,
                    ),
                }?;
                Ok(game.format_action(action))
            }
        }
    }
}

fn v2_action<const HISTORY: usize>(
    evaluator: algorithms::alphazero::BatcherClient,
    game: &ChessAzGame<HISTORY>,
    simulations: usize,
    sampled: bool,
) -> Result<engine_core::game::Action> {
    let root_hash = game.position().hash();
    let result = Mcts::new(
        evaluator,
        MctsConfig {
            simulations,
            variant: MctsVariant::Puct,
            eps: 0.0,
            ..Default::default()
        },
    )
    .search_with_repetitions_mode(
        &game.search_state(),
        |hash| game.repetitions_before_root(hash, root_hash),
        if sampled {
            PolicyMode::Explore
        }
        else {
            PolicyMode::Deterministic
        },
    );
    Ok(if sampled {
        result.sample_action(&mut rand::rng())
    }
    else {
        result.best_action()
    })
}

fn load_config_and_model(
    run_dir: &Path,
    model: &str,
) -> Result<(PathBuf, algorithms::alphazero::RunConfig)> {
    let config_path = run_dir.join("config.json");
    let config: algorithms::alphazero::RunConfig = serde_json::from_str(
        &std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?,
    )
    .with_context(|| format!("parsing {}", config_path.display()))?;
    anyhow::ensure!(
        config.game == ChessGame::NAME,
        "run is for {}, not chess",
        config.game
    );
    let path = PathBuf::from(model);
    let weights = if path.exists() {
        path
    }
    else {
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
        assert!(engine.game.parse_move("h1g1").is_some());

        let before = engine.game.to_string();
        assert!(engine.set_position(Some("not a fen"), &[]).is_err());
        assert_eq!(engine.game.to_string(), before);
    }
}

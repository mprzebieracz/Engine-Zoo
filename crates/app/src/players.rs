use alphazero::representation::{
    AlphaZeroRepresentation, ChessV1Representation, Connect4AzRepresentation,
};
use alphazero::{
    Batcher, Mcts, MctsConfig, MctsVariant, NetworkConfig, RepresentedEvaluator,
};
use search::SearchRules;
use anyhow::{Context, Result};
use engine_core::agent::{Agent, PolicyMode};
use engine_core::game::GameState;
use engine_core::notation::GameNotation;
use games::chess::notation::ChessUciNotation;
use games::{ChessGame, ChessPosition, ChessRepetitionContext, Connect4, Connect4Notation};
use std::io::Write;
use std::path::Path;
use std::time::Duration;
use tch::Device;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentSpec {
    User,
    AlphaZero { model: String },
}

impl AgentSpec {
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("user") || s.eq_ignore_ascii_case("human") {
            return Ok(AgentSpec::User);
        }

        for prefix in ["alphazero:", "az:"] {
            if let Some(model) = s.strip_prefix(prefix) {
                anyhow::ensure!(!model.is_empty(), "missing model in agent spec {s}");
                return Ok(AgentSpec::AlphaZero {
                    model: model.to_owned(),
                });
            }
        }

        if s.eq_ignore_ascii_case("alphazero") || s.eq_ignore_ascii_case("az") {
            return Ok(AgentSpec::AlphaZero {
                model: "best".into(),
            });
        }

        anyhow::bail!("unknown agent {s}; expected user or alphazero[:model]")
    }
}

pub struct HumanAgent;

impl<G: InteractiveGame> Agent<G> for HumanAgent {
    fn select_move(&mut self, game: &G, _mode: PolicyMode) -> G::Move {
        loop {
            print!("your move: ");
            std::io::stdout().flush().unwrap();
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
                panic!("stdin closed");
            }
            if let Some(mv) = game.parse_native_move(&line) {
                return mv;
            }
            println!("illegal or unparsable move");
        }
    }
}

pub trait InteractiveGame: GameState + Clone + Default + std::fmt::Display {
    const NAME: &'static str;

    type SearchState: GameState<Move = Self::Move> + Clone;
    type Representation: AlphaZeroRepresentation<Self::SearchState> + Default;
    type Rules: SearchRules<Self::SearchState> + Default;

    fn search_state(&self) -> Self::SearchState;
    fn search_context(&self) -> <Self::Rules as SearchRules<Self::SearchState>>::Context<'_>;
    fn parse_native_move(&self, text: &str) -> Option<Self::Move>;
    fn format_native_move(&self, mv: Self::Move) -> String;
}

impl InteractiveGame for Connect4 {
    const NAME: &'static str = "connect4";

    type SearchState = Connect4;
    type Representation = Connect4AzRepresentation;
    type Rules = search::NoExtraRules;

    fn search_state(&self) -> Self::SearchState {
        *self
    }
    fn search_context(&self) {}
    fn parse_native_move(&self, text: &str) -> Option<Self::Move> {
        Connect4Notation.parse_move(self, text)
    }
    fn format_native_move(&self, mv: Self::Move) -> String {
        Connect4Notation.format_move(self, mv)
    }
}

impl InteractiveGame for ChessGame {
    const NAME: &'static str = "chess";

    type SearchState = ChessPosition;
    type Representation = ChessV1Representation;
    type Rules = alphazero::ChessRepetitionRules;

    fn search_state(&self) -> Self::SearchState {
        self.position_state()
    }
    fn search_context(&self) -> ChessRepetitionContext<'_> {
        self.repetition_context()
    }
    fn parse_native_move(&self, text: &str) -> Option<Self::Move> {
        ChessUciNotation.parse_move(&self.position_state(), text)
    }
    fn format_native_move(&self, mv: Self::Move) -> String {
        ChessUciNotation.format_move(&self.position_state(), mv)
    }
}

type NativeMcts<G> = Mcts<
    <G as InteractiveGame>::SearchState,
    RepresentedEvaluator<
        <G as InteractiveGame>::SearchState,
        <G as InteractiveGame>::Representation,
        alphazero::BatcherClient,
    >,
    <G as InteractiveGame>::Rules,
>;

pub struct AlphaZeroAgent<G: InteractiveGame> {
    _batcher: Batcher,
    mcts: NativeMcts<G>,
}

impl<G: InteractiveGame> AlphaZeroAgent<G> {
    pub fn new(
        cfg: &NetworkConfig,
        weights: &Path,
        device: Device,
        simulations: usize,
        wait_for_count: usize,
        timeout: Duration,
    ) -> Result<Self> {
        let batcher =
            Batcher::new_with_network(cfg, weights, device, wait_for_count.max(1), timeout)
                .with_context(|| format!("loading AlphaZero agent from {}", weights.display()))?;
        let mcts = Mcts::new(
            RepresentedEvaluator::new(G::Representation::default(), batcher.client()),
            MctsConfig {
                simulations,
                eps: 0.0,
                ..Default::default()
            },
            G::Rules::default(),
        );
        Ok(AlphaZeroAgent {
            _batcher: batcher,
            mcts,
        })
    }
}

impl<G: InteractiveGame> Agent<G> for AlphaZeroAgent<G> {
    fn select_move(&mut self, game: &G, mode: PolicyMode) -> G::Move {
        let variant = self.mcts.config().variant;
        let state = game.search_state();
        let result = self.mcts.search(&state, game.search_context(), mode);
        if matches!(variant, MctsVariant::Puct) && mode == PolicyMode::Explore {
            result.sample_move(&mut rand::rng())
        } else {
            result.best_move()
        }
    }
}

pub enum PlayerAgent<G: InteractiveGame> {
    Human(HumanAgent),
    AlphaZero(Box<AlphaZeroAgent<G>>),
}

impl<G: InteractiveGame> Agent<G> for PlayerAgent<G> {
    fn select_move(&mut self, game: &G, mode: PolicyMode) -> G::Move {
        match self {
            PlayerAgent::Human(agent) => agent.select_move(game, mode),
            PlayerAgent::AlphaZero(agent) => agent.select_move(game, mode),
        }
    }
}

use algorithms::alphazero::representation::{
    AlphaZeroRepresentation, ChessV1Representation, Connect4AzRepresentation,
};
use algorithms::alphazero::{
    Batcher, Mcts, MctsConfig, MctsVariant, NetConfig, RepresentedEvaluator,
};
use algorithms::search::SearchRules;
use anyhow::{Context, Result};
use engine_core::agent::{Agent, PolicyMode};
use engine_core::game::{Action, Game, GameState};
use games::{ChessGame, ChessPosition, ChessRepetitionContext, Connect4};
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

impl<G: Game> Agent<G> for HumanAgent {
    fn act_with_mode(&mut self, game: &G, _mode: PolicyMode) -> Action {
        loop {
            print!("your move: ");
            std::io::stdout().flush().unwrap();
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
                panic!("stdin closed");
            }
            if let Some(action) = game.parse_move(&line) {
                return action;
            }
            println!("illegal or unparsable move");
        }
    }
}

pub trait InteractiveGame: Game {
    type SearchState: GameState + Clone;
    type Representation: AlphaZeroRepresentation<Self::SearchState> + Default;
    type Rules: SearchRules<Self::SearchState> + Default;

    fn search_state(&self) -> Self::SearchState;
    fn search_context(&self) -> <Self::Rules as SearchRules<Self::SearchState>>::Context<'_>;
    fn to_action(
        &self,
        state: &Self::SearchState,
        mv: <Self::SearchState as GameState>::Move,
    ) -> Action;
}

impl InteractiveGame for Connect4 {
    type SearchState = Connect4;
    type Representation = Connect4AzRepresentation;
    type Rules = algorithms::search::NoExtraRules;

    fn search_state(&self) -> Self::SearchState {
        *self
    }
    fn search_context(&self) {}
    fn to_action(&self, state: &Self::SearchState, mv: <Connect4 as GameState>::Move) -> Action {
        Connect4AzRepresentation.move_to_action(state, mv).as_u32()
    }
}

impl InteractiveGame for ChessGame {
    type SearchState = ChessPosition;
    type Representation = ChessV1Representation;
    type Rules = algorithms::search::ChessRepetitionRules;

    fn search_state(&self) -> Self::SearchState {
        self.position_state()
    }
    fn search_context(&self) -> ChessRepetitionContext<'_> {
        self.repetition_context()
    }
    fn to_action(
        &self,
        state: &Self::SearchState,
        mv: <ChessPosition as GameState>::Move,
    ) -> Action {
        ChessV1Representation.move_to_action(state, mv).as_u32()
    }
}

type NativeMcts<G> = Mcts<
    <G as InteractiveGame>::SearchState,
    RepresentedEvaluator<
        <G as InteractiveGame>::SearchState,
        <G as InteractiveGame>::Representation,
        algorithms::alphazero::BatcherClient,
    >,
    <G as InteractiveGame>::Rules,
>;

pub struct AlphaZeroAgent<G: InteractiveGame> {
    _batcher: Batcher,
    mcts: NativeMcts<G>,
}

impl<G: InteractiveGame> AlphaZeroAgent<G> {
    pub fn new(
        cfg: &NetConfig,
        weights: &Path,
        device: Device,
        simulations: usize,
        wait_for_count: usize,
        timeout: Duration,
    ) -> Result<Self> {
        let batcher = Batcher::new(cfg, weights, device, wait_for_count.max(1), timeout)
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
    fn act_with_mode(&mut self, game: &G, mode: PolicyMode) -> Action {
        let variant = self.mcts.config().variant;
        let state = game.search_state();
        let result = self.mcts.search(&state, game.search_context(), mode);
        if matches!(variant, MctsVariant::Puct) && mode == PolicyMode::Explore {
            game.to_action(&state, result.sample_move(&mut rand::rng()))
        }
        else {
            game.to_action(&state, result.best_move())
        }
    }
}

pub enum PlayerAgent<G: InteractiveGame> {
    Human(HumanAgent),
    AlphaZero(Box<AlphaZeroAgent<G>>),
}

impl<G: InteractiveGame> Agent<G> for PlayerAgent<G> {
    fn act_with_mode(&mut self, game: &G, mode: PolicyMode) -> Action {
        match self {
            PlayerAgent::Human(agent) => agent.act_with_mode(game, mode),
            PlayerAgent::AlphaZero(agent) => agent.act_with_mode(game, mode),
        }
    }
}

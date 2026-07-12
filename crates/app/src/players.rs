use algorithms::alphazero::{Batcher, Mcts, MctsConfig, MctsVariant, NetConfig};
use anyhow::{Context, Result};
use engine_core::agent::{Agent, PolicyMode};
use engine_core::game::{Action, Game};
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

pub struct AlphaZeroAgent {
    _batcher: Batcher,
    mcts: Mcts<algorithms::alphazero::BatcherClient>,
}

impl AlphaZeroAgent {
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
            batcher.client(),
            MctsConfig {
                simulations,
                eps: 0.0,
                ..Default::default()
            },
        );
        Ok(AlphaZeroAgent {
            _batcher: batcher,
            mcts,
        })
    }
}

impl<G: Game> Agent<G> for AlphaZeroAgent {
    fn act_with_mode(&mut self, game: &G, mode: PolicyMode) -> Action {
        let variant = self.mcts.config().variant;
        let result = self.mcts.search_with_mode(game, mode);
        if matches!(variant, MctsVariant::Puct) && mode == PolicyMode::Explore {
            result.sample_action(&mut rand::rng())
        }
        else {
            result.best_action()
        }
    }
}

pub enum PlayerAgent {
    Human(HumanAgent),
    AlphaZero(Box<AlphaZeroAgent>),
}

impl<G: Game> Agent<G> for PlayerAgent {
    fn act_with_mode(&mut self, game: &G, mode: PolicyMode) -> Action {
        match self {
            PlayerAgent::Human(agent) => agent.act_with_mode(game, mode),
            PlayerAgent::AlphaZero(agent) => agent.act_with_mode(game, mode),
        }
    }
}

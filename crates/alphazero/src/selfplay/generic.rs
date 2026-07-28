use super::coordinator::{
    CompletedGame, GameRequest, SelfPlayWorker, SelfPlayWorkerFactory, BUDGET_SEED, MOVE_SEED,
};
use super::{select_temperature_action, SelfPlayConfig, SelfPlayStats};
use crate::{AlphaZeroRepresentation, Batcher};
use crate::{
    BatcherClient, Outcome, ReplaySample, RepresentedEvaluator, SampleMetadata, SearchKind,
    TrainingWeights,
};
use anyhow::Result;
use engine_core::agent::PolicyMode;
use engine_core::game::{GameState, TerminalValue};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use search::{Mcts, NoExtraRules};

pub struct GenericSelfPlayWorkerFactory<G, Rep> {
    batcher: BatcherClient,
    config: SelfPlayConfig,
    marker: std::marker::PhantomData<fn() -> (G, Rep)>,
}

impl<G, Rep> GenericSelfPlayWorkerFactory<G, Rep> {
    pub fn new(batcher: &Batcher, config: SelfPlayConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            batcher: batcher.client(),
            config,
            marker: std::marker::PhantomData,
        })
    }
}

pub struct GenericSelfPlayWorker<G, Rep>
where
    G: GameState + Clone,
    Rep: AlphaZeroRepresentation<G> + Default,
{
    representation: Rep,
    mcts: Mcts<G, RepresentedEvaluator<G, Rep, BatcherClient>, NoExtraRules>,
    config: SelfPlayConfig,
}

impl<G, Rep> SelfPlayWorkerFactory<G> for GenericSelfPlayWorkerFactory<G, Rep>
where
    G: GameState + Clone + Send + 'static,
    G::Move: Send + Sync,
    Rep: AlphaZeroRepresentation<G> + Default,
{
    type Worker = GenericSelfPlayWorker<G, Rep>;

    fn create(&self, _: usize, seed: u64) -> Result<Self::Worker> {
        let representation = Rep::default();
        let evaluator = RepresentedEvaluator::new(representation.clone(), self.batcher.clone());
        Ok(GenericSelfPlayWorker {
            representation,
            mcts: Mcts::new(evaluator, self.config.search.clone(), NoExtraRules).with_seed(seed),
            config: self.config.clone(),
        })
    }
}

impl<G, Rep> SelfPlayWorker<G> for GenericSelfPlayWorker<G, Rep>
where
    G: GameState + Clone + Send + 'static,
    G::Move: Send + Sync,
    Rep: AlphaZeroRepresentation<G> + Default,
{
    fn play_game(&mut self, game: GameRequest) -> Result<CompletedGame<G>> {
        self.mcts
            .reseed(game.seed_for(super::coordinator::SEARCH_SEED));
        let mut position = G::initial();
        let mut trajectory = Vec::with_capacity(self.config.max_moves.min(256));
        let mut budget_rng = SmallRng::seed_from_u64(game.seed_for(BUDGET_SEED));
        let mut move_rng = SmallRng::seed_from_u64(game.seed_for(MOVE_SEED));
        let mut stats = SelfPlayStats {
            games: 1,
            ..Default::default()
        };

        while !position.is_terminal() && trajectory.len() < self.config.max_moves {
            let (budget, policy_weight, full) = self.config.budget_schedule.choose(&mut budget_rng);
            if full {
                stats.full_searches += 1;
            }
            else {
                stats.fast_searches += 1;
            }
            let result = self.mcts.search(
                &position,
                (),
                super::config::search_request(budget, PolicyMode::Explore),
            )?;
            let action = select_temperature_action(
                &result,
                self.config.temperature.at_ply(trajectory.len()),
                &mut move_rng,
            );
            trajectory.push(ReplaySample {
                state: position.clone(),
                policy: result
                    .policy
                    .iter()
                    .map(|&(mv, probability)| {
                        (
                            self.representation.move_to_action(&position, mv),
                            probability,
                        )
                    })
                    .collect(),
                outcome: Outcome::Draw,
                weights: TrainingWeights {
                    policy: policy_weight,
                    value: 1.0,
                },
                metadata: SampleMetadata {
                    search_kind: if full {
                        SearchKind::Full
                    }
                    else {
                        SearchKind::Fast
                    },
                    simulations: budget.simulations() as u32,
                    model_generation: game.model_generation,
                    game_id: game.game_id,
                    ply: trajectory.len() as u16,
                },
            });
            position.play(action);
        }
        assign_outcomes(
            &mut trajectory,
            last_mover_outcome(position.terminal_value(), position.is_terminal()),
        );
        stats.moves = trajectory.len();
        Ok(CompletedGame { stats, trajectory })
    }
}

pub(crate) fn assign_outcomes<S>(trajectory: &mut [ReplaySample<S>], mut outcome: Outcome) {
    for sample in trajectory.iter_mut().rev() {
        sample.outcome = outcome;
        outcome = outcome.flipped();
    }
}

pub(crate) fn last_mover_outcome(value: Option<TerminalValue>, terminal: bool) -> Outcome {
    if !terminal {
        return Outcome::Draw;
    }
    match value.unwrap_or(TerminalValue::Draw) {
        TerminalValue::Win => Outcome::Loss,
        TerminalValue::Draw => Outcome::Draw,
        TerminalValue::Loss => Outcome::Win,
    }
}

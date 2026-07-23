use super::coordinator::{
    CompletedGame, GameRequest, SelfPlayWorker, SelfPlayWorkerFactory, BUDGET_SEED, MOVE_SEED,
    RESIGNATION_SEED, SEARCH_SEED,
};
use super::generic::{assign_outcomes, last_mover_outcome, set_budget};
use super::{select_temperature_action, GumbelMoveSelection, SelfPlayConfig, SelfPlayStats};
use crate::representation::{ChessAzRepresentation, ChessAzState};
use crate::{
    AlphaZeroRepresentation, Batcher, BatcherClient, ChessRepetitionRules, EvalTable, Outcome,
    ReplaySample, RepresentedEvaluator, SampleMetadata, SearchKind, TrainingWeights,
};
use anyhow::Result;
use engine_core::agent::PolicyMode;
use engine_core::game::GameState;
use games::ChessGame;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use search::{Mcts, SearchConfig};
use std::sync::Arc;

pub struct ChessSelfPlayWorkerFactory<const HISTORY: usize> {
    batcher: BatcherClient,
    config: SelfPlayConfig,
    cache: Option<Arc<EvalTable<chess::ChessMove>>>,
}

impl<const HISTORY: usize> ChessSelfPlayWorkerFactory<HISTORY> {
    pub fn new(batcher: &Batcher, config: SelfPlayConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            batcher: batcher.client(),
            cache: (config.evaluation_cache_entries > 0)
                .then(|| Arc::new(EvalTable::new(config.evaluation_cache_entries))),
            config,
        })
    }
}

pub struct ChessSelfPlayWorker<const HISTORY: usize> {
    mcts: Mcts<
        ChessAzState<HISTORY>,
        RepresentedEvaluator<ChessAzState<HISTORY>, ChessAzRepresentation<HISTORY>, BatcherClient>,
        ChessRepetitionRules,
    >,
    representation: ChessAzRepresentation<HISTORY>,
    config: SelfPlayConfig,
}

impl<const HISTORY: usize> SelfPlayWorkerFactory<ChessAzState<HISTORY>>
    for ChessSelfPlayWorkerFactory<HISTORY>
{
    type Worker = ChessSelfPlayWorker<HISTORY>;

    fn create(&self, _: usize, seed: u64) -> Result<Self::Worker> {
        let representation = ChessAzRepresentation::<HISTORY>;
        let evaluator = RepresentedEvaluator::new(representation, self.batcher.clone());
        let mut mcts =
            Mcts::new(evaluator, self.config.search.clone(), ChessRepetitionRules).with_seed(seed);
        if let Some(cache) = &self.cache {
            mcts = mcts.with_eval_cache(Arc::clone(cache));
        }
        Ok(ChessSelfPlayWorker {
            mcts,
            representation,
            config: self.config.clone(),
        })
    }
}

impl<const HISTORY: usize> SelfPlayWorker<ChessAzState<HISTORY>> for ChessSelfPlayWorker<HISTORY> {
    fn play_game(&mut self, request: GameRequest) -> Result<CompletedGame<ChessAzState<HISTORY>>> {
        self.mcts.reseed(request.seed_for(SEARCH_SEED));
        let mut game = ChessGame::default();
        let mut trajectory = Vec::with_capacity(self.config.max_moves.min(256));
        let mut budget_rng = SmallRng::seed_from_u64(request.seed_for(BUDGET_SEED));
        let mut move_rng = SmallRng::seed_from_u64(request.seed_for(MOVE_SEED));
        let mut resignation_rng = SmallRng::seed_from_u64(request.seed_for(RESIGNATION_SEED));
        let resignation_disabled =
            resignation_rng.random_bool(f64::from(self.config.resignation.disable_probability));
        let mut resignation_streak = 0;
        let mut resigned = false;
        let mut stats = SelfPlayStats {
            games: 1,
            ..Default::default()
        };

        while !game.is_terminal() && trajectory.len() < self.config.max_moves {
            let state = ChessAzState::from_game(&game);
            let (budget, policy_weight, full) = self.config.budget_schedule.choose(&mut budget_rng);
            set_budget(&mut self.mcts, budget)?;
            if full {
                stats.full_searches += 1;
            } else {
                stats.fast_searches += 1;
            }
            let result =
                self.mcts
                    .search(&state, game.repetition_context(), PolicyMode::Explore)?;
            let action = match (self.mcts.config(), self.config.gumbel_move_selection) {
                (SearchConfig::Gumbel(_), GumbelMoveSelection::ProposedAction) => {
                    result.selected_move
                }
                _ => select_temperature_action(
                    &result,
                    self.config.temperature.at_ply(trajectory.len()),
                    &mut move_rng,
                ),
            };
            trajectory.push(ReplaySample {
                state,
                policy: result
                    .policy
                    .iter()
                    .map(|&(mv, probability)| {
                        (self.representation.move_to_action(&state, mv), probability)
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
                    } else {
                        SearchKind::Fast
                    },
                    simulations: budget.simulations() as u32,
                    model_generation: request.model_generation,
                    game_id: request.game_id,
                    ply: trajectory.len() as u16,
                },
            });
            if should_resign(
                &self.config,
                result.root_value.as_f32(),
                trajectory.len(),
                resignation_disabled,
                &mut resignation_streak,
            ) {
                resigned = true;
                stats.resignations += 1;
                break;
            }
            game.play(action);
        }
        assign_outcomes(
            &mut trajectory,
            if resigned {
                Outcome::Loss
            } else {
                last_mover_outcome(game.terminal_value(), game.is_terminal())
            },
        );
        stats.moves = trajectory.len();
        Ok(CompletedGame { stats, trajectory })
    }
}

fn should_resign(
    config: &SelfPlayConfig,
    value: f32,
    ply: usize,
    disabled: bool,
    streak: &mut usize,
) -> bool {
    if config.resignation.enabled
        && !disabled
        && ply >= config.resignation.minimum_ply
        && value < config.resignation.threshold
    {
        *streak += 1;
    } else {
        *streak = 0;
    }
    *streak >= config.resignation.consecutive_moves
}

use super::coordinator::{
    CompletedGame, GameRequest, SelfPlayWorker, SelfPlayWorkerFactory, BUDGET_SEED, MOVE_SEED,
    RESIGNATION_SEED, SEARCH_SEED,
};
use super::domain::SelfPlayDomain;
use super::{
    select_temperature_action, GumbelMoveSelection, SearchBudget, SelfPlayConfig, SelfPlayStats,
};
use crate::{
    AlphaZeroRepresentation, Batcher, BatcherClient, EvalTable, Outcome, ReplaySample,
    RepresentedEvaluator, SampleMetadata, SearchKind, TrainingWeights,
};
use anyhow::Result;
use engine_core::agent::PolicyMode;
use engine_core::game::GameState;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use search::{Mcts, SearchConfig, SearchResult};
use std::sync::Arc;

type DomainMove<D> = <<D as SelfPlayDomain>::State as GameState>::Move;
type DomainEvaluator<D> = RepresentedEvaluator<
    <D as SelfPlayDomain>::State,
    <D as SelfPlayDomain>::Representation,
    BatcherClient,
>;
type DomainMcts<D> =
    Mcts<<D as SelfPlayDomain>::State, DomainEvaluator<D>, <D as SelfPlayDomain>::Rules>;

pub struct DomainSelfPlayWorkerFactory<D: SelfPlayDomain> {
    batcher: BatcherClient,
    config: SelfPlayConfig,
    cache: Option<Arc<EvalTable<DomainMove<D>>>>,
}

impl<D: SelfPlayDomain> DomainSelfPlayWorkerFactory<D> {
    pub fn new(batcher: &Batcher, config: SelfPlayConfig) -> Result<Self> {
        config.validate()?;

        let cache = D::uses_evaluation_cache()
            .then_some(config.evaluation_cache_entries)
            .filter(|&entries| entries > 0)
            .map(EvalTable::new)
            .map(Arc::new);

        Ok(Self {
            batcher: batcher.client(),
            config,
            cache,
        })
    }
}

pub struct DomainSelfPlayWorker<D: SelfPlayDomain> {
    representation: D::Representation,
    mcts: DomainMcts<D>,
    config: SelfPlayConfig,
}

impl<D: SelfPlayDomain> SelfPlayWorkerFactory<D::State> for DomainSelfPlayWorkerFactory<D>
where
    DomainMove<D>: Send + Sync,
    <D::Rules as search::SearchRules<D::State>>::PathState: Send,
    <D::Rules as search::SearchRules<D::State>>::NodeMeta: Send,
{
    type Worker = DomainSelfPlayWorker<D>;

    fn create(&self, _: usize, seed: u64) -> Result<Self::Worker> {
        let representation = D::Representation::default();
        let evaluator = RepresentedEvaluator::new(representation.clone(), self.batcher.clone());
        let mut mcts =
            Mcts::new(evaluator, self.config.search.clone(), D::Rules::default()).with_seed(seed);

        if let Some(cache) = &self.cache {
            mcts = mcts.with_eval_cache(Arc::clone(cache));
        }

        Ok(DomainSelfPlayWorker {
            representation,
            mcts,
            config: self.config.clone(),
        })
    }
}

impl<D: SelfPlayDomain> SelfPlayWorker<D::State> for DomainSelfPlayWorker<D>
where
    DomainMove<D>: Send + Sync,
    <D::Rules as search::SearchRules<D::State>>::PathState: Send,
    <D::Rules as search::SearchRules<D::State>>::NodeMeta: Send,
{
    fn play_game(&mut self, request: GameRequest) -> Result<CompletedGame<D::State>> {
        self.mcts.reseed(request.seed_for(SEARCH_SEED));

        let mut game = D::initial_game();
        let mut trajectory = Vec::with_capacity(self.config.max_moves.min(256));
        let mut budget_rng = SmallRng::seed_from_u64(request.seed_for(BUDGET_SEED));
        let mut move_rng = SmallRng::seed_from_u64(request.seed_for(MOVE_SEED));
        let mut resignation = ResignationState::new(&self.config, request);
        let mut stats = SelfPlayStats {
            games: 1,
            ..Default::default()
        };

        while !D::is_terminal(&game) && trajectory.len() < self.config.max_moves {
            let turn = TurnPlan::choose(&self.config.budget_schedule, &mut budget_rng);
            turn.record_search_kind(&mut stats);

            let (state, result) = self.search_turn(&game, turn.budget)?;
            stats.add_search_diagnostics(result.diagnostics);

            let action = self.select_action(&result, trajectory.len(), &mut move_rng);
            trajectory.push(self.replay_sample(&state, &result, turn, request, trajectory.len()));

            if D::supports_resignation()
                && resignation.should_resign(
                    &self.config,
                    result.root_value.as_f32(),
                    trajectory.len(),
                )
            {
                stats.resignations += 1;
                return Ok(completed_game(stats, trajectory, Outcome::Loss));
            }

            D::play(&mut game, action);
        }

        let outcome = last_mover_outcome(D::terminal_value(&game), D::is_terminal(&game));
        Ok(completed_game(stats, trajectory, outcome))
    }
}

impl<D: SelfPlayDomain> DomainSelfPlayWorker<D>
where
    DomainMove<D>: Send + Sync,
{
    fn search_turn(
        &mut self,
        game: &D::Game,
        budget: SearchBudget,
    ) -> Result<(D::State, SearchResult<DomainMove<D>>)> {
        let state = D::search_state(game);
        let result = self.mcts.search(
            &state,
            D::search_context(game),
            super::config::search_request(budget, PolicyMode::Explore),
        )?;

        Ok((state, result))
    }

    fn select_action(
        &self,
        result: &SearchResult<DomainMove<D>>,
        ply: usize,
        rng: &mut SmallRng,
    ) -> DomainMove<D> {
        match (self.mcts.config(), self.config.gumbel_move_selection) {
            (
                SearchConfig::RootGumbelPuct(_) | SearchConfig::FullGumbel(_),
                GumbelMoveSelection::ProposedAction,
            ) => result.selected_move,
            _ => select_temperature_action(result, self.config.temperature.at_ply(ply), rng),
        }
    }

    fn replay_sample(
        &self,
        state: &D::State,
        result: &SearchResult<DomainMove<D>>,
        turn: TurnPlan,
        request: GameRequest,
        ply: usize,
    ) -> ReplaySample<D::State> {
        ReplaySample {
            state: state.clone(),
            policy: result
                .policy
                .iter()
                .map(|&(mv, probability)| {
                    (self.representation.move_to_action(state, mv), probability)
                })
                .collect(),
            outcome: Outcome::Draw,
            weights: turn.weights(),
            metadata: turn.metadata(request, ply),
        }
    }
}

#[derive(Clone, Copy)]
struct TurnPlan {
    policy_weight: f32,
    full: bool,
    budget: SearchBudget,
}

impl TurnPlan {
    fn choose(schedule: &super::SearchBudgetSchedule, rng: &mut SmallRng) -> Self {
        let (budget, policy_weight, full) = schedule.choose(rng);
        Self {
            policy_weight,
            full,
            budget,
        }
    }

    fn record_search_kind(self, stats: &mut SelfPlayStats) {
        if self.full {
            stats.full_searches += 1;
        }
        else {
            stats.fast_searches += 1;
        }
    }

    fn weights(self) -> TrainingWeights {
        TrainingWeights {
            policy: self.policy_weight,
            value: 1.0,
        }
    }

    fn metadata(self, request: GameRequest, ply: usize) -> SampleMetadata {
        SampleMetadata {
            search_kind: if self.full {
                SearchKind::Full
            }
            else {
                SearchKind::Fast
            },
            simulations: self.budget.simulations() as u32,
            model_generation: request.model_generation,
            game_id: request.game_id,
            ply: ply as u16,
        }
    }
}

struct ResignationState {
    disabled: bool,
    streak: usize,
}

impl ResignationState {
    fn new(config: &SelfPlayConfig, request: GameRequest) -> Self {
        let mut rng = SmallRng::seed_from_u64(request.seed_for(RESIGNATION_SEED));
        let disabled = rng.random_bool(f64::from(config.resignation.disable_probability));

        Self {
            disabled,
            streak: 0,
        }
    }

    fn should_resign(&mut self, config: &SelfPlayConfig, value: f32, ply: usize) -> bool {
        let is_losing = config.resignation.enabled
            && !self.disabled
            && ply >= config.resignation.minimum_ply
            && value < config.resignation.threshold;

        self.streak = if is_losing { self.streak + 1 } else { 0 };
        self.streak >= config.resignation.consecutive_moves
    }
}

fn finish_stats(mut stats: SelfPlayStats, moves: usize) -> SelfPlayStats {
    stats.moves = moves;
    stats
}

fn completed_game<S>(
    stats: SelfPlayStats,
    trajectory: Vec<ReplaySample<S>>,
    outcome: Outcome,
) -> CompletedGame<S> {
    CompletedGame {
        stats: finish_stats(stats, trajectory.len()),
        trajectory: with_outcomes(trajectory, outcome),
    }
}

fn with_outcomes<S>(
    mut trajectory: Vec<ReplaySample<S>>,
    outcome: Outcome,
) -> Vec<ReplaySample<S>> {
    assign_outcomes(&mut trajectory, outcome);
    trajectory
}

fn assign_outcomes<S>(trajectory: &mut [ReplaySample<S>], mut outcome: Outcome) {
    for sample in trajectory.iter_mut().rev() {
        sample.outcome = outcome;
        outcome = outcome.flipped();
    }
}

fn last_mover_outcome(value: Option<engine_core::game::TerminalValue>, terminal: bool) -> Outcome {
    if !terminal {
        return Outcome::Draw;
    }

    match value.unwrap_or(engine_core::game::TerminalValue::Draw) {
        engine_core::game::TerminalValue::Win => Outcome::Loss,
        engine_core::game::TerminalValue::Draw => Outcome::Draw,
        engine_core::game::TerminalValue::Loss => Outcome::Win,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SearchKind, TrainingWeights};

    #[test]
    fn turn_plan_keeps_fast_search_weight_and_metadata() {
        let schedule = super::super::SearchBudgetSchedule::PlayoutCapRandomization {
            full: SearchBudget::Puct { simulations: 32 },
            fast: SearchBudget::Puct { simulations: 8 },
            full_probability: 0.0,
            fast_policy_weight: 0.25,
        };
        let request = GameRequest {
            game_id: 4,
            model_generation: 3,
            worker_id: 0,
            experiment_seed: 7,
        };
        let mut rng = SmallRng::seed_from_u64(9);
        let turn = TurnPlan::choose(&schedule, &mut rng);

        assert_eq!(
            turn.weights(),
            TrainingWeights {
                policy: 0.25,
                value: 1.0,
            }
        );
        assert_eq!(
            turn.metadata(request, 2),
            SampleMetadata {
                search_kind: SearchKind::Fast,
                simulations: 8,
                model_generation: 3,
                game_id: 4,
                ply: 2,
            }
        );
    }
}

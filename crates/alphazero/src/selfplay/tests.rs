use super::{
    select_temperature_action, CompletedGame, GameRequest, SearchBudget, SearchBudgetSchedule,
    SelfPlayConfig, SelfPlayCoordinator, SelfPlayEpoch, SelfPlayStats, SelfPlayWorker,
    SelfPlayWorkerFactory, TemperaturePhase, TemperatureSchedule,
};
use crate::{Action, Outcome, ReplayBuffer, ReplaySample, SampleMetadata, TrainingWeights};
use anyhow::Result;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use search::{PositionValue, SearchDiagnostics, SearchResult};
use std::time::Duration;

#[test]
fn game_id_and_purpose_produce_independent_deterministic_seeds() {
    let first = GameRequest {
        game_id: 3,
        model_generation: 0,
        worker_id: 0,
        experiment_seed: 7,
    };
    let second = GameRequest {
        game_id: 4,
        model_generation: 1,
        worker_id: 9,
        experiment_seed: 7,
    };
    assert_eq!(first.seed_for(11), first.seed_for(11));
    assert_ne!(first.seed_for(11), second.seed_for(11));
    assert_ne!(first.seed_for(11), first.seed_for(12));
    assert_ne!(
        first.seed_for(11),
        GameRequest {
            model_generation: 1,
            ..first
        }
        .seed_for(11)
    );
    assert_eq!(
        first.seed_for(11),
        GameRequest {
            worker_id: 99,
            ..first
        }
        .seed_for(11)
    );
}

#[test]
fn zero_games_is_rejected_before_self_play_starts() {
    assert!(SelfPlayConfig {
        num_games: 0,
        ..SelfPlayConfig::default()
    }
    .validate()
    .is_err());
}

#[test]
fn aggregate_search_diagnostics_are_preserved() {
    let mut stats = SelfPlayStats::default();
    stats.add_search_diagnostics(SearchDiagnostics {
        completed_simulations: 12,
        backend_evaluations: 4,
        evaluation_cache_hits: 8,
        duplicate_leaves: 2,
        nodes_created: 9,
        max_depth: 5,
        ..Default::default()
    });
    stats.add_search_diagnostics(SearchDiagnostics {
        completed_simulations: 7,
        backend_evaluations: 3,
        evaluation_cache_hits: 1,
        duplicate_leaves: 4,
        nodes_created: 6,
        max_depth: 8,
        ..Default::default()
    });

    assert_eq!(stats.completed_simulations, 19);
    assert_eq!(stats.backend_evaluations, 7);
    assert_eq!(stats.evaluation_cache_hits, 9);
    assert_eq!(stats.duplicate_leaves, 6);
    assert_eq!(stats.nodes_created, 15);
    assert_eq!(stats.maximum_search_depth, 8);
}

#[derive(Clone, Copy)]
struct ScriptedFactory;

struct ScriptedWorker;

impl SelfPlayWorkerFactory<u64> for ScriptedFactory {
    type Worker = ScriptedWorker;

    fn create(&self, _: usize, _: u64) -> Result<Self::Worker> {
        Ok(ScriptedWorker)
    }
}

impl SelfPlayWorker<u64> for ScriptedWorker {
    fn play_game(&mut self, request: GameRequest) -> Result<CompletedGame<u64>> {
        let delay = (3 - request.game_id % 4) * 2;
        std::thread::sleep(Duration::from_millis(delay));
        Ok(CompletedGame {
            stats: SelfPlayStats {
                games: 1,
                moves: 1,
                ..Default::default()
            },
            trajectory: vec![ReplaySample {
                state: request.game_id,
                policy: vec![(Action::new(0), 1.0)].into(),
                outcome: Outcome::Draw,
                weights: TrainingWeights::default(),
                metadata: SampleMetadata {
                    model_generation: request.model_generation,
                    game_id: request.game_id,
                    ..Default::default()
                },
            }],
        })
    }
}

fn scripted_run(threads: usize, epoch: SelfPlayEpoch) -> Vec<(u64, u64)> {
    let coordinator = SelfPlayCoordinator::new(
        SelfPlayConfig {
            num_games: 8,
            threads,
            ..SelfPlayConfig::default()
        },
        17,
    )
    .unwrap();
    let replay = ReplayBuffer::new(16, 1);
    let stats = coordinator.run(&ScriptedFactory, &replay, epoch).unwrap();
    assert_eq!(stats.games, 8);
    replay
        .export_filled()
        .into_iter()
        .map(|sample| (sample.metadata.game_id, sample.metadata.model_generation))
        .collect()
}

#[test]
fn ordered_commit_makes_thread_counts_and_generation_metadata_reproducible() {
    let epoch = SelfPlayEpoch {
        model_generation: 6,
        first_game_id: 40,
    };
    let one_thread = scripted_run(1, epoch);
    let four_threads = scripted_run(4, epoch);
    assert_eq!(one_thread, four_threads);
    assert_eq!(
        four_threads,
        (40..48).map(|game_id| (game_id, 6)).collect::<Vec<_>>()
    );
}

#[test]
fn epochs_never_reuse_game_ids() {
    let first = scripted_run(
        2,
        SelfPlayEpoch {
            model_generation: 0,
            first_game_id: 0,
        },
    );
    let second = scripted_run(
        2,
        SelfPlayEpoch {
            model_generation: 1,
            first_game_id: first.len() as u64,
        },
    );
    assert_eq!(first.last().unwrap().0, 7);
    assert_eq!(second.first().unwrap().0, 8);
    assert!(first.iter().all(|(_, generation)| *generation == 0));
    assert!(second.iter().all(|(_, generation)| *generation == 1));
}

#[test]
fn randomized_budget_choice_is_reproducible_and_keeps_fast_weight() {
    let schedule = SearchBudgetSchedule::PlayoutCapRandomization {
        full: SearchBudget::Gumbel {
            simulations: 32,
            max_considered_actions: 8,
        },
        fast: SearchBudget::Gumbel {
            simulations: 8,
            max_considered_actions: 4,
        },
        full_probability: 0.25,
        fast_policy_weight: 0.0,
    };
    let mut left = SmallRng::seed_from_u64(9);
    let mut right = SmallRng::seed_from_u64(9);
    for _ in 0..20 {
        assert_eq!(schedule.choose(&mut left), schedule.choose(&mut right));
    }
}

#[test]
fn temperature_schedule_uses_argmax_after_final_phase() {
    let schedule = TemperatureSchedule {
        phases: vec![
            TemperaturePhase {
                until_ply_exclusive: 2,
                temperature: 1.0,
            },
            TemperaturePhase {
                until_ply_exclusive: 4,
                temperature: 0.5,
            },
        ],
    };
    assert_eq!(schedule.at_ply(0), Some(1.0));
    assert_eq!(schedule.at_ply(2), Some(0.5));
    assert_eq!(schedule.at_ply(4), None);
}

#[test]
fn temperature_sampling_returns_the_sparse_action_id() {
    let result = SearchResult {
        policy: vec![(42, 1.0)],
        selected_move: 42,
        root_value: PositionValue::DRAW,
        diagnostics: Default::default(),
    };
    let mut rng = SmallRng::seed_from_u64(7);
    assert_eq!(select_temperature_action(&result, Some(0.5), &mut rng), 42);
}

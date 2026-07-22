use super::{
    select_temperature_action, GameRequest, SearchBudget, SearchBudgetSchedule, TemperaturePhase,
    TemperatureSchedule,
};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use search::{PositionValue, SearchResult};

#[test]
fn game_id_and_purpose_produce_independent_deterministic_seeds() {
    let first = GameRequest {
        game_id: 3,
        worker_id: 0,
        seed: 7,
    };
    let second = GameRequest {
        game_id: 4,
        worker_id: 0,
        seed: 7,
    };
    assert_eq!(first.seed_for(11), first.seed_for(11));
    assert_ne!(first.seed_for(11), second.seed_for(11));
    assert_ne!(first.seed_for(11), first.seed_for(12));
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

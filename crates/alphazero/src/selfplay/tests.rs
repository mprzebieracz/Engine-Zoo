use super::Transition;
use super::{
    assign_trajectory_values, select_temperature_action, ChessV2GumbelProfiles, SelfPlayConfig,
    SelfPlayTemperature,
};
use crate::SearchResult;
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn transition() -> Transition {
    Transition {
        state: Vec::new(),
        policy: Vec::new(),
        value: 0.0,
    }
}

#[test]
fn values_alternate_backwards() {
    let mut traj: Vec<Transition> = (0..5).map(|_| transition()).collect();
    assign_trajectory_values(&mut traj, 1.0);
    let values: Vec<f32> = traj.iter().map(|t| t.value).collect();
    assert_eq!(values, vec![1.0, -1.0, 1.0, -1.0, 1.0]);
}

#[test]
fn draw_leaves_zeros() {
    let mut traj: Vec<Transition> = (0..4).map(|_| transition()).collect();
    assign_trajectory_values(&mut traj, 0.0);
    assert!(traj.iter().all(|t| t.value == 0.0));
}

#[test]
fn chess_v2_defaults_use_paired_gumbel_budgets_and_temperature_schedule() {
    let cfg = SelfPlayConfig::chess_v2_defaults();
    assert_eq!(
        cfg.chess_v2_gumbel_profiles,
        Some(ChessV2GumbelProfiles::DEFAULT)
    );
    assert_eq!(cfg.full_simulation_probability, 0.5);
    assert_eq!(
        cfg.chess_v2_temperature,
        Some(SelfPlayTemperature::default())
    );
    assert!(cfg.validate_chess_v2().is_ok());
}

#[test]
fn temperature_schedule_transitions_to_argmax() {
    let schedule = SelfPlayTemperature::default();
    assert_eq!(schedule.at_ply(0), Some(1.0));
    assert_eq!(schedule.at_ply(20), Some(0.5));
    assert_eq!(schedule.at_ply(40), None);
}

#[test]
fn temperature_sampling_returns_the_sparse_action_id() {
    let result = SearchResult {
        policy: vec![(42, 1.0)],
        selected_move: 42,
        value: 0.0,
    };
    let mut rng = SmallRng::seed_from_u64(7);
    for _ in 0..100 {
        assert_eq!(select_temperature_action(&result, Some(0.5), &mut rng), 42);
    }
}

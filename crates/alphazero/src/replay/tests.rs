use super::*;
use crate::representation::Connect4AzRepresentation;
use games::Connect4;
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn sample(id: u64) -> ReplaySample<Connect4> {
    ReplaySample {
        state: Connect4::default(),
        policy: vec![(Action::new(0), 1.0)].into(),
        outcome: Outcome::Draw,
        weights: TrainingWeights::default(),
        metadata: SampleMetadata {
            game_id: id,
            ..Default::default()
        },
    }
}

#[test]
fn ring_retains_compact_states_in_order() {
    let replay = ReplayBuffer::new(2, 7);
    replay.add([sample(1), sample(2), sample(3)]);
    let ids: Vec<_> = replay
        .export_filled()
        .into_iter()
        .map(|entry| entry.metadata.game_id)
        .collect();
    assert_eq!(ids, [2, 3]);
}

#[test]
fn sampled_states_are_encoded_only_at_load_time() {
    let replay = ReplayBuffer::new(1, 7);
    replay.add([sample(1)]);
    let mut rng = SmallRng::seed_from_u64(1);
    let batch = replay
        .sample(1, &Connect4AzRepresentation, &mut rng)
        .unwrap();
    assert_eq!(batch.states.size(), [1, 42]);
    assert_eq!(batch.outcomes.int64_value(&[0]), Outcome::Draw.wdl_index());
    assert_eq!(batch.policy_weight_sum, 1.0);
    assert_eq!(batch.value_weight_sum, 1.0);
}

#[test]
fn sampled_batch_keeps_cpu_weight_sums_for_loss_activation() {
    let replay = ReplayBuffer::new(2, 7);
    let mut policy_only = sample(1);
    policy_only.weights.value = 0.0;

    let mut value_only = sample(2);
    value_only.weights.policy = 0.0;
    value_only.policy = SparsePolicy::new([]).unwrap();

    replay.add([policy_only, value_only]);
    let mut rng = SmallRng::seed_from_u64(1);
    let batch = replay
        .sample(2, &Connect4AzRepresentation, &mut rng)
        .unwrap();

    assert_eq!(batch.policy_weight_sum, 1.0);
    assert_eq!(batch.value_weight_sum, 1.0);
}

#[test]
fn seeded_sampling_reproduces_the_same_sparse_batch() {
    let replay = ReplayBuffer::new(7, 7);
    replay.add((0..7).map(|id| ReplaySample {
        policy: vec![(Action::new(id), 1.0)].into(),
        ..sample(id as u64)
    }));
    let mut first_rng = SmallRng::seed_from_u64(42);
    let mut second_rng = SmallRng::seed_from_u64(42);
    let first = replay
        .sample(4, &Connect4AzRepresentation, &mut first_rng)
        .unwrap();
    let second = replay
        .sample(4, &Connect4AzRepresentation, &mut second_rng)
        .unwrap();
    assert_eq!(
        Vec::<i64>::try_from(&first.policies.actions).unwrap(),
        Vec::<i64>::try_from(&second.policies.actions).unwrap()
    );
    assert_eq!(
        Vec::<i64>::try_from(&first.policies.rows).unwrap(),
        Vec::<i64>::try_from(&second.policies.rows).unwrap()
    );
}

#[test]
fn outcome_is_not_a_float_protocol() {
    assert_eq!(Outcome::Win.flipped(), Outcome::Loss);
    assert_eq!(Outcome::Draw.wdl_index(), 1);
}

#[test]
fn sparse_policy_merges_duplicates_normalizes_and_round_trips_serde() {
    let policy = SparsePolicy::new([
        (Action::new(2), 0.25),
        (Action::new(0), 0.25),
        (Action::new(2), 0.5),
        (Action::new(1), 0.0),
    ])
    .unwrap();
    assert_eq!(policy.len(), 2);
    assert_eq!(policy[0], (Action::new(0), 0.25));
    assert_eq!(policy[1], (Action::new(2), 0.75));
    let decoded: SparsePolicy = serde_json::from_str(r#"[[2,0.5],[0,0.25],[2,0.25]]"#).unwrap();
    assert_eq!(decoded, policy);
}

#[test]
fn positive_policy_weight_requires_policy_mass() {
    let replay = ReplayBuffer::new(1, 7);
    let mut empty = sample(1);
    empty.policy = SparsePolicy::new([]).unwrap();
    assert!(std::panic::catch_unwind(|| replay.add([empty])).is_err());
}

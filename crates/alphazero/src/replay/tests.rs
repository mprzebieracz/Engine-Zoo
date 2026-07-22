use super::*;
use crate::representation::Connect4AzRepresentation;
use games::Connect4;

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
    let batch = replay.sample(1, &Connect4AzRepresentation).unwrap();
    assert_eq!(batch.states.size(), [1, 42]);
    assert_eq!(batch.outcomes.int64_value(&[0]), Outcome::Draw.wdl_index());
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

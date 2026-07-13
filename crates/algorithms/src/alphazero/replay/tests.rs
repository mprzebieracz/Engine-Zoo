use super::{ReplayBuffer, Transition};

fn tensor_1d(tensor: &tch::Tensor) -> Vec<f32> {
    tensor.flatten(0, -1).try_into().unwrap()
}

fn transition(id: f32) -> Transition {
    Transition {
        state: vec![id],
        policy: vec![(0, id)],
        reward: id,
    }
}

#[test]
fn len_grows_until_capacity() {
    let replay = ReplayBuffer::new(3, 1, 4);
    assert!(replay.is_empty());

    replay.add(vec![transition(1.0)]);
    assert_eq!(replay.len(), 1);

    replay.add(vec![transition(2.0), transition(3.0)]);
    assert_eq!(replay.len(), 3);
}

#[test]
fn ring_overwrites_oldest_entry() {
    let replay = ReplayBuffer::new(3, 1, 4);
    replay.add(vec![transition(1.0), transition(2.0), transition(3.0)]);
    replay.add(vec![transition(4.0)]);

    let exported: Vec<_> = replay
        .export_filled()
        .into_iter()
        .map(|t| t.reward)
        .collect();
    assert_eq!(exported, vec![2.0, 3.0, 4.0]);

    let batch = replay.sample(3).unwrap();
    let mut rewards = tensor_1d(&batch.rewards);
    rewards.sort_by(|a, b| a.total_cmp(b));
    assert_eq!(rewards, vec![2.0, 3.0, 4.0]);

    let mut action_zero = tensor_1d(&batch.policies.probabilities);
    action_zero.sort_by(|a, b| a.total_cmp(b));
    assert_eq!(action_zero, vec![2.0, 3.0, 4.0]);
}

#[test]
fn sample_keeps_policy_sparse() {
    let replay = ReplayBuffer::new(4, 1, 5);
    replay.add(vec![Transition {
        state: vec![7.0],
        policy: vec![(1, 0.25), (3, 0.75)],
        reward: 0.5,
    }]);

    let batch = replay.sample(1).unwrap();
    let actions: Vec<i64> = batch.policies.actions.try_into().unwrap();
    let policies = tensor_1d(&batch.policies.probabilities);
    let rows: Vec<i64> = batch.policies.rows.try_into().unwrap();
    let rewards = tensor_1d(&batch.rewards);

    assert_eq!(actions, vec![1, 3]);
    assert_eq!(policies, vec![0.25, 0.75]);
    assert_eq!(rows, vec![0, 0]);
    assert_eq!(batch.policies.offsets, vec![0, 2]);
    assert_eq!(rewards, vec![0.5]);
}

#[test]
fn duplicate_policy_actions_preserve_last_write_wins_semantics() {
    let replay = ReplayBuffer::new(1, 1, 4);
    replay.add(vec![Transition {
        state: vec![0.0],
        policy: vec![(2, 0.25), (2, 0.75)],
        reward: 0.0,
    }]);

    let stored = replay.export_filled();
    assert_eq!(stored[0].policy, vec![(2, 0.75)]);
}

#[test]
#[should_panic(expected = "replay policy action is out of range")]
fn replay_rejects_out_of_range_policy_actions_in_release_builds() {
    let replay = ReplayBuffer::new(1, 1, 4);
    replay.add(vec![Transition {
        state: vec![0.0],
        policy: vec![(4, 1.0)],
        reward: 0.0,
    }]);
}

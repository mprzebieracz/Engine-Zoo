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

    let (_, policies, rewards) = replay.sample(3).unwrap();
    let mut rewards = tensor_1d(&rewards);
    rewards.sort_by(|a, b| a.total_cmp(b));
    assert_eq!(rewards, vec![2.0, 3.0, 4.0]);

    let policies = tensor_1d(&policies);
    let mut action_zero: Vec<f32> = (0..3).map(|row| policies[row * 4]).collect();
    action_zero.sort_by(|a, b| a.total_cmp(b));
    assert_eq!(action_zero, vec![2.0, 3.0, 4.0]);
}

#[test]
fn sample_densifies_sparse_policy() {
    let replay = ReplayBuffer::new(4, 1, 5);
    replay.add(vec![Transition {
        state: vec![7.0],
        policy: vec![(1, 0.25), (3, 0.75)],
        reward: 0.5,
    }]);

    let (_, policies, rewards) = replay.sample(1).unwrap();
    let policies = tensor_1d(&policies);
    let rewards = tensor_1d(&rewards);

    assert_eq!(policies, vec![0.0, 0.25, 0.0, 0.75, 0.0]);
    assert_eq!(rewards, vec![0.5]);
}

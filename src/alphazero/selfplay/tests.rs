use super::assign_trajectory_rewards;
use super::Transition;

fn transition() -> Transition {
    Transition {
        state: Vec::new(),
        policy: Vec::new(),
        reward: 0.0,
    }
}

#[test]
fn rewards_alternate_backwards() {
    let mut traj: Vec<Transition> = (0..5).map(|_| transition()).collect();
    assign_trajectory_rewards(&mut traj, 1.0);
    let rewards: Vec<f32> = traj.iter().map(|t| t.reward).collect();
    assert_eq!(rewards, vec![1.0, -1.0, 1.0, -1.0, 1.0]);
}

#[test]
fn draw_leaves_zeros() {
    let mut traj: Vec<Transition> = (0..4).map(|_| transition()).collect();
    assign_trajectory_rewards(&mut traj, 0.0);
    assert!(traj.iter().all(|t| t.reward == 0.0));
}

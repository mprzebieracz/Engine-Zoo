use super::batcher::{Batcher, BatcherClient};
use super::mcts::{Mcts, MctsConfig};
use super::replay::{ReplayBuffer, Transition};
use crate::game::{Action, Game};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    pub num_games: usize,
    pub threads: usize,
    /// Games longer than this are truncated and scored as draws.
    pub max_moves: usize,
    /// Moves sampled from the visit distribution before switching to argmax.
    pub temperature_moves: usize,
    pub mcts: MctsConfig,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            num_games: 100,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            max_moves: 256,
            temperature_moves: 30,
            mcts: MctsConfig::default(),
        }
    }
}

/// Plays `cfg.num_games` games of self-play across `cfg.threads` threads, all
/// sharing `batcher` for network evaluation, appending trajectories to `replay`.
pub fn self_play<G: Game>(batcher: &Batcher, replay: &ReplayBuffer, cfg: &SelfPlayConfig) {
    let next_game = AtomicUsize::new(0);
    let finished = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        for _ in 0..cfg.threads.max(1) {
            scope.spawn(|| {
                let mut mcts = Mcts::new(batcher.client(), cfg.mcts);
                while next_game.fetch_add(1, Ordering::Relaxed) < cfg.num_games {
                    play_game::<G>(&mut mcts, replay, cfg);
                    let done = finished.fetch_add(1, Ordering::Relaxed) + 1;
                    println!("Games played: {done}/{}", cfg.num_games);
                }
            });
        }
    });
}

fn play_game<G: Game>(
    mcts: &mut Mcts<BatcherClient>,
    replay: &ReplayBuffer,
    cfg: &SelfPlayConfig,
) {
    let mut game = G::default();
    let mut trajectory: Vec<Transition> = Vec::new();
    let mut rng = rand::rng();

    while !game.is_terminal() && trajectory.len() < cfg.max_moves {
        let mut state = vec![0.0f32; G::state_size()];
        game.encode_state(&mut state);

        let result = mcts.search(&game);
        let action = if trajectory.len() < cfg.temperature_moves {
            result.sample_action(&mut rng)
        }
        else {
            result.best_action()
        };

        let policy = result
            .policy
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(a, &p)| (a as Action, p))
            .collect();

        trajectory.push(Transition {
            state,
            policy,
            reward: 0.0,
        });
        game.step(action);
    }

    assign_trajectory_rewards(&mut trajectory, -game.reward());
    replay.add(trajectory);
}

/// Walks the trajectory backwards from the terminal reward, flipping sign
/// each ply so every transition's target value is from its own player's
/// perspective.
fn assign_trajectory_rewards(trajectory: &mut [Transition], terminal_reward: f32) {
    let mut value = terminal_reward;
    for t in trajectory.iter_mut().rev() {
        t.reward = value;
        value = -value;
    }
}

#[cfg(test)]
mod tests;

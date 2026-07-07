use super::*;
use crate::alphazero::evaluator::UniformEvaluator;
use crate::games::Connect4;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::fmt;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct ImmediateOutcomeGame {
    reward: Option<f32>,
}

impl fmt::Display for ImmediateOutcomeGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.reward)
    }
}

impl Game for ImmediateOutcomeGame {
    const ACTION_SIZE: usize = 3;
    const STATE_SHAPE: [crate::game::TensorDim; 3] = [1, 1, 1];
    const NAME: &'static str = "immediate";

    fn legal_actions(&self) -> impl Iterator<Item = Action> + '_ {
        (0..Self::ACTION_SIZE as Action).filter(|_| self.reward.is_none())
    }

    fn step(&mut self, action: Action) {
        self.reward = Some(match action {
            0 => -1.0, // The mover won; after step, side to move has lost.
            1 => 0.0,
            2 => 1.0,
            _ => panic!("illegal action {action}"),
        });
    }

    fn is_terminal(&self) -> bool {
        self.reward.is_some()
    }

    fn reward(&self) -> f32 {
        self.reward.unwrap_or(0.0)
    }

    fn encode_state(&self, out: &mut [f32]) {
        out[0] = self.reward();
    }

    fn parse_move(&self, s: &str) -> Option<Action> {
        s.parse().ok().filter(|&a| a < Self::ACTION_SIZE as Action)
    }

    fn format_action(&self, action: Action) -> String {
        action.to_string()
    }
}

#[derive(Clone, Default)]
struct SinglePathGame {
    ply: u8,
}

impl fmt::Display for SinglePathGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ply={}", self.ply)
    }
}

impl Game for SinglePathGame {
    const ACTION_SIZE: usize = 1;
    const STATE_SHAPE: [crate::game::TensorDim; 3] = [1, 1, 1];
    const NAME: &'static str = "single-path";

    fn legal_actions(&self) -> impl Iterator<Item = Action> + '_ {
        (0..1).filter(|_| self.ply < 2)
    }

    fn step(&mut self, action: Action) {
        assert_eq!(action, 0);
        self.ply += 1;
    }

    fn is_terminal(&self) -> bool {
        self.ply >= 2
    }

    fn reward(&self) -> f32 {
        0.0
    }

    fn encode_state(&self, out: &mut [f32]) {
        out[0] = self.ply as f32;
    }

    fn parse_move(&self, s: &str) -> Option<Action> {
        (s == "0").then_some(0)
    }

    fn format_action(&self, action: Action) -> String {
        action.to_string()
    }
}

#[derive(Clone)]
struct TerminalGame;

impl Default for TerminalGame {
    fn default() -> Self {
        TerminalGame
    }
}

impl fmt::Display for TerminalGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "terminal")
    }
}

impl Game for TerminalGame {
    const ACTION_SIZE: usize = 2;
    const STATE_SHAPE: [crate::game::TensorDim; 3] = [1, 1, 1];
    const NAME: &'static str = "terminal";

    fn legal_actions(&self) -> impl Iterator<Item = Action> + '_ {
        std::iter::empty()
    }

    fn step(&mut self, action: Action) {
        panic!("terminal game has no legal action {action}");
    }

    fn is_terminal(&self) -> bool {
        true
    }

    fn reward(&self) -> f32 {
        -1.0
    }

    fn encode_state(&self, out: &mut [f32]) {
        out[0] = 1.0;
    }

    fn parse_move(&self, _s: &str) -> Option<Action> {
        None
    }

    fn format_action(&self, action: Action) -> String {
        action.to_string()
    }
}

struct RecordingEvaluator {
    calls: Arc<Mutex<Vec<usize>>>,
    favor_action_zero: bool,
}

impl RecordingEvaluator {
    fn uniform(calls: Arc<Mutex<Vec<usize>>>) -> Self {
        RecordingEvaluator {
            calls,
            favor_action_zero: false,
        }
    }

    fn favor_action_zero(calls: Arc<Mutex<Vec<usize>>>) -> Self {
        RecordingEvaluator {
            calls,
            favor_action_zero: true,
        }
    }
}

impl Evaluator for RecordingEvaluator {
    fn evaluate(&mut self, batch: &EvalBatch) -> Vec<Evaluation> {
        self.calls.lock().unwrap().push(batch.len());
        (0..batch.len())
            .map(|i| {
                let legal =
                    &batch.legal[batch.offsets[i] as usize..batch.offsets[i + 1] as usize];
                let logits = legal
                    .iter()
                    .map(|&a| {
                        if self.favor_action_zero && a == 0 {
                            100.0
                        }
                        else {
                            0.0
                        }
                    })
                    .collect();
                Evaluation { logits, value: 0.0 }
            })
            .collect()
    }
}

struct EmptyResultEvaluator;

impl Evaluator for EmptyResultEvaluator {
    fn evaluate(&mut self, _batch: &EvalBatch) -> Vec<Evaluation> {
        Vec::new()
    }
}

struct BadLogitEvaluator;

impl Evaluator for BadLogitEvaluator {
    fn evaluate(&mut self, batch: &EvalBatch) -> Vec<Evaluation> {
        (0..batch.len())
            .map(|_| Evaluation {
                logits: Vec::new(),
                value: 0.0,
            })
            .collect()
    }
}

fn play(moves: &[u32]) -> Connect4 {
    let mut g = Connect4::default();
    for &m in moves {
        g.step(m);
    }
    g
}

fn mcts(simulations: usize) -> Mcts<UniformEvaluator> {
    Mcts::new(
        UniformEvaluator,
        MctsConfig {
            simulations,
            batch_size: 8,
            eps: 0.0,
            ..Default::default()
        },
    )
}

#[test]
fn terminal_root_returns_reward_without_evaluator() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut mcts = Mcts::new(
        RecordingEvaluator::uniform(calls.clone()),
        MctsConfig {
            simulations: 4,
            batch_size: 2,
            eps: 0.0,
            ..Default::default()
        },
    );

    let result = mcts.search(&TerminalGame);

    assert_eq!(result.value, -1.0);
    assert_eq!(result.policy, vec![0.0, 0.0]);
    assert!(calls.lock().unwrap().is_empty());
}

#[test]
fn puct_finds_forced_terminal_win() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut mcts = Mcts::new(
        RecordingEvaluator::uniform(calls),
        MctsConfig {
            simulations: 32,
            batch_size: 8,
            eps: 0.0,
            ..Default::default()
        },
    );

    let result = mcts.search(&ImmediateOutcomeGame::default());

    assert_eq!(result.best_action(), 0);
    assert!(result.policy[0] > result.policy[1], "{:?}", result.policy);
    assert!(result.policy[0] > result.policy[2], "{:?}", result.policy);
}

#[test]
fn batching_deduplicates_colliding_leaves() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut mcts = Mcts::new(
        RecordingEvaluator::uniform(calls.clone()),
        MctsConfig {
            simulations: 4,
            batch_size: 4,
            eps: 0.0,
            ..Default::default()
        },
    );

    let result = mcts.search(&SinglePathGame::default());

    assert_eq!(result.policy, vec![1.0]);
    assert_eq!(*calls.lock().unwrap(), vec![1, 1]);
}

#[test]
fn finds_immediate_win() {
    // X has three stones in column 0 and is to move.
    let game = play(&[0, 1, 0, 1, 0, 1]);
    let result = mcts(400).search(&game);
    assert_eq!(
        result.best_action(),
        0,
        "should pick the winning column: {:?}",
        result.policy
    );
}

#[test]
fn blocks_opponent_threat() {
    // X threatens to complete column 0; O to move must block it.
    let game = play(&[0, 6, 0, 5, 0]);
    let result = mcts(2000).search(&game);
    assert_eq!(
        result.best_action(),
        0,
        "should block column 0: {:?}",
        result.policy
    );
}

#[test]
fn policy_is_a_distribution() {
    let game = Connect4::default();
    let mut mcts = Mcts::new(
        UniformEvaluator,
        MctsConfig {
            simulations: 200,
            batch_size: 16,
            eps: 0.0,
            ..Default::default()
        },
    );
    let result = mcts.search(&game);
    let sum: f32 = result.policy.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
    assert!(result.policy.iter().all(|&p| p >= 0.0));
}

#[test]
fn gumbel_policy_stays_on_sampled_root_actions() {
    let game = Connect4::default();
    let mut mcts = Mcts::new(
        UniformEvaluator,
        MctsConfig {
            variant: MctsVariant::Gumbel { sampled_actions: 3 },
            simulations: 48,
            batch_size: 8,
            eps: 0.0,
            ..Default::default()
        },
    );
    mcts.rng = SmallRng::seed_from_u64(11);

    let result = mcts.search(&game);
    let sum: f32 = result.policy.iter().sum();
    let nonzero = result.policy.iter().filter(|&&p| p > 0.0).count();

    assert!((sum - 1.0).abs() < 1e-4);
    assert!((1..=3).contains(&nonzero), "{:?}", result.policy);
    assert!(result.policy.iter().all(|&p| p >= 0.0));
}

#[test]
fn gumbel_zero_sampled_actions_still_searches_one_root_action() {
    let game = Connect4::default();
    let mut mcts = Mcts::new(
        UniformEvaluator,
        MctsConfig {
            variant: MctsVariant::Gumbel { sampled_actions: 0 },
            simulations: 24,
            batch_size: 8,
            eps: 0.0,
            ..Default::default()
        },
    );
    mcts.rng = SmallRng::seed_from_u64(7);

    let result = mcts.search(&game);

    assert_eq!(result.policy.iter().filter(|&&p| p > 0.0).count(), 1);
    assert!((result.policy.iter().sum::<f32>() - 1.0).abs() < 1e-4);
}

#[test]
fn gumbel_with_all_root_actions_finds_forced_terminal_win() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut mcts = Mcts::new(
        RecordingEvaluator::favor_action_zero(calls),
        MctsConfig {
            variant: MctsVariant::Gumbel { sampled_actions: 3 },
            simulations: 32,
            batch_size: 8,
            eps: 0.0,
            ..Default::default()
        },
    );
    mcts.rng = SmallRng::seed_from_u64(3);

    let result = mcts.search(&ImmediateOutcomeGame::default());

    assert_eq!(result.best_action(), 0);
    assert!(result.policy[0] > result.policy[1], "{:?}", result.policy);
    assert!(result.policy[0] > result.policy[2], "{:?}", result.policy);
}

#[test]
#[should_panic(expected = "MCTS batch_size must be positive")]
fn rejects_zero_batch_size() {
    let _ = Mcts::new(
        UniformEvaluator,
        MctsConfig {
            batch_size: 0,
            ..Default::default()
        },
    );
}

#[test]
#[should_panic(expected = "evaluator must return one result per input state")]
fn rejects_evaluator_result_count_mismatch() {
    let mut mcts = Mcts::new(
        EmptyResultEvaluator,
        MctsConfig {
            simulations: 1,
            batch_size: 1,
            eps: 0.0,
            ..Default::default()
        },
    );

    let _ = mcts.search(&ImmediateOutcomeGame::default());
}

#[test]
#[should_panic(expected = "evaluator logits must match the legal actions for each state")]
fn rejects_evaluator_logit_count_mismatch() {
    let mut mcts = Mcts::new(
        BadLogitEvaluator,
        MctsConfig {
            simulations: 1,
            batch_size: 1,
            eps: 0.0,
            ..Default::default()
        },
    );

    let _ = mcts.search(&ImmediateOutcomeGame::default());
}

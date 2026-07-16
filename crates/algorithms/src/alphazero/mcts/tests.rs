use super::core::MctsKind;
use super::*;
use engine_core::rules::RepetitionGame;
use games::Connect4;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

impl<E: EncodedEvaluator> Mcts<E> {
    fn seed_rng(&mut self, seed: u64) {
        match &mut self.inner {
            MctsKind::Puct(core) => core.rng = SmallRng::seed_from_u64(seed),
            MctsKind::Gumbel(core) => core.rng = SmallRng::seed_from_u64(seed),
        }
    }

    fn gumbel_root_actions(&self) -> Vec<Action> {
        match &self.inner {
            MctsKind::Puct(_) => Vec::new(),
            MctsKind::Gumbel(core) => core
                .variant
                .root_actions
                .iter()
                .map(|a| core.nodes[a.node as usize].action_from_parent)
                .collect(),
        }
    }

    fn virtual_loss_total(&self) -> u32 {
        match &self.inner {
            MctsKind::Puct(core) => core.nodes.iter().map(|n| n.virtual_loss_count).sum(),
            MctsKind::Gumbel(core) => core.nodes.iter().map(|n| n.virtual_loss_count).sum(),
        }
    }

    fn root_visits(&self) -> u32 {
        match &self.inner {
            MctsKind::Puct(core) => core.nodes[0].visits,
            MctsKind::Gumbel(core) => core.nodes[0].visits,
        }
    }

    fn node_count(&self) -> usize {
        match &self.inner {
            MctsKind::Puct(core) => core.nodes.len(),
            MctsKind::Gumbel(core) => core.nodes.len(),
        }
    }
}

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
    const STATE_SHAPE: [engine_core::game::TensorDim; 3] = [1, 1, 1];
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

#[derive(Clone, Copy, Default)]
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
    const STATE_SHAPE: [engine_core::game::TensorDim; 3] = [1, 1, 1];
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

impl RepetitionGame for SinglePathGame {
    fn repetition_hash(&self) -> u64 {
        u64::from(self.ply)
    }

    fn halfmove_clock(&self) -> usize {
        100
    }

    fn set_repetition_draw(&mut self) {}
}

#[derive(Clone, Copy, Default)]
struct CycleGame {
    ply: u8,
    repetition_draw: bool,
}

#[derive(Clone, Copy, Default)]
struct FeatureRepetitionGame {
    ply: u8,
    repetitions_before: u8,
}

impl fmt::Display for FeatureRepetitionGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ply={}", self.ply)
    }
}

impl Game for FeatureRepetitionGame {
    const ACTION_SIZE: usize = 1;
    const STATE_SHAPE: [engine_core::game::TensorDim; 3] = [1, 1, 1];
    const NAME: &'static str = "feature-repetition";

    fn legal_actions(&self) -> impl Iterator<Item = Action> + '_ {
        (0..1).filter(|_| self.ply == 0)
    }

    fn step(&mut self, action: Action) {
        assert_eq!(action, 0);
        self.ply += 1;
    }

    fn is_terminal(&self) -> bool {
        false
    }

    fn reward(&self) -> f32 {
        0.0
    }

    fn encode_state(&self, out: &mut [f32]) {
        out[0] = f32::from(self.repetitions_before);
    }

    fn parse_move(&self, s: &str) -> Option<Action> {
        (s == "0").then_some(0)
    }

    fn format_action(&self, action: Action) -> String {
        action.to_string()
    }
}

impl RepetitionGame for FeatureRepetitionGame {
    fn repetition_hash(&self) -> u64 {
        u64::from(self.ply)
    }

    fn halfmove_clock(&self) -> usize {
        100
    }

    fn set_repetitions_before_current(&mut self, count: u8) {
        self.repetitions_before = count;
    }

    fn set_repetition_draw(&mut self) {}
}

impl fmt::Display for CycleGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ply={}", self.ply)
    }
}

impl Game for CycleGame {
    const ACTION_SIZE: usize = 1;
    const STATE_SHAPE: [engine_core::game::TensorDim; 3] = [1, 1, 1];
    const NAME: &'static str = "cycle";

    fn legal_actions(&self) -> impl Iterator<Item = Action> + '_ {
        (0..1).filter(|_| self.ply < 2 && !self.repetition_draw)
    }

    fn step(&mut self, action: Action) {
        assert_eq!(action, 0);
        self.ply += 1;
    }

    fn is_terminal(&self) -> bool {
        self.repetition_draw
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

impl RepetitionGame for CycleGame {
    fn repetition_hash(&self) -> u64 {
        if self.ply.is_multiple_of(2) {
            1
        }
        else {
            2
        }
    }

    fn halfmove_clock(&self) -> usize {
        100
    }

    fn set_repetition_draw(&mut self) {
        self.repetition_draw = true;
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
    const STATE_SHAPE: [engine_core::game::TensorDim; 3] = [1, 1, 1];
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

/// Uniform policy, zero value: MCTS degenerates to a plain PUCT tree search.
struct UniformEvaluator;

impl EncodedEvaluator for UniformEvaluator {
    fn evaluate(&mut self, batch: &mut EncodedEvalBatch) -> Vec<Evaluation> {
        (0..batch.len())
            .map(|i| {
                let n = (batch.offsets[i + 1] - batch.offsets[i]) as usize;
                Evaluation {
                    logits: vec![0.0; n],
                    value: 0.0,
                }
            })
            .collect()
    }
}

struct CountingEvaluator {
    calls: Arc<AtomicUsize>,
}

impl EncodedEvaluator for CountingEvaluator {
    fn evaluate(&mut self, batch: &mut EncodedEvalBatch) -> Vec<Evaluation> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        UniformEvaluator.evaluate(batch)
    }
}

struct RecordingEvaluator {
    calls: Arc<Mutex<Vec<usize>>>,
    favor_action_zero: bool,
}

struct FeatureRecordingEvaluator {
    encoded: Arc<Mutex<Vec<f32>>>,
}

impl EncodedEvaluator for FeatureRecordingEvaluator {
    fn evaluate(&mut self, batch: &mut EncodedEvalBatch) -> Vec<Evaluation> {
        self.encoded.lock().unwrap().extend(
            batch
                .states
                .chunks_exact(FeatureRepetitionGame::state_size())
                .map(|state| state[0]),
        );
        UniformEvaluator.evaluate(batch)
    }
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

impl EncodedEvaluator for RecordingEvaluator {
    fn evaluate(&mut self, batch: &mut EncodedEvalBatch) -> Vec<Evaluation> {
        self.calls.lock().unwrap().push(batch.len());
        (0..batch.len())
            .map(|i| {
                let legal_actions =
                    &batch.legal_actions[batch.offsets[i] as usize..batch.offsets[i + 1] as usize];
                let logits = legal_actions
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

impl EncodedEvaluator for EmptyResultEvaluator {
    fn evaluate(&mut self, _batch: &mut EncodedEvalBatch) -> Vec<Evaluation> {
        Vec::new()
    }
}

struct BadLogitEvaluator;

impl EncodedEvaluator for BadLogitEvaluator {
    fn evaluate(&mut self, batch: &mut EncodedEvalBatch) -> Vec<Evaluation> {
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
            eps: 0.0,
            ..Default::default()
        },
    )
}

mod cases;

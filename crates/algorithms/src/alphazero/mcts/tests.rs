use super::*;
use crate::search::{Evaluation, NoExtraRules, PolicyValueEvaluator, RuleResult, SearchRules};
use engine_core::agent::PolicyMode;
use engine_core::game::{GameState, TerminalValue};
use games::{Connect4, Connect4Move};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TestMove {
    Left,
    Right,
}

#[test]
fn search_result_supports_native_move_types() {
    let result = SearchResult {
        policy: vec![(TestMove::Left, 0.0), (TestMove::Right, 1.0)],
        selected_move: TestMove::Right,
        value: 0.5,
    };
    assert_eq!(result.best_move(), TestMove::Right);
    assert_eq!(result.probability(TestMove::Left), 0.0);
    assert_eq!(result.probability(TestMove::Right), 1.0);

    let mut rng = SmallRng::seed_from_u64(7);
    for _ in 0..100 {
        assert_eq!(result.sample_move(&mut rng), TestMove::Right);
    }
}

#[test]
fn node_storage_accepts_non_legacy_move_types() {
    let root = super::Node::<TestMove>::new(None, None, 0.5, 0.0, false, 0.0);
    let node =
        super::Node::<TestMove, ()>::new(Some(0), Some(TestMove::Left), 0.5, 0.0, false, 0.0);
    assert_eq!(root.move_from_parent, None);
    assert_eq!(node.move_from_parent, Some(TestMove::Left));
    assert_eq!(
        std::mem::size_of_val(&node.move_from_parent),
        std::mem::size_of::<Option<TestMove>>()
    );
}

#[test]
fn generic_cache_and_policy_keep_native_move_order() {
    let table = EvalTable::new(1);
    table.insert(
        7,
        CachedEvaluation {
            legal: vec![TestMove::Right, TestMove::Left],
            eval: Evaluation {
                logits: vec![2.0, 1.0],
                value: 0.0,
            },
        },
    );
    assert_eq!(
        table.get(7).unwrap().legal,
        vec![TestMove::Right, TestMove::Left]
    );
    assert_eq!(table.stats().hits, 1);

    let mut policy = Vec::new();
    let mut rng = SmallRng::seed_from_u64(1);
    super::evaluation::build_policy(
        &mut policy,
        &[TestMove::Right, TestMove::Left],
        &Evaluation {
            logits: vec![2.0, 1.0],
            value: 0.0,
        },
        false,
        0.0,
        1.0,
        &mut rng,
    );
    assert_eq!(
        policy.iter().map(|entry| entry.0).collect::<Vec<_>>(),
        vec![TestMove::Right, TestMove::Left]
    );
    assert!((policy.iter().map(|entry| entry.1).sum::<f32>() - 1.0).abs() < 1e-6);
}

impl<G, E, R> Mcts<G, E, R>
where
    G: GameState + Clone,
    E: PolicyValueEvaluator<G>,
    R: SearchRules<G>,
{
    fn seed_rng(&mut self, seed: u64) {
        match &mut self.inner {
            super::core::MctsKind::Puct(core) => core.rng = SmallRng::seed_from_u64(seed),
            super::core::MctsKind::Gumbel(core) => core.rng = SmallRng::seed_from_u64(seed),
        }
    }

    fn gumbel_root_actions(&self) -> Vec<G::Move> {
        match &self.inner {
            super::core::MctsKind::Puct(_) => Vec::new(),
            super::core::MctsKind::Gumbel(core) => core
                .variant
                .root_actions
                .iter()
                .map(|a| {
                    core.nodes[a.node as usize]
                        .move_from_parent
                        .expect("root action must be a child move")
                })
                .collect(),
        }
    }

    fn virtual_loss_total(&self) -> u32 {
        match &self.inner {
            super::core::MctsKind::Puct(core) => {
                core.nodes.iter().map(|n| n.virtual_loss_count).sum()
            }
            super::core::MctsKind::Gumbel(core) => {
                core.nodes.iter().map(|n| n.virtual_loss_count).sum()
            }
        }
    }

    fn root_visits(&self) -> u32 {
        match &self.inner {
            super::core::MctsKind::Puct(core) => core.nodes[0].visits,
            super::core::MctsKind::Gumbel(core) => core.nodes[0].visits,
        }
    }

    fn node_count(&self) -> usize {
        match &self.inner {
            super::core::MctsKind::Puct(core) => core.nodes.len(),
            super::core::MctsKind::Gumbel(core) => core.nodes.len(),
        }
    }
}

#[derive(Clone, Default, Debug)]
struct ImmediateOutcomeGame {
    reward: Option<f32>,
}

impl GameState for ImmediateOutcomeGame {
    type Move = u8;
    fn initial() -> Self {
        Self::default()
    }
    fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
        (0..3).filter(|_| self.reward.is_none())
    }
    fn play(&mut self, action: Self::Move) {
        self.reward = Some(match action {
            0 => -1.0, // The mover won; after step, side to move has lost.
            1 => 0.0,
            2 => 1.0,
            _ => panic!("illegal action {action}"),
        });
    }

    fn terminal_value(&self) -> Option<TerminalValue> {
        self.reward.map(|r| {
            if r < 0.0 {
                TerminalValue::Loss
            } else if r > 0.0 {
                TerminalValue::Win
            } else {
                TerminalValue::Draw
            }
        })
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

impl GameState for SinglePathGame {
    type Move = u8;
    fn initial() -> Self {
        Self::default()
    }
    fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
        (0..1).filter(|_| self.ply < 2)
    }

    fn play(&mut self, action: Self::Move) {
        assert_eq!(action, 0);
        self.ply += 1;
    }

    fn terminal_value(&self) -> Option<TerminalValue> {
        (self.ply >= 2).then_some(TerminalValue::Draw)
    }
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

impl GameState for FeatureRepetitionGame {
    type Move = u8;
    fn initial() -> Self {
        Self::default()
    }
    fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
        (0..1).filter(|_| self.ply == 0)
    }
    fn play(&mut self, action: Self::Move) {
        assert_eq!(action, 0);
        self.ply += 1;
    }
    fn terminal_value(&self) -> Option<TerminalValue> {
        None
    }
}

#[derive(Clone, Copy, Default)]
struct FeatureRules;

impl SearchRules<FeatureRepetitionGame> for FeatureRules {
    type Context<'a>
        = ()
    where
        FeatureRepetitionGame: 'a;
    type PathState = ();
    type NodeMeta = ();

    fn reset_path<'a>(&self, _: Self::Context<'a>, _: &FeatureRepetitionGame, _: &mut ()) {}

    fn enter_state<'a>(
        &self,
        _: Self::Context<'a>,
        state: &mut FeatureRepetitionGame,
        _: &mut (),
        _: &mut (),
    ) -> RuleResult {
        state.repetitions_before = u8::from(state.ply > 0);
        RuleResult::Continue
    }
}

#[derive(Clone, Copy, Default)]
struct CycleRules;

impl SearchRules<CycleGame> for CycleRules {
    type Context<'a>
        = ()
    where
        CycleGame: 'a;
    type PathState = ();
    type NodeMeta = ();

    fn reset_path<'a>(&self, _: Self::Context<'a>, _: &CycleGame, _: &mut ()) {}

    fn enter_state<'a>(
        &self,
        _: Self::Context<'a>,
        state: &mut CycleGame,
        _: &mut (),
        _: &mut (),
    ) -> RuleResult {
        if state.ply >= 2 {
            state.repetition_draw = true;
            RuleResult::Terminal(TerminalValue::Draw)
        } else {
            RuleResult::Continue
        }
    }
}
/* legacy repetition hooks intentionally removed; SearchRules below owns them. */

impl fmt::Display for CycleGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ply={}", self.ply)
    }
}

impl GameState for CycleGame {
    type Move = u8;
    fn initial() -> Self {
        Self::default()
    }
    fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
        (0..1).filter(|_| self.ply < 2 && !self.repetition_draw)
    }

    fn play(&mut self, action: Self::Move) {
        assert_eq!(action, 0);
        self.ply += 1;
    }

    fn terminal_value(&self) -> Option<TerminalValue> {
        self.repetition_draw.then_some(TerminalValue::Draw)
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

impl GameState for TerminalGame {
    type Move = u8;
    fn initial() -> Self {
        Self
    }
    fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
        std::iter::empty()
    }

    fn play(&mut self, action: Self::Move) {
        panic!("terminal game has no legal action {action}");
    }

    fn terminal_value(&self) -> Option<TerminalValue> {
        Some(TerminalValue::Loss)
    }
}

/// Uniform policy, zero value: MCTS degenerates to a plain PUCT tree search.
struct UniformEvaluator;
impl<G: GameState> PolicyValueEvaluator<G> for UniformEvaluator {
    fn evaluate(&mut self, _states: &[G], _legal: &[G::Move], offsets: &[u32]) -> Vec<Evaluation> {
        offsets
            .windows(2)
            .map(|w| Evaluation {
                logits: vec![0.0; (w[1] - w[0]) as usize],
                value: 0.0,
            })
            .collect()
    }
}

struct CountingEvaluator {
    calls: Arc<AtomicUsize>,
}

impl<G: GameState> PolicyValueEvaluator<G> for CountingEvaluator {
    fn evaluate(&mut self, states: &[G], legal: &[G::Move], offsets: &[u32]) -> Vec<Evaluation> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        UniformEvaluator.evaluate(states, legal, offsets)
    }
}

struct RecordingEvaluator {
    calls: Arc<Mutex<Vec<usize>>>,
    favor_action_zero: bool,
}

struct FeatureRecordingEvaluator {
    encoded: Arc<Mutex<Vec<f32>>>,
}

impl PolicyValueEvaluator<FeatureRepetitionGame> for FeatureRecordingEvaluator {
    fn evaluate(
        &mut self,
        states: &[FeatureRepetitionGame],
        legal: &[u8],
        offsets: &[u32],
    ) -> Vec<Evaluation> {
        self.encoded
            .lock()
            .unwrap()
            .extend(states.iter().map(|state| state.repetitions_before as f32));
        UniformEvaluator.evaluate(states, legal, offsets)
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

impl<G: GameState> PolicyValueEvaluator<G> for RecordingEvaluator {
    fn evaluate(&mut self, _states: &[G], legal: &[G::Move], offsets: &[u32]) -> Vec<Evaluation> {
        self.calls.lock().unwrap().push(offsets.len() - 1);
        (0..offsets.len() - 1)
            .map(|i| {
                let logits = legal[offsets[i] as usize..offsets[i + 1] as usize]
                    .iter()
                    .map(|a| {
                        if self.favor_action_zero && format!("{a:?}") == "0" {
                            100.0
                        } else {
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
impl<G: GameState> PolicyValueEvaluator<G> for EmptyResultEvaluator {
    fn evaluate(&mut self, _states: &[G], _legal: &[G::Move], _offsets: &[u32]) -> Vec<Evaluation> {
        Vec::new()
    }
}

struct BadLogitEvaluator;

impl<G: GameState> PolicyValueEvaluator<G> for BadLogitEvaluator {
    fn evaluate(&mut self, states: &[G], _legal: &[G::Move], _offsets: &[u32]) -> Vec<Evaluation> {
        (0..states.len())
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
        g.play(Connect4Move::new(m as u8).unwrap());
    }
    g
}

fn mcts<G: GameState + Clone>(simulations: usize) -> Mcts<G, UniformEvaluator, NoExtraRules> {
    Mcts::new(
        UniformEvaluator,
        MctsConfig {
            simulations,
            eps: 0.0,
            ..Default::default()
        },
        NoExtraRules,
    )
}

mod cases;

use super::representation::{Action, AlphaZeroRepresentation};
use engine_core::GameState;
pub use search::{Evaluation, EvaluationError, PolicyValueEvaluator};
use std::marker::PhantomData;

/// A batch of canonically-encoded states to evaluate, each with its legal
/// actions. The search computes legal actions anyway to expand its nodes, so
/// they travel with the request rather than being re-derived by the
/// evaluator; a network evaluator then only returns logits for the actions
/// that will actually be read.
pub struct EncodedEvalBatch {
    /// `n * state_size` floats, states back to back.
    pub states: Vec<f32>,
    /// Legal action ids for all states, flattened.
    pub legal_actions: Vec<Action>,
    /// `n + 1` offsets into `legal_actions`.
    pub offsets: Vec<u32>,
}

impl EncodedEvalBatch {
    pub fn new() -> Self {
        EncodedEvalBatch {
            states: Vec::new(),
            legal_actions: Vec::new(),
            offsets: vec![0],
        }
    }

    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.legal_actions.clear();
        self.offsets.truncate(1);
    }
}

impl Default for EncodedEvalBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Position evaluator used by MCTS: policy logits over legal actions plus a
/// value in [-1, 1] from the side to move's perspective.
pub trait EncodedEvaluator: Send {
    /// Evaluates `batch` while allowing implementations to temporarily take
    /// ownership of its backing allocations. Implementations must restore a
    /// reusable batch before returning.
    fn evaluate(
        &mut self,
        batch: &mut EncodedEvalBatch,
    ) -> Result<Vec<Evaluation>, EvaluationError>;
}

/// Adapts native states and moves through a representation to an encoded evaluator.
pub struct RepresentedEvaluator<G, R, E> {
    representation: R,
    encoded: E,
    batch: EncodedEvalBatch,
    _game: PhantomData<fn() -> G>,
}

impl<G, R, E> RepresentedEvaluator<G, R, E> {
    /// Creates an adapter with an owned, reusable encoded batch.
    pub fn new(representation: R, encoded: E) -> Self {
        Self {
            representation,
            encoded,
            batch: EncodedEvalBatch::new(),
            _game: PhantomData,
        }
    }
}

impl<G, R, E> PolicyValueEvaluator<G> for RepresentedEvaluator<G, R, E>
where
    G: GameState,
    R: AlphaZeroRepresentation<G>,
    E: EncodedEvaluator,
{
    fn evaluate(
        &mut self,
        states: &[G],
        legal_moves: &[G::Move],
        offsets: &[u32],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        assert_eq!(
            offsets.len(),
            states.len() + 1,
            "offsets must have one entry per state plus a sentinel"
        );
        assert_eq!(offsets.first(), Some(&0), "offsets must start at zero");
        assert!(
            offsets
                .windows(2)
                .all(|range| { range[0] <= range[1] && range[1] as usize <= legal_moves.len() }),
            "offsets must be monotonic and in range"
        );
        assert_eq!(
            offsets.last().copied().unwrap_or(0) as usize,
            legal_moves.len(),
            "offsets must cover legal moves"
        );
        self.batch.clear();
        for (row, state) in states.iter().enumerate() {
            let start = self.batch.states.len();
            self.batch.states.resize(start + R::state_size(), 0.0);
            self.representation
                .encode_state(state, &mut self.batch.states[start..]);
            self.batch.legal_actions.extend(
                legal_moves[offsets[row] as usize..offsets[row + 1] as usize]
                    .iter()
                    .copied()
                    .map(|mv| self.representation.move_to_action(state, mv)),
            );
            self.batch
                .offsets
                .push(self.batch.legal_actions.len() as u32);
        }
        self.encoded.evaluate(&mut self.batch)
    }

    fn evaluation_key(&self, state: &G) -> Option<u64> {
        Some(self.representation.encoded_state_key(state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::TerminalValue;
    use search::PositionValue;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Copy)]
    struct TestState(u32);

    impl GameState for TestState {
        type Move = u32;

        fn initial() -> Self {
            Self(0)
        }

        fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
            std::iter::empty()
        }

        fn play(&mut self, _mv: Self::Move) {}

        fn terminal_value(&self) -> Option<TerminalValue> {
            None
        }
    }

    #[derive(Clone)]
    struct TestRepresentation;

    impl AlphaZeroRepresentation<TestState> for TestRepresentation {
        const STATE_SHAPE: [usize; 3] = [1, 1, 2];
        const ACTION_SIZE: usize = 1_000;

        fn encode_state(&self, state: &TestState, output: &mut [f32]) {
            output.copy_from_slice(&[state.0 as f32, state.0 as f32 + 0.5]);
        }

        fn move_to_action(&self, state: &TestState, mv: u32) -> Action {
            Action::new(state.0 * 100 + mv)
        }

        fn action_to_move(&self, _state: &TestState, _action: Action) -> Option<u32> {
            None
        }

        fn encoded_state_key(&self, state: &TestState) -> u64 {
            u64::from(state.0) * 7
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct RecordedBatch {
        states: Vec<u32>,
        actions: Vec<u32>,
        offsets: Vec<u32>,
        pointers: [usize; 3],
        capacities: [usize; 3],
    }

    struct RecordingEvaluator(Arc<Mutex<Vec<RecordedBatch>>>);

    impl EncodedEvaluator for RecordingEvaluator {
        fn evaluate(
            &mut self,
            batch: &mut EncodedEvalBatch,
        ) -> Result<Vec<Evaluation>, EvaluationError> {
            self.0.lock().unwrap().push(RecordedBatch {
                states: batch.states.iter().map(|value| value.to_bits()).collect(),
                actions: batch
                    .legal_actions
                    .iter()
                    .map(|action| action.as_u32())
                    .collect(),
                offsets: batch.offsets.clone(),
                pointers: [
                    batch.states.as_ptr() as usize,
                    batch.legal_actions.as_ptr() as usize,
                    batch.offsets.as_ptr() as usize,
                ],
                capacities: [
                    batch.states.capacity(),
                    batch.legal_actions.capacity(),
                    batch.offsets.capacity(),
                ],
            });
            Ok((0..batch.len())
                .map(|row| {
                    let begin = batch.offsets[row] as usize;
                    let end = batch.offsets[row + 1] as usize;
                    Evaluation {
                        logits: batch.legal_actions[begin..end]
                            .iter()
                            .map(|action| action.as_u32() as f32)
                            .collect(),
                        value: PositionValue::new_clamped(row as f32 + 0.25),
                    }
                })
                .collect())
        }
    }

    fn adapter(
        records: &Arc<Mutex<Vec<RecordedBatch>>>,
    ) -> RepresentedEvaluator<TestState, TestRepresentation, RecordingEvaluator> {
        RepresentedEvaluator::new(TestRepresentation, RecordingEvaluator(Arc::clone(records)))
    }

    #[test]
    fn maps_rows_preserves_order_delegates_keys_and_reuses_allocations() {
        let records = Arc::new(Mutex::new(Vec::new()));
        let mut evaluator = adapter(&records);
        let evaluations = evaluator
            .evaluate(&[TestState(2), TestState(5)], &[9, 3, 8], &[0, 2, 3])
            .unwrap();

        assert_eq!(evaluations[0].logits, [209.0, 203.0]);
        assert_eq!(evaluations[0].value, PositionValue::new_clamped(0.25));
        assert_eq!(evaluations[1].logits, [508.0]);
        assert_eq!(evaluations[1].value, PositionValue::WIN);
        assert_eq!(evaluator.evaluation_key(&TestState(6)), Some(42));

        evaluator.evaluate(&[TestState(1)], &[4], &[0, 1]).unwrap();
        assert!(evaluator.evaluate(&[], &[], &[0]).unwrap().is_empty());
        let records = records.lock().unwrap();
        assert_eq!(
            records[0].states,
            [
                2.0f32.to_bits(),
                2.5f32.to_bits(),
                5.0f32.to_bits(),
                5.5f32.to_bits()
            ]
        );
        assert_eq!(records[0].actions, [209, 203, 508]);
        assert_eq!(records[0].offsets, [0, 2, 3]);
        assert_eq!(records[1].pointers, records[0].pointers);
        assert_eq!(records[1].capacities, records[0].capacities);
        assert_eq!(records[2].offsets, [0]);
    }

    #[test]
    #[should_panic(expected = "offsets must start at zero")]
    fn rejects_nonzero_first_offset() {
        let _ = adapter(&Arc::new(Mutex::new(Vec::new()))).evaluate(&[TestState(1)], &[2], &[1, 1]);
    }

    #[test]
    #[should_panic(expected = "offsets must be monotonic and in range")]
    fn rejects_nonmonotonic_offsets() {
        let _ = adapter(&Arc::new(Mutex::new(Vec::new()))).evaluate(
            &[TestState(1), TestState(2)],
            &[3],
            &[0, 1, 0],
        );
    }

    #[test]
    #[should_panic(expected = "offsets must be monotonic and in range")]
    fn rejects_out_of_range_offsets() {
        let _ = adapter(&Arc::new(Mutex::new(Vec::new()))).evaluate(&[TestState(1)], &[2], &[0, 2]);
    }
}

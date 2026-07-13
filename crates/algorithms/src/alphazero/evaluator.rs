use engine_core::game::Action;

/// A batch of canonically-encoded states to evaluate, each with its legal
/// actions. The search computes legal actions anyway to expand its nodes, so
/// they travel with the request rather than being re-derived by the
/// evaluator; a network evaluator then only returns logits for the actions
/// that will actually be read.
pub struct EvalBatch {
    /// `n * state_size` floats, states back to back.
    pub states: Vec<f32>,
    /// Legal action ids for all states, flattened.
    pub legal: Vec<Action>,
    /// `n + 1` offsets into `legal`.
    pub offsets: Vec<u32>,
}

impl EvalBatch {
    pub fn new() -> Self {
        EvalBatch {
            states: Vec::new(),
            legal: Vec::new(),
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
        self.legal.clear();
        self.offsets.truncate(1);
    }
}

impl Default for EvalBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Network output for one state: `logits[k]` corresponds to the k-th legal
/// action the request supplied for that state.
#[derive(Clone)]
pub struct Evaluation {
    pub logits: Vec<f32>,
    pub value: f32,
}

/// Position evaluator used by MCTS: policy logits over legal actions plus a
/// value in [-1, 1] from the side to move's perspective.
pub trait Evaluator: Send {
    /// Evaluates `batch` while allowing implementations to temporarily take
    /// ownership of its backing allocations. Implementations must restore a
    /// reusable batch before returning.
    fn evaluate(&mut self, batch: &mut EvalBatch) -> Vec<Evaluation>;
}

use crate::PositionValue;
use engine_core::GameState;
use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvaluationError {
    Message(String),
    ResultCardinality { expected: usize, actual: usize },
}

impl EvaluationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    pub const fn result_cardinality(expected: usize, actual: usize) -> Self {
        Self::ResultCardinality { expected, actual }
    }
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => formatter.write_str(message),
            Self::ResultCardinality { expected, actual } => {
                write!(
                    formatter,
                    "evaluator returned {actual} results for {expected} states"
                )
            }
        }
    }
}

impl Error for EvaluationError {}

/// Policy/value evaluation for one state in native legal-move order.
#[derive(Clone)]
pub struct Evaluation {
    pub logits: Vec<f32>,
    /// Value from the evaluated state's player-to-move perspective.
    pub value: PositionValue,
}

/// Evaluates native game states and their flattened legal moves for search.
pub trait PolicyValueEvaluator<G: GameState>: Send {
    /// `legal_moves[offsets[i]..offsets[i + 1]]` belongs to `states[i]`.
    fn evaluate(
        &mut self,
        states: &[G],
        legal_moves: &[G::Move],
        offsets: &[u32],
    ) -> Result<Vec<Evaluation>, EvaluationError>;

    /// Returns a complete evaluation-cache key, or `None` to disable caching.
    fn evaluation_key(&self, state: &G) -> Option<u64> {
        let _ = state;
        None
    }
}

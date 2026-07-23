use crate::PositionValue;
use engine_core::GameState;
use std::error::Error;
use std::fmt;

/// Identifies one evaluator result. The namespace changes whenever the model
/// behind an evaluator changes, so cache entries cannot cross model reloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EvaluationKey {
    pub state: u64,
    pub namespace: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EvaluationError {
    Message(String),
    ResultCardinality {
        expected: usize,
        actual: usize,
    },
    LogitCardinality {
        row: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteLogit {
        row: usize,
        index: usize,
    },
    InvalidValue {
        row: usize,
        value: f32,
    },
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
            Self::LogitCardinality {
                row,
                expected,
                actual,
            } => write!(
                formatter,
                "evaluator row {row} returned {actual} logits for {expected} legal moves"
            ),
            Self::NonFiniteLogit { row, index } => {
                write!(
                    formatter,
                    "evaluator row {row} returned a non-finite logit at {index}"
                )
            }
            Self::InvalidValue { row, value } => {
                write!(
                    formatter,
                    "evaluator row {row} returned invalid value {value}"
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
    fn evaluation_key(&self, state: &G) -> Option<EvaluationKey> {
        let _ = state;
        None
    }
}

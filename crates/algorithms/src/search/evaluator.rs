use engine_core::GameState;

/// Policy/value evaluation for one state in native legal-move order.
#[derive(Clone)]
pub struct Evaluation {
    pub logits: Vec<f32>,
    pub value: f32,
}

/// Evaluates native game states and their flattened legal moves for search.
pub trait PolicyValueEvaluator<G: GameState>: Send {
    /// `legal_moves[offsets[i]..offsets[i + 1]]` belongs to `states[i]`.
    fn evaluate(
        &mut self,
        states: &[G],
        legal_moves: &[G::Move],
        offsets: &[u32],
    ) -> Vec<Evaluation>;

    /// Returns a complete evaluation-cache key, or `None` to disable caching.
    fn evaluation_key(&self, state: &G) -> Option<u64> {
        let _ = state;
        None
    }
}

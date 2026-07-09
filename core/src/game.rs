use std::fmt::Display;

pub type Action = u32;
pub type TensorDim = i64;

/// A two-player, zero-sum, perfect-information game
/// Conventions:
/// - Actions are ids in a fixed action space `[0, ACTION_SIZE)`; at any
///   position the legal actions are a subset of it.
/// - `step` always flips the side to move, including on a game-ending move.
/// - `reward` is from the perspective of the player to move at the current
///   state. Because `step` flips side-to-move even on a winning move,
///   decisive terminal states have `reward() == -1.0`; draws and non-terminal
///   states have `reward() == 0.0`.
/// - `Default` is the starting position; `Display` renders the position
pub trait Game: Clone + Default + Display + Send + 'static {
    /// Size of the fixed action space.
    const ACTION_SIZE: usize;
    /// Canonical state tensor shape: [channels, height, width].
    const STATE_SHAPE: [TensorDim; 3];
    /// Short identifier used in run configs ("connect4", "chess").
    const NAME: &'static str;

    /// Flat length of the canonical state encoding.
    fn state_size() -> usize {
        (Self::STATE_SHAPE[0] * Self::STATE_SHAPE[1] * Self::STATE_SHAPE[2]) as usize
    }

    /// Legal action ids in the current position.
    fn legal_actions(&self) -> impl Iterator<Item = Action> + '_;

    /// Applies a legal action. May panic on illegal input.
    fn step(&mut self, action: Action);

    fn is_terminal(&self) -> bool;
    fn reward(&self) -> f32;

    /// Writes the canonical (side-to-move perspective) state encoding into
    /// `out`, which must have length `state_size()`.
    fn encode_state(&self, out: &mut [f32]);

    /// Parses a human-entered move into an action id, if legal in the current position.
    fn parse_move(&self, s: &str) -> Option<Action>;

    /// Human-readable form of an action in the current position.
    fn format_action(&self, action: Action) -> String;
}

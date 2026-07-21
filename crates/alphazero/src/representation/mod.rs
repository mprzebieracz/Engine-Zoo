use engine_core::GameState;

pub mod chess_v2;
mod chess_v2_state;
pub mod compat_v1;
pub mod connect4;
pub use chess_v2::ChessAzRepresentation;
pub use chess_v2_state::ChessAzState;
pub use compat_v1::ChessV1Representation;
pub use connect4::Connect4AzRepresentation;

/// Index into the fixed policy vector of one AlphaZero representation.
/// This is not a native game move.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Action(u32);

impl Action {
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

pub trait AlphaZeroRepresentation<G: GameState>: Clone + Send + Sync + 'static {
    const STATE_SHAPE: [usize; 3];
    const ACTION_SIZE: usize;

    fn encode_state(&self, state: &G, output: &mut [f32]);

    fn move_to_action(&self, state: &G, mv: G::Move) -> Action;

    fn action_to_move(&self, state: &G, action: Action) -> Option<G::Move>;

    /// Identifies every field that can change encoding or legal-action policy lookup.
    fn encoded_state_key(&self, state: &G) -> u64;

    fn state_size() -> usize {
        Self::STATE_SHAPE.into_iter().product()
    }
}

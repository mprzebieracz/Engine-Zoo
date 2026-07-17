//! Chess game states and policy encodings.

mod action;
mod az;
mod game;
mod legacy;
pub mod notation;
mod position;
mod repetition;
mod zobrist;

pub use action::{
    decode_v1_action, decode_v2_action, encode_v1_action, encode_v2_action, AzActionError,
    AZ_ACTION_SIZE,
};
pub use az::ChessHistoryState;
pub use game::{ChessGame, ChessRepetitionContext};
pub use notation::ChessUciNotation;
pub use position::ChessPosition;
#[cfg(test)]
use position::Status;
pub use repetition::RepetitionTracker;

use engine_core::game::GameState;

pub trait ChessRepetitionState: GameState<Move = chess::ChessMove> {
    fn repetition_hash(&self) -> u64;
    fn reversible_plies(&self) -> usize;

    fn set_current_repetitions_before(&mut self, count: u8) {
        let _ = count;
    }
}

#[cfg(test)]
use chess::{Board, BoardStatus, ChessMove, Color, MoveGen, Piece, Square};

#[cfg(test)]
mod tests;

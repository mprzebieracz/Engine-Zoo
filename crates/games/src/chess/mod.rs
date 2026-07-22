//! Native chess rules, state, repetition, and notation.

mod game;
pub mod notation;
mod position;
mod repetition;
mod zobrist;

pub use game::{ChessAdjudication, ChessGame, ChessRepetitionContext};
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
use chess::{ChessMove, Color, Piece, Square};

#[cfg(test)]
mod tests;

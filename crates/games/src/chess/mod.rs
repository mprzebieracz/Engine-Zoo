//! Chess game states and policy encodings.

mod action;
mod az;
mod az_game;
mod game;
mod legacy;
pub mod notation;
mod position;
mod zobrist;

pub use action::{
    decode_v1_action, decode_v2_action, encode_v1_action, encode_v2_action, AzActionError,
    AZ_ACTION_SIZE,
};
pub use az::ChessAzState;
pub use az_game::ChessAzGame;
pub use game::ChessGame;
pub use legacy::ChessLegacyState;
pub use notation::ChessUciNotation;
pub use position::ChessPosition;
#[cfg(test)]
use position::Status;

#[cfg(test)]
use chess::{Board, BoardStatus, ChessMove, Color, MoveGen, Piece, Square};
#[cfg(test)]
use engine_core::game::{Action, Game};
#[cfg(test)]
use std::collections::HashMap;

#[cfg(test)]
mod tests;

//! Chess game states and policy encodings.

mod action;
mod az;
mod az_game;
mod game;
mod legacy;
pub mod notation;
mod position;

pub use action::{
    decode_az_move, decode_move, encode_az_move, encode_move, AzActionError, AZ_ACTION_SIZE,
};
pub use az::ChessAzState;
pub use az_game::ChessAzGame;
pub use game::ChessGame;
pub use legacy::ChessLegacyState;
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

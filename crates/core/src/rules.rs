use crate::game::Game;
use serde::Serialize;

/// Optional frontend-oriented state projection for a game.
pub trait BoardView: Game {
    type View: Serialize;

    fn board_view(&self) -> Self::View;
}

/// Optional typed position loading/saving for a game.
pub trait PositionCodec: Game {
    type Position;

    fn from_position(position: &Self::Position) -> anyhow::Result<Self>
    where
        Self: Sized;
}

/// Game capability for repetition-aware search.
///
/// This intentionally lives in `core` so games can expose the capability
/// without depending on a particular search algorithm crate.
pub trait RepetitionGame: Game + Copy {
    fn repetition_hash(&self) -> u64;
    fn halfmove_clock(&self) -> usize;
    fn set_repetition_draw(&mut self);
}

/// Hook point for future per-game search rules.
pub trait SearchRules<G: Game>: Copy {
    fn root_hash(self, _game: &G) -> u64 {
        0
    }
}

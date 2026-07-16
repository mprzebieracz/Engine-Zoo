use crate::game::Game;

/// Game capability for repetition-aware search.
///
/// This intentionally lives in `core` so games can expose the capability
/// without depending on a particular search algorithm crate.
pub trait RepetitionGame: Game + Copy {
    fn repetition_hash(&self) -> u64;
    /// Key for cached network evaluations. By default a game's position hash
    /// is sufficient, but encodings that include history must override this
    /// with every feature that can affect network output.
    fn evaluation_cache_key(&self) -> u64 {
        self.repetition_hash()
    }
    fn halfmove_clock(&self) -> usize;
    /// Supplies the full-game repetition count for the current search state.
    /// Games whose encoding has no repetition feature can keep the default.
    fn set_repetitions_before_current(&mut self, _count: u8) {}
    fn set_repetition_draw(&mut self);
}

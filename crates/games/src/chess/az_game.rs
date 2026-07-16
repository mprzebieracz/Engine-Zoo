use super::action::decode_v2_action;
use super::az::ChessAzState;
use super::game::ChessGame;
use super::notation;
use super::position::ChessPosition;
use super::zobrist::ZobristBuildHasher;
use chess::Board;
use engine_core::game::{Action, Game};
use engine_core::rules::RepetitionGame;
use std::collections::HashMap;

/// Authoritative full-game state for AlphaZero play.
///
/// It owns full repetition adjudication and the bounded, allocation-free
/// search snapshot together. Callers step this type once, then pass
/// [`search_state`](Self::search_state) into MCTS.
#[derive(Clone)]
pub struct ChessAzGame<const HISTORY: usize> {
    search: ChessAzState<HISTORY>,
    position_counts: HashMap<u64, u8, ZobristBuildHasher>,
}

impl<const HISTORY: usize> ChessAzGame<HISTORY> {
    pub fn new(position: ChessPosition) -> Self {
        let hash = position.hash();
        Self {
            search: ChessAzState::new(position),
            position_counts: HashMap::from_iter([(hash, 1)]),
        }
    }

    /// Converts a legacy game at its current position.
    /// Earlier AZ feature frames are intentionally unavailable, while the
    /// full repetition table is preserved for MCTS adjudication.
    pub fn from_game(game: &ChessGame) -> Self {
        let mut this = Self::new(game.position());
        this.position_counts = game.position_counts.clone();
        this.search.set_repetitions_before_current(
            game.repetitions_before_current(game.position().hash()),
        );
        this
    }

    pub fn from_fen(fen: &str) -> anyhow::Result<Self> {
        Ok(Self::from_game(&ChessGame::from_fen(fen)?))
    }

    pub fn search_state(&self) -> ChessAzState<HISTORY> {
        self.search
    }

    pub fn board(&self) -> &Board {
        self.search.board()
    }

    pub fn position(&self) -> ChessPosition {
        self.search.position()
    }

    /// Number of times `hash` occurred before the current search root.
    ///
    /// MCTS supplies this for every descendant hash, then combines it with
    /// repetitions along its simulated branch to adjudicate threefold draws.
    pub fn repetitions_before(&self, hash: u64) -> u8 {
        self.repetitions_before_root(hash, self.position().hash())
    }

    /// Number of occurrences before an explicitly cached search root.
    pub fn repetitions_before_root(&self, hash: u64, root_hash: u64) -> u8 {
        let count = self.position_counts.get(&hash).copied().unwrap_or(0);
        // The table includes the current root only when this is its hash.
        // Every other position occurred strictly before the root and must be
        // passed through unchanged for a descendant MCTS repetition check.
        if hash == root_hash {
            count.saturating_sub(1)
        }
        else {
            count
        }
    }

    pub fn repetitions_before_current(&self) -> u8 {
        self.repetitions_before(self.position().hash())
    }

    pub fn parse_move(&self, text: &str) -> Option<Action> {
        self.search.parse_move(text)
    }

    pub fn format_action(&self, action: Action) -> String {
        self.search.format_action(action)
    }

    pub fn san_for_action(&self, action: Action) -> String {
        match decode_v2_action(self.board(), action) {
            Ok(mv) => notation::san(self.board(), mv),
            Err(_) => self.format_action(action),
        }
    }

    pub fn step(&mut self, action: Action) {
        self.search.step(action);
        if self.search.is_terminal() {
            return;
        }
        let position = self.search.position();
        if position.halfmove_clock() == 0 {
            self.position_counts.clear();
        }
        let count = self.position_counts.entry(position.hash()).or_insert(0);
        *count += 1;
        self.search
            .set_repetitions_before_current(count.saturating_sub(1));
        if *count >= 3 {
            self.search.set_repetition_draw();
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.search.is_terminal()
    }

    pub fn reward(&self) -> f32 {
        self.search.reward()
    }
}

impl<const HISTORY: usize> Default for ChessAzGame<HISTORY> {
    fn default() -> Self {
        Self::new(ChessPosition::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::game::Game;

    fn assert_rules_for_history<const HISTORY: usize>() {
        let mut game = ChessAzGame::<HISTORY>::default();
        assert_eq!(game.search_state().legal_actions().count(), 20);
        for mv in [
            "b1c3", "b8c6", "c3b1", "c6b8", "b1c3", "b8c6", "c3b1", "c6b8",
        ] {
            game.step(game.parse_move(mv).unwrap());
        }
        assert!(game.is_terminal());
        assert_eq!(game.reward(), 0.0);
        assert_eq!(game.repetitions_before_current(), 2);
    }

    #[test]
    fn legal_moves_and_full_repetition_work_for_every_supported_history() {
        assert_rules_for_history::<1>();
        assert_rules_for_history::<4>();
        assert_rules_for_history::<8>();
    }

    #[test]
    fn one_step_updates_search_history_and_full_repetition() {
        let mut game = ChessAzGame::<8>::default();
        for mv in [
            "b1c3", "b8c6", "c3b1", "c6b8", "b1c3", "b8c6", "c3b1", "c6b8",
        ] {
            let action = game.parse_move(mv).unwrap();
            game.step(action);
        }
        assert!(game.is_terminal());
        assert_eq!(game.reward(), 0.0);
        assert!(game.search_state().is_terminal());
        assert_eq!(game.repetitions_before_current(), 2);
        assert_eq!(game.board(), game.search_state().board());
    }

    #[test]
    fn aggregate_and_full_game_stay_in_lockstep() {
        let mut aggregate = ChessAzGame::<8>::default();
        let mut full = ChessGame::default();
        for mv in [
            "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6",
        ] {
            let san = full.san_for_action(full.parse_move(mv).unwrap());
            assert_eq!(
                aggregate.san_for_action(aggregate.parse_move(mv).unwrap()),
                san
            );
            aggregate.step(aggregate.parse_move(mv).unwrap());
            full.step(full.parse_move(mv).unwrap());
            assert_eq!(aggregate.position().hash(), full.position().hash());
            assert_eq!(
                aggregate.position().halfmove_clock(),
                full.position().halfmove_clock()
            );
            assert_eq!(aggregate.is_terminal(), full.is_terminal());
        }
    }

    #[test]
    fn conversion_preserves_current_repetition_count() {
        let mut full = ChessGame::default();
        let start = full.position().hash();
        for mv in ["b1c3", "b8c6", "c3b1", "c6b8"] {
            full.step(full.parse_move(mv).unwrap());
        }
        let aggregate = ChessAzGame::<4>::from_game(&full);
        assert_eq!(aggregate.position().hash(), full.position().hash());
        assert_eq!(aggregate.repetitions_before_current(), 1);
        assert_eq!(aggregate.repetitions_before(start), 1);
        let mut encoded = vec![0.0; ChessAzState::<4>::state_size()];
        aggregate.search_state().encode_state(&mut encoded);
        assert!(encoded[12 * 64..13 * 64].iter().all(|&value| value == 1.0));
        assert!(encoded[13 * 64..14 * 64].iter().all(|&value| value == 0.0));
    }

    #[test]
    fn irreversible_move_discards_old_repetition_counts() {
        let mut game = ChessAzGame::<8>::default();
        for mv in ["b1c3", "b8c6", "c3b1", "c6b8"] {
            game.step(game.parse_move(mv).unwrap());
        }
        assert_eq!(game.repetitions_before_current(), 1);
        game.step(game.parse_move("a2a3").unwrap());
        assert_eq!(game.repetitions_before_current(), 0);
    }

    #[test]
    fn retains_counts_for_non_current_positions_for_mcts() {
        let mut game = ChessAzGame::<4>::default();
        let start = game.position().hash();
        for mv in ["b1c3", "b8c6", "c3b1", "c6b8", "g1f3"] {
            game.step(game.parse_move(mv).unwrap());
        }

        assert_ne!(game.position().hash(), start);
        assert_eq!(game.repetitions_before(start), 2);
        assert_eq!(game.repetitions_before_current(), 0);
    }
}

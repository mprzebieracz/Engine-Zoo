use chess::{Board, ChessMove};
use engine_core::game::{GameState, TerminalValue};
use games::{ChessGame, ChessPosition, ChessRepetitionState};
use std::fmt;

/// Branch-local chess state used by the AlphaZero v2 representation.
///
/// `HISTORY` includes the current position and is deliberately fixed-size so
/// MCTS children remain allocation-free. Only 1, 4, and 8 are supported.
#[derive(Clone, Copy, Debug)]
pub struct ChessAzState<const HISTORY: usize> {
    frames: [Option<Frame>; HISTORY],
}

#[derive(Clone, Copy, Debug)]
struct Frame {
    position: ChessPosition,
    repetitions_before: u8,
}

impl<const HISTORY: usize> ChessAzState<HISTORY> {
    pub const INPUT_PLANES: usize = 14 * HISTORY + 7;

    pub fn new(current: ChessPosition) -> Self {
        assert_supported_history::<HISTORY>();
        let mut frames = [None; HISTORY];
        frames[0] = Some(Frame {
            position: current,
            repetitions_before: 0,
        });
        Self { frames }
    }

    pub fn from_game(game: &ChessGame) -> Self {
        assert_supported_history::<HISTORY>();
        let mut frames = [None; HISTORY];
        for (slot, (position, repetitions_before)) in frames.iter_mut().zip(game.recent_positions())
        {
            *slot = Some(Frame {
                position,
                repetitions_before,
            });
        }
        Self { frames }
    }

    pub fn position(&self) -> ChessPosition {
        self.current().position
    }

    pub fn board(&self) -> &Board {
        self.current().position.board()
    }

    pub fn for_each_frame(&self, mut f: impl FnMut(usize, Option<ChessPosition>, u8)) {
        for (index, frame) in self.frames.iter().enumerate() {
            f(
                index,
                frame.map(|frame| frame.position),
                frame.map_or(0, |frame| frame.repetitions_before),
            );
        }
    }

    fn current(&self) -> &Frame {
        self.frames[0]
            .as_ref()
            .expect("current frame always exists")
    }

    fn play_native(&mut self, mv: ChessMove) {
        assert!(self.board().legal(mv), "illegal chess move {mv}");
        let mut next = self.current().position;
        next.play(mv);
        let repetitions_before = self
            .frames
            .iter()
            .flatten()
            .filter(|frame| frame.position.hash() == next.hash())
            .count()
            .min(2) as u8;
        self.frames.rotate_right(1);
        self.frames[0] = Some(Frame {
            position: next,
            repetitions_before,
        });
    }

    fn set_repetitions_before_current(&mut self, count: u8) {
        self.frames[0]
            .as_mut()
            .expect("current frame always exists")
            .repetitions_before = count;
    }
}

fn assert_supported_history<const HISTORY: usize>() {
    assert!(
        matches!(HISTORY, 1 | 4 | 8),
        "ChessAzState history must be one of 1, 4, or 8, got {HISTORY}"
    );
}

impl<const HISTORY: usize> Default for ChessAzState<HISTORY> {
    fn default() -> Self {
        Self::new(ChessPosition::default())
    }
}

impl<const HISTORY: usize> ChessRepetitionState for ChessAzState<HISTORY> {
    fn repetition_hash(&self) -> u64 {
        self.current().position.hash()
    }

    fn reversible_plies(&self) -> usize {
        self.current().position.halfmove_clock()
    }

    fn set_current_repetitions_before(&mut self, count: u8) {
        self.set_repetitions_before_current(count);
    }
}

impl<const HISTORY: usize> GameState for ChessAzState<HISTORY> {
    type Move = ChessMove;

    fn initial() -> Self {
        Self::default()
    }

    fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
        self.current().position.legal_moves()
    }

    fn play(&mut self, mv: Self::Move) {
        self.play_native(mv);
    }

    fn terminal_value(&self) -> Option<TerminalValue> {
        self.current().position.terminal_value()
    }
}

impl<const HISTORY: usize> fmt::Display for ChessAzState<HISTORY> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.current().position.fmt(f)
    }
}

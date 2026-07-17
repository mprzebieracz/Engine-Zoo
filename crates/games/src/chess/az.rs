use super::action::square_to_az_cell;
use super::position::ChessPosition;
use chess::{Board, ChessMove, Color, MoveGen, Piece};
use engine_core::game::{GameState, TerminalValue};
use std::fmt;

const PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

const AZ_HISTORY_PLANES: usize = 14;
const AZ_AUXILIARY_PLANES: usize = 7;

/// A chess position for the AlphaZero v2 model format.
///
/// `HISTORY` is the number of positions encoded, including the current one.
/// Only 1, 4, and 8 are supported.  The fixed array keeps MCTS child creation
/// allocation-free; absent positions at the start of a game encode as zero.
#[derive(Clone, Copy, Debug)]
pub struct ChessHistoryState<const HISTORY: usize> {
    pub(crate) frames: [Option<HistoryFrame>; HISTORY],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct HistoryFrame {
    pub(crate) position: ChessPosition,
    pub(crate) repetitions_before: u8,
}

impl<const HISTORY: usize> ChessHistoryState<HISTORY> {
    pub const INPUT_PLANES: usize = AZ_HISTORY_PLANES * HISTORY + AZ_AUXILIARY_PLANES;

    pub fn repetition_hash(&self) -> u64 {
        self.current().position.hash()
    }

    pub fn new(current: ChessPosition) -> Self {
        assert_supported_history::<HISTORY>();
        let mut frames = [None; HISTORY];
        frames[0] = Some(HistoryFrame {
            position: current,
            repetitions_before: 0,
        });
        Self { frames }
    }

    pub fn position(&self) -> ChessPosition {
        self.current().position
    }

    pub fn board(&self) -> &Board {
        &self.current().position.board
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

    pub(super) fn set_repetitions_before_current(&mut self, count: u8) {
        self.frames[0].as_mut().unwrap().repetitions_before = count;
    }

    fn current(&self) -> &HistoryFrame {
        self.frames[0]
            .as_ref()
            .expect("current frame always exists")
    }

    fn current_mut(&mut self) -> &mut HistoryFrame {
        self.frames[0]
            .as_mut()
            .expect("current frame always exists")
    }

    fn play_native(&mut self, mv: ChessMove) {
        assert!(self.board().legal(mv), "illegal chess move {mv}");
        let mut next = self.current().position;
        next.play_with_effect(mv);
        let repetitions_before = self
            .frames
            .iter()
            .flatten()
            .filter(|frame| frame.position.hash() == next.hash())
            .count()
            .min(2) as u8;
        self.frames.rotate_right(1);
        self.frames[0] = Some(HistoryFrame {
            position: next,
            repetitions_before,
        });
    }
}

fn assert_supported_history<const HISTORY: usize>() {
    assert!(
        matches!(HISTORY, 1 | 4 | 8),
        "ChessHistoryState history must be one of 1, 4, or 8, got {HISTORY}"
    );
}

impl<const HISTORY: usize> Default for ChessHistoryState<HISTORY> {
    fn default() -> Self {
        Self::new(ChessPosition::default())
    }
}

impl<const HISTORY: usize> GameState for ChessHistoryState<HISTORY> {
    type Move = ChessMove;

    fn initial() -> Self {
        Self::default()
    }

    fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
        MoveGen::new_legal(self.board())
    }

    fn play(&mut self, mv: Self::Move) {
        self.play_native(mv);
    }

    fn terminal_value(&self) -> Option<TerminalValue> {
        self.current().position.terminal_value()
    }
}

fn mix_cache_key(key: u64, value: u64) -> u64 {
    let mixed = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (key ^ mixed)
        .rotate_left(27)
        .wrapping_mul(0x94D0_49BB_1331_11EB)
}

impl<const HISTORY: usize> fmt::Display for ChessHistoryState<HISTORY> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.current().position.fmt(f)
    }
}

/// Encodes one historical frame from `perspective`, which is always the side
/// to move in the current state. The last two planes mark whether this exact
/// position occurred once or twice earlier in the retained game history.
fn encode_az_position(
    position: &ChessPosition,
    perspective: Color,
    repetitions_before: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), AZ_HISTORY_PLANES * 64);
    for (color, base_plane) in [(perspective, 0), (!perspective, 6)] {
        let color_squares = *position.board.color_combined(color);
        for piece in PIECES {
            for square in *position.board.pieces(piece) & color_squares {
                let (row, col) = square_to_az_cell(perspective, square);
                out[(base_plane + piece.to_index()) * 64 + (row * 8 + col) as usize] = 1.0;
            }
        }
    }
    if repetitions_before >= 1 {
        out[12 * 64..13 * 64].fill(1.0);
    }
    if repetitions_before >= 2 {
        out[13 * 64..14 * 64].fill(1.0);
    }
}

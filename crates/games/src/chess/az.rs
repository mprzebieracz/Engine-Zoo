use super::action::{decode_v2_action, encode_v2_action, square_to_az_cell, AZ_ACTION_SIZE};
use super::position::ChessPosition;
use chess::{Board, ChessMove, Color, MoveGen, Piece};
use engine_core::game::{Action, Game, TensorDim};
use engine_core::rules::RepetitionGame;
use std::fmt;
use std::str::FromStr;

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
pub struct ChessAzState<const HISTORY: usize> {
    frames: [Option<HistoryFrame>; HISTORY],
}

#[derive(Clone, Copy, Debug)]
struct HistoryFrame {
    position: ChessPosition,
    repetitions_before: u8,
}

impl<const HISTORY: usize> ChessAzState<HISTORY> {
    pub const INPUT_PLANES: usize = AZ_HISTORY_PLANES * HISTORY + AZ_AUXILIARY_PLANES;

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

impl<const HISTORY: usize> Game for ChessAzState<HISTORY> {
    const ACTION_SIZE: usize = AZ_ACTION_SIZE;
    const STATE_SHAPE: [TensorDim; 3] = [Self::INPUT_PLANES as TensorDim, 8, 8];
    const NAME: &'static str = "chess-az-v2";

    fn legal_actions(&self) -> impl Iterator<Item = Action> + '_ {
        MoveGen::new_legal(self.board()).map(|mv| encode_v2_action(self.board(), mv))
    }

    fn step(&mut self, action: Action) {
        let mv = decode_v2_action(self.board(), action)
            .unwrap_or_else(|error| panic!("invalid v2 chess action: {error}"));
        assert!(self.board().legal(mv), "illegal v2 chess move {mv}");
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

    fn is_terminal(&self) -> bool {
        self.current().position.is_terminal()
    }

    fn reward(&self) -> f32 {
        self.current().position.reward()
    }

    fn encode_state(&self, out: &mut [f32]) {
        assert_eq!(out.len(), Self::state_size(), "invalid state buffer length");
        out.fill(0.0);
        let current = self.current();
        let perspective = current.position.board.side_to_move();

        for (index, frame) in self.frames.iter().enumerate() {
            let Some(frame) = frame
            else {
                continue;
            };
            let base = index * AZ_HISTORY_PLANES * 64;
            encode_az_position(
                &frame.position,
                perspective,
                usize::from(frame.repetitions_before),
                &mut out[base..base + AZ_HISTORY_PLANES * 64],
            );
        }

        let base = HISTORY * AZ_HISTORY_PLANES * 64;
        let rights = |color: Color| {
            let rights = current.position.board.castle_rights(color);
            (rights.has_kingside(), rights.has_queenside())
        };
        let (own_kingside, own_queenside) = rights(perspective);
        let (opponent_kingside, opponent_queenside) = rights(!perspective);
        out[base..base + 64].fill(f32::from(perspective == Color::White));
        out[base + 64..base + 128].fill(f32::from(own_kingside));
        out[base + 128..base + 192].fill(f32::from(own_queenside));
        out[base + 192..base + 256].fill(f32::from(opponent_kingside));
        out[base + 256..base + 320].fill(f32::from(opponent_queenside));
        out[base + 320..base + 384].fill((current.position.halfmove_clock as f32 / 100.0).min(1.0));
        let fullmove = current.position.ply / 2 + 1;
        out[base + 384..base + 448].fill((f32::from(fullmove) / 200.0).min(1.0));
    }

    fn parse_move(&self, s: &str) -> Option<Action> {
        let mv = ChessMove::from_str(s.trim()).ok()?;
        self.board()
            .legal(mv)
            .then(|| encode_v2_action(self.board(), mv))
    }

    fn format_action(&self, action: Action) -> String {
        decode_v2_action(self.board(), action)
            .map(|mv| mv.to_string())
            .unwrap_or_else(|error| error.to_string())
    }
}

impl<const HISTORY: usize> RepetitionGame for ChessAzState<HISTORY> {
    fn repetition_hash(&self) -> u64 {
        self.current().position.hash()
    }

    fn evaluation_cache_key(&self) -> u64 {
        // The v2 encoding includes every retained board frame, its repetition
        // planes, and the current halfmove/fullmove planes. A board hash alone
        // would incorrectly share evaluations between distinct histories.
        let mut key = 0x9E37_79B9_7F4A_7C15_u64;
        for (index, frame) in self.frames.iter().enumerate() {
            let value = frame.map_or(0, |frame| {
                frame.position.hash() ^ (u64::from(frame.repetitions_before) << 61)
            });
            key = mix_cache_key(key, value ^ index as u64);
        }
        let current = self.current().position;
        mix_cache_key(
            key,
            (current.halfmove_clock() as u64) << 16 | u64::from(current.ply),
        )
    }

    fn halfmove_clock(&self) -> usize {
        self.current().position.halfmove_clock()
    }

    fn set_repetitions_before_current(&mut self, count: u8) {
        self.set_repetitions_before_current(count);
    }

    fn set_repetition_draw(&mut self) {
        self.current_mut().position.set_repetition_draw();
    }
}

fn mix_cache_key(key: u64, value: u64) -> u64 {
    let mixed = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (key ^ mixed)
        .rotate_left(27)
        .wrapping_mul(0x94D0_49BB_1331_11EB)
}

impl<const HISTORY: usize> fmt::Display for ChessAzState<HISTORY> {
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

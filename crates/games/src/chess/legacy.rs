use super::notation;
use super::position::ChessPosition;
use super::{decode_v1_action, encode_v1_action};
use chess::{Color, MoveGen, Piece};
use engine_core::game::{Action, Game, TensorDim};
use engine_core::notation::GameNotation;
use engine_core::rules::RepetitionGame;
use std::fmt;

const PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

/// Writes the original 19-plane checkpoint-compatible representation.
pub(super) fn encode(position: &ChessPosition, out: &mut [f32]) {
    assert_eq!(out.len(), 19 * 64, "invalid state buffer length");
    out.fill(0.0);
    let me = position.board.side_to_move();
    let white_to_move = me == Color::White;
    let rights = |color: Color| {
        let rights = position.board.castle_rights(color);
        (rights.has_kingside(), rights.has_queenside())
    };
    let (own_k, own_q) = rights(me);
    let (opp_k, opp_q) = rights(!me);

    out[12 * 64..13 * 64].fill(f32::from(white_to_move));
    out[13 * 64..14 * 64].fill(f32::from(position.ply));
    out[14 * 64..15 * 64].fill(f32::from(own_k));
    out[15 * 64..16 * 64].fill(f32::from(own_q));
    out[16 * 64..17 * 64].fill(f32::from(opp_k));
    out[17 * 64..18 * 64].fill(f32::from(opp_q));
    if let Some(file) = position
        .board
        .en_passant()
        .map(|square| square.get_file().to_index())
    {
        for row in 0..8 {
            out[18 * 64 + row * 8 + file] = 1.0;
        }
    }

    for (color, base_plane) in [(me, 0), (!me, 6)] {
        let occupied = *position.board.color_combined(color);
        for piece in PIECES {
            for square in *position.board.pieces(piece) & occupied {
                let index = square.to_index();
                let file = index & 7;
                let rank = index >> 3;
                let row = if white_to_move { 7 - rank } else { rank };
                out[(base_plane + piece.to_index()) * 64 + row * 8 + file] = 1.0;
            }
        }
    }
}

/// Allocation-free legacy policy/feature view used by search.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChessLegacyState(pub(crate) ChessPosition);

impl Game for ChessLegacyState {
    const ACTION_SIZE: usize = 64 * 64 * 5;
    const STATE_SHAPE: [TensorDim; 3] = [19, 8, 8];
    const NAME: &'static str = "chess";

    fn legal_actions(&self) -> impl Iterator<Item = Action> + '_ {
        MoveGen::new_legal(&self.0.board).map(encode_v1_action)
    }

    fn step(&mut self, action: Action) {
        self.0.play_with_effect(decode_v1_action(action));
    }

    fn is_terminal(&self) -> bool {
        self.0.is_terminal()
    }

    fn reward(&self) -> f32 {
        self.0.reward()
    }

    fn encode_state(&self, out: &mut [f32]) {
        encode(&self.0, out);
    }

    fn parse_move(&self, text: &str) -> Option<Action> {
        notation::ChessUciNotation
            .parse_move(&self.0, text)
            .map(encode_v1_action)
    }

    fn format_action(&self, action: Action) -> String {
        notation::ChessUciNotation.format_move(&self.0, decode_v1_action(action))
    }
}

impl RepetitionGame for ChessLegacyState {
    fn repetition_hash(&self) -> u64 {
        self.0.hash()
    }

    fn halfmove_clock(&self) -> usize {
        self.0.halfmove_clock()
    }

    fn set_repetition_draw(&mut self) {
        self.0.set_repetition_draw();
    }
}

impl fmt::Display for ChessLegacyState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

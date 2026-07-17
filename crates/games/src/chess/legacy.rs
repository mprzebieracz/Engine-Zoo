use super::position::ChessPosition;
use chess::{Color, Piece};

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

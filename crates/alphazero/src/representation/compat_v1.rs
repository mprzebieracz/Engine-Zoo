//! Classic chess representation used by the original scalar AlphaZero model.

use super::{Action, AlphaZeroRepresentation};
use chess::{ChessMove, Color, File, Piece, Rank, Square};
use games::ChessPosition;

const PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

/// Zero-sized adapter for the 19-plane, 64×64×5 classic chess model.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChessClassicRepresentation;

fn cell(side: Color, square: Square) -> (usize, usize) {
    let rank = square.get_rank().to_index();
    (
        if side == Color::White { 7 - rank } else { rank },
        square.get_file().to_index(),
    )
}

fn action_for(mv: ChessMove) -> Action {
    let square = |s: Square| {
        let (row, file) = cell(Color::White, s);
        row * 8 + file
    };
    let promotion = match mv.get_promotion() {
        None => 0,
        Some(Piece::Queen) => 1,
        Some(Piece::Rook) => 2,
        Some(Piece::Knight) => 3,
        Some(Piece::Bishop) => 4,
        Some(_) => unreachable!("illegal promotion piece"),
    };
    Action::new(((square(mv.get_source()) * 64 + square(mv.get_dest())) * 5 + promotion) as u32)
}

fn square(index: usize) -> Option<Square> {
    (index < 64)
        .then(|| Square::make_square(Rank::from_index(7 - index / 8), File::from_index(index % 8)))
}

impl AlphaZeroRepresentation<ChessPosition> for ChessClassicRepresentation {
    const STATE_SHAPE: [usize; 3] = [19, 8, 8];
    const ACTION_SIZE: usize = 64 * 64 * 5;

    fn encode_state(&self, state: &ChessPosition, out: &mut [f32]) {
        assert_eq!(out.len(), Self::state_size());
        out.fill(0.0);
        let board = state.board();
        let side = board.side_to_move();
        let rights = |color| {
            (
                board.castle_rights(color).has_kingside(),
                board.castle_rights(color).has_queenside(),
            )
        };
        let (own_k, own_q) = rights(side);
        let (opp_k, opp_q) = rights(!side);
        out[12 * 64..13 * 64].fill(f32::from(side == Color::White));
        out[13 * 64..14 * 64].fill(f32::from(state.ply()));
        for (plane, value) in [(14, own_k), (15, own_q), (16, opp_k), (17, opp_q)] {
            out[plane * 64..(plane + 1) * 64].fill(f32::from(value));
        }
        if let Some(file) = board.en_passant().map(|s| s.get_file().to_index()) {
            for row in 0..8 {
                out[18 * 64 + row * 8 + file] = 1.0;
            }
        }
        for (color, base) in [(side, 0), (!side, 6)] {
            let occupied = *board.color_combined(color);
            for piece in PIECES {
                for sq in *board.pieces(piece) & occupied {
                    let (row, file) = cell(side, sq);
                    out[(base + piece.to_index()) * 64 + row * 8 + file] = 1.0;
                }
            }
        }
    }

    fn move_to_action(&self, _: &ChessPosition, mv: ChessMove) -> Action {
        action_for(mv)
    }

    fn action_to_move(&self, state: &ChessPosition, action: Action) -> Option<ChessMove> {
        if action.index() >= Self::ACTION_SIZE {
            return None;
        }
        let promotion = match action.index() % 5 {
            0 => None,
            1 => Some(Piece::Queen),
            2 => Some(Piece::Rook),
            3 => Some(Piece::Knight),
            _ => Some(Piece::Bishop),
        };
        let pair = action.index() / 5;
        let mv = ChessMove::new(square(pair / 64)?, square(pair % 64)?, promotion);
        state.board().legal(mv).then_some(mv)
    }

    fn encoded_state_key(&self, state: &ChessPosition) -> u64 {
        (state.hash() ^ 0x9E37_79B9_7F4A_7C15)
            .rotate_left(27)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9)
            ^ u64::from(state.ply())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use games::ChessGame;
    use std::collections::HashSet;
    use std::str::FromStr;

    fn check_move(state: &ChessPosition, mv: ChessMove) {
        let r = ChessClassicRepresentation;
        let action = r.move_to_action(state, mv);
        assert_eq!(r.action_to_move(state, action), Some(mv));
    }

    fn golden(game: &ChessGame) {
        let state = game.position();
        let r = ChessClassicRepresentation;
        let mut actual = vec![0.0; 19 * 64];
        r.encode_state(&state, &mut actual);
        let mut actions = HashSet::new();
        for mv in state.legal_moves() {
            let action = r.move_to_action(&state, mv);
            assert!(actions.insert(action));
            assert_eq!(r.action_to_move(&state, action), Some(mv));
        }
        assert!(r.action_to_move(&state, Action::new(20_480)).is_none());
    }

    #[test]
    fn golden_fens_and_progression() {
        for fen in [
            "startpos",
            "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq e6 0 1",
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        ] {
            let game = if fen == "startpos" {
                ChessGame::default()
            } else {
                ChessGame::from_fen(fen).unwrap()
            };
            golden(&game);
        }
        let mut game = ChessGame::default();
        for _ in 0..24 {
            golden(&game);
            let mv = game.position().legal_moves().next().unwrap();
            game.play(mv);
        }
    }

    #[test]
    fn promotions_roundtrip_for_both_colors() {
        for fen in [
            "1r3r1k/P1P1P3/8/8/8/8/8/K7 w - - 0 1",
            "k7/8/8/8/8/8/p1p1p3/1R3R1K b - - 0 1",
        ] {
            let state = ChessGame::from_fen(fen).unwrap().position();
            let mut pieces = HashSet::new();
            for mv in state
                .legal_moves()
                .filter(|mv| mv.get_promotion().is_some())
            {
                pieces.insert(mv.get_promotion().unwrap());
                check_move(&state, mv);
            }
            assert_eq!(
                pieces,
                HashSet::from([Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight])
            );
        }
    }

    #[test]
    fn castling_and_en_passant_match_legacy_actions_for_both_colors() {
        for (fen, moves) in [
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                &["e1g1", "e1c1"][..],
            ),
            (
                "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
                &["e8g8", "e8c8"][..],
            ),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", &["e5d6"][..]),
            ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 2", &["e4d3"][..]),
        ] {
            let state = ChessGame::from_fen(fen).unwrap().position();
            for text in moves {
                check_move(&state, ChessMove::from_str(text).unwrap());
            }
        }
    }

    #[test]
    fn key_includes_encoded_ply_for_identical_boards() {
        let r = ChessClassicRepresentation;
        let a = ChessGame::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .unwrap()
            .position();
        let b = ChessGame::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 3")
            .unwrap()
            .position();
        assert_eq!(a.board(), b.board());
        assert_ne!(r.encoded_state_key(&a), r.encoded_state_key(&b));
        let mut encoded_a = vec![0.0; 19 * 64];
        let mut encoded_b = vec![0.0; 19 * 64];
        r.encode_state(&a, &mut encoded_a);
        r.encode_state(&b, &mut encoded_b);
        assert_eq!(&encoded_a[..13 * 64], &encoded_b[..13 * 64]);
        assert!(encoded_a[13 * 64..14 * 64].iter().all(|&v| v == 0.0));
        assert!(encoded_b[13 * 64..14 * 64].iter().all(|&v| v == 4.0));
        assert_eq!(&encoded_a[14 * 64..], &encoded_b[14 * 64..]);
    }

    #[test]
    fn rejects_actions_from_the_wrong_position() {
        let source = ChessGame::default().position();
        let other = ChessGame::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
            .unwrap()
            .position();
        let r = ChessClassicRepresentation;
        let mv = ChessMove::from_str("e2e4").unwrap();
        let action = r.move_to_action(&source, mv);
        assert_eq!(r.action_to_move(&source, action), Some(mv));
        assert_eq!(r.action_to_move(&other, action), None);
    }
}

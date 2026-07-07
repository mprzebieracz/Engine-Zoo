use super::*;
use std::str::FromStr;

/// Load a FEN for unit tests. Restores the board and mate/stalemate only;
/// ply and halfmove_clock start at zero (the `chess` crate drops FEN clocks).
fn from_fen(fen: &str) -> ChessGame {
    let board = Board::from_str(fen).expect("test FEN should parse");
    let status = match board.status() {
        BoardStatus::Ongoing => Status::Ongoing,
        BoardStatus::Checkmate => Status::Checkmate,
        BoardStatus::Stalemate => Status::Stalemate,
    };
    let mut game = ChessGame {
        board,
        ply: 0,
        status,
        halfmove_clock: 0,
        position_counts: HashMap::with_capacity(16),
    };
    game.record_position();
    game
}

#[test]
fn startpos_has_twenty_moves() {
    let g = ChessGame::default();
    assert_eq!(g.legal_actions().count(), 20);
}

#[test]
fn action_roundtrip_over_random_games() {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..20 {
        let mut g = ChessGame::default();
        for _ in 0..80 {
            if g.is_terminal() {
                break;
            }
            let legal: Vec<u32> = g.legal_actions().collect();
            for &a in &legal {
                assert_eq!(encode_move(decode_move(a)), a);
                assert!((a as usize) < ChessGame::ACTION_SIZE);
            }
            g.step(*legal.choose(&mut rng).unwrap());
        }
    }
}

#[test]
fn scholars_mate_reward_convention() {
    let mut g = ChessGame::default();
    for mv in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"] {
        let a = g
            .parse_move(mv)
            .unwrap_or_else(|| panic!("{mv} should be legal"));
        g.step(a);
    }
    // Black is to move and has been mated.
    assert!(g.is_terminal());
    assert_eq!(g.reward(), -1.0);
    assert_eq!(g.board().side_to_move(), Color::Black);
}

#[test]
fn stalemate_is_a_draw() {
    let g = from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    assert!(g.is_terminal());
    assert_eq!(g.status, Status::Stalemate);
    assert_eq!(g.reward(), 0.0);
}

#[test]
fn canonical_encoding_flips_for_black() {
    let mut g = ChessGame::default();
    let mut white_view = vec![0.0f32; ChessGame::state_size()];
    g.encode_state(&mut white_view);
    // White's own pawns (plane 0) appear on output row 6 (rank 2).
    assert_eq!(white_view[6 * 8], 1.0);
    assert_eq!(white_view[12 * 64], 1.0); // white-to-move plane

    g.step(g.parse_move("e2e4").unwrap());
    let mut black_view = vec![0.0f32; ChessGame::state_size()];
    g.encode_state(&mut black_view);
    // Black's own pawns also appear on row 6 after the vertical flip.
    assert_eq!(black_view[6 * 8], 1.0);
    assert_eq!(black_view[12 * 64], 0.0);
    // No black pawn can capture on e3, so the en-passant plane stays empty.
    assert!(black_view[18 * 64..].iter().all(|&v| v == 0.0));
}

#[test]
fn en_passant_plane_when_capturable() {
    // 1.e4 a6 2.e5 d5 — exd6 e.p. is available; the d-file is flagged.
    let mut g = ChessGame::default();
    for mv in ["e2e4", "a7a6", "e4e5", "d7d5"] {
        g.step(g.parse_move(mv).unwrap());
    }
    let mut v = vec![0.0f32; ChessGame::state_size()];
    g.encode_state(&mut v);
    assert_eq!(v[18 * 64 + 3], 1.0);
    assert_eq!(v[18 * 64 + 4], 0.0);
}

#[test]
fn promotions_encode_distinctly() {
    let g = from_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1");
    let promos: Vec<u32> = g.legal_actions().map(|a| a % 5).collect();
    for p in [1u32, 2, 3, 4] {
        assert!(promos.contains(&p), "missing promotion code {p}");
    }
}

#[test]
fn threefold_repetition_is_a_draw() {
    let mut g = ChessGame::default();
    for mv in [
        "b1c3", "b8c6", "c3b1", "c6b8", "b1c3", "b8c6", "c3b1", "c6b8",
    ] {
        assert!(!g.is_terminal(), "draw too early before {mv}");
        g.step(g.parse_move(mv).unwrap());
    }
    assert!(g.is_terminal());
    assert_eq!(g.status, Status::DrawRepetition);
    assert_eq!(g.reward(), 0.0);
}

#[test]
fn fifty_move_rule_is_a_draw() {
    let mut g = from_fen("k7/8/8/8/8/8/8/K7 w - - 0 1");
    g.halfmove_clock = 99;
    g.step(g.parse_move("a1a2").unwrap());
    assert!(g.is_terminal());
    assert_eq!(g.status, Status::DrawFiftyMoveRule);
    assert_eq!(g.reward(), 0.0);
}

#[test]
fn pawn_move_resets_halfmove_clock() {
    let mut g = from_fen("k7/8/8/8/8/8/P7/K7 w - - 0 1");
    g.halfmove_clock = 99;
    g.step(g.parse_move("a2a3").unwrap());
    assert_eq!(g.halfmove_clock, 0);
    assert!(!g.is_terminal());
}

#[test]
fn capture_resets_halfmove_clock() {
    let mut g = from_fen("4k3/8/8/8/8/8/4p3/4K3 w - - 0 1");
    g.halfmove_clock = 99;
    g.step(g.parse_move("e1e2").unwrap());
    assert_eq!(g.halfmove_clock, 0);
    assert!(!g.is_terminal());
}

#[test]
fn castling_rights_change_resets_halfmove_clock() {
    let mut g = from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    g.halfmove_clock = 99;
    g.step(g.parse_move("e1f1").unwrap());
    assert_eq!(g.halfmove_clock, 0);
    assert!(!g.is_terminal());
}

#[test]
fn irreversible_move_clears_repetition_history() {
    let mut g = ChessGame::default();
    for mv in ["b1c3", "b8c6", "c3b1", "c6b8"] {
        g.step(g.parse_move(mv).unwrap());
    }
    assert!(!g.is_terminal());
    g.step(g.parse_move("a2a3").unwrap());
    g.step(g.parse_move("a7a6").unwrap());
    for mv in ["b1c3", "b8c6", "c3b1", "c6b8"] {
        g.step(g.parse_move(mv).unwrap());
    }
    assert!(!g.is_terminal());
}

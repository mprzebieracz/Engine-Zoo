use super::*;
use engine_core::game::{GameState, TerminalValue};
use engine_core::notation::GameNotation;
use std::str::FromStr;

/// Parses a legal UCI move against the current game position and plays it.
fn play_uci(game: &mut ChessGame, text: &str) {
    let mv = ChessUciNotation
        .parse_move(&game.position(), text)
        .unwrap_or_else(|| panic!("{text} should be legal"));
    game.play(mv);
}

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
        pos: ChessPosition {
            board,
            ply: 0,
            status,
            halfmove_clock: 0,
        },
        repetitions: RepetitionTracker::new(board.get_hash()),
        history: [None; 8],
    };
    game.history[0] = Some((game.pos, 0));
    game
}

#[test]
fn startpos_has_twenty_moves() {
    let g = ChessGame::default();
    assert_eq!(g.legal_moves().count(), 20);
}

#[test]
fn authoritative_game_state_tracks_repetition_terminal_value() {
    let mut game = <ChessGame as GameState>::initial();
    for text in [
        "b1c3", "b8c6", "c3b1", "c6b8", "b1c3", "b8c6", "c3b1", "c6b8",
    ] {
        let mv = game
            .legal_moves()
            .find(|mv| mv == &ChessMove::from_str(text).unwrap())
            .unwrap();
        GameState::play(&mut game, mv);
    }

    assert_eq!(game.legal_moves().count(), 20);
    assert_eq!(GameState::terminal_value(&game), Some(TerminalValue::Draw));
}

#[test]
fn native_terminal_values_use_side_to_move_perspective() {
    let mut checkmate = from_fen("7k/R7/6K1/8/8/8/8/8 w - - 0 1").position();
    checkmate.play(ChessMove::from_str("a7a8").unwrap());
    let stalemate = from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").position();
    let mut fifty_move = from_fen("k7/8/8/8/8/8/8/K6R w - - 0 1").position();
    fifty_move.status = Status::DrawFiftyMoveRule;
    let mut repetition = checkmate;
    repetition.status = Status::DrawRepetition;

    assert_eq!(checkmate.terminal_value(), Some(TerminalValue::Loss));
    assert_eq!(stalemate.terminal_value(), Some(TerminalValue::Draw));
    assert_eq!(fifty_move.terminal_value(), Some(TerminalValue::Draw));
    assert_eq!(repetition.terminal_value(), Some(TerminalValue::Draw));
    assert_eq!(ChessPosition::initial().terminal_value(), None);
}

#[test]
fn public_fen_loader_restores_clocks() {
    let g = ChessGame::from_fen("8/8/8/8/8/8/8/K6k b - - 17 23").unwrap();
    assert_eq!(g.pos.ply, 45);
    assert_eq!(g.pos.halfmove_clock, 17);
}

#[test]
fn public_fen_loader_rejects_invalid_clocks() {
    assert!(ChessGame::from_fen("8/8/8/8/8/8/8/K6k w - - nope 1").is_err());
    assert!(ChessGame::from_fen("8/8/8/8/8/8/8/K6k w - - 0 nope").is_err());
    assert!(ChessGame::from_fen("8/8/8/8/8/8/8/K6k w - - 0 0").is_err());
    assert!(ChessGame::from_fen("8/8/8/8/8/8/8/K6k w - - 0 65535").is_err());
}

#[test]
fn scholars_mate_terminal_value_uses_side_to_move_perspective() {
    let mut g = ChessGame::default();
    for text in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"] {
        play_uci(&mut g, text);
    }
    // Black is to move and has been mated.
    assert!(g.is_terminal());
    assert_eq!(g.terminal_value(), Some(TerminalValue::Loss));
    assert_eq!(g.board().side_to_move(), Color::Black);
}

#[test]
fn stalemate_is_a_draw() {
    let g = from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    assert!(g.is_terminal());
    assert_eq!(g.pos.status, Status::Stalemate);
    assert_eq!(g.terminal_value(), Some(TerminalValue::Draw));
}

#[test]
fn castling_moves_are_exposed_to_policy() {
    let g = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    let position = g.position();
    let kingside = ChessUciNotation
        .parse_move(&position, "e1g1")
        .expect("white O-O should be legal");
    let queenside = ChessUciNotation
        .parse_move(&position, "e1c1")
        .expect("white O-O-O should be legal");
    let legal: Vec<_> = g.legal_moves().collect();

    assert!(legal.contains(&kingside));
    assert!(legal.contains(&queenside));
    assert_eq!(ChessUciNotation.format_move(&position, kingside), "e1g1");
    assert_eq!(ChessUciNotation.format_move(&position, queenside), "e1c1");
    assert_eq!(notation::san(g.board(), kingside), "O-O");
    assert_eq!(notation::san(g.board(), queenside), "O-O-O");
}

#[test]
fn castling_step_moves_king_and_rook() {
    let mut g = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    g.pos.halfmove_clock = 17;

    play_uci(&mut g, "e1g1");
    assert_eq!(g.board().piece_on(Square::G1), Some(Piece::King));
    assert_eq!(g.board().piece_on(Square::F1), Some(Piece::Rook));
    assert_eq!(g.board().piece_on(Square::E1), None);
    assert_eq!(g.board().piece_on(Square::H1), None);
    assert!(!g.board().castle_rights(Color::White).has_kingside());
    assert!(!g.board().castle_rights(Color::White).has_queenside());
    assert_eq!(g.pos.halfmove_clock, 18);

    play_uci(&mut g, "e8c8");
    assert_eq!(g.board().piece_on(Square::C8), Some(Piece::King));
    assert_eq!(g.board().piece_on(Square::D8), Some(Piece::Rook));
    assert_eq!(g.board().piece_on(Square::E8), None);
    assert_eq!(g.board().piece_on(Square::A8), None);
    assert!(!g.board().castle_rights(Color::Black).has_kingside());
    assert!(!g.board().castle_rights(Color::Black).has_queenside());
    assert_eq!(g.pos.halfmove_clock, 19);
}

#[test]
fn threefold_repetition_is_a_draw() {
    let mut g = ChessGame::default();
    for mv in [
        "b1c3", "b8c6", "c3b1", "c6b8", "b1c3", "b8c6", "c3b1", "c6b8",
    ] {
        assert!(!g.is_terminal(), "draw too early before {mv}");
        play_uci(&mut g, mv);
    }
    assert!(g.is_terminal());
    assert_eq!(g.pos.status, Status::DrawRepetition);
    assert_eq!(g.terminal_value(), Some(TerminalValue::Draw));
}

#[test]
fn fifty_move_rule_is_a_draw() {
    let mut g = from_fen("k7/8/8/8/8/8/8/K7 w - - 0 1");
    g.pos.halfmove_clock = 99;
    play_uci(&mut g, "a1a2");
    assert!(g.is_terminal());
    assert_eq!(g.pos.status, Status::DrawFiftyMoveRule);
    assert_eq!(g.terminal_value(), Some(TerminalValue::Draw));
}

#[test]
fn move_effect_describes_move_properties_even_when_terminal() {
    let mut position = from_fen("7k/R7/6K1/8/8/8/8/8 w - - 0 1").position();
    let effect = position.play_with_effect(ChessMove::from_str("a7a8").unwrap());

    assert_eq!(position.status, Status::Checkmate);
    assert!(!effect.clears_repetition_history());
}

#[test]
fn pawn_move_resets_halfmove_clock() {
    let mut g = from_fen("k7/8/8/8/8/8/P7/K7 w - - 0 1");
    g.pos.halfmove_clock = 99;
    play_uci(&mut g, "a2a3");
    assert_eq!(g.pos.halfmove_clock, 0);
    assert!(!g.is_terminal());
}

#[test]
fn en_passant_capture_resets_halfmove_clock() {
    let mut g = ChessGame::default();
    for text in ["e2e4", "a7a6", "e4e5", "d7d5"] {
        play_uci(&mut g, text);
    }
    g.pos.halfmove_clock = 99;
    play_uci(&mut g, "e5d6");
    assert_eq!(g.pos.halfmove_clock, 0);
    assert!(!g.is_terminal());
}

#[test]
fn capture_resets_halfmove_clock() {
    let mut g = from_fen("4k3/8/8/8/8/8/4p3/4K3 w - - 0 1");
    g.pos.halfmove_clock = 99;
    play_uci(&mut g, "e1e2");
    assert_eq!(g.pos.halfmove_clock, 0);
    assert!(!g.is_terminal());
}

#[test]
fn castling_rights_change_keeps_the_halfmove_clock() {
    let mut g = from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    g.pos.halfmove_clock = 99;
    play_uci(&mut g, "e1f1");
    assert_eq!(g.pos.halfmove_clock, 100);
    assert!(g.is_terminal());
    assert_eq!(g.terminal_value(), Some(TerminalValue::Draw));
}

#[test]
fn irreversible_move_clears_repetition_history() {
    let mut g = ChessGame::default();
    for text in ["b1c3", "b8c6", "c3b1", "c6b8"] {
        play_uci(&mut g, text);
    }
    assert!(!g.is_terminal());
    play_uci(&mut g, "a2a3");
    play_uci(&mut g, "a7a6");
    for text in ["b1c3", "b8c6", "c3b1", "c6b8"] {
        play_uci(&mut g, text);
    }
    assert!(!g.is_terminal());
}

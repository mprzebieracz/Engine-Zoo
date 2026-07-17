use super::az::HistoryFrame;
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
    game.history[0] = Some(HistoryFrame {
        position: game.pos,
        repetitions_before: 0,
    });
    game
}

#[test]
fn startpos_has_twenty_moves() {
    let g = ChessGame::default();
    assert_eq!(g.legal_moves().count(), 20);
}

#[test]
fn native_chess_states_match_policy_stepping() {
    let mv = ChessMove::from_str("e2e4").unwrap();
    let mut position = ChessPosition::initial();
    let mut az_native = ChessHistoryState::<4>::initial();
    let mut az_from_codec = az_native;

    assert_eq!(position.legal_moves().count(), 20);
    assert_eq!(az_native.legal_moves().count(), 20);
    position.play(mv);
    az_native.play(mv);
    let action = encode_v2_action(az_from_codec.board(), mv);
    az_from_codec.play(decode_v2_action(az_from_codec.board(), action).unwrap());

    assert_eq!(position.hash(), az_native.position().hash());
    assert_eq!(az_native.position().hash(), az_from_codec.position().hash());
    assert_eq!(
        az_native.position().halfmove_clock(),
        az_from_codec.position().halfmove_clock()
    );
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
fn az_actions_roundtrip_over_random_games() {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..20 {
        let mut game = ChessGame::default();
        for _ in 0..80 {
            if game.is_terminal() {
                break;
            }
            let legal: Vec<_> = game.legal_moves().collect();
            for &mv in &legal {
                let action = encode_v2_action(game.board(), mv);
                assert!((action as usize) < AZ_ACTION_SIZE);
                assert_eq!(decode_v2_action(game.board(), action).unwrap(), mv);
            }
            game.play(*legal.choose(&mut rng).unwrap());
        }
    }
}

#[test]
fn history_state_matches_authoritative_game_position() {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(31);
    for _ in 0..16 {
        let mut game = ChessGame::default();
        for _ in 0..60 {
            if game.is_terminal() {
                break;
            }
            let history = game.history_state::<8>();
            assert_eq!(
                game.legal_moves().collect::<Vec<_>>(),
                history.legal_moves().collect::<Vec<_>>()
            );
            assert_eq!(game.position().hash(), history.repetition_hash());
            assert_eq!(game.position().halfmove_clock(), history.reversible_plies());

            let legal: Vec<_> = game.legal_moves().collect();
            game.play(*legal.choose(&mut rng).unwrap());
        }
    }
}

#[test]
fn az_actions_roundtrip_over_random_games_without_collisions() {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(19);
    for _ in 0..20 {
        let mut game = ChessGame::default();
        for _ in 0..80 {
            if game.is_terminal() {
                break;
            }
            let legal: Vec<_> = game.legal_moves().collect();
            let actions: Vec<_> = legal
                .iter()
                .map(|&mv| encode_v2_action(game.board(), mv))
                .collect();
            assert!(actions
                .iter()
                .all(|&action| (action as usize) < AZ_ACTION_SIZE));
            let mut unique = actions.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), legal.len(), "legal moves collided in {game}");
            for (&mv, &action) in legal.iter().zip(&actions) {
                assert_eq!(decode_v2_action(game.board(), action).unwrap(), mv);
            }
            game.play(*legal.choose(&mut rng).unwrap());
        }
    }
}

#[test]
fn history_state_retains_frames_for_alpha_zero_evaluation() {
    let initial = ChessHistoryState::<4>::default();
    let mut returned = initial;
    for text in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        returned.play(ChessMove::from_str(text).unwrap());
    }

    assert_eq!(initial.repetition_hash(), returned.repetition_hash());
    assert_ne!(initial.position().ply(), returned.position().ply());
    assert_eq!(returned.frames.iter().flatten().count(), 4);
    assert_eq!(returned.position().halfmove_clock(), 4);
}

#[test]
fn az_codec_handles_promotions_castling_and_en_passant() {
    let promotions = from_fen("1r3r1k/P1P1P3/8/8/8/8/8/K7 w - - 0 1");
    let legal: Vec<_> = MoveGen::new_legal(promotions.board()).collect();
    for promotion in [Piece::Queen, Piece::Knight, Piece::Bishop, Piece::Rook] {
        assert!(legal.iter().any(|mv| mv.get_promotion() == Some(promotion)));
    }
    assert!(legal.iter().any(|mv| {
        mv.get_promotion().is_some() && mv.get_source().get_file() != mv.get_dest().get_file()
    }));
    for mv in legal.into_iter().filter(|mv| mv.get_promotion().is_some()) {
        let action = encode_v2_action(promotions.board(), mv);
        assert_eq!(decode_v2_action(promotions.board(), action).unwrap(), mv);
    }

    let castles = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    for text in ["e1g1", "e1c1"] {
        let mv = ChessMove::from_str(text).unwrap();
        let action = encode_v2_action(castles.board(), mv);
        assert_eq!(decode_v2_action(castles.board(), action).unwrap(), mv);
    }

    let mut ep = ChessGame::default();
    for text in ["e2e4", "a7a6", "e4e5", "d7d5"] {
        let mv = ChessUciNotation.parse_move(&ep.position(), text).unwrap();
        ep.play(mv);
    }
    let mv = ChessMove::from_str("e5d6").unwrap();
    let action = encode_v2_action(ep.board(), mv);
    assert_eq!(decode_v2_action(ep.board(), action).unwrap(), mv);
}

#[test]
fn az_codec_is_color_canonical_and_rejects_invalid_actions() {
    let white = from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    let black = from_fen("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1");
    let white_move = ChessMove::from_str("e2e4").unwrap();
    let black_move = ChessMove::from_str("e7e5").unwrap();
    assert_eq!(
        encode_v2_action(white.board(), white_move),
        encode_v2_action(black.board(), black_move)
    );
    assert_eq!(
        decode_v2_action(white.board(), AZ_ACTION_SIZE as u32),
        Err(AzActionError::OutOfRange(AZ_ACTION_SIZE as u32))
    );
}

#[test]
fn history_state_shapes_and_history_are_canonical() {
    assert_eq!(ChessHistoryState::<1>::INPUT_PLANES, 21);
    assert_eq!(ChessHistoryState::<4>::INPUT_PLANES, 63);
    assert_eq!(ChessHistoryState::<8>::INPUT_PLANES, 119);

    let mut state = ChessGame::default().history_state::<4>();
    assert_eq!(state.frames.iter().flatten().count(), 1);
    state.play(ChessMove::from_str("e2e4").unwrap());
    assert_eq!(state.frames.iter().flatten().count(), 2);
    assert_eq!(state.position().board().side_to_move(), Color::Black);
}

#[test]
fn history_state_marks_repeated_current_positions() {
    let mut state = ChessGame::default().history_state::<8>();
    for text in ["b1c3", "b8c6", "c3b1", "c6b8"] {
        state.play(ChessMove::from_str(text).unwrap());
    }
    assert_eq!(state.frames[0].unwrap().repetitions_before, 1);
    assert_eq!(state.frames[1].unwrap().repetitions_before, 0);
}

#[test]
fn fen_has_only_its_current_real_frame() {
    let game = ChessGame::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 17 42").unwrap();
    let state = game.history_state::<8>();
    assert_eq!(state.frames.iter().flatten().count(), 1);
    assert_eq!(
        state.frames[0].unwrap().position.hash(),
        game.position().hash()
    );
    assert_eq!(state.frames[0].unwrap().repetitions_before, 0);
}

#[test]
fn authoritative_snapshots_support_one_four_and_eight_frames() {
    let mut game = ChessGame::default();
    for text in ["b1c3", "b8c6", "g1f3", "g8f6"] {
        let mv = ChessUciNotation.parse_move(&game.position(), text).unwrap();
        game.play(mv);
    }

    let h1 = game.history_state::<1>();
    let h4 = game.history_state::<4>();
    let h8 = game.history_state::<8>();
    assert_eq!(h1.frames.iter().flatten().count(), 1);
    assert_eq!(h4.frames.iter().flatten().count(), 4);
    assert_eq!(h8.frames.iter().flatten().count(), 5);
    assert_eq!(h1.position().hash(), game.position().hash());
    assert_eq!(h4.position().hash(), game.position().hash());
    assert_eq!(h8.position().hash(), game.position().hash());
}

#[test]
fn irreversible_move_resets_counts_but_retains_real_neural_history() {
    let mut game = ChessGame::default();
    game.play(
        ChessUciNotation
            .parse_move(&game.position(), "b1c3")
            .unwrap(),
    );
    let previous_hash = game.position().hash();
    game.play(
        ChessUciNotation
            .parse_move(&game.position(), "a7a6")
            .unwrap(),
    );

    let state = game.history_state::<8>();
    assert_eq!(state.frames.iter().flatten().count(), 3);
    assert_eq!(state.frames[1].unwrap().position.hash(), previous_hash);
    assert_eq!(
        game.repetition_context()
            .occurrences_before_root(previous_hash),
        0
    );
    assert_eq!(
        game.repetition_context()
            .occurrences_before_root(game.position().hash()),
        0
    );
}

#[test]
fn authoritative_frames_store_repetition_count_and_terminal_position() {
    let mut game = ChessGame::default();
    for mv in [
        "b1c3", "b8c6", "c3b1", "c6b8", "b1c3", "b8c6", "c3b1", "c6b8",
    ] {
        game.play(ChessUciNotation.parse_move(&game.position(), mv).unwrap());
    }

    let state = game.history_state::<8>();
    assert_eq!(state.frames[0].unwrap().repetitions_before, 2);
    assert_eq!(state.frames[4].unwrap().repetitions_before, 1);
    assert_eq!(
        state.frames[0].unwrap().position.status,
        Status::DrawRepetition
    );
    assert!(game.is_terminal());
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
fn history_state_tracks_side_to_move_for_alpha_zero_encoding() {
    let mut game = ChessGame::default();
    assert_eq!(
        game.history_state::<1>().position().board().side_to_move(),
        Color::White
    );

    play_uci(&mut game, "e2e4");
    let state = game.history_state::<1>();
    assert_eq!(state.position().board().side_to_move(), Color::Black);
    assert_eq!(state.position().board().en_passant(), None);
}

#[test]
fn en_passant_move_is_exposed_to_alpha_zero_policy() {
    // 1.e4 a6 2.e5 d5 — exd6 e.p. is available.
    let mut g = ChessGame::default();
    for text in ["e2e4", "a7a6", "e4e5", "d7d5"] {
        play_uci(&mut g, text);
    }
    let mv = ChessMove::from_str("e5d6").unwrap();
    assert!(g.legal_moves().any(|legal| legal == mv));
    let action = encode_v2_action(g.board(), mv);
    assert_eq!(decode_v2_action(g.board(), action), Ok(mv));
}

#[test]
fn promotions_encode_distinctly() {
    let g = from_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1");
    let promotions: Vec<_> = g
        .legal_moves()
        .filter(|mv| mv.get_promotion().is_some())
        .collect();
    let actions: Vec<_> = promotions
        .iter()
        .map(|&mv| encode_v2_action(g.board(), mv))
        .collect();
    for piece in [Piece::Queen, Piece::Rook, Piece::Knight, Piece::Bishop] {
        assert!(promotions
            .iter()
            .any(|mv| mv.get_promotion() == Some(piece)));
    }
    let mut unique = actions;
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), promotions.len());
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
    assert!((encode_v2_action(g.board(), kingside) as usize) < AZ_ACTION_SIZE);
    assert!((encode_v2_action(g.board(), queenside) as usize) < AZ_ACTION_SIZE);
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
    assert_eq!(g.pos.halfmove_clock, 0);

    play_uci(&mut g, "e8c8");
    assert_eq!(g.board().piece_on(Square::C8), Some(Piece::King));
    assert_eq!(g.board().piece_on(Square::D8), Some(Piece::Rook));
    assert_eq!(g.board().piece_on(Square::E8), None);
    assert_eq!(g.board().piece_on(Square::A8), None);
    assert!(!g.board().castle_rights(Color::Black).has_kingside());
    assert!(!g.board().castle_rights(Color::Black).has_queenside());
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
    assert!(!effect.is_irreversible());
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
fn castling_rights_change_resets_halfmove_clock() {
    let mut g = from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    g.pos.halfmove_clock = 99;
    play_uci(&mut g, "e1f1");
    assert_eq!(g.pos.halfmove_clock, 0);
    assert!(!g.is_terminal());
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

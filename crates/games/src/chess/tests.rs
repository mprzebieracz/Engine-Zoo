use super::*;
use engine_core::rules::RepetitionGame;
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
        pos: ChessPosition {
            board,
            ply: 0,
            status,
            halfmove_clock: 0,
        },
        position_counts: HashMap::with_capacity_and_hasher(
            16,
            zobrist::ZobristBuildHasher::default(),
        ),
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
#[should_panic(expected = "invalid state buffer length")]
fn encoding_rejects_wrong_buffer_length() {
    ChessGame::default().encode_state(&mut []);
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
                assert_eq!(encode_v1_action(decode_v1_action(a)), a);
                assert!((a as usize) < ChessGame::ACTION_SIZE);
            }
            g.step(*legal.choose(&mut rng).unwrap());
        }
    }
}

#[test]
fn legacy_search_view_matches_full_game_until_repetition_adjudication() {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(31);
    for _ in 0..16 {
        let mut game = ChessGame::default();
        let mut legacy = game.legacy_state();
        for _ in 0..60 {
            if game.is_terminal() {
                break;
            }
            let legal: Vec<_> = game.legal_actions().collect();
            assert_eq!(legal, legacy.legal_actions().collect::<Vec<_>>());
            let mut game_encoding = vec![0.0; ChessGame::state_size()];
            let mut legacy_encoding = vec![0.0; ChessLegacyState::state_size()];
            game.encode_state(&mut game_encoding);
            legacy.encode_state(&mut legacy_encoding);
            assert_eq!(game_encoding, legacy_encoding);

            let action = *legal.choose(&mut rng).unwrap();
            game.step(action);
            legacy.step(action);
            assert_eq!(game.position().hash(), legacy.repetition_hash());
            assert_eq!(game.position().halfmove_clock(), legacy.halfmove_clock());
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
            let legal: Vec<_> = MoveGen::new_legal(game.board()).collect();
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
            game.step(encode_v1_action(*legal.choose(&mut rng).unwrap()));
        }
    }
}

#[test]
fn az_evaluation_cache_key_includes_history_and_clock_features() {
    let initial = ChessAzState::<4>::default();
    let mut returned = initial;
    for mv in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        returned.step(returned.parse_move(mv).unwrap());
    }

    assert_eq!(initial.repetition_hash(), returned.repetition_hash());
    assert_ne!(
        initial.evaluation_cache_key(),
        returned.evaluation_cache_key(),
        "identical boards with different v2 feature histories must not share an evaluation"
    );
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
        ep.step(ep.parse_move(text).unwrap());
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
        decode_v2_action(white.board(), AZ_ACTION_SIZE as Action),
        Err(AzActionError::OutOfRange(AZ_ACTION_SIZE as Action))
    );
}

#[test]
fn az_state_shapes_and_history_are_canonical() {
    assert_eq!(ChessAzState::<1>::STATE_SHAPE, [21, 8, 8]);
    assert_eq!(ChessAzState::<4>::STATE_SHAPE, [63, 8, 8]);
    assert_eq!(ChessAzState::<8>::STATE_SHAPE, [119, 8, 8]);

    let mut state = ChessGame::default().az_state::<4>();
    let mut encoded = vec![0.0; ChessAzState::<4>::state_size()];
    state.encode_state(&mut encoded);
    assert!(encoded[14 * 64..56 * 64].iter().all(|&value| value == 0.0));
    assert_eq!(encoded[6 * 8], 1.0, "own pawns face north for white");

    let action = state.parse_move("e2e4").unwrap();
    state.step(action);
    state.encode_state(&mut encoded);
    assert_eq!(encoded[6 * 8], 1.0, "own pawns face north for black too");
    assert!(encoded[14 * 64..28 * 64].iter().any(|&value| value != 0.0));
}

#[test]
fn az_state_marks_repeated_current_positions() {
    let mut state = ChessGame::default().az_state::<8>();
    for text in ["b1c3", "b8c6", "c3b1", "c6b8"] {
        state.step(state.parse_move(text).unwrap());
    }
    let mut encoded = vec![0.0; ChessAzState::<8>::state_size()];
    state.encode_state(&mut encoded);
    assert!(encoded[12 * 64..13 * 64].iter().all(|&value| value == 1.0));
    assert!(encoded[13 * 64..14 * 64].iter().all(|&value| value == 0.0));
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
    assert_eq!(g.pos.status, Status::Stalemate);
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
fn castling_actions_are_exposed_to_policy() {
    let g = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    let kingside = g.parse_move("e1g1").expect("white O-O should be legal");
    let queenside = g.parse_move("e1c1").expect("white O-O-O should be legal");
    let legal: Vec<u32> = g.legal_actions().collect();

    assert!(legal.contains(&kingside));
    assert!(legal.contains(&queenside));
    assert_eq!(g.format_action(kingside), "e1g1");
    assert_eq!(g.format_action(queenside), "e1c1");
    assert_eq!(g.san_for_action(kingside), "O-O");
    assert_eq!(g.san_for_action(queenside), "O-O-O");
    assert!((kingside as usize) < ChessGame::ACTION_SIZE);
    assert!((queenside as usize) < ChessGame::ACTION_SIZE);
}

#[test]
fn castling_step_moves_king_and_rook() {
    let mut g = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    g.pos.halfmove_clock = 17;

    g.step(g.parse_move("e1g1").unwrap());
    assert_eq!(g.board().piece_on(Square::G1), Some(Piece::King));
    assert_eq!(g.board().piece_on(Square::F1), Some(Piece::Rook));
    assert_eq!(g.board().piece_on(Square::E1), None);
    assert_eq!(g.board().piece_on(Square::H1), None);
    assert!(!g.board().castle_rights(Color::White).has_kingside());
    assert!(!g.board().castle_rights(Color::White).has_queenside());
    assert_eq!(g.pos.halfmove_clock, 0);

    g.step(g.parse_move("e8c8").unwrap());
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
        g.step(g.parse_move(mv).unwrap());
    }
    assert!(g.is_terminal());
    assert_eq!(g.pos.status, Status::DrawRepetition);
    assert_eq!(g.reward(), 0.0);
}

#[test]
fn fifty_move_rule_is_a_draw() {
    let mut g = from_fen("k7/8/8/8/8/8/8/K7 w - - 0 1");
    g.pos.halfmove_clock = 99;
    g.step(g.parse_move("a1a2").unwrap());
    assert!(g.is_terminal());
    assert_eq!(g.pos.status, Status::DrawFiftyMoveRule);
    assert_eq!(g.reward(), 0.0);
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
    g.step(g.parse_move("a2a3").unwrap());
    assert_eq!(g.pos.halfmove_clock, 0);
    assert!(!g.is_terminal());
}

#[test]
fn en_passant_capture_resets_halfmove_clock() {
    let mut g = ChessGame::default();
    for mv in ["e2e4", "a7a6", "e4e5", "d7d5"] {
        g.step(g.parse_move(mv).unwrap());
    }
    g.pos.halfmove_clock = 99;
    g.step(g.parse_move("e5d6").unwrap());
    assert_eq!(g.pos.halfmove_clock, 0);
    assert!(!g.is_terminal());
}

#[test]
fn capture_resets_halfmove_clock() {
    let mut g = from_fen("4k3/8/8/8/8/8/4p3/4K3 w - - 0 1");
    g.pos.halfmove_clock = 99;
    g.step(g.parse_move("e1e2").unwrap());
    assert_eq!(g.pos.halfmove_clock, 0);
    assert!(!g.is_terminal());
}

#[test]
fn castling_rights_change_resets_halfmove_clock() {
    let mut g = from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    g.pos.halfmove_clock = 99;
    g.step(g.parse_move("e1f1").unwrap());
    assert_eq!(g.pos.halfmove_clock, 0);
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

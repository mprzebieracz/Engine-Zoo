use super::{Action, AlphaZeroRepresentation};
use chess::{Board, ChessMove, Color, File, Piece, Rank, Square};
use games::ChessHistoryState;

const DIRS: [(i32, i32); 8] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
];
const KNIGHTS: [(i32, i32); 8] = [
    (-2, 1),
    (-1, 2),
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
];
const PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

#[derive(Clone, Copy, Debug, Default)]
pub struct ChessAzRepresentation<const HISTORY: usize>;

fn cell(side: Color, square: Square) -> (i32, i32) {
    (
        if side == Color::White {
            7 - square.get_rank().to_index() as i32
        }
        else {
            square.get_rank().to_index() as i32
        },
        square.get_file().to_index() as i32,
    )
}
fn square(side: Color, row: i32, col: i32) -> Option<Square> {
    (0..8).contains(&row).then_some(())?;
    (0..8).contains(&col).then_some(())?;
    Some(Square::make_square(
        Rank::from_index(if side == Color::White {
            7 - row as usize
        }
        else {
            row as usize
        }),
        File::from_index(col as usize),
    ))
}
fn plane(side: Color, mv: ChessMove) -> u32 {
    let (r, c) = cell(side, mv.get_source());
    let (tr, tc) = cell(side, mv.get_dest());
    let d = (tr - r, tc - c);
    if let Some(p) = mv.get_promotion() {
        if p != Piece::Queen {
            let m = match d {
                (-1, 0) => 0,
                (-1, -1) => 1,
                (-1, 1) => 2,
                _ => panic!("invalid promotion"),
            };
            return 64
                + m * 3
                + match p {
                    Piece::Knight => 0,
                    Piece::Bishop => 1,
                    Piece::Rook => 2,
                    _ => unreachable!(),
                };
        }
    }
    if let Some((i, n)) = DIRS
        .iter()
        .enumerate()
        .find_map(|(i, &(dr, dc))| (1..=7).find_map(|n| (d == (dr * n, dc * n)).then_some((i, n))))
    {
        return i as u32 * 7 + n as u32 - 1;
    }
    56 + KNIGHTS
        .iter()
        .position(|&x| x == d)
        .expect("unsupported chess move") as u32
}
fn encode_move(board: &Board, mv: ChessMove) -> Action {
    let (r, c) = cell(board.side_to_move(), mv.get_source());
    Action::new(plane(board.side_to_move(), mv) * 64 + (r * 8 + c) as u32)
}
fn decode_move(board: &Board, action: Action) -> Option<ChessMove> {
    if action.index() >= 4672 {
        return None;
    }
    let p = action.index() / 64;
    let s = action.index() % 64;
    let r = (s / 8) as i32;
    let c = (s % 8) as i32;
    let (dr, dc, mut promo) = match p {
        0..=55 => {
            let (a, b) = DIRS[p / 7];
            let n = (p % 7 + 1) as i32;
            (a * n, b * n, None)
        }
        56..=63 => {
            let (a, b) = KNIGHTS[p - 56];
            (a, b, None)
        }
        64..=72 => {
            let q = match (p - 64) % 3 {
                0 => Piece::Knight,
                1 => Piece::Bishop,
                _ => Piece::Rook,
            };
            let d = match (p - 64) / 3 {
                0 => (-1, 0),
                1 => (-1, -1),
                _ => (-1, 1),
            };
            (d.0, d.1, Some(q))
        }
        _ => unreachable!(),
    };
    let from = square(board.side_to_move(), r, c)?;
    let to = square(board.side_to_move(), r + dr, c + dc)?;
    if promo.is_none()
        && board.piece_on(from) == Some(Piece::Pawn)
        && r == 1
        && dr == -1
        && (-1..=1).contains(&dc)
    {
        promo = Some(Piece::Queen);
    }
    Some(ChessMove::new(from, to, promo))
}

impl<const HISTORY: usize> AlphaZeroRepresentation<ChessHistoryState<HISTORY>>
    for ChessAzRepresentation<HISTORY>
{
    const STATE_SHAPE: [usize; 3] = [14 * HISTORY + 7, 8, 8];
    const ACTION_SIZE: usize = 4672;
    fn encode_state(&self, state: &ChessHistoryState<HISTORY>, out: &mut [f32]) {
        assert_eq!(out.len(), Self::state_size());
        out.fill(0.0);
        let current = state.position();
        let side = current.board().side_to_move();
        state.for_each_frame(|i, frame, reps| {
            if let Some(pos) = frame {
                for (color, base) in [(side, 0), (!side, 6)] {
                    let occupied = *pos.board().color_combined(color);
                    for piece in PIECES {
                        for sq in *pos.board().pieces(piece) & occupied {
                            let (r, c) = cell(side, sq);
                            out[(i * 14 + base + piece.to_index()) * 64 + (r * 8 + c) as usize] =
                                1.0;
                        }
                    }
                }
                if reps >= 1 {
                    out[(i * 14 + 12) * 64..(i * 14 + 13) * 64].fill(1.0);
                }
                if reps >= 2 {
                    out[(i * 14 + 13) * 64..(i * 14 + 14) * 64].fill(1.0);
                }
            }
        });
        let b = HISTORY * 14 * 64;
        let cr = |c| current.board().castle_rights(c);
        for (n, v) in [
            side == Color::White,
            cr(side).has_kingside(),
            cr(side).has_queenside(),
            cr(!side).has_kingside(),
            cr(!side).has_queenside(),
        ]
        .into_iter()
        .enumerate()
        {
            out[(b + n * 64)..(b + (n + 1) * 64)].fill(f32::from(v));
        }
        out[b + 320..b + 384].fill((current.halfmove_clock() as f32 / 100.0).min(1.0));
        out[b + 384..b + 448].fill((f32::from(current.ply() / 2 + 1) / 200.0).min(1.0));
    }
    fn move_to_action(&self, state: &ChessHistoryState<HISTORY>, mv: ChessMove) -> Action {
        encode_move(state.board(), mv)
    }
    fn action_to_move(&self, state: &ChessHistoryState<HISTORY>, a: Action) -> Option<ChessMove> {
        let mv = decode_move(state.board(), a)?;
        state.board().legal(mv).then_some(mv)
    }
    fn encoded_state_key(&self, state: &ChessHistoryState<HISTORY>) -> u64 {
        let mut k = 0x9E37_79B9_7F4A_7C15;
        state.for_each_frame(|i, p, r| {
            let v = p.map_or(0, |x| x.hash() ^ (u64::from(r) << 61));
            k = mix(k, v ^ i as u64);
        });
        let p = state.position();
        mix(k, (p.halfmove_clock() as u64) << 16 | u64::from(p.ply()))
    }
}
fn mix(k: u64, v: u64) -> u64 {
    (k ^ (v.wrapping_mul(0xBF58_476D_1CE4_E5B9)))
        .rotate_left(27)
        .wrapping_mul(0x94D0_49BB_1331_11EB)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{Game, GameState, RepetitionGame};
    use games::{ChessGame, ChessHistoryState};
    use std::collections::HashSet;
    use std::str::FromStr;

    fn check<const H: usize>(state: ChessHistoryState<H>) {
        let r = ChessAzRepresentation::<H>;
        let mut expected = vec![0.0; ChessHistoryState::<H>::state_size()];
        let mut actual = vec![
            0.0;
            <ChessAzRepresentation<H> as AlphaZeroRepresentation<
                ChessHistoryState<H>,
            >>::state_size()
        ];
        state.encode_state(&mut expected);
        r.encode_state(&state, &mut actual);
        assert_eq!(actual, expected);
        assert_eq!(r.encoded_state_key(&state), state.evaluation_cache_key());
        let mut actions = HashSet::new();
        for mv in state.legal_moves() {
            let action = r.move_to_action(&state, mv);
            assert!(actions.insert(action));
            assert_eq!(r.action_to_move(&state, action), Some(mv));
        }
        assert!(r.action_to_move(&state, Action::new(4672)).is_none());
    }

    #[test]
    fn golden_h1_h4_h8_fens_and_populated_histories() {
        for fen in [
            "rn1qk2r/ppp2ppp/2pb1n2/8/2B1P3/2N2N2/PPP2PPP/R1BQ1RK1 b kq - 3 8",
            "r3k2r/ppp1bppp/2n1p3/8/2BPP3/2N2N2/PPP2PPP/R1BQ1RK1 w kq - 0 9",
        ] {
            let game = ChessGame::from_fen(fen).unwrap();
            check(game.history_state::<1>());
            check(game.history_state::<4>());
            check(game.history_state::<8>());
        }

        let mut game = ChessGame::default();
        let mut seed = 0xA17E_5EED_u32;
        for _ in 0..48 {
            check(game.history_state::<1>());
            check(game.history_state::<4>());
            check(game.history_state::<8>());
            let moves: Vec<_> = chess::MoveGen::new_legal(game.board()).collect();
            if moves.is_empty() {
                break;
            }
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            game.play(moves[seed as usize % moves.len()]);
        }
    }

    #[test]
    fn promotions_castling_and_en_passant_cover_both_colors() {
        for fen in [
            "1r3r1k/P1P1P3/8/8/8/8/8/K7 w - - 0 1",
            "k7/8/8/8/8/8/p1p1p3/1R3R1K b - - 0 1",
        ] {
            let pos = ChessGame::from_fen(fen).unwrap().position();
            let state = ChessHistoryState::<1>::new(pos);
            let r = ChessAzRepresentation::<1>;
            let mut pieces = HashSet::new();
            for mv in state
                .legal_moves()
                .filter(|mv| mv.get_promotion().is_some())
            {
                pieces.insert(mv.get_promotion().unwrap());
                assert_eq!(
                    r.action_to_move(&state, r.move_to_action(&state, mv)),
                    Some(mv)
                );
            }
            assert_eq!(
                pieces,
                HashSet::from([Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight])
            );
        }

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
            let state = ChessGame::from_fen(fen).unwrap().history_state::<1>();
            let r = ChessAzRepresentation::<1>;
            for text in moves {
                let mv = chess::ChessMove::from_str(text).unwrap();
                assert_eq!(
                    r.action_to_move(&state, r.move_to_action(&state, mv)),
                    Some(mv)
                );
            }
        }
    }

    #[test]
    fn rejects_actions_from_the_wrong_position() {
        let source = ChessHistoryState::<1>::default();
        let other = ChessGame::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
            .unwrap()
            .history_state::<1>();
        let r = ChessAzRepresentation::<1>;
        let action = r.move_to_action(&source, chess::ChessMove::from_str("e2e4").unwrap());
        assert_eq!(r.action_to_move(&other, action), None);
        assert_eq!(r.action_to_move(&source, Action::new(4672)), None);
    }

    #[test]
    fn key_changes_for_clock_and_history_features() {
        let r = ChessAzRepresentation::<4>;
        let clock_a =
            ChessGame::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                .unwrap()
                .history_state::<4>();
        let clock_b =
            ChessGame::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 4 3")
                .unwrap()
                .history_state::<4>();
        check(clock_a);
        check(clock_b);
        assert_ne!(r.encoded_state_key(&clock_a), r.encoded_state_key(&clock_b));

        let mut game = ChessGame::default();
        for text in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            game.play(chess::ChessMove::from_str(text).unwrap());
        }
        let populated = game.history_state::<4>();
        check(populated);
        assert_eq!(populated.position().hash(), clock_b.position().hash());
        assert_eq!(
            populated.position().halfmove_clock(),
            clock_b.position().halfmove_clock()
        );
        assert_eq!(populated.position().ply(), clock_b.position().ply());
        assert_ne!(
            r.encoded_state_key(&populated),
            r.encoded_state_key(&clock_b)
        );
    }
}

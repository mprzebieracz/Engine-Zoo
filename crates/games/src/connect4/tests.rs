use super::*;
use engine_core::{
    game::{GameState, TerminalValue},
    notation::GameNotation,
};
use std::mem::size_of;

/// Straightforward array-based implementation used as an oracle for randomized
/// cross-checking of the bitboard version.
#[derive(Clone)]
struct Naive {
    board: [[i8; COLS]; ROWS],
    current: i8,
    status: Status,
}

fn mv(column: u8) -> Connect4Move {
    Connect4Move::new(column).unwrap()
}

fn moves(columns: &[u8]) -> Vec<Connect4Move> {
    columns.iter().copied().map(mv).collect()
}

#[test]
fn move_is_checked_and_compact() {
    assert_eq!(mv(3).column(), 3);
    assert!(Connect4Move::new(7).is_none());
    assert_eq!(size_of::<Connect4Move>(), 1);
    // A private u8 has no invalid bit pattern for Option to use as a niche.
    assert_eq!(size_of::<Option<Connect4Move>>(), 2);
}

#[test]
fn native_move_flow_and_terminal_values() {
    let mut game = Connect4::initial();
    assert_eq!(game.legal_moves().count(), COLS);
    game.play(mv(0));
    assert_eq!(game.legal_moves().next().unwrap().column(), 0);
    assert_eq!(game.terminal_value(), None);

    for column in [1, 0, 1, 0, 1, 0] {
        game.play(mv(column));
    }
    assert_eq!(game.terminal_value(), Some(TerminalValue::Loss));
}

impl Naive {
    fn new() -> Self {
        Naive {
            board: [[0; COLS]; ROWS],
            current: 1,
            status: Status::Ongoing,
        }
    }

    fn legal(&self) -> Vec<Connect4Move> {
        (0..COLS)
            .filter(|&c| self.board[0][c] == 0)
            .map(|c| mv(c as u8))
            .collect()
    }

    fn step(&mut self, mv: Connect4Move) {
        let col = mv.column() as usize;
        let row = (0..ROWS).rev().find(|&r| self.board[r][col] == 0).unwrap();
        self.board[row][col] = self.current;
        let won = self.check_win(row as i32, col as i32);
        let full = (0..COLS).all(|c| self.board[0][c] != 0);
        self.current = -self.current;
        if won {
            self.status = Status::Loss;
        }
        else if full {
            self.status = Status::Draw;
        }
    }

    fn check_win(&self, row: i32, col: i32) -> bool {
        let me = self.board[row as usize][col as usize];
        for (dr, dc) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
            let mut count = 0;
            for i in -3..=3 {
                let (r, c) = (row + i * dr, col + i * dc);
                if r >= 0
                    && r < ROWS as i32
                    && c >= 0
                    && c < COLS as i32
                    && self.board[r as usize][c as usize] == me
                {
                    count += 1;
                    if count == 4 {
                        return true;
                    }
                }
                else {
                    count = 0;
                }
            }
        }
        false
    }
}

#[test]
fn startpos_has_seven_moves() {
    let game = Connect4::initial();
    assert_eq!(game.legal_moves().count(), COLS);
}

#[test]
fn loads_position_from_move_list() {
    let game = Connect4::from_moves(&moves(&[3, 3, 4])).unwrap();
    assert_eq!(game.legal_moves().count(), COLS);
    assert_eq!(game.current_player(), -1);
}

#[test]
fn vertical_win_terminal_value_convention() {
    let mut game = Connect4::initial();
    for column in [0, 1, 0, 1, 0, 1, 0] {
        assert!(!game.is_terminal());
        game.play(mv(column));
    }
    // X just completed four-in-a-row in column 0; O to move has lost.
    assert!(game.is_terminal());
    assert_eq!(game.status, Status::Loss);
    assert_eq!(game.terminal_value(), Some(TerminalValue::Loss));
    assert_eq!(game.current_player(), -1);
}

#[test]
fn horizontal_win_is_a_loss() {
    let mut game = Connect4::initial();
    for column in [0, 0, 1, 1, 2, 2, 3] {
        game.play(mv(column));
    }
    assert!(game.is_terminal());
    assert_eq!(game.status, Status::Loss);
    assert_eq!(game.terminal_value(), Some(TerminalValue::Loss));
}

#[test]
fn detects_both_diagonal_win_directions() {
    for columns in [
        [0, 1, 1, 2, 4, 2, 2, 3, 5, 3, 5, 3, 3].as_slice(),
        [6, 5, 5, 4, 2, 4, 4, 3, 1, 3, 1, 3, 3].as_slice(),
    ] {
        let game = Connect4::from_moves(&moves(columns)).unwrap();
        assert_eq!(game.status, Status::Loss);
        assert_eq!(game.terminal_value(), Some(TerminalValue::Loss));
        assert!(game.is_terminal());
    }
}

#[test]
fn completely_filled_board_is_a_draw() {
    let columns = [
        1, 4, 6, 6, 6, 0, 2, 0, 3, 6, 3, 3, 5, 3, 6, 1, 0, 3, 0, 4, 3, 5, 0, 6, 5, 2, 2, 5, 1, 2,
        2, 0, 2, 5, 4, 5, 4, 4, 4, 1, 1, 1,
    ];
    let game = Connect4::from_moves(&moves(&columns)).unwrap();
    assert_eq!(game.mask, FULL_MASK);
    assert_eq!(game.status, Status::Draw);
    assert_eq!(game.terminal_value(), Some(TerminalValue::Draw));
    assert!(game.legal_moves().next().is_none());
}

#[test]
fn state_bits_are_from_the_side_to_move_perspective() {
    let mut game = Connect4::initial();
    game.play(mv(3));
    // O is now to move, so X's stone is not in the side-to-move bitboard.
    assert_eq!(game.position_bits(), 0);
    assert_eq!(game.occupied_bits(), 1 << (3 * 7));
}

#[test]
fn full_column_is_not_legal() {
    let mut game = Connect4::initial();
    for _ in 0..ROWS {
        game.play(mv(0));
    }
    assert!(!game.legal_moves().any(|move_| move_ == mv(0)));
}

#[test]
fn legal_moves_parse_and_format_roundtrip() {
    let mut game = Connect4::initial();
    let notation = notation::Connect4Notation;
    for played in [3, 3, 0, 6] {
        for move_ in game.legal_moves() {
            let text = notation.format_move(&game, move_);
            assert_eq!(notation.parse_move(&game, &text), Some(move_));
            assert_eq!(
                notation.parse_move(&game, &format!("  {text}  ")),
                Some(move_)
            );
        }
        game.play(mv(played));
    }
    for invalid in ["", "-1", "7", "3.0", "garbage"] {
        assert_eq!(notation.parse_move(&game, invalid), None);
    }
}

#[test]
fn terminal_position_has_no_legal_moves() {
    let mut game = Connect4::initial();
    for column in [0, 1, 0, 1, 0, 1, 0] {
        game.play(mv(column));
    }

    assert!(game.is_terminal());
    assert_eq!(game.legal_moves().count(), 0);
}

#[test]
fn illegal_plays_panic_without_mutating_the_position() {
    let mut full_column = Connect4::initial();
    for _ in 0..ROWS {
        full_column.play(mv(0));
    }
    let before = (
        full_column.pos,
        full_column.mask,
        full_column.ply,
        full_column.status,
    );
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        full_column.play(mv(0));
    }))
    .is_err());
    assert_eq!(
        before,
        (
            full_column.pos,
            full_column.mask,
            full_column.ply,
            full_column.status
        )
    );

    let mut terminal = Connect4::initial();
    for column in [0, 1, 0, 1, 0, 1, 0] {
        terminal.play(mv(column));
    }
    let before = (terminal.pos, terminal.mask, terminal.ply, terminal.status);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        terminal.play(mv(2));
    }))
    .is_err());
    assert_eq!(
        before,
        (terminal.pos, terminal.mask, terminal.ply, terminal.status)
    );
}

#[test]
fn matches_naive_oracle_on_random_games() {
    use rand::prelude::*;

    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..300 {
        let mut fast = Connect4::initial();
        let mut naive = Naive::new();
        while !fast.is_terminal() {
            let legal: Vec<_> = fast.legal_moves().collect();
            assert_eq!(legal, naive.legal());
            assert_eq!(fast.current_player(), naive.current);

            let move_ = *legal.choose(&mut rng).unwrap();
            fast.play(move_);
            naive.step(move_);
        }
        assert_ne!(naive.status, Status::Ongoing);
        assert_eq!(fast.status, naive.status);
        assert_eq!(
            fast.terminal_value(),
            match naive.status {
                Status::Ongoing => None,
                Status::Loss => Some(TerminalValue::Loss),
                Status::Draw => Some(TerminalValue::Draw),
            }
        );
    }
}

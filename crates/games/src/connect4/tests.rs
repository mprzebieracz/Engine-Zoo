use super::*;
use engine_core::game::{GameState, TerminalValue};
use std::mem::size_of;

/// Straightforward array-based implementation used as an oracle for randomized
/// cross-checking of the bitboard version.
#[derive(Clone)]
struct Naive {
    board: [[i8; COLS]; ROWS],
    current: i8,
    status: Status,
}

#[test]
fn move_is_checked_and_compact() {
    assert_eq!(Connect4Move::new(3).unwrap().column(), 3);
    assert!(Connect4Move::new(7).is_none());
    assert_eq!(size_of::<Connect4Move>(), 1);
    // A private u8 has no invalid bit pattern for Option to use as a niche.
    assert_eq!(size_of::<Option<Connect4Move>>(), 2);
}

#[test]
fn native_move_flow_and_terminal_values() {
    let mut game = Connect4::initial();
    assert_eq!(game.legal_moves().count(), COLS);
    game.play(Connect4Move::new(0).unwrap());
    assert_eq!(game.legal_moves().next().unwrap().column(), 0);
    assert_eq!(game.terminal_value(), None);

    for column in [1, 0, 1, 0, 1, 0] {
        game.play(Connect4Move::new(column).unwrap());
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

    fn legal(&self) -> Vec<Action> {
        (0..COLS)
            .filter(|&c| self.board[0][c] == 0)
            .map(|c| c as Action)
            .collect()
    }

    fn step(&mut self, col: usize) {
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

    fn encode(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; ROWS * COLS];
        for r in 0..ROWS {
            for c in 0..COLS {
                out[r * COLS + c] = (self.board[r][c] * self.current) as f32;
            }
        }
        out
    }

    fn reward(&self) -> f32 {
        if self.status == Status::Loss {
            -1.0
        }
        else {
            0.0
        }
    }
}

#[test]
fn startpos_has_seven_moves() {
    let g = Connect4::default();
    assert_eq!(g.legal_actions().count(), COLS);
}

#[test]
fn loads_position_from_move_list() {
    let g = Connect4::from_moves(&[3, 3, 4]).unwrap();
    assert_eq!(g.legal_actions().count(), COLS);
    assert_eq!(g.current_player(), -1);
}

#[test]
fn vertical_win_reward_convention() {
    let mut g = Connect4::default();
    for a in [0u32, 1, 0, 1, 0, 1, 0] {
        assert!(!engine_core::Game::is_terminal(&g));
        g.step(a);
    }
    // X just completed four-in-a-row in column 0; O to move has lost.
    assert!(engine_core::Game::is_terminal(&g));
    assert_eq!(g.status, Status::Loss);
    assert_eq!(g.reward(), -1.0);
    assert_eq!(g.current_player(), -1);
}

#[test]
fn horizontal_win_is_a_loss() {
    let mut g = Connect4::default();
    for a in [0u32, 0, 1, 1, 2, 2, 3] {
        g.step(a);
    }
    assert!(engine_core::Game::is_terminal(&g));
    assert_eq!(g.status, Status::Loss);
    assert_eq!(g.reward(), -1.0);
}

#[test]
fn detects_both_diagonal_win_directions() {
    for moves in [
        vec![0, 1, 1, 2, 4, 2, 2, 3, 5, 3, 5, 3, 3],
        vec![6, 5, 5, 4, 2, 4, 4, 3, 1, 3, 1, 3, 3],
    ] {
        let game = Connect4::from_moves(&moves).unwrap();
        assert_eq!(game.status, Status::Loss);
        assert_eq!(game.reward(), -1.0);
        assert!(engine_core::Game::is_terminal(&game));
    }
}

#[test]
fn completely_filled_board_is_a_draw() {
    let moves = [
        1, 4, 6, 6, 6, 0, 2, 0, 3, 6, 3, 3, 5, 3, 6, 1, 0, 3, 0, 4, 3, 5, 0, 6, 5, 2, 2, 5, 1, 2,
        2, 0, 2, 5, 4, 5, 4, 4, 4, 1, 1, 1,
    ];
    let game = Connect4::from_moves(&moves).unwrap();
    assert_eq!(game.mask, FULL_MASK);
    assert_eq!(game.status, Status::Draw);
    assert_eq!(game.reward(), 0.0);
    assert!(game.legal_actions().next().is_none());
}

#[test]
fn encoding_is_canonical() {
    let mut g = Connect4::default();
    g.step(3);
    // From O's perspective the X stone at bottom row, column 3 is -1.
    let mut out = vec![0.0f32; Connect4::state_size()];
    g.encode_state(&mut out);
    assert_eq!(out[5 * COLS + 3], -1.0);
    assert_eq!(out.iter().filter(|&&v| v != 0.0).count(), 1);
}

#[test]
fn full_column_is_not_legal() {
    let mut g = Connect4::default();
    for _ in 0..ROWS {
        g.step(0);
    }
    assert!(!g.legal_actions().any(|a| a == 0));
}

#[test]
fn legal_actions_parse_and_format_roundtrip() {
    let mut game = Connect4::default();
    for played in [3, 3, 0, 6] {
        for action in game.legal_actions() {
            let text = game.format_action(action);
            assert_eq!(game.parse_move(&text), Some(action));
            assert_eq!(game.parse_move(&format!("  {text}  ")), Some(action));
        }
        game.step(played);
    }
    for invalid in ["", "-1", "7", "3.0", "garbage"] {
        assert_eq!(game.parse_move(invalid), None);
    }
}

#[test]
fn terminal_position_has_no_legal_actions() {
    let mut g = Connect4::default();
    for action in [0, 1, 0, 1, 0, 1, 0] {
        g.step(action);
    }

    assert!(engine_core::Game::is_terminal(&g));
    assert_eq!(g.legal_actions().count(), 0);
}

#[test]
fn illegal_steps_panic_without_mutating_the_position() {
    let mut full_column = Connect4::default();
    for _ in 0..ROWS {
        full_column.step(0);
    }
    let before = (
        full_column.pos,
        full_column.mask,
        full_column.ply,
        full_column.status,
    );
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        full_column.step(0);
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

    let mut terminal = Connect4::default();
    for action in [0, 1, 0, 1, 0, 1, 0] {
        terminal.step(action);
    }
    let before = (terminal.pos, terminal.mask, terminal.ply, terminal.status);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        terminal.step(2);
    }))
    .is_err());
    assert_eq!(
        before,
        (terminal.pos, terminal.mask, terminal.ply, terminal.status)
    );
}

#[test]
#[should_panic(expected = "invalid state buffer length")]
fn encoding_rejects_wrong_buffer_length() {
    Connect4::default().encode_state(&mut []);
}

#[test]
fn matches_naive_oracle_on_random_games() {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..300 {
        let mut fast = Connect4::default();
        let mut naive = Naive::new();
        while !engine_core::Game::is_terminal(&fast) {
            let legal: Vec<u32> = fast.legal_actions().collect();
            assert_eq!(legal, naive.legal());
            assert_eq!(fast.current_player(), naive.current);
            let mut enc = vec![0.0f32; Connect4::state_size()];
            fast.encode_state(&mut enc);
            assert_eq!(enc, naive.encode());

            let a = *legal.choose(&mut rng).unwrap();
            fast.step(a);
            naive.step(a as usize);
        }
        assert_ne!(naive.status, Status::Ongoing);
        assert_eq!(fast.status, naive.status);
        assert_eq!(fast.reward(), naive.reward());
    }
}

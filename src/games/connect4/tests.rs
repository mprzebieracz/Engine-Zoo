use super::*;

/// Straightforward array-based implementation used as an oracle for randomized
/// cross-checking of the bitboard version.
#[derive(Clone)]
struct Naive {
    board: [[i8; COLS]; ROWS],
    current: i8,
    status: Status,
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
        assert!(!g.is_terminal());
        g.step(a);
    }
    // X just completed four-in-a-row in column 0; O to move has lost.
    assert!(g.is_terminal());
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
    assert!(g.is_terminal());
    assert_eq!(g.status, Status::Loss);
    assert_eq!(g.reward(), -1.0);
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
fn matches_naive_oracle_on_random_games() {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..300 {
        let mut fast = Connect4::default();
        let mut naive = Naive::new();
        while !fast.is_terminal() {
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

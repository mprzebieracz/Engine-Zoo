use crate::game::{Action, Game, TensorDim};
use std::fmt;

const ROWS: usize = 6;
const COLS: usize = 7;

/// Bit index of (col, row-from-bottom): each column occupies 7 bits (6 cells +
/// 1 sentinel) so that `mask + bottom_bit(col)` carries into the lowest empty
/// cell and the shift-based alignment checks can't wrap between columns.
const fn bit(col: usize, row_from_bottom: usize) -> u64 {
    1u64 << (col * 7 + row_from_bottom)
}

const fn column_mask(col: usize) -> u64 {
    0b011_1111u64 << (col * 7)
}

const fn bottom_bit(col: usize) -> u64 {
    bit(col, 0)
}

const FULL_MASK: u64 = {
    let mut m = 0u64;
    let mut c = 0;
    while c < COLS {
        m |= column_mask(c);
        c += 1;
    }
    m
};

/// Four-in-a-row test on one player's bitboard: fold pairs, then pairs of
/// pairs, along each of the four directions (vertical, horizontal, both
/// diagonals — shifts 1, 7, 6, 8 in this 7-bits-per-column layout).
fn has_alignment(bb: u64) -> bool {
    for shift in [1u32, 7, 6, 8] {
        let m = bb & (bb >> shift);
        if m & (m >> (2 * shift)) != 0 {
            return true;
        }
    }
    false
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum Status {
    #[default]
    Ongoing,
    /// Side to move has lost (opponent completed four in a row).
    Loss,
    Draw,
}

/// Connect Four with bitboard move generation.
#[derive(Clone, Copy, Default)]
pub struct Connect4 {
    /// Stones of the player to move.
    pos: u64,
    /// All stones.
    mask: u64,
    ply: u16,
    status: Status,
}

impl Connect4 {
    pub fn from_moves(moves: &[Action]) -> anyhow::Result<Self> {
        let mut game = Connect4::default();
        for &action in moves {
            anyhow::ensure!(
                game.parse_move(&action.to_string()).is_some(),
                "illegal connect4 move {action}"
            );
            game.step(action);
        }
        Ok(game)
    }

    /// 1 if the first player (X) is to move, -1 otherwise.
    pub fn current_player(&self) -> i8 {
        if self.ply.is_multiple_of(2) {
            1
        }
        else {
            -1
        }
    }

    fn player1_stones(&self) -> u64 {
        if self.ply.is_multiple_of(2) {
            self.pos
        }
        else {
            self.pos ^ self.mask
        }
    }

    fn column_playable(&self, col: usize) -> bool {
        self.mask & bit(col, ROWS - 1) == 0
    }
}

impl Game for Connect4 {
    const ACTION_SIZE: usize = COLS;
    const STATE_SHAPE: [TensorDim; 3] = [1, ROWS as TensorDim, COLS as TensorDim];
    const NAME: &'static str = "connect4";

    fn legal_actions(&self) -> impl Iterator<Item = Action> + '_ {
        (0..COLS)
            .filter(|&c| self.column_playable(c))
            .map(|c| c as Action)
    }

    fn step(&mut self, action: Action) {
        let col = action as usize;
        debug_assert!(self.status == Status::Ongoing && col < COLS && self.column_playable(col));

        let move_bit = (self.mask + bottom_bit(col)) & column_mask(col);
        let placed = self.pos | move_bit;
        self.mask |= move_bit;
        // Perspective flip: the new player to move is the opponent of the mover.
        self.pos = placed ^ self.mask;
        self.ply += 1;

        if has_alignment(placed) {
            self.status = Status::Loss;
        }
        else if self.mask == FULL_MASK {
            self.status = Status::Draw;
        }
    }

    fn is_terminal(&self) -> bool {
        self.status != Status::Ongoing
    }

    fn reward(&self) -> f32 {
        if self.status == Status::Loss {
            -1.0
        }
        else {
            0.0
        }
    }

    fn encode_state(&self, out: &mut [f32]) {
        let own = self.pos;
        let opp = self.pos ^ self.mask;
        for row_top in 0..ROWS {
            for col in 0..COLS {
                let b = bit(col, ROWS - 1 - row_top);
                out[row_top * COLS + col] = if own & b != 0 {
                    1.0
                }
                else if opp & b != 0 {
                    -1.0
                }
                else {
                    0.0
                };
            }
        }
    }

    fn parse_move(&self, s: &str) -> Option<Action> {
        let col: usize = s.trim().parse().ok()?;
        (col < COLS && self.column_playable(col) && self.status == Status::Ongoing)
            .then_some(col as Action)
    }

    fn format_action(&self, action: Action) -> String {
        action.to_string()
    }
}

impl fmt::Display for Connect4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p1 = self.player1_stones();
        let p2 = p1 ^ self.mask;
        for row_top in 0..ROWS {
            for col in 0..COLS {
                let b = bit(col, ROWS - 1 - row_top);
                let cell = if p1 & b != 0 {
                    'X'
                }
                else if p2 & b != 0 {
                    'O'
                }
                else {
                    '.'
                };
                write!(f, "{cell} ")?;
            }
            writeln!(f)?;
        }
        writeln!(f, "0 1 2 3 4 5 6")?;
        writeln!(
            f,
            "{} to move",
            if self.current_player() == 1 { 'X' } else { 'O' }
        )
    }
}

#[cfg(test)]
mod tests;

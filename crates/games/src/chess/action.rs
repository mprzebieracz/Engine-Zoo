use chess::{Board, ChessMove, Color, File, Piece, Rank, Square};
use engine_core::game::Action;
use std::fmt;

pub const AZ_ACTION_SIZE: usize = 8 * 8 * 73;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzActionError {
    OutOfRange(Action),
    OffBoard { action: Action },
}

impl fmt::Display for AzActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange(action) => write!(f, "v2 chess action {action} is out of range"),
            Self::OffBoard { action } => write!(f, "v2 chess action {action} leaves the board"),
        }
    }
}

impl std::error::Error for AzActionError {}

const SLIDING_DIRECTIONS: [(i32, i32); 8] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
];

const KNIGHT_OFFSETS: [(i32, i32); 8] = [
    (-2, 1),
    (-1, 2),
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
];

pub(super) fn square_to_az_cell(side_to_move: Color, square: Square) -> (i32, i32) {
    let file = square.get_file().to_index() as i32;
    let rank = square.get_rank().to_index() as i32;
    let row = if side_to_move == Color::White {
        7 - rank
    }
    else {
        rank
    };
    (row, file)
}

fn az_cell_to_square(side_to_move: Color, row: i32, col: i32) -> Option<Square> {
    if !(0..8).contains(&row) || !(0..8).contains(&col) {
        return None;
    }
    let rank = if side_to_move == Color::White {
        7 - row
    }
    else {
        row
    };
    Some(Square::make_square(
        Rank::from_index(rank as usize),
        File::from_index(col as usize),
    ))
}

fn az_plane_for_move(side_to_move: Color, mv: ChessMove) -> u32 {
    let (from_row, from_col) = square_to_az_cell(side_to_move, mv.get_source());
    let (to_row, to_col) = square_to_az_cell(side_to_move, mv.get_dest());
    let delta = (to_row - from_row, to_col - from_col);

    if let Some(promotion) = mv.get_promotion() {
        if promotion != Piece::Queen {
            let movement = match delta {
                (-1, 0) => 0,
                (-1, -1) => 1,
                (-1, 1) => 2,
                _ => panic!("underpromotion has invalid geometry: {mv}"),
            };
            let piece = match promotion {
                Piece::Knight => 0,
                Piece::Bishop => 1,
                Piece::Rook => 2,
                _ => unreachable!("queen promotions use sliding planes"),
            };
            return 64 + movement * 3 + piece;
        }
    }

    if let Some((direction, distance)) =
        SLIDING_DIRECTIONS
            .iter()
            .enumerate()
            .find_map(|(direction, &(dr, dc))| {
                (1..=7).find_map(|distance| {
                    (delta == (dr * distance, dc * distance)).then_some((direction, distance))
                })
            })
    {
        return direction as u32 * 7 + distance as u32 - 1;
    }
    if let Some(knight) = KNIGHT_OFFSETS.iter().position(|&offset| offset == delta) {
        return 56 + knight as u32;
    }
    panic!("chess move has unsupported AlphaZero geometry: {mv}");
}

pub fn encode_az_move(board: &Board, mv: ChessMove) -> Action {
    let side_to_move = board.side_to_move();
    let (row, col) = square_to_az_cell(side_to_move, mv.get_source());
    az_plane_for_move(side_to_move, mv) * 64 + row as u32 * 8 + col as u32
}

pub fn decode_az_move(board: &Board, action: Action) -> Result<ChessMove, AzActionError> {
    if action as usize >= AZ_ACTION_SIZE {
        return Err(AzActionError::OutOfRange(action));
    }
    let plane = action / 64;
    let source = action % 64;
    let row = (source / 8) as i32;
    let col = (source % 8) as i32;
    let (dr, dc, mut promotion) = match plane {
        0..=55 => {
            let direction = (plane / 7) as usize;
            let distance = (plane % 7 + 1) as i32;
            let (dr, dc) = SLIDING_DIRECTIONS[direction];
            (dr * distance, dc * distance, None)
        }
        56..=63 => {
            let (dr, dc) = KNIGHT_OFFSETS[(plane - 56) as usize];
            (dr, dc, None)
        }
        64..=72 => {
            let promotion = match (plane - 64) % 3 {
                0 => Piece::Knight,
                1 => Piece::Bishop,
                _ => Piece::Rook,
            };
            let (dr, dc) = match (plane - 64) / 3 {
                0 => (-1, 0),
                1 => (-1, -1),
                _ => (-1, 1),
            };
            (dr, dc, Some(promotion))
        }
        _ => unreachable!("range checked above"),
    };
    let side_to_move = board.side_to_move();
    let source =
        az_cell_to_square(side_to_move, row, col).ok_or(AzActionError::OffBoard { action })?;
    let dest = az_cell_to_square(side_to_move, row + dr, col + dc)
        .ok_or(AzActionError::OffBoard { action })?;
    if promotion.is_none()
        && board.piece_on(source) == Some(Piece::Pawn)
        && row == 1
        && row + dr == 0
        && dr == -1
        && (-1..=1).contains(&dc)
    {
        promotion = Some(Piece::Queen);
    }
    Ok(ChessMove::new(source, dest, promotion))
}

fn square_to_action_cell(square: Square) -> u32 {
    let index = square.to_index() as u32;
    let file = index & 7;
    let rank = index >> 3;
    (7 - rank) * 8 + file
}

fn action_cell_to_square(row: u32, column: u32) -> Square {
    Square::make_square(
        Rank::from_index(7 - row as usize),
        File::from_index(column as usize),
    )
}

pub fn encode_move(mv: ChessMove) -> Action {
    let from = square_to_action_cell(mv.get_source());
    let to = square_to_action_cell(mv.get_dest());
    let promotion = match mv.get_promotion() {
        None => 0,
        Some(Piece::Queen) => 1,
        Some(Piece::Rook) => 2,
        Some(Piece::Knight) => 3,
        Some(Piece::Bishop) => 4,
        Some(other) => unreachable!("illegal promotion piece {other:?}"),
    };
    (from * 64 + to) * 5 + promotion
}

pub fn decode_move(action: Action) -> ChessMove {
    let promotion = match action % 5 {
        0 => None,
        1 => Some(Piece::Queen),
        2 => Some(Piece::Rook),
        3 => Some(Piece::Knight),
        _ => Some(Piece::Bishop),
    };
    let from_to = action / 5;
    let (from, to) = (from_to / 64, from_to % 64);
    ChessMove::new(
        action_cell_to_square(from / 8, from % 8),
        action_cell_to_square(to / 8, to % 8),
        promotion,
    )
}

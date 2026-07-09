use chess::{Board, BoardStatus, ChessMove, Color, File, MoveGen, Piece, Rank, Square};
use engine_core::game::{Action, Game, TensorDim};
use engine_core::rules::RepetitionGame;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// Chess with move generation from the `chess` crate.
///
/// Actions are encoded as `(from * 64 + to) * 5 + promotion` in a board frame
/// where row 0 is rank 8 (`from = (7 - rank) * 8 + file`);
/// promotion codes (0 none, 1 queen, 2 rook, 3 knight, 4 bishop):
/// See `encode_state` for the 19-plane tensor encoding (planes 0-5 are own
/// pieces, 6-11 are opponent's, 12 is white-to-move, 13 is ply count, 14-17
/// are castling rights, 18 is en-passant file).
///
/// Fifty-move, threefold repetition, and stalemate are auto-draws (terminal, reward 0).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Ongoing,
    Checkmate,
    Stalemate,
    DrawRepetition,
    DrawFiftyMoveRule,
}

#[derive(Clone, Copy)]
pub struct ChessPosition {
    board: Board,
    ply: u16,
    status: Status,
    halfmove_clock: u16,
}

#[derive(Clone)]
pub struct ChessGame {
    pos: ChessPosition,
    /// Zobrist hash occurrence counts since the last irreversible move.
    position_counts: HashMap<u64, u8>,
}

fn square_to_rc(sq: Square) -> (u32, u32) {
    (
        7 - sq.get_rank().to_index() as u32,
        sq.get_file().to_index() as u32,
    )
}

fn rc_to_square(r: u32, c: u32) -> Square {
    Square::make_square(
        Rank::from_index(7 - r as usize),
        File::from_index(c as usize),
    )
}

pub fn encode_move(mv: ChessMove) -> Action {
    let (r1, c1) = square_to_rc(mv.get_source());
    let (r2, c2) = square_to_rc(mv.get_dest());
    let promo = match mv.get_promotion() {
        None => 0,
        Some(Piece::Queen) => 1,
        Some(Piece::Rook) => 2,
        Some(Piece::Knight) => 3,
        Some(Piece::Bishop) => 4,
        Some(other) => unreachable!("illegal promotion piece {other:?}"),
    };
    ((r1 * 8 + c1) * 64 + (r2 * 8 + c2)) * 5 + promo
}

pub fn decode_move(action: Action) -> ChessMove {
    let promo = match action % 5 {
        0 => None,
        1 => Some(Piece::Queen),
        2 => Some(Piece::Rook),
        3 => Some(Piece::Knight),
        _ => Some(Piece::Bishop),
    };
    let from_to = action / 5;
    let (from, to) = (from_to / 64, from_to % 64);
    ChessMove::new(
        rc_to_square(from / 8, from % 8),
        rc_to_square(to / 8, to % 8),
        promo,
    )
}

impl ChessGame {
    pub fn board(&self) -> &Board {
        &self.pos.board
    }

    pub fn position(&self) -> ChessPosition {
        self.pos
    }

    pub fn san_for_action(&self, action: Action) -> String {
        self.pos.san_for_action(action)
    }

    pub fn repetitions_before_current(&self, hash: u64) -> u8 {
        let count = self.position_counts.get(&hash).copied().unwrap_or(0);
        if hash == self.pos.hash() {
            count.saturating_sub(1)
        } else {
            count
        }
    }

    pub fn from_fen(fen: &str) -> anyhow::Result<Self> {
        let board = Board::from_str(fen).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let parts: Vec<_> = fen.split_whitespace().collect();
        let halfmove_clock = parts
            .get(4)
            .map_or(Ok(0), |s| s.parse::<u16>())
            .unwrap_or(0);
        let fullmove = parts
            .get(5)
            .map_or(Ok(1), |s| s.parse::<u16>())
            .unwrap_or(1)
            .max(1);
        let ply = (fullmove - 1) * 2
            + if board.side_to_move() == Color::Black {
                1
            } else {
                0
            };
        let status = match board.status() {
            BoardStatus::Ongoing => {
                if halfmove_clock >= 100 {
                    Status::DrawFiftyMoveRule
                } else {
                    Status::Ongoing
                }
            }
            BoardStatus::Checkmate => Status::Checkmate,
            BoardStatus::Stalemate => Status::Stalemate,
        };
        let mut game = ChessGame {
            pos: ChessPosition {
                board,
                ply,
                status,
                halfmove_clock,
            },
            position_counts: HashMap::with_capacity(16),
        };
        game.record_position();
        Ok(game)
    }

    fn record_position(&mut self) {
        let count = self.position_counts.entry(self.pos.hash()).or_insert(0);
        *count += 1;
        if *count >= 3 {
            self.pos.status = Status::DrawRepetition;
        }
    }

    fn after_irreversible_move(&mut self) {
        self.pos.halfmove_clock = 0;
        self.position_counts.clear();
        self.record_position();
    }

    fn after_reversible_move(&mut self) {
        self.pos.halfmove_clock += 1;
        self.record_position();
        if self.pos.halfmove_clock >= 100 {
            self.pos.status = Status::DrawFiftyMoveRule;
        }
    }
}

impl Default for ChessGame {
    fn default() -> Self {
        let mut game = ChessGame {
            pos: ChessPosition::default(),
            position_counts: HashMap::with_capacity(16),
        };
        game.record_position();
        game
    }
}

impl ChessPosition {
    pub fn hash(&self) -> u64 {
        self.board.get_hash()
    }

    pub fn halfmove_clock(&self) -> usize {
        self.halfmove_clock as usize
    }

    pub(crate) fn set_repetition_draw(&mut self) {
        self.status = Status::DrawRepetition;
    }

    pub fn san_for_action(&self, action: Action) -> String {
        let mv = decode_move(action);
        if !self.board.legal(mv) {
            return mv.to_string();
        }
        let Some(piece) = self.board.piece_on(mv.get_source()) else {
            return mv.to_string();
        };

        if piece == Piece::King {
            let from_file = mv.get_source().get_file().to_index();
            let to_file = mv.get_dest().get_file().to_index();
            if from_file == 4 && to_file == 6 {
                return self.san_suffix("O-O", mv);
            }
            if from_file == 4 && to_file == 2 {
                return self.san_suffix("O-O-O", mv);
            }
        }

        let capture = self.board.piece_on(mv.get_dest()).is_some()
            || (piece == Piece::Pawn && self.board.en_passant() == Some(mv.get_dest()));
        let mut san = String::new();
        if piece == Piece::Pawn {
            if capture {
                san.push(file_char(mv.get_source()));
            }
        } else {
            san.push(piece_char(piece));
            san.push_str(&self.disambiguation(piece, mv));
        }
        if capture {
            san.push('x');
        }
        san.push_str(&square_name(mv.get_dest()));
        if let Some(promo) = mv.get_promotion() {
            san.push('=');
            san.push(piece_char(promo));
        }
        self.san_suffix(&san, mv)
    }

    fn disambiguation(&self, piece: Piece, mv: ChessMove) -> String {
        let same_target: Vec<_> = MoveGen::new_legal(&self.board)
            .filter(|&other| {
                other != mv
                    && other.get_dest() == mv.get_dest()
                    && self.board.piece_on(other.get_source()) == Some(piece)
            })
            .collect();
        if same_target.is_empty() {
            return String::new();
        }
        let source = mv.get_source();
        let same_file = same_target
            .iter()
            .any(|other| other.get_source().get_file() == source.get_file());
        let same_rank = same_target
            .iter()
            .any(|other| other.get_source().get_rank() == source.get_rank());
        if !same_file {
            file_char(source).to_string()
        } else if !same_rank {
            rank_char(source).to_string()
        } else {
            format!("{}{}", file_char(source), rank_char(source))
        }
    }

    fn san_suffix(&self, base: &str, mv: ChessMove) -> String {
        let mut san = base.to_owned();
        match self.board.make_move_new(mv).status() {
            BoardStatus::Checkmate => san.push('#'),
            BoardStatus::Ongoing => {
                let moved = self.board.make_move_new(mv);
                if moved.checkers().popcnt() > 0 {
                    san.push('+');
                }
            }
            BoardStatus::Stalemate => {}
        }
        san
    }

    fn step_without_repetition(&mut self, action: Action) -> bool {
        let mv = decode_move(action);
        debug_assert!(self.board.legal(mv), "illegal move {mv} in {}", self.board);

        let white_cr = self.board.castle_rights(Color::White);
        let black_cr = self.board.castle_rights(Color::Black);
        let is_pawn = self.board.piece_on(mv.get_source()) == Some(Piece::Pawn);
        let is_capture = self.board.piece_on(mv.get_dest()).is_some();

        self.board = self.board.make_move_new(mv);
        self.ply += 1;

        match self.board.status() {
            BoardStatus::Checkmate => {
                self.status = Status::Checkmate;
                return true;
            }
            BoardStatus::Stalemate => {
                self.status = Status::Stalemate;
                return true;
            }
            BoardStatus::Ongoing => {}
        }

        let irreversible = is_pawn
            || is_capture
            || self.board.castle_rights(Color::White) != white_cr
            || self.board.castle_rights(Color::Black) != black_cr;

        if irreversible {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
            if self.halfmove_clock >= 100 {
                self.status = Status::DrawFiftyMoveRule;
            }
        }
        irreversible
    }
}

fn piece_char(piece: Piece) -> char {
    match piece {
        Piece::Knight => 'N',
        Piece::Bishop => 'B',
        Piece::Rook => 'R',
        Piece::Queen => 'Q',
        Piece::King => 'K',
        Piece::Pawn => unreachable!("pawns have no SAN piece letter"),
    }
}

fn file_char(square: Square) -> char {
    (b'a' + square.get_file().to_index() as u8) as char
}

fn rank_char(square: Square) -> char {
    (b'1' + square.get_rank().to_index() as u8) as char
}

fn square_name(square: Square) -> String {
    format!("{}{}", file_char(square), rank_char(square))
}

impl Default for ChessPosition {
    fn default() -> Self {
        ChessPosition {
            board: Board::default(),
            ply: 0,
            status: Status::Ongoing,
            halfmove_clock: 0,
        }
    }
}

impl Game for ChessPosition {
    const ACTION_SIZE: usize = 64 * 64 * 5;
    const STATE_SHAPE: [TensorDim; 3] = [19, 8, 8];
    const NAME: &'static str = "chess";

    fn legal_actions(&self) -> impl Iterator<Item = Action> + '_ {
        MoveGen::new_legal(&self.board).map(encode_move)
    }

    fn step(&mut self, action: Action) {
        self.step_without_repetition(action);
    }

    fn is_terminal(&self) -> bool {
        self.status != Status::Ongoing
    }

    fn reward(&self) -> f32 {
        // Checkmate: the player to move has been mated.
        if self.status == Status::Checkmate {
            -1.0
        } else {
            0.0
        }
    }

    /// 19 planes of 8x8, from the side to move's perspective (vertical flip
    /// for black, columns unflipped):
    /// 0-5 own P N B R Q K, 6-11 opponent's, 12 all-ones iff white to move,
    /// 13 ply count (raw), 14-17 castling rights (own K/Q, opp K/Q),
    /// 18 en-passant file (set only when the capture is actually legal).
    fn encode_state(&self, out: &mut [f32]) {
        out.fill(0.0);
        let me = self.board.side_to_move();
        let white_to_move = me == Color::White;

        let rights = |color: Color| {
            let r = self.board.castle_rights(color);
            (r.has_kingside(), r.has_queenside())
        };
        let (own_k, own_q) = rights(me);
        let (opp_k, opp_q) = rights(!me);
        let ep_file = self.board.en_passant().map(|sq| sq.get_file().to_index());

        for i in 0..8usize {
            for j in 0..8usize {
                // Output row i -> board rank: no flip for white, vertical flip for black.
                let rank = if white_to_move { 7 - i } else { i };
                let sq = Square::make_square(Rank::from_index(rank), File::from_index(j));
                if let Some(piece) = self.board.piece_on(sq) {
                    let own = self.board.color_on(sq) == Some(me);
                    let plane = if own { 0 } else { 6 } + piece.to_index();
                    out[plane * 64 + i * 8 + j] = 1.0;
                }
                let cell = i * 8 + j;
                out[12 * 64 + cell] = if white_to_move { 1.0 } else { 0.0 };
                out[13 * 64 + cell] = f32::from(self.ply);
                out[14 * 64 + cell] = f32::from(own_k);
                out[15 * 64 + cell] = f32::from(own_q);
                out[16 * 64 + cell] = f32::from(opp_k);
                out[17 * 64 + cell] = f32::from(opp_q);
                if ep_file == Some(j) {
                    out[18 * 64 + cell] = 1.0;
                }
            }
        }
    }

    fn parse_move(&self, s: &str) -> Option<Action> {
        let mv = ChessMove::from_str(s.trim()).ok()?;
        self.board.legal(mv).then(|| encode_move(mv))
    }

    fn format_action(&self, action: Action) -> String {
        decode_move(action).to_string()
    }
}

impl RepetitionGame for ChessPosition {
    fn repetition_hash(&self) -> u64 {
        self.hash()
    }

    fn halfmove_clock(&self) -> usize {
        ChessPosition::halfmove_clock(self)
    }

    fn set_repetition_draw(&mut self) {
        ChessPosition::set_repetition_draw(self);
    }
}

impl Game for ChessGame {
    const ACTION_SIZE: usize = ChessPosition::ACTION_SIZE;
    const STATE_SHAPE: [TensorDim; 3] = ChessPosition::STATE_SHAPE;
    const NAME: &'static str = ChessPosition::NAME;

    fn legal_actions(&self) -> impl Iterator<Item = Action> + '_ {
        self.pos.legal_actions()
    }

    fn step(&mut self, action: Action) {
        let irreversible = self.pos.step_without_repetition(action);
        if self.pos.is_terminal() {
            return;
        }
        if irreversible {
            self.after_irreversible_move();
        } else {
            self.after_reversible_move();
        }
    }

    fn is_terminal(&self) -> bool {
        self.pos.is_terminal()
    }

    fn reward(&self) -> f32 {
        self.pos.reward()
    }

    fn encode_state(&self, out: &mut [f32]) {
        self.pos.encode_state(out);
    }

    fn parse_move(&self, s: &str) -> Option<Action> {
        self.pos.parse_move(s)
    }

    fn format_action(&self, action: Action) -> String {
        self.pos.format_action(action)
    }
}

impl fmt::Display for ChessPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            write!(f, "{} ", rank + 1)?;
            for file in 0..8 {
                let sq = Square::make_square(Rank::from_index(rank), File::from_index(file));
                let cell = match (self.board.piece_on(sq), self.board.color_on(sq)) {
                    (Some(p), Some(Color::White)) => b"PNBRQK"[p.to_index()] as char,
                    (Some(p), Some(Color::Black)) => b"pnbrqk"[p.to_index()] as char,
                    _ => '.',
                };
                write!(f, "{cell} ")?;
            }
            writeln!(f)?;
        }
        writeln!(f, "  a b c d e f g h")?;
        writeln!(
            f,
            "{} to move",
            if self.board.side_to_move() == Color::White {
                "White"
            } else {
                "Black"
            }
        )
    }
}

impl fmt::Display for ChessGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.pos.fmt(f)
    }
}

#[cfg(test)]
mod tests;

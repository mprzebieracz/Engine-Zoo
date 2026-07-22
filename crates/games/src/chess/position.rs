use chess::{Board, BoardStatus, ChessMove, Color, File, Piece, Rank, Square};
use engine_core::game::{GameState, TerminalValue};
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Status {
    Ongoing,
    Checkmate,
    Stalemate,
    DrawRepetition,
    DrawFiftyMoveRule,
    DrawInsufficientMaterial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MoveEffect {
    resets_halfmove_clock: bool,
    clears_repetition_history: bool,
}

impl MoveEffect {
    pub(super) fn clears_repetition_history(self) -> bool {
        self.clears_repetition_history
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ChessPosition {
    pub(super) board: Board,
    pub(super) ply: u16,
    pub(super) status: Status,
    pub(super) halfmove_clock: u16,
}

impl ChessPosition {
    pub fn initial() -> Self {
        Self::default()
    }

    pub fn legal_moves(&self) -> impl Iterator<Item = ChessMove> + '_ {
        (self.status == Status::Ongoing)
            .then(|| chess::MoveGen::new_legal(&self.board))
            .into_iter()
            .flatten()
    }

    pub fn play(&mut self, mv: ChessMove) {
        self.play_with_effect(mv);
    }

    pub fn terminal_value(&self) -> Option<TerminalValue> {
        match self.status {
            Status::Ongoing => None,
            Status::Checkmate => Some(TerminalValue::Loss),
            Status::Stalemate
            | Status::DrawRepetition
            | Status::DrawFiftyMoveRule
            | Status::DrawInsufficientMaterial => Some(TerminalValue::Draw),
        }
    }

    pub fn hash(&self) -> u64 {
        self.board.get_hash()
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn ply(&self) -> u16 {
        self.ply
    }

    pub fn halfmove_clock(&self) -> usize {
        self.halfmove_clock as usize
    }

    pub(super) fn play_with_effect(&mut self, mv: ChessMove) -> MoveEffect {
        debug_assert!(self.board.legal(mv), "illegal move {mv} in {}", self.board);

        let source = mv.get_source();
        let dest = mv.get_dest();

        let moved_piece = self.board.piece_on(source);

        let is_pawn = moved_piece == Some(Piece::Pawn);
        let is_capture = self.board.piece_on(dest).is_some();

        let castle_rights_before = (
            self.board.castle_rights(Color::White),
            self.board.castle_rights(Color::Black),
        );

        self.board = self.board.make_move_new(mv);
        self.ply += 1;

        let castle_rights_after = (
            self.board.castle_rights(Color::White),
            self.board.castle_rights(Color::Black),
        );

        let castle_rights_changed = castle_rights_before != castle_rights_after;

        let effect = MoveEffect {
            resets_halfmove_clock: is_pawn || is_capture,
            clears_repetition_history: is_pawn || is_capture || castle_rights_changed,
        };

        if effect.resets_halfmove_clock {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        self.refresh_status_without_repetition();

        effect
    }

    /// Resolves automatic terminals other than authoritative repetition.
    ///
    /// This project deliberately adjudicates claimable fifty-move draws in
    /// self-play. Checkmate and stalemate still take precedence over that
    /// adjudication, as they do over every other draw status here.
    pub(super) fn refresh_status_without_repetition(&mut self) {
        self.status = match self.board.status() {
            BoardStatus::Checkmate => Status::Checkmate,
            BoardStatus::Stalemate => Status::Stalemate,
            BoardStatus::Ongoing if self.has_insufficient_material() => {
                Status::DrawInsufficientMaterial
            }
            BoardStatus::Ongoing if self.halfmove_clock >= 100 => Status::DrawFiftyMoveRule,
            BoardStatus::Ongoing => Status::Ongoing,
        };
    }

    /// The canonical automatic subset of FIDE dead positions we support:
    /// bare kings, a single bishop or knight, and one bishop per side on the
    /// same colour complex. This intentionally does not attempt the general
    /// dead-position problem.
    fn has_insufficient_material(&self) -> bool {
        if [Piece::Pawn, Piece::Rook, Piece::Queen]
            .into_iter()
            .any(|piece| self.board.pieces(piece).popcnt() != 0)
        {
            return false;
        }

        let bishops = *self.board.pieces(Piece::Bishop);
        let knights = self.board.pieces(Piece::Knight).popcnt();
        let minor_count = bishops.popcnt() + knights;

        if minor_count <= 1 {
            return true;
        }

        if knights != 0
            || bishops.popcnt() != 2
            || (bishops & self.board.color_combined(Color::White)).popcnt() != 1
            || (bishops & self.board.color_combined(Color::Black)).popcnt() != 1
        {
            return false;
        }

        let first = bishops.into_iter().next().expect("two bishops exist");
        let colour = (first.get_file().to_index() + first.get_rank().to_index()) % 2;
        bishops.into_iter().all(|square| {
            (square.get_file().to_index() + square.get_rank().to_index()) % 2 == colour
        })
    }
}

impl super::ChessRepetitionState for ChessPosition {
    fn repetition_hash(&self) -> u64 {
        self.hash()
    }

    fn reversible_plies(&self) -> usize {
        self.halfmove_clock()
    }
}

impl GameState for ChessPosition {
    type Move = ChessMove;

    fn initial() -> Self {
        Self::default()
    }

    fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
        self.legal_moves()
    }

    fn play(&mut self, mv: Self::Move) {
        self.play_with_effect(mv);
    }

    fn terminal_value(&self) -> Option<TerminalValue> {
        self.terminal_value()
    }
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

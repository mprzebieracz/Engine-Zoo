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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MoveEffect {
    irreversible: bool,
}

impl MoveEffect {
    pub(super) fn is_irreversible(self) -> bool {
        self.irreversible
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
        chess::MoveGen::new_legal(&self.board)
    }

    pub fn play(&mut self, mv: ChessMove) {
        self.play_with_effect(mv);
    }

    pub fn terminal_value(&self) -> Option<TerminalValue> {
        match self.status {
            Status::Ongoing => None,
            Status::Checkmate => Some(TerminalValue::Loss),
            Status::Stalemate | Status::DrawRepetition | Status::DrawFiftyMoveRule => {
                Some(TerminalValue::Draw)
            }
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

    pub(crate) fn set_repetition_draw(&mut self) {
        self.status = Status::DrawRepetition;
    }

    pub(super) fn play_with_effect(&mut self, mv: ChessMove) -> MoveEffect {
        debug_assert!(self.board.legal(mv), "illegal move {mv} in {}", self.board);

        let source = mv.get_source();
        let dest = mv.get_dest();
        let moved_piece = self.board.piece_on(source);
        let is_pawn = moved_piece == Some(Piece::Pawn);
        let is_capture = self.board.piece_on(dest).is_some();
        let is_en_passant_capture = is_pawn && !is_capture && self.board.en_passant() == Some(dest);
        let castle_rights_before =
            matches!(moved_piece, Some(Piece::King | Piece::Rook)).then(|| {
                (
                    self.board.castle_rights(Color::White),
                    self.board.castle_rights(Color::Black),
                )
            });

        self.board = self.board.make_move_new(mv);
        self.ply += 1;

        let castle_rights_changed = castle_rights_before.is_some_and(|(white_cr, black_cr)| {
            self.board.castle_rights(Color::White) != white_cr
                || self.board.castle_rights(Color::Black) != black_cr
        });
        let irreversible = is_pawn || is_capture || is_en_passant_capture || castle_rights_changed;

        match self.board.status() {
            BoardStatus::Checkmate => {
                self.status = Status::Checkmate;
                return MoveEffect { irreversible };
            }
            BoardStatus::Stalemate => {
                self.status = Status::Stalemate;
                return MoveEffect { irreversible };
            }
            BoardStatus::Ongoing => {}
        }

        if irreversible {
            self.halfmove_clock = 0;
        }
        else {
            self.halfmove_clock += 1;
            if self.halfmove_clock >= 100 {
                self.status = Status::DrawFiftyMoveRule;
            }
        }
        MoveEffect { irreversible }
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

impl ChessPosition {
    pub(crate) fn is_terminal(&self) -> bool {
        self.status != Status::Ongoing
    }

    pub(crate) fn reward(&self) -> f32 {
        // Checkmate: the player to move has been mated.
        if self.status == Status::Checkmate {
            -1.0
        }
        else {
            0.0
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
            }
            else {
                "Black"
            }
        )
    }
}

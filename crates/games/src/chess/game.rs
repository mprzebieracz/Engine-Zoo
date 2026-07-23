use super::notation;
use super::position::{ChessPosition, Status};
use super::repetition::RepetitionTracker;
use crate::setup::ChessSetup;
use chess::{Board, ChessMove, Color};
use engine_core::game::{GameState, TerminalValue};
use engine_core::notation::GameNotation;
use std::fmt;
use std::str::FromStr;

/// The deliberate terminal policy used for chess self-play.
///
/// Threefold repetition and the fifty-move rule are claimable in FIDE play,
/// but self-play has no player claim protocol, so the engine adjudicates them
/// immediately once their thresholds are reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChessAdjudication {
    SelfPlayClaimableDraws,
}

#[derive(Clone)]
pub struct ChessGame {
    pub(super) pos: ChessPosition,
    /// Zobrist hash occurrence counts since the last irreversible move.
    pub(super) repetitions: RepetitionTracker,
    /// Recent authoritative positions. Consumers that need a bounded history
    /// snapshot (such as a neural representation) copy it at their boundary.
    pub(super) history: [Option<(ChessPosition, u8)>; 8],
}

impl ChessGame {
    pub const ADJUDICATION: ChessAdjudication = ChessAdjudication::SelfPlayClaimableDraws;

    pub fn from_setup(setup: &ChessSetup) -> anyhow::Result<Self> {
        let mut game = match &setup.fen {
            Some(fen) => Self::from_fen(fen)?,
            None => Self::default(),
        };
        for mv in &setup.moves {
            let mv = notation::ChessUciNotation
                .parse_move(&game.pos, mv)
                .ok_or_else(|| anyhow::anyhow!("illegal chess move {mv}"))?;
            game.play(mv);
        }
        Ok(game)
    }

    pub fn board(&self) -> &Board {
        &self.pos.board
    }

    pub fn position(&self) -> ChessPosition {
        self.pos
    }

    /// Recent positions, newest first, with their preceding repetition count.
    pub fn recent_positions(&self) -> impl Iterator<Item = (ChessPosition, u8)> + '_ {
        self.history.iter().copied().flatten()
    }

    pub fn repetition_context(&self) -> ChessRepetitionContext<'_> {
        ChessRepetitionContext {
            tracker: &self.repetitions,
            root_hash: self.pos.hash(),
        }
    }

    pub fn repetitions_before_current(&self, hash: u64) -> u8 {
        self.repetitions_before_root(hash, self.pos.hash())
    }

    /// Number of occurrences before an explicitly cached search root.
    pub fn repetitions_before_root(&self, hash: u64, root_hash: u64) -> u8 {
        self.repetitions.root_count(hash, root_hash)
    }

    pub fn from_fen(fen: &str) -> anyhow::Result<Self> {
        let parts: Vec<_> = fen.split_whitespace().collect();
        let board_fen = chess_crate_fen(&parts);
        let board = Board::from_str(&board_fen).map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let halfmove_clock = parts
            .get(4)
            .ok_or_else(|| anyhow::anyhow!("FEN is missing the halfmove clock"))?
            .parse::<u16>()
            .map_err(|e| anyhow::anyhow!("invalid FEN halfmove clock: {e}"))?;

        let fullmove = parts
            .get(5)
            .ok_or_else(|| anyhow::anyhow!("FEN is missing the fullmove number"))?
            .parse::<u16>()
            .map_err(|e| anyhow::anyhow!("invalid FEN fullmove number: {e}"))?;
        anyhow::ensure!(fullmove > 0, "FEN fullmove number must be positive");

        let ply = fullmove
            .checked_sub(1)
            .and_then(|moves| moves.checked_mul(2))
            .and_then(|ply| ply.checked_add(u16::from(board.side_to_move() == Color::Black)))
            .ok_or_else(|| anyhow::anyhow!("FEN fullmove number exceeds supported game length"))?;

        let mut game = ChessGame {
            pos: ChessPosition {
                board,
                ply,
                status: Status::Ongoing,
                halfmove_clock,
            },
            repetitions: RepetitionTracker::new(board.get_hash()),
            history: [None; 8],
        };

        game.pos.refresh_status_without_repetition();

        game.history[0] = Some((game.pos, 0));

        Ok(game)
    }

    pub(super) fn record_position(&mut self) {
        let count = self.repetitions.record(self.pos.hash());
        if count >= 3 && self.pos.status == Status::Ongoing {
            self.pos.status = Status::DrawRepetition;
        }
    }

    fn after_irreversible_move(&mut self) {
        self.repetitions.reset(self.pos.hash());
    }

    fn after_reversible_move(&mut self) {
        self.record_position();
    }

    pub fn play(&mut self, mv: ChessMove) {
        assert!(self.pos.board.legal(mv), "illegal chess move {mv}");
        let effect = self.pos.play_with_effect(mv);
        if effect.clears_repetition_history() {
            self.after_irreversible_move();
        }
        else {
            self.after_reversible_move();
        }
        self.history.rotate_right(1);
        self.history[0] = Some((
            self.pos,
            self.repetitions
                .current_count(self.pos.hash())
                .saturating_sub(1),
        ));
    }
}

/// The `chess` crate stores en-passant as the double-pushed pawn's square,
/// while standard FEN records the square crossed by that pawn. Convert only
/// at this dependency boundary; public callers always use standard FEN.
fn chess_crate_fen(parts: &[&str]) -> String {
    let mut fields: Vec<_> = parts.iter().map(|field| (*field).to_owned()).collect();
    let Some([file, rank]) = fields.get(3).and_then(|square| square.as_bytes().get(..2))
    else {
        return fields.join(" ");
    };
    let pawn_rank = match (fields.get(1).map(String::as_str), rank) {
        (Some("w"), b'6') => b'5',
        (Some("b"), b'3') => b'4',
        _ => *rank,
    };
    if pawn_rank != *rank {
        fields[3] = format!("{}{}", *file as char, pawn_rank as char);
    }
    fields.join(" ")
}

#[derive(Clone, Copy)]
pub struct ChessRepetitionContext<'a> {
    tracker: &'a RepetitionTracker,
    root_hash: u64,
}

impl ChessRepetitionContext<'_> {
    pub fn occurrences_before_root(&self, hash: u64) -> u8 {
        self.tracker.root_count(hash, self.root_hash)
    }
}

impl Default for ChessGame {
    fn default() -> Self {
        let mut game = ChessGame {
            pos: ChessPosition::default(),
            repetitions: RepetitionTracker::new(ChessPosition::default().hash()),
            history: [None; 8],
        };
        game.history[0] = Some((game.pos, 0));
        game
    }
}

impl GameState for ChessGame {
    type Move = ChessMove;

    fn initial() -> Self {
        Self::default()
    }

    fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
        self.pos.legal_moves()
    }

    fn play(&mut self, mv: Self::Move) {
        ChessGame::play(self, mv);
    }

    fn terminal_value(&self) -> Option<TerminalValue> {
        self.pos.terminal_value()
    }
}

impl fmt::Display for ChessGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.pos.fmt(f)
    }
}

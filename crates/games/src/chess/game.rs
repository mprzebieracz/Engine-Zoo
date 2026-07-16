use super::action::{decode_v1_action, encode_v1_action};
use super::az::{ChessHistoryState, HistoryFrame};
use super::legacy::{self, ChessLegacyState};
use super::notation;
use super::position::{ChessPosition, Status};
use super::repetition::RepetitionTracker;
use crate::setup::ChessSetup;
use chess::{Board, BoardStatus, ChessMove, Color, MoveGen};
use engine_core::game::{Action, Game, TensorDim};
use engine_core::notation::GameNotation;
use std::fmt;
use std::str::FromStr;

#[derive(Clone)]
pub struct ChessGame {
    pub(super) pos: ChessPosition,
    /// Zobrist hash occurrence counts since the last irreversible move.
    pub(super) repetitions: RepetitionTracker,
    pub(super) history: [Option<HistoryFrame>; 8],
}

impl ChessGame {
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

    pub fn legacy_state(&self) -> ChessLegacyState {
        ChessLegacyState(self.pos)
    }

    /// Returns the latest real game frames for neural evaluation.
    pub fn history_state<const HISTORY: usize>(&self) -> ChessHistoryState<HISTORY> {
        let mut state = ChessHistoryState::new(self.pos);
        state.frames.copy_from_slice(&self.history[..HISTORY]);
        state
    }

    pub fn position_state(&self) -> ChessPosition {
        self.pos
    }
    pub fn repetition_context(&self) -> ChessRepetitionContext<'_> {
        ChessRepetitionContext {
            tracker: &self.repetitions,
            root_hash: self.pos.hash(),
        }
    }

    pub fn san_for_action(&self, action: Action) -> String {
        notation::san(self.board(), decode_v1_action(action))
    }

    pub fn repetitions_before_current(&self, hash: u64) -> u8 {
        self.repetitions_before_root(hash, self.pos.hash())
    }

    /// Number of occurrences before an explicitly cached search root.
    pub fn repetitions_before_root(&self, hash: u64, root_hash: u64) -> u8 {
        self.repetitions.root_count(hash, root_hash)
    }

    pub fn from_fen(fen: &str) -> anyhow::Result<Self> {
        let board = Board::from_str(fen).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let parts: Vec<_> = fen.split_whitespace().collect();
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
        let status = match board.status() {
            BoardStatus::Ongoing => {
                if halfmove_clock >= 100 {
                    Status::DrawFiftyMoveRule
                }
                else {
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
            repetitions: RepetitionTracker::new(board.get_hash()),
            history: [None; 8],
        };
        game.history[0] = Some(HistoryFrame {
            position: game.pos,
            repetitions_before: 0,
        });
        Ok(game)
    }

    pub(super) fn record_position(&mut self) {
        let count = self.repetitions.record(self.pos.hash());
        if count >= 3 {
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
        if effect.is_irreversible() {
            self.after_irreversible_move();
        }
        else {
            self.after_reversible_move();
        }
        self.history.rotate_right(1);
        self.history[0] = Some(HistoryFrame {
            position: self.pos,
            repetitions_before: self
                .repetitions
                .current_count(self.pos.hash())
                .saturating_sub(1),
        });
    }
}

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
        game.history[0] = Some(HistoryFrame {
            position: game.pos,
            repetitions_before: 0,
        });
        game
    }
}

impl Game for ChessGame {
    const ACTION_SIZE: usize = 64 * 64 * 5;
    const STATE_SHAPE: [TensorDim; 3] = [19, 8, 8];
    const NAME: &'static str = "chess";

    fn legal_actions(&self) -> impl Iterator<Item = Action> + '_ {
        MoveGen::new_legal(&self.pos.board).map(encode_v1_action)
    }

    fn step(&mut self, action: Action) {
        self.play(decode_v1_action(action));
    }

    fn is_terminal(&self) -> bool {
        self.pos.is_terminal()
    }

    fn reward(&self) -> f32 {
        self.pos.reward()
    }

    fn encode_state(&self, out: &mut [f32]) {
        legacy::encode(&self.pos, out);
    }

    fn parse_move(&self, s: &str) -> Option<Action> {
        notation::ChessUciNotation
            .parse_move(&self.pos, s)
            .map(encode_v1_action)
    }

    fn format_action(&self, action: Action) -> String {
        notation::ChessUciNotation.format_move(&self.pos, decode_v1_action(action))
    }
}

impl fmt::Display for ChessGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.pos.fmt(f)
    }
}

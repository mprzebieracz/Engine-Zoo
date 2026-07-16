use super::action::{decode_v1_action, encode_v1_action};
use super::az::ChessAzState;
use super::legacy::{self, ChessLegacyState};
use super::notation;
use super::position::{ChessPosition, Status};
use super::zobrist::ZobristBuildHasher;
use crate::setup::ChessSetup;
use chess::{Board, BoardStatus, Color, MoveGen};
use engine_core::game::{Action, Game, TensorDim};
use engine_core::notation::GameNotation;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

#[derive(Clone)]
pub struct ChessGame {
    pub(super) pos: ChessPosition,
    /// Zobrist hash occurrence counts since the last irreversible move.
    pub(super) position_counts: HashMap<u64, u8, ZobristBuildHasher>,
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
            game.play_native(mv);
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

    /// Starts a v2 AlphaZero state at this game's current position.
    ///
    /// `ChessGame` deliberately keeps repetition counts rather than an
    /// unbounded position list, so a state constructed mid-game has no prior
    /// feature frames. Self-play should carry `ChessAzState` forward instead.
    pub fn az_state<const HISTORY: usize>(&self) -> ChessAzState<HISTORY> {
        let mut state = ChessAzState::new(self.pos);
        state.set_repetitions_before_current(self.repetitions_before_current(self.pos.hash()));
        state
    }

    pub fn san_for_action(&self, action: Action) -> String {
        notation::san(self.board(), decode_v1_action(action))
    }

    pub fn repetitions_before_current(&self, hash: u64) -> u8 {
        self.repetitions_before_root(hash, self.pos.hash())
    }

    /// Number of occurrences before an explicitly cached search root.
    pub fn repetitions_before_root(&self, hash: u64, root_hash: u64) -> u8 {
        let count = self.position_counts.get(&hash).copied().unwrap_or(0);
        if hash == root_hash {
            count.saturating_sub(1)
        }
        else {
            count
        }
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
            position_counts: HashMap::with_capacity_and_hasher(16, ZobristBuildHasher::default()),
        };
        game.record_position();
        Ok(game)
    }

    pub(super) fn record_position(&mut self) {
        let count = self.position_counts.entry(self.pos.hash()).or_insert(0);
        *count += 1;
        if *count >= 3 {
            self.pos.status = Status::DrawRepetition;
        }
    }

    fn after_irreversible_move(&mut self) {
        self.position_counts.clear();
        self.record_position();
    }

    fn after_reversible_move(&mut self) {
        self.record_position();
    }

    fn play_native(&mut self, mv: chess::ChessMove) {
        let effect = self.pos.play_with_effect(mv);
        if self.pos.is_terminal() {
            return;
        }
        if effect.is_irreversible() {
            self.after_irreversible_move();
        }
        else {
            self.after_reversible_move();
        }
    }
}

impl Default for ChessGame {
    fn default() -> Self {
        let mut game = ChessGame {
            pos: ChessPosition::default(),
            position_counts: HashMap::with_capacity_and_hasher(16, ZobristBuildHasher::default()),
        };
        game.record_position();
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
        self.play_native(decode_v1_action(action));
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

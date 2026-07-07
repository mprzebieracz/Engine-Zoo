use crate::game::{Action, Game};
use crate::games::{ChessGame, Connect4};
use anyhow::Result;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChessPosition {
    /// If omitted, the standard start position is used.
    pub fen: Option<String>,
    /// Legal moves, in UCI form, applied after `fen` or the start position.
    #[serde(default)]
    pub moves: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Connect4Position {
    /// Columns played from the empty board.
    #[serde(default)]
    pub moves: Vec<Action>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "game", content = "position", rename_all = "lowercase")]
pub enum PositionSpec {
    Chess(ChessPosition),
    Connect4(Connect4Position),
}

pub trait PositionGame: Game {
    type Position: Clone + Serialize + DeserializeOwned + Send + Sync + 'static;

    fn from_position(position: &Self::Position) -> Result<Self>;
}

impl PositionGame for ChessGame {
    type Position = ChessPosition;

    fn from_position(position: &Self::Position) -> Result<Self> {
        let mut game = match &position.fen {
            Some(fen) => ChessGame::from_fen(fen)?,
            None => ChessGame::default(),
        };
        for mv in &position.moves {
            let action = game
                .parse_move(mv)
                .ok_or_else(|| anyhow::anyhow!("illegal chess move {mv}"))?;
            game.step(action);
        }
        Ok(game)
    }
}

impl PositionGame for Connect4 {
    type Position = Connect4Position;

    fn from_position(position: &Self::Position) -> Result<Self> {
        Connect4::from_moves(&position.moves)
    }
}

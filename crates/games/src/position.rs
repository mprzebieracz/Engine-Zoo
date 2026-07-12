use crate::{ChessGame, Connect4};
use anyhow::Result;
use engine_core::game::{Action, Game};
use engine_core::rules::PositionCodec;
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

impl PositionCodec for ChessGame {
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

impl PositionCodec for Connect4 {
    type Position = Connect4Position;

    fn from_position(position: &Self::Position) -> Result<Self> {
        Connect4::from_moves(&position.moves)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_chess_and_connect4_positions() {
        let chess = ChessGame::from_position(&ChessPosition {
            fen: None,
            moves: vec!["e2e4".into(), "e7e5".into()],
        })
        .unwrap();
        assert!(!chess.is_terminal());

        let connect4 = Connect4::from_position(&Connect4Position {
            moves: vec![3, 3, 2],
        })
        .unwrap();
        assert!(!connect4.is_terminal());
    }

    #[test]
    fn chess_position_applies_moves_after_fen_and_preserves_clocks() {
        let game = ChessGame::from_position(&ChessPosition {
            fen: Some("8/8/8/8/8/8/P7/K6k w - - 17 23".into()),
            moves: vec!["a2a3".into(), "h1g1".into()],
        })
        .unwrap();

        assert_eq!(game.board().to_string(), "8/8/8/8/8/P7/8/K5k1 w - - 0 1");
        assert_eq!(game.position().halfmove_clock(), 1);
    }

    #[test]
    fn position_loading_reports_bad_setups_and_illegal_replays() {
        assert!(ChessGame::from_position(&ChessPosition {
            fen: Some("not a fen".into()),
            moves: vec![],
        })
        .is_err());
        assert!(ChessGame::from_position(&ChessPosition {
            fen: None,
            moves: vec!["e2e5".into()],
        })
        .is_err());
        assert!(Connect4::from_position(&Connect4Position {
            moves: vec![0, 0, 0, 0, 0, 0, 0],
        })
        .is_err());
        assert!(Connect4::from_position(&Connect4Position { moves: vec![7] }).is_err());
    }
}

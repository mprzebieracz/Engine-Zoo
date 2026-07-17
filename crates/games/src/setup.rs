use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChessSetup {
    /// If omitted, the standard start position is used.
    pub fen: Option<String>,
    /// Legal moves, in UCI form, applied after `fen` or the start position.
    #[serde(default)]
    pub moves: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Connect4Setup {
    /// Columns played from the empty board.
    #[serde(default)]
    pub moves: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "game", content = "position", rename_all = "lowercase")]
pub enum GameSetup {
    Chess(ChessSetup),
    Connect4(Connect4Setup),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChessGame, Connect4};
    use engine_core::game::GameState;

    #[test]
    fn loads_chess_and_connect4_positions() {
        let chess = ChessGame::from_setup(&ChessSetup {
            fen: None,
            moves: vec!["e2e4".into(), "e7e5".into()],
        })
        .unwrap();
        assert!(!chess.is_terminal());

        let connect4 = Connect4::from_setup(&Connect4Setup {
            moves: vec![3, 3, 2],
        })
        .unwrap();
        assert!(!connect4.is_terminal());
    }

    #[test]
    fn chess_position_applies_moves_after_fen_and_preserves_clocks() {
        let game = ChessGame::from_setup(&ChessSetup {
            fen: Some("8/8/8/8/8/8/P7/K6k w - - 17 23".into()),
            moves: vec!["a2a3".into(), "h1g1".into()],
        })
        .unwrap();

        assert_eq!(game.board().to_string(), "8/8/8/8/8/P7/8/K5k1 w - - 0 1");
        assert_eq!(game.position().halfmove_clock(), 1);
    }

    #[test]
    fn position_loading_reports_bad_setups_and_illegal_replays() {
        assert!(ChessGame::from_setup(&ChessSetup {
            fen: Some("not a fen".into()),
            moves: vec![],
        })
        .is_err());
        assert!(ChessGame::from_setup(&ChessSetup {
            fen: None,
            moves: vec!["e2e5".into()],
        })
        .is_err());
        assert!(Connect4::from_setup(&Connect4Setup {
            moves: vec![0, 0, 0, 0, 0, 0, 0],
        })
        .is_err());
        assert!(Connect4::from_setup(&Connect4Setup { moves: vec![7] }).is_err());
    }
}

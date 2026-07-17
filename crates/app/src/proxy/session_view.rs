use super::*;
use engine_core::game::{Game, GameState, TerminalValue};
use engine_core::notation::GameNotation;

pub(super) fn session_id(session: &LiveSession) -> u64 {
    match session {
        LiveSession::Chess(s) => s.id,
        LiveSession::Connect4(s) => s.id,
    }
}

pub(super) fn session_human_turn(session: &LiveSession) -> bool {
    match session {
        LiveSession::Chess(s) => s.human_turn,
        LiveSession::Connect4(s) => s.human_turn,
    }
}

pub(super) fn session_terminal(session: &LiveSession) -> bool {
    match session {
        LiveSession::Chess(s) => s.game.terminal_value().is_some(),
        LiveSession::Connect4(s) => s.game.terminal_value().is_some(),
    }
}

pub(super) fn session_view(session: &LiveSession) -> serde_json::Value {
    match session {
        LiveSession::Chess(s) => session_view_chess(s),
        LiveSession::Connect4(s) => session_view_connect4(s),
    }
}

pub(super) fn session_view_chess(session: &SessionState<ChessGame>) -> serde_json::Value {
    if let Some(history) = session.v2_history {
        let legal_moves = match history {
            1 => v2_legal_moves::<1>(&session.game),
            4 => v2_legal_moves::<4>(&session.game),
            8 => v2_legal_moves::<8>(&session.game),
            _ => unreachable!("validated v2 history length"),
        };
        return session_view_chess_with_legal(session, legal_moves);
    }
    session_view_chess_for(session)
}

fn v2_legal_moves<const HISTORY: usize>(game: &ChessGame) -> Vec<serde_json::Value> {
    let state = game.history_state::<HISTORY>();
    state
        .legal_actions()
        .map(|action| json!({ "action": action, "move": state.format_action(action) }))
        .collect()
}

fn session_view_chess_with_legal(
    session: &SessionState<ChessGame>,
    legal_moves: Vec<serde_json::Value>,
) -> serde_json::Value {
    json!({
        "id": session.id,
        "board": session.game.to_string(),
        "moves": session.moves,
        "san_moves": session.san_moves,
        "pgn": notation::movetext(&session.san_moves, 0, ""),
        "human_turn": session.human_turn,
        "terminal": session.game.terminal_value().is_some(),
        "reward": session.game.terminal_value().map_or(0.0, TerminalValue::as_f32),
        "legal_moves": legal_moves,
    })
}

pub(super) fn session_view_chess_for(session: &SessionState<ChessGame>) -> serde_json::Value {
    let legal_moves: Vec<_> = session
        .game
        .legal_moves()
        .map(|mv| {
            let action = games::encode_v1_action(mv);
            json!({
                "action": action,
                "move": games::chess::ChessUciNotation.format_move(&session.game.position(), mv),
            })
        })
        .collect();
    json!({
        "id": session.id,
        "board": session.game.to_string(),
        "moves": session.moves,
        "san_moves": session.san_moves,
        "pgn": notation::movetext(&session.san_moves, 0, ""),
        "human_turn": session.human_turn,
        "terminal": session.game.terminal_value().is_some(),
        "reward": session.game.terminal_value().map_or(0.0, TerminalValue::as_f32),
        "legal_moves": legal_moves,
    })
}

pub(super) fn session_view_connect4(session: &SessionState<Connect4>) -> serde_json::Value {
    let legal_moves: Vec<_> = session.game.legal_moves().map(|mv| {
        let action = mv.column() as u32;
        json!({ "action": action, "move": games::connect4::notation::Connect4Notation.format_move(&session.game, mv) })
    }).collect();
    json!({
        "id": session.id, "board": session.game.to_string(), "moves": session.moves,
        "san_moves": session.san_moves, "pgn": notation::movetext(&session.san_moves, 0, ""),
        "human_turn": session.human_turn, "terminal": session.game.terminal_value().is_some(),
        "reward": session.game.terminal_value().map_or(0.0, TerminalValue::as_f32), "legal_moves": legal_moves,
    })
}

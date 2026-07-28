use super::*;
use alphazero::representation::{
    AlphaZeroRepresentation, ChessAzRepresentation, ChessAzState, ChessClassicRepresentation,
};
use engine_core::game::{GameState, TerminalValue};
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
    if let Some(history) = session.chess_history {
        let legal_moves = match history {
            alphazero::ChessHistory::One => chess_legal_moves::<1>(&session.game),
            alphazero::ChessHistory::Four => chess_legal_moves::<4>(&session.game),
            alphazero::ChessHistory::Eight => chess_legal_moves::<8>(&session.game),
        };
        return session_view_chess_with_legal(session, legal_moves);
    }
    session_view_chess_with_legal(session, chess_classic_legal_moves(&session.game))
}

fn chess_classic_legal_moves(game: &ChessGame) -> Vec<serde_json::Value> {
    let state = game.position();
    let representation = ChessClassicRepresentation;
    state
        .legal_moves()
        .map(|mv| {
            json!({
                "action": representation.move_to_action(&state, mv).as_u32(),
                "move": games::chess::ChessUciNotation.format_move(&state, mv),
            })
        })
        .collect()
}

fn chess_legal_moves<const HISTORY: usize>(game: &ChessGame) -> Vec<serde_json::Value> {
    let state = ChessAzState::from_game(game);
    let representation = ChessAzRepresentation::<HISTORY>;
    state
        .legal_moves()
        .map(|mv| {
            json!({
                "action": representation.move_to_action(&state, mv).as_u32(),
                "move": games::chess::ChessUciNotation.format_move(&state.position(), mv),
            })
        })
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

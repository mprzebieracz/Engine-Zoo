use super::*;

pub(super) fn session_id(session: &LiveSession) -> u64 {
    match session {
        LiveSession::Chess(s) => s.id,
        LiveSession::ChessAzV2(s) => s.id,
        LiveSession::Connect4(s) => s.id,
    }
}

pub(super) fn session_human_turn(session: &LiveSession) -> bool {
    match session {
        LiveSession::Chess(s) => s.human_turn,
        LiveSession::ChessAzV2(s) => s.human_turn,
        LiveSession::Connect4(s) => s.human_turn,
    }
}

pub(super) fn session_terminal(session: &LiveSession) -> bool {
    match session {
        LiveSession::Chess(s) => s.game.is_terminal(),
        LiveSession::ChessAzV2(s) => s.game.is_terminal(),
        LiveSession::Connect4(s) => s.game.is_terminal(),
    }
}

pub(super) fn session_view(session: &LiveSession) -> serde_json::Value {
    match session {
        LiveSession::Chess(s) => session_view_for(s),
        LiveSession::ChessAzV2(s) => session_view_chess_az_v2(s),
        LiveSession::Connect4(s) => session_view_for(s),
    }
}

pub(super) fn session_view_chess_az_v2(session: &ChessAzV2Session) -> serde_json::Value {
    let legal_moves: Vec<_> = session
        .game
        .legal_moves()
        .into_iter()
        .map(|(action, move_text)| {
            json!({
                "action": action,
                "move": move_text,
            })
        })
        .collect();
    json!({
        "id": session.id,
        "board": session.game.board_text(),
        "moves": session.moves,
        "san_moves": session.san_moves,
        "pgn": notation::movetext(&session.san_moves, 0, ""),
        "human_turn": session.human_turn,
        "terminal": session.game.is_terminal(),
        "reward": session.game.reward(),
        "legal_moves": legal_moves,
    })
}

pub(super) fn session_view_for<G: Game>(session: &SessionState<G>) -> serde_json::Value {
    let legal_moves: Vec<_> = session
        .game
        .legal_actions()
        .map(|action| {
            json!({
                "action": action,
                "move": session.game.format_action(action),
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
        "terminal": session.game.is_terminal(),
        "reward": session.game.reward(),
        "legal_moves": legal_moves,
    })
}

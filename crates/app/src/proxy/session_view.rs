use super::*;

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
        LiveSession::Chess(s) => s.game.is_terminal(),
        LiveSession::Connect4(s) => s.game.is_terminal(),
    }
}

pub(super) fn session_view(session: &LiveSession) -> serde_json::Value {
    match session {
        LiveSession::Chess(s) => session_view_chess(s),
        LiveSession::Connect4(s) => session_view_for(s),
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
        return session_view_with_legal(session, legal_moves);
    }
    session_view_for(session)
}

fn v2_legal_moves<const HISTORY: usize>(game: &ChessGame) -> Vec<serde_json::Value> {
    let state = game.history_state::<HISTORY>();
    state
        .legal_actions()
        .map(|action| json!({ "action": action, "move": state.format_action(action) }))
        .collect()
}

fn session_view_with_legal<G: Game>(
    session: &SessionState<G>,
    legal_moves: Vec<serde_json::Value>,
) -> serde_json::Value {
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

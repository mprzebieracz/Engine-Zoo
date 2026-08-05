use super::*;
use alphazero::representation::Connect4AzRepresentation;
use engine_core::notation::GameNotation;
use search::{Mcts, NoExtraRules, SearchRequest};

pub(super) fn create_session_inner(
    state: &AppState,
    req: CreateSessionRequest,
) -> Result<serde_json::Value> {
    let id = state.next_session.fetch_add(1, Ordering::Relaxed);
    let session = match state.game {
        GameKind::Chess => {
            let position = match req.position {
                Some(GameSetup::Chess(position)) => position,
                Some(_) => anyhow::bail!("session position does not match chess"),
                None => ChessSetup::default(),
            };
            let model = resolve_server_model(state, &req.model)?;
            anyhow::ensure!(
                model.model.chess_history().is_some() || model.model.is_chess_classic(),
                "model is not a chess model"
            );
            LiveSession::Chess(Box::new(SessionState {
                id,
                game: ChessGame::from_setup(&position)?,
                moves: position.moves.clone(),
                san_moves: Vec::new(),
                model: req.model,
                human_turn: !req.engine_first,
                simulations: req.simulations,
                wait_for_count: req.wait_for_count,
                chess_history: model.model.chess_history(),
                chess_engine: None,
            }))
        }
        GameKind::Connect4 => {
            let position = match req.position {
                Some(GameSetup::Connect4(position)) => position,
                Some(_) => anyhow::bail!("session position does not match connect4"),
                None => Connect4Setup::default(),
            };
            let moves = position.moves.iter().map(ToString::to_string).collect();
            LiveSession::Connect4(SessionState {
                id,
                game: Connect4::from_setup(&position)?,
                moves,
                san_moves: Vec::new(),
                model: req.model,
                human_turn: !req.engine_first,
                simulations: req.simulations,
                wait_for_count: req.wait_for_count,
                chess_history: None,
                chess_engine: None,
            })
        }
    };
    let value = session_view(&session);
    state.sessions.lock().unwrap().push(session);
    Ok(value)
}

pub(super) fn session_move_inner(
    state: &AppState,
    id: u64,
    req: MoveRequest,
) -> Result<serde_json::Value> {
    let mut sessions = state.sessions.lock().unwrap();
    let session = sessions
        .iter_mut()
        .find(|s| session_id(s) == id)
        .ok_or_else(|| anyhow::anyhow!("unknown session"))?;
    match session {
        LiveSession::Chess(s) => play_chess_human_turn(s, &req.mv)?,
        LiveSession::Connect4(s) => play_connect4_human_turn(s, &req.mv)?,
    }
    Ok(session_view(session))
}

pub(super) fn session_engine_move_inner(state: AppState, id: u64) -> Result<serde_json::Value> {
    let mut sessions = state.sessions.lock().unwrap();
    let session = sessions
        .iter_mut()
        .find(|s| session_id(s) == id)
        .ok_or_else(|| anyhow::anyhow!("unknown session"))?;
    anyhow::ensure!(!session_human_turn(session), "not the engine's turn");
    if !session_terminal(session) && !session_human_turn(session) {
        play_engine_turn(&state, session)?;
    }
    Ok(session_view(session))
}

pub(super) fn play_connect4_human_turn(
    session: &mut SessionState<Connect4>,
    mv: &str,
) -> Result<()> {
    anyhow::ensure!(session.human_turn, "not the human's turn");
    let native = games::connect4::notation::Connect4Notation
        .parse_move(&session.game, mv)
        .ok_or_else(|| anyhow::anyhow!("illegal or unparsable move {mv}"))?;
    session.game.play(native);
    session
        .moves
        .push(games::connect4::notation::Connect4Notation.format_move(&session.game, native));
    session.san_moves.push(mv.to_owned());
    session.human_turn = false;
    Ok(())
}

pub(super) fn play_chess_human_turn(session: &mut SessionState<ChessGame>, mv: &str) -> Result<()> {
    anyhow::ensure!(session.human_turn, "not the human's turn");
    let native = games::chess::ChessUciNotation
        .parse_move(&session.game.position(), mv)
        .ok_or_else(|| anyhow::anyhow!("illegal or unparsable move {mv}"))?;
    let san = games::chess::notation::san(session.game.board(), native);
    session.game.play(native);
    session.moves.push(native.to_string());
    session.san_moves.push(san);
    session.human_turn = false;
    Ok(())
}

pub(super) fn play_engine_turn(state: &AppState, session: &mut LiveSession) -> Result<()> {
    match session {
        LiveSession::Chess(s) => play_chess_engine_turn(state, s),
        LiveSession::Connect4(s) => play_engine_turn_for(state, s),
    }
}

pub(super) fn play_chess_engine_turn(
    state: &AppState,
    session: &mut SessionState<ChessGame>,
) -> Result<()> {
    if session.game.is_terminal() {
        return Ok(());
    }
    let model = resolve_server_model(state, &session.model)?;
    if let Some(history) = session.chess_history {
        anyhow::ensure!(
            model.model.chess_history() == Some(history),
            "session model no longer matches its selected model"
        );
    }
    else {
        anyhow::ensure!(
            model.model.is_chess_classic(),
            "chess session model no longer matches its selected model"
        );
    }

    if session.chess_engine.is_none() {
        let config = state.models.prepared_config(
            &model,
            &state.repository,
            session.wait_for_count,
            Duration::from_millis(1),
            state.device,
            state.backend,
        )?;
        let loaded = state
            .models
            .load(&model.model, &model.checkpoint, state.device, &config)?;
        let engine = alphazero::ChessAlphaZeroEngine::from_client(
            loaded.model.clone(),
            loaded.inference.client(),
            puct_search(),
        )?;

        session.chess_engine = Some(Box::new(engine));
    }

    let native = session
        .chess_engine
        .as_deref_mut()
        .expect("chess engine was initialized")
        .select_move(
            &session.game,
            SearchRequest::deterministic_puct(session.simulations.max(1)),
        )?;
    let uci = games::chess::ChessUciNotation.format_move(&session.game.position(), native);
    let san = games::chess::notation::san(session.game.board(), native);
    session.game.play(native);
    session.moves.push(uci);
    session.san_moves.push(san);
    session.human_turn = true;
    Ok(())
}

pub(super) fn play_engine_turn_for(
    state: &AppState,
    session: &mut SessionState<Connect4>,
) -> Result<()> {
    if session.game.is_terminal() {
        return Ok(());
    }
    let model = resolve_server_model(state, &session.model)?;
    anyhow::ensure!(
        model.model.game() == alphazero::GameKind::Connect4,
        "model is not Connect4"
    );
    let config = state.models.prepared_config(
        &model,
        &state.repository,
        session.wait_for_count,
        Duration::from_millis(1),
        state.device,
        state.backend,
    )?;
    let loaded = state
        .models
        .load(&model.model, &model.checkpoint, state.device, &config)?;
    let mut mcts = Mcts::new(
        alphazero::RepresentedEvaluator::new(Connect4AzRepresentation, loaded.inference.client()),
        puct_search(),
        NoExtraRules,
    );
    let native = mcts
        .search(
            &session.game,
            (),
            SearchRequest::deterministic_puct(session.simulations.max(1)),
        )?
        .best_move();
    let mv = games::connect4::notation::Connect4Notation.format_move(&session.game, native);
    session.game.play(native);
    session.moves.push(mv);
    session.human_turn = true;
    Ok(())
}

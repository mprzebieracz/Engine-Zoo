use super::*;
use alphazero::representation::{
    ChessAzRepresentation, ChessClassicRepresentation, Connect4AzRepresentation,
};
use alphazero::ChessRepetitionRules;
use engine_core::notation::GameNotation;
use search::NoExtraRules;

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
            let (_, cfg) = open_existing_run(&state.run_dir, "chess")?;
            anyhow::ensure!(
                cfg.model.chess_history().is_some() || cfg.model.is_chess_classic(),
                "run config is not a chess model"
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
                chess_history: cfg.model.chess_history(),
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
        play_engine_turn(&state.run_dir, state.device, session)?;
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

pub(super) fn play_engine_turn(
    run_dir: &Path,
    device: Device,
    session: &mut LiveSession,
) -> Result<()> {
    match session {
        LiveSession::Chess(s) => play_chess_engine_turn(run_dir, device, s),
        LiveSession::Connect4(s) => play_engine_turn_for(run_dir, device, s),
    }
}

pub(super) fn play_chess_engine_turn(
    run_dir: &Path,
    device: Device,
    session: &mut SessionState<ChessGame>,
) -> Result<()> {
    if session.game.is_terminal() {
        return Ok(());
    }
    let (_, cfg) = open_existing_run(run_dir, "chess")?;
    if let Some(history) = session.chess_history {
        anyhow::ensure!(
            cfg.model.chess_history() == Some(history),
            "session model no longer matches its run"
        );
        let batcher = Batcher::new_with_model_precision(
            cfg.model.clone(),
            &resolve_model(run_dir, &session.model),
            device,
            batcher_config(session.wait_for_count, Duration::from_millis(1)),
            if device.is_cuda() {
                InferencePrecision::Fp16
            }
            else {
                InferencePrecision::Fp32
            },
        )?;
        let mv = match history {
            alphazero::ChessHistory::One => {
                chess_best_action_for::<1>(batcher.client(), &session.game, session.simulations)?
            }
            alphazero::ChessHistory::Four => {
                chess_best_action_for::<4>(batcher.client(), &session.game, session.simulations)?
            }
            alphazero::ChessHistory::Eight => {
                chess_best_action_for::<8>(batcher.client(), &session.game, session.simulations)?
            }
        };
        let uci = games::chess::ChessUciNotation.format_move(&session.game.position(), mv);
        let san = games::chess::notation::san(session.game.board(), mv);
        session.game.play(mv);
        session.moves.push(uci);
        session.san_moves.push(san);
        session.human_turn = true;
        return Ok(());
    }
    anyhow::ensure!(
        cfg.model.is_chess_classic(),
        "chess session model no longer matches its run"
    );
    let batcher = Batcher::new_with_model_precision(
        cfg.model.clone(),
        &resolve_model(run_dir, &session.model),
        device,
        batcher_config(session.wait_for_count, Duration::from_millis(1)),
        if device.is_cuda() {
            InferencePrecision::Fp16
        }
        else {
            InferencePrecision::Fp32
        },
    )?;
    let native = Mcts::new(
        alphazero::RepresentedEvaluator::new(ChessClassicRepresentation, batcher.client()),
        puct_search(session.simulations),
        NoExtraRules,
    )
    .search(
        &session.game.position(),
        (),
        search_request(session.simulations),
    )?
    .best_move();
    let uci = games::chess::ChessUciNotation.format_move(&session.game.position(), native);
    let san = games::chess::notation::san(session.game.board(), native);
    session.game.play(native);
    session.moves.push(uci);
    session.san_moves.push(san);
    session.human_turn = true;
    Ok(())
}

pub(super) fn play_engine_turn_for(
    run_dir: &Path,
    device: Device,
    session: &mut SessionState<Connect4>,
) -> Result<()> {
    if session.game.is_terminal() {
        return Ok(());
    }
    let (_, cfg) = open_existing_run(run_dir, "connect4")?;
    anyhow::ensure!(
        cfg.model.game() == alphazero::GameKind::Connect4,
        "run model is not Connect4"
    );
    let batcher = Batcher::new_with_model(
        cfg.model.clone(),
        &resolve_model(run_dir, &session.model),
        device,
        batcher_config(session.wait_for_count, Duration::from_millis(1)),
    )?;
    let mut mcts = Mcts::new(
        alphazero::RepresentedEvaluator::new(Connect4AzRepresentation, batcher.client()),
        puct_search(session.simulations),
        NoExtraRules,
    );
    let native = mcts
        .search(&session.game, (), search_request(session.simulations))?
        .best_move();
    let mv = games::connect4::notation::Connect4Notation.format_move(&session.game, native);
    session.game.play(native);
    session.moves.push(mv);
    session.human_turn = true;
    Ok(())
}

fn chess_best_action_for<const HISTORY: usize>(
    evaluator: alphazero::BatcherClient,
    game: &ChessGame,
    simulations: usize,
) -> Result<<ChessGame as GameState>::Move> {
    let state = alphazero::representation::ChessAzState::from_game(game);
    let context = game.repetition_context();
    let mv = Mcts::new(
        alphazero::RepresentedEvaluator::new(ChessAzRepresentation::<HISTORY>, evaluator),
        puct_search(simulations),
        ChessRepetitionRules,
    )
    .search(&state, context, search_request(simulations))?
    .best_move();
    Ok(mv)
}

fn search_request(simulations: usize) -> alphazero::SearchRequest {
    alphazero::SearchRequest {
        mode: engine_core::agent::PolicyMode::Deterministic,
        budget: alphazero::MctsSearchBudget::Puct {
            simulations: simulations.max(1),
        },
    }
}

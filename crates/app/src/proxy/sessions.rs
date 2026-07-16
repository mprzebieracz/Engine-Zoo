use super::*;
use anyhow::Context;

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
            let (_, cfg) = open_existing_run::<ChessGame>(&state.run_dir)?;
            let v2_history = if let RunArchitecture::ChessAzV2(v2) = cfg.architecture {
                v2.validate()?;
                Some(v2.history)
            }
            else {
                None
            };
            LiveSession::Chess(Box::new(SessionState {
                id,
                game: ChessGame::from_setup(&position)?,
                moves: position.moves.clone(),
                san_moves: Vec::new(),
                model: req.model,
                human_turn: !req.engine_first,
                simulations: req.simulations,
                wait_for_count: req.wait_for_count,
                v2_history,
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
                v2_history: None,
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
        LiveSession::Connect4(s) => play_human_turn(s, &req.mv)?,
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

pub(super) fn play_human_turn<G: Game>(session: &mut SessionState<G>, mv: &str) -> Result<()> {
    anyhow::ensure!(session.human_turn, "not the human's turn");
    let action = session
        .game
        .parse_move(mv)
        .ok_or_else(|| anyhow::anyhow!("illegal or unparsable move {mv}"))?;
    session.game.step(action);
    session.moves.push(mv.to_owned());
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
        LiveSession::Connect4(s) => play_engine_turn_for::<Connect4>(run_dir, device, s),
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
    let (_, cfg) = open_existing_run::<ChessGame>(run_dir)?;
    if let Some(history) = session.v2_history {
        let RunArchitecture::ChessAzV2(v2) = &cfg.architecture
        else {
            anyhow::bail!("session architecture no longer matches its run");
        };
        anyhow::ensure!(
            v2.history == history,
            "session history {history} no longer matches run history {}",
            v2.history
        );
        let batcher = Batcher::new_with_network_precision(
            &cfg.network_config(),
            &resolve_model(run_dir, &session.model),
            device,
            session.wait_for_count.max(1),
            Duration::from_millis(1),
            if device.is_cuda() {
                InferencePrecision::Fp16
            }
            else {
                InferencePrecision::Fp32
            },
        )?;
        let action = match history {
            1 => v2_best_action_for::<1>(batcher.client(), &session.game, session.simulations)?,
            4 => v2_best_action_for::<4>(batcher.client(), &session.game, session.simulations)?,
            8 => v2_best_action_for::<8>(batcher.client(), &session.game, session.simulations)?,
            history => anyhow::bail!("unsupported chess history {history}"),
        };
        let mv =
            decode_v2_action(session.game.board(), action).context("decoding v2 engine action")?;
        let uci = games::chess::ChessUciNotation.format_move(&session.game.position(), mv);
        let san = games::chess::notation::san(session.game.board(), mv);
        session.game.play(mv);
        session.moves.push(uci);
        session.san_moves.push(san);
        session.human_turn = true;
        return Ok(());
    }
    anyhow::ensure!(
        matches!(&cfg.architecture, RunArchitecture::Legacy),
        "session architecture no longer matches its run"
    );
    let batcher = Batcher::new(
        &cfg.net,
        &resolve_model(run_dir, &session.model),
        device,
        session.wait_for_count.max(1),
        Duration::from_millis(1),
    )?;
    let mut mcts = Mcts::new(
        batcher.client(),
        MctsConfig {
            simulations: session.simulations,
            eps: 0.0,
            ..Default::default()
        },
    );
    let action = mcts
        .search_with_mode(&session.game, PolicyMode::Deterministic)
        .best_action();
    let mv = session.game.format_action(action);
    let san = session.game.san_for_action(action);
    session.game.step(action);
    session.moves.push(mv);
    session.san_moves.push(san);
    session.human_turn = true;
    Ok(())
}

pub(super) fn play_engine_turn_for<G: Game>(
    run_dir: &Path,
    device: Device,
    session: &mut SessionState<G>,
) -> Result<()> {
    if session.game.is_terminal() {
        return Ok(());
    }
    let (_, cfg) = open_existing_run::<G>(run_dir)?;
    let batcher = Batcher::new(
        &cfg.net,
        &resolve_model(run_dir, &session.model),
        device,
        session.wait_for_count.max(1),
        Duration::from_millis(1),
    )?;
    let mut mcts = Mcts::new(
        batcher.client(),
        MctsConfig {
            simulations: session.simulations,
            eps: 0.0,
            ..Default::default()
        },
    );
    let action = mcts
        .search_with_mode(&session.game, PolicyMode::Deterministic)
        .best_action();
    let mv = session.game.format_action(action);
    session.game.step(action);
    session.moves.push(mv);
    session.human_turn = true;
    Ok(())
}

fn v2_best_action_for<const HISTORY: usize>(
    evaluator: algorithms::alphazero::BatcherClient,
    game: &ChessGame,
    simulations: usize,
) -> Result<engine_core::game::Action> {
    let state = game.history_state::<HISTORY>();
    let context = game.repetition_context();
    Ok(Mcts::new(
        evaluator,
        MctsConfig {
            simulations: simulations.max(1),
            variant: MctsVariant::Puct,
            eps: 0.0,
            ..Default::default()
        },
    )
    .search_with_repetitions_mode(
        &state,
        |hash| context.occurrences_before_root(hash),
        PolicyMode::Deterministic,
    )
    .best_action())
}

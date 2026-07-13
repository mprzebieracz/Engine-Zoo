use super::*;

pub(super) fn create_session_inner(
    state: &AppState,
    req: CreateSessionRequest,
) -> Result<serde_json::Value> {
    let id = state.next_session.fetch_add(1, Ordering::Relaxed);
    let session = match state.game {
        GameKind::Chess => {
            let position = match req.position {
                Some(PositionSpec::Chess(position)) => position,
                Some(_) => anyhow::bail!("session position does not match chess"),
                None => ChessPosition::default(),
            };
            let (_, cfg) = open_existing_run::<ChessGame>(&state.run_dir)?;
            match cfg.architecture {
                RunArchitecture::Legacy => LiveSession::Chess(SessionState {
                    id,
                    game: ChessGame::from_position(&position)?,
                    moves: position.moves.clone(),
                    san_moves: Vec::new(),
                    model: req.model,
                    human_turn: !req.engine_first,
                    simulations: req.simulations,
                    wait_for_count: req.wait_for_count,
                }),
                RunArchitecture::ChessAzV2(v2) => {
                    cfg.validate()?;
                    LiveSession::ChessAzV2(Box::new(create_chess_az_v2_session(
                        id,
                        position,
                        req.model,
                        !req.engine_first,
                        req.simulations,
                        req.wait_for_count,
                        v2.history,
                    )?))
                }
            }
        }
        GameKind::Connect4 => {
            let position = match req.position {
                Some(PositionSpec::Connect4(position)) => position,
                Some(_) => anyhow::bail!("session position does not match connect4"),
                None => Connect4Position::default(),
            };
            let moves = position.moves.iter().map(ToString::to_string).collect();
            LiveSession::Connect4(SessionState {
                id,
                game: Connect4::from_position(&position)?,
                moves,
                san_moves: Vec::new(),
                model: req.model,
                human_turn: !req.engine_first,
                simulations: req.simulations,
                wait_for_count: req.wait_for_count,
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
        LiveSession::ChessAzV2(s) => play_chess_az_v2_human_turn(s, &req.mv)?,
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
    let action = session
        .game
        .parse_move(mv)
        .ok_or_else(|| anyhow::anyhow!("illegal or unparsable move {mv}"))?;
    let san = session.game.san_for_action(action);
    session.game.step(action);
    session.moves.push(session.game.format_action(action));
    session.san_moves.push(san);
    session.human_turn = false;
    Ok(())
}

pub(super) fn create_chess_az_v2_session(
    id: u64,
    position: ChessPosition,
    model: String,
    human_turn: bool,
    simulations: usize,
    wait_for_count: usize,
    history: usize,
) -> Result<ChessAzV2Session> {
    // A bare FEN has no recoverable earlier positions, so its v2 history is
    // intentionally zero-padded. When `moves` are supplied, replaying them
    // here reconstructs the same feature frames used during self-play.
    let setup = ChessPosition {
        fen: position.fen,
        moves: Vec::new(),
    };
    let initial = ChessGame::from_position(&setup)?.position();
    let mut game = ChessAzGameKind::new(history, initial)?;
    for mv in &position.moves {
        let action = game
            .parse_move(mv)
            .ok_or_else(|| anyhow::anyhow!("illegal v2 chess move {mv}"))?;
        game.step(action);
    }
    Ok(ChessAzV2Session {
        id,
        game,
        moves: position.moves,
        san_moves: Vec::new(),
        model,
        human_turn,
        simulations,
        wait_for_count,
    })
}

pub(super) fn play_chess_az_v2_human_turn(session: &mut ChessAzV2Session, mv: &str) -> Result<()> {
    anyhow::ensure!(session.human_turn, "not the human's turn");
    let action = session
        .game
        .parse_move(mv)
        .ok_or_else(|| anyhow::anyhow!("illegal or unparsable move {mv}"))?;
    let san = session.game.san_for_action(action);
    let uci = session.game.format_action(action);
    session.game.step(action);
    session.moves.push(uci);
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
        LiveSession::ChessAzV2(s) => play_chess_az_v2_engine_turn(run_dir, device, s),
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
    anyhow::ensure!(
        matches!(
            cfg.architecture,
            algorithms::alphazero::RunArchitecture::Legacy
        ),
        "interactive proxy play does not yet support chess-az-v2 runs"
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

pub(super) fn play_chess_az_v2_engine_turn(
    run_dir: &Path,
    device: Device,
    session: &mut ChessAzV2Session,
) -> Result<()> {
    if session.game.is_terminal() {
        return Ok(());
    }
    let (_, cfg) = open_existing_run::<ChessGame>(run_dir)?;
    let RunArchitecture::ChessAzV2(v2) = &cfg.architecture
    else {
        anyhow::bail!("v2 chess session cannot use a legacy run");
    };
    let precision = if device.is_cuda() {
        InferencePrecision::Fp16
    }
    else {
        InferencePrecision::Fp32
    };
    let batcher = Batcher::new_with_network_precision(
        &cfg.network_config(),
        &resolve_model(run_dir, &session.model),
        device,
        session.wait_for_count.max(1),
        Duration::from_millis(1),
        precision,
    )?;
    let action = match &session.game {
        ChessAzGameKind::H1(game) => {
            v2_best_action(batcher.client(), game, session.simulations, v2.history)
        }
        ChessAzGameKind::H4(game) => {
            v2_best_action(batcher.client(), game, session.simulations, v2.history)
        }
        ChessAzGameKind::H8(game) => {
            v2_best_action(batcher.client(), game, session.simulations, v2.history)
        }
    }?;
    let mv = session.game.format_action(action);
    let san = session.game.san_for_action(action);
    session.game.step(action);
    session.moves.push(mv);
    session.san_moves.push(san);
    session.human_turn = true;
    Ok(())
}

pub(super) fn v2_best_action<const HISTORY: usize>(
    evaluator: algorithms::alphazero::BatcherClient,
    game: &ChessAzGame<HISTORY>,
    simulations: usize,
    history: usize,
) -> Result<engine_core::game::Action> {
    anyhow::ensure!(
        HISTORY == history,
        "v2 session history does not match its run config"
    );
    let mut mcts = Mcts::new(
        evaluator,
        MctsConfig {
            simulations: simulations.max(1),
            variant: MctsVariant::Puct,
            eps: 0.0,
            ..Default::default()
        },
    );
    let root_hash = game.position().hash();
    Ok(mcts
        .search_with_repetitions_mode(
            &game.search_state(),
            |hash| game.repetitions_before_root(hash, root_hash),
            PolicyMode::Deterministic,
        )
        .best_action())
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

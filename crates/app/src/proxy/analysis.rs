use super::*;

pub fn analyze_request(
    game: GameKind,
    run_dir: PathBuf,
    req: AnalyzeRequest,
    device: Device,
) -> Result<Analysis> {
    let mode = req.mode.unwrap_or(AnalyzeMode::Net);
    let cfg = AnalyzeConfig {
        mode,
        mcts: MctsConfig {
            simulations: req.simulations,
            eps: 0.0,
            ..Default::default()
        },
        wait_for_count: req.wait_for_count.max(1),
        timeout: BATCH_TIMEOUT,
    };
    match (game, req.position) {
        (GameKind::Chess, PositionSpec::Chess(position)) => {
            let (_, run_cfg) = open_existing_run::<ChessGame>(&run_dir)?;
            let weights = resolve_model(&run_dir, &req.model);
            match run_cfg.architecture {
                RunArchitecture::Legacy => {
                    analyze_position::<ChessGame>(&run_cfg.net, &weights, &position, device, &cfg)
                }
                RunArchitecture::ChessAzV2(v2) => match v2.history {
                    1 => analyze_chess_az_v2::<1>(&run_cfg, &weights, &position, device, &cfg),
                    4 => analyze_chess_az_v2::<4>(&run_cfg, &weights, &position, device, &cfg),
                    8 => analyze_chess_az_v2::<8>(&run_cfg, &weights, &position, device, &cfg),
                    history => anyhow::bail!("unsupported chess az v2 history {history}"),
                },
            }
        }
        (GameKind::Connect4, PositionSpec::Connect4(position)) => {
            let (_, run_cfg) = open_existing_run::<Connect4>(&run_dir)?;
            let weights = resolve_model(&run_dir, &req.model);
            analyze_position::<Connect4>(&run_cfg.net, &weights, &position, device, &cfg)
        }
        (expected, other) => anyhow::bail!(
            "server is configured for {}, but request position is {:?}",
            game_name(expected),
            other
        ),
    }
}

/// Replays a request from its setup. A FEN with no accompanying moves has no
/// recoverable preceding frames, so the unavailable feature history is padded.
pub(super) fn chess_az_state<const HISTORY: usize>(
    position: &ChessPosition,
) -> Result<ChessAzGame<HISTORY>> {
    let mut game = match &position.fen {
        Some(fen) => ChessAzGame::from_fen(fen)?,
        None => ChessAzGame::default(),
    };
    for text in &position.moves {
        let action = game
            .parse_move(text)
            .ok_or_else(|| anyhow::anyhow!("illegal v2 chess move {text}"))?;
        game.step(action);
    }
    Ok(game)
}

fn inference_precision(device: Device) -> InferencePrecision {
    if device.is_cuda() {
        InferencePrecision::Fp16
    }
    else {
        InferencePrecision::Fp32
    }
}

fn analyze_chess_az_v2<const HISTORY: usize>(
    run_cfg: &RunConfig,
    weights: &Path,
    position: &ChessPosition,
    device: Device,
    cfg: &AnalyzeConfig,
) -> Result<Analysis> {
    let game = chess_az_state::<HISTORY>(position)?;
    let state = game.search_state();
    if state.is_terminal() {
        return Ok(Analysis {
            value: state.reward(),
            network_value: state.reward(),
            mcts_value: None,
            best_action: None,
            best_move: None,
            policy: Vec::new(),
            network_policy: Vec::new(),
            mcts_policy: Vec::new(),
        });
    }
    let batcher = Batcher::new_with_network_precision(
        &run_cfg.network_config(),
        weights,
        device,
        cfg.wait_for_count,
        cfg.timeout,
        inference_precision(device),
    )?;
    let network = analyze_game_net(state, batcher.client())?;
    if cfg.mode == AnalyzeMode::Net {
        return Ok(network);
    }
    let mut mcts_cfg = cfg.mcts;
    mcts_cfg.eps = 0.0;
    // Training uses Gumbel self-play, but served models are evaluated with
    // deterministic PUCT for stable, comparable browser results.
    mcts_cfg.variant = MctsVariant::Puct;
    let mut mcts = Mcts::new(batcher.client(), mcts_cfg);
    analyze_game_mcts_with_repetitions(state, &mut mcts, network, |hash| {
        game.repetitions_before(hash)
    })
}

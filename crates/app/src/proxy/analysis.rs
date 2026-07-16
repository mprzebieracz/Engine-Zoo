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
        (GameKind::Chess, GameSetup::Chess(position)) => {
            let (_, run_cfg) = open_existing_run::<ChessGame>(&run_dir)?;
            let weights = resolve_model(&run_dir, &req.model);
            match run_cfg.architecture {
                RunArchitecture::Legacy => analyze_game(
                    ChessGame::from_setup(&position)?,
                    &run_cfg.net,
                    &weights,
                    device,
                    &cfg,
                ),
                RunArchitecture::ChessAzV2(v2) => match v2.history {
                    1 => analyze_chess_az_v2::<1>(&run_cfg, &weights, &position, device, &cfg),
                    4 => analyze_chess_az_v2::<4>(&run_cfg, &weights, &position, device, &cfg),
                    8 => analyze_chess_az_v2::<8>(&run_cfg, &weights, &position, device, &cfg),
                    history => anyhow::bail!("unsupported chess az v2 history {history}"),
                },
            }
        }
        (GameKind::Connect4, GameSetup::Connect4(position)) => {
            let (_, run_cfg) = open_existing_run::<Connect4>(&run_dir)?;
            let weights = resolve_model(&run_dir, &req.model);
            analyze_game(
                Connect4::from_setup(&position)?,
                &run_cfg.net,
                &weights,
                device,
                &cfg,
            )
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
fn chess_az_state_snapshot<const HISTORY: usize>(
    position: &ChessSetup,
) -> Result<(ChessGame, ChessHistoryState<HISTORY>)> {
    let game = ChessGame::from_setup(position)?;
    let state = game.history_state::<HISTORY>();
    Ok((game, state))
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
    position: &ChessSetup,
    device: Device,
    cfg: &AnalyzeConfig,
) -> Result<Analysis> {
    let (game, state) = chess_az_state_snapshot::<HISTORY>(position)?;
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
    let repetition_context = game.repetition_context();
    analyze_game_mcts_with_repetitions(state, &mut mcts, network, |hash| {
        repetition_context.occurrences_before_root(hash)
    })
}

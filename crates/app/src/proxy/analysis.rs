use super::*;
use algorithms::alphazero::representation::{
    ChessAzRepresentation, ChessV1Representation, Connect4AzRepresentation,
};
use algorithms::search::{ChessRepetitionRules, NoExtraRules};
use engine_core::notation::GameNotation;
use engine_core::Game;

struct ChessHistoryNotation;

impl<const HISTORY: usize> GameNotation<ChessHistoryState<HISTORY>> for ChessHistoryNotation {
    fn parse_move(
        &self,
        state: &ChessHistoryState<HISTORY>,
        text: &str,
    ) -> Option<<ChessHistoryState<HISTORY> as engine_core::game::GameState>::Move> {
        notation::ChessUciNotation.parse_move(&state.position(), text)
    }

    fn format_move(
        &self,
        state: &ChessHistoryState<HISTORY>,
        mv: <ChessHistoryState<HISTORY> as engine_core::game::GameState>::Move,
    ) -> String {
        notation::ChessUciNotation.format_move(&state.position(), mv)
    }
}

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
            let (_, run_cfg) = open_existing_run(&run_dir, "chess")?;
            let weights = resolve_model(&run_dir, &req.model);
            match run_cfg.architecture {
                RunArchitecture::Legacy => analyze_chess_legacy(
                    &run_cfg,
                    &weights,
                    &ChessGame::from_setup(&position)?,
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
            let (_, run_cfg) = open_existing_run(&run_dir, "connect4")?;
            let weights = resolve_model(&run_dir, &req.model);
            analyze_connect4(
                &run_cfg,
                &weights,
                &Connect4::from_setup(&position)?,
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

fn analyze_chess_legacy(
    run_cfg: &RunConfig,
    weights: &Path,
    game: &ChessGame,
    device: Device,
    cfg: &AnalyzeConfig,
) -> Result<Analysis> {
    let batcher = Batcher::new(
        &run_cfg.net,
        weights,
        device,
        cfg.wait_for_count,
        cfg.timeout,
    )?;
    let position = game.position_state();
    let network = analyze_game_net(
        position,
        batcher.client(),
        &ChessV1Representation,
        &notation::ChessUciNotation,
    )?;
    if cfg.mode == AnalyzeMode::Net {
        return Ok(network);
    }
    let mut mcts = Mcts::new(
        algorithms::alphazero::RepresentedEvaluator::new(ChessV1Representation, batcher.client()),
        cfg.mcts,
        ChessRepetitionRules,
    );
    analyze_game_mcts(
        position,
        &mut mcts,
        game.repetition_context(),
        network,
        &ChessV1Representation,
        &notation::ChessUciNotation,
    )
}

fn analyze_connect4(
    run_cfg: &RunConfig,
    weights: &Path,
    game: &Connect4,
    device: Device,
    cfg: &AnalyzeConfig,
) -> Result<Analysis> {
    let batcher = Batcher::new(
        &run_cfg.net,
        weights,
        device,
        cfg.wait_for_count,
        cfg.timeout,
    )?;
    let network = analyze_game_net(
        *game,
        batcher.client(),
        &Connect4AzRepresentation,
        &games::connect4::notation::Connect4Notation,
    )?;
    if cfg.mode == AnalyzeMode::Net {
        return Ok(network);
    }
    let mut mcts = Mcts::new(
        algorithms::alphazero::RepresentedEvaluator::new(
            Connect4AzRepresentation,
            batcher.client(),
        ),
        cfg.mcts,
        NoExtraRules,
    );
    analyze_game_mcts(
        *game,
        &mut mcts,
        (),
        network,
        &Connect4AzRepresentation,
        &games::connect4::notation::Connect4Notation,
    )
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
    if Game::is_terminal(&state) {
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
    let network = analyze_game_net(
        state,
        batcher.client(),
        &ChessAzRepresentation::<HISTORY>,
        &ChessHistoryNotation,
    )?;
    if cfg.mode == AnalyzeMode::Net {
        return Ok(network);
    }
    let mut mcts_cfg = cfg.mcts;
    mcts_cfg.eps = 0.0;
    // Training uses Gumbel self-play, but served models are evaluated with
    // deterministic PUCT for stable, comparable browser results.
    mcts_cfg.variant = MctsVariant::Puct;
    let mut mcts = Mcts::new(
        algorithms::alphazero::RepresentedEvaluator::new(
            ChessAzRepresentation::<HISTORY>,
            batcher.client(),
        ),
        mcts_cfg,
        ChessRepetitionRules,
    );
    let repetition_context = game.repetition_context();
    analyze_game_mcts(
        state,
        &mut mcts,
        repetition_context,
        network,
        &ChessAzRepresentation::<HISTORY>,
        &ChessHistoryNotation,
    )
}

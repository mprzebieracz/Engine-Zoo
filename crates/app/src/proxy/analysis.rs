use super::*;
use alphazero::representation::{
    ChessAzRepresentation, ChessAzState, ChessV1Representation, Connect4AzRepresentation,
};
use alphazero::ChessRepetitionRules;
use engine_core::game::{GameState, TerminalValue};
use engine_core::notation::GameNotation;
use search::NoExtraRules;

struct ChessHistoryNotation;

impl<const HISTORY: usize> GameNotation<ChessAzState<HISTORY>> for ChessHistoryNotation {
    fn parse_move(
        &self,
        state: &ChessAzState<HISTORY>,
        text: &str,
    ) -> Option<<ChessAzState<HISTORY> as engine_core::game::GameState>::Move> {
        notation::ChessUciNotation.parse_move(&state.position(), text)
    }

    fn format_move(
        &self,
        state: &ChessAzState<HISTORY>,
        mv: <ChessAzState<HISTORY> as engine_core::game::GameState>::Move,
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
            match &run_cfg.model {
                ModelConfig::ChessScalarAzV1(_) => analyze_chess_legacy(
                    &run_cfg.network_config(),
                    &weights,
                    &ChessGame::from_setup(&position)?,
                    device,
                    &cfg,
                ),
                ModelConfig::ChessAzV2(v2) => match v2.history {
                    1 => analyze_chess_az_v2::<1>(
                        &run_cfg.network_config(),
                        &weights,
                        &position,
                        device,
                        &cfg,
                    ),
                    4 => analyze_chess_az_v2::<4>(
                        &run_cfg.network_config(),
                        &weights,
                        &position,
                        device,
                        &cfg,
                    ),
                    8 => analyze_chess_az_v2::<8>(
                        &run_cfg.network_config(),
                        &weights,
                        &position,
                        device,
                        &cfg,
                    ),
                    history => anyhow::bail!("unsupported chess az v2 history {history}"),
                },
                ModelConfig::Connect4ScalarAz(_) => {
                    anyhow::bail!("run config is not a chess model")
                }
            }
        }
        (GameKind::Connect4, GameSetup::Connect4(position)) => {
            let (_, run_cfg) = open_existing_run(&run_dir, "connect4")?;
            let weights = resolve_model(&run_dir, &req.model);
            anyhow::ensure!(
                matches!(&run_cfg.model, ModelConfig::Connect4ScalarAz(_)),
                "run config is not a Connect4 scalar AlphaZero model"
            );
            analyze_connect4(
                &run_cfg.network_config(),
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
    network_cfg: &NetworkConfig,
    weights: &Path,
    game: &ChessGame,
    device: Device,
    cfg: &AnalyzeConfig,
) -> Result<Analysis> {
    let batcher = Batcher::new_with_network(
        network_cfg,
        weights,
        device,
        cfg.wait_for_count,
        cfg.timeout,
    )?;
    let position = game.position();
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
        alphazero::RepresentedEvaluator::new(ChessV1Representation, batcher.client()),
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
    network_cfg: &NetworkConfig,
    weights: &Path,
    game: &Connect4,
    device: Device,
    cfg: &AnalyzeConfig,
) -> Result<Analysis> {
    let batcher = Batcher::new_with_network(
        network_cfg,
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
        alphazero::RepresentedEvaluator::new(Connect4AzRepresentation, batcher.client()),
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
) -> Result<(ChessGame, ChessAzState<HISTORY>)> {
    let game = ChessGame::from_setup(position)?;
    let state = ChessAzState::from_game(&game);
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
    network_cfg: &NetworkConfig,
    weights: &Path,
    position: &ChessSetup,
    device: Device,
    cfg: &AnalyzeConfig,
) -> Result<Analysis> {
    let (game, state) = chess_az_state_snapshot::<HISTORY>(position)?;
    if state.is_terminal() {
        return Ok(Analysis {
            value: state.terminal_value().map_or(0.0, TerminalValue::as_f32),
            network_value: state.terminal_value().map_or(0.0, TerminalValue::as_f32),
            mcts_value: None,
            best_action: None,
            best_move: None,
            policy: Vec::new(),
            network_policy: Vec::new(),
            mcts_policy: Vec::new(),
        });
    }
    let batcher = Batcher::new_with_network_precision(
        network_cfg,
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
        alphazero::RepresentedEvaluator::new(ChessAzRepresentation::<HISTORY>, batcher.client()),
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

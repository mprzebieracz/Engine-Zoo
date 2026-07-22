use super::*;
use alphazero::representation::{
    ChessAzRepresentation, ChessAzState, ChessClassicRepresentation, Connect4AzRepresentation,
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
        mcts: puct_search(req.simulations),
        wait_for_count: req.wait_for_count.max(1),
        timeout: BATCH_TIMEOUT,
    };
    match (game, req.position) {
        (GameKind::Chess, GameSetup::Chess(position)) => {
            let (_, run_cfg) = open_existing_run(&run_dir, "chess")?;
            let weights = resolve_model(&run_dir, &req.model);
            match run_cfg.model.chess_history() {
                Some(alphazero::ChessHistoryLength::One) => {
                    analyze_chess::<1>(&run_cfg.model, &weights, &position, device, &cfg)
                }
                Some(alphazero::ChessHistoryLength::Four) => {
                    analyze_chess::<4>(&run_cfg.model, &weights, &position, device, &cfg)
                }
                Some(alphazero::ChessHistoryLength::Eight) => {
                    analyze_chess::<8>(&run_cfg.model, &weights, &position, device, &cfg)
                }
                None if run_cfg.model.is_chess_classic() => {
                    analyze_classic_chess(&run_cfg.model, &weights, &position, device, &cfg)
                }
                None => anyhow::bail!("run config is not a chess model"),
            }
        }
        (GameKind::Connect4, GameSetup::Connect4(position)) => {
            let (_, run_cfg) = open_existing_run(&run_dir, "connect4")?;
            let weights = resolve_model(&run_dir, &req.model);
            anyhow::ensure!(
                run_cfg.model.game == alphazero::GameSpec::Connect4,
                "run config is not a Connect4 model"
            );
            analyze_connect4(
                &run_cfg.model,
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

fn analyze_classic_chess(
    model: &alphazero::ModelSpec,
    weights: &Path,
    position: &ChessSetup,
    device: Device,
    cfg: &AnalyzeConfig,
) -> Result<Analysis> {
    let game = ChessGame::from_setup(position)?;
    let state = game.position();
    let batcher = Batcher::new_with_model_precision(
        model.clone(),
        weights,
        device,
        batcher_config(cfg.wait_for_count, cfg.timeout),
        inference_precision(device),
    )?;
    let network = analyze_game_net(
        state,
        batcher.client(),
        &ChessClassicRepresentation,
        &notation::ChessUciNotation,
    )?;
    if cfg.mode == AnalyzeMode::Net {
        return Ok(network);
    }
    let mut mcts = Mcts::new(
        alphazero::RepresentedEvaluator::new(ChessClassicRepresentation, batcher.client()),
        puct_search(match &cfg.mcts {
            SearchConfig::Puct(search) => search.common.simulations,
            SearchConfig::Gumbel(search) => search.simulations,
        }),
        NoExtraRules,
    );
    analyze_game_mcts(
        state,
        &mut mcts,
        (),
        network,
        &ChessClassicRepresentation,
        &notation::ChessUciNotation,
    )
}

fn analyze_connect4(
    model: &alphazero::ModelSpec,
    weights: &Path,
    game: &Connect4,
    device: Device,
    cfg: &AnalyzeConfig,
) -> Result<Analysis> {
    let batcher = Batcher::new_with_model(
        model.clone(),
        weights,
        device,
        batcher_config(cfg.wait_for_count, cfg.timeout),
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
        cfg.mcts.clone(),
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
    } else {
        InferencePrecision::Fp32
    }
}

fn analyze_chess<const HISTORY: usize>(
    model: &alphazero::ModelSpec,
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
    let batcher = Batcher::new_with_model_precision(
        model.clone(),
        weights,
        device,
        batcher_config(cfg.wait_for_count, cfg.timeout),
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
    let mcts_cfg = puct_search(match &cfg.mcts {
        SearchConfig::Puct(search) => search.common.simulations,
        SearchConfig::Gumbel(search) => search.simulations,
    });
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

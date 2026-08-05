use super::*;
use alphazero::representation::{
    ChessAzRepresentation, ChessAzState, ChessClassicRepresentation, Connect4AzRepresentation,
};
use alphazero::ChessRepetitionRules;
use engine_core::game::{GameState, TerminalValue};
use engine_core::notation::GameNotation;
use engine_model_runtime::BackendPreference;
use search::{Mcts, NoExtraRules};

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
    let registry = ModelRegistry::new();
    let repository = RepositoryConfig::discover(std::env::current_dir()?)?;
    analyze_request_with_registry(
        game,
        run_dir,
        req,
        device,
        &registry,
        &repository,
        "latest",
        BackendPreference::Auto,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn analyze_request_with_registry(
    game: GameKind,
    run_dir: PathBuf,
    req: AnalyzeRequest,
    device: Device,
    registry: &ModelRegistry,
    repository: &RepositoryConfig,
    default_model: &str,
    backend: BackendPreference,
) -> Result<Analysis> {
    let mode = req.mode.unwrap_or(AnalyzeMode::Net);
    let cfg = AnalyzeConfig {
        mode,
        mcts: puct_search(),
        wait_for_count: req.wait_for_count.max(1),
        timeout: BATCH_TIMEOUT,
    };
    match (game, req.position) {
        (GameKind::Chess, GameSetup::Chess(position)) => {
            let model = resolve_server_model_at(repository, &run_dir, default_model, &req.model)?;
            let inference = inference_for_model(
                &model,
                repository,
                cfg.wait_for_count,
                cfg.timeout,
                device,
                backend,
            )?;
            let loaded = registry.load(&model.model, &model.checkpoint, device, &inference)?;
            match model.model.chess_history() {
                Some(alphazero::ChessHistory::One) => {
                    analyze_chess::<1>(loaded.inference.client(), &position, &cfg)
                }
                Some(alphazero::ChessHistory::Four) => {
                    analyze_chess::<4>(loaded.inference.client(), &position, &cfg)
                }
                Some(alphazero::ChessHistory::Eight) => {
                    analyze_chess::<8>(loaded.inference.client(), &position, &cfg)
                }
                None if model.model.is_chess_classic() => {
                    analyze_classic_chess(loaded.inference.client(), &position, &cfg)
                }
                None => anyhow::bail!("model is not a chess model"),
            }
        }
        (GameKind::Connect4, GameSetup::Connect4(position)) => {
            let model = resolve_server_model_at(repository, &run_dir, default_model, &req.model)?;
            anyhow::ensure!(
                model.model.game() == alphazero::GameKind::Connect4,
                "model is not Connect4"
            );
            let inference = inference_for_model(
                &model,
                repository,
                cfg.wait_for_count,
                cfg.timeout,
                device,
                backend,
            )?;
            let loaded = registry.load(&model.model, &model.checkpoint, device, &inference)?;
            analyze_connect4(
                loaded.inference.client(),
                &Connect4::from_setup(&position)?,
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
    client: alphazero::InferenceClient,
    position: &ChessSetup,
    cfg: &AnalyzeConfig,
) -> Result<Analysis> {
    let game = ChessGame::from_setup(position)?;
    let state = game.position();
    let network = analyze_game_net(
        state,
        client.clone(),
        &ChessClassicRepresentation,
        &notation::ChessUciNotation,
    )?;
    if cfg.mode == AnalyzeMode::Net {
        return Ok(network);
    }
    let mut mcts = Mcts::new(
        alphazero::RepresentedEvaluator::new(ChessClassicRepresentation, client),
        cfg.mcts.clone(),
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
    client: alphazero::InferenceClient,
    game: &Connect4,
    cfg: &AnalyzeConfig,
) -> Result<Analysis> {
    let network = analyze_game_net(
        *game,
        client.clone(),
        &Connect4AzRepresentation,
        &games::connect4::notation::Connect4Notation,
    )?;
    if cfg.mode == AnalyzeMode::Net {
        return Ok(network);
    }
    let mut mcts = Mcts::new(
        alphazero::RepresentedEvaluator::new(Connect4AzRepresentation, client),
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

fn analyze_chess<const HISTORY: usize>(
    client: alphazero::InferenceClient,
    position: &ChessSetup,
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
    let network = analyze_game_net(
        state,
        client.clone(),
        &ChessAzRepresentation::<HISTORY>,
        &ChessHistoryNotation,
    )?;
    if cfg.mode == AnalyzeMode::Net {
        return Ok(network);
    }
    let mcts_cfg = cfg.mcts.clone();
    let mut mcts = Mcts::new(
        alphazero::RepresentedEvaluator::new(ChessAzRepresentation::<HISTORY>, client),
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

use alphazero::{analyze_game_mcts, analyze_game_net, Analysis, AnalyzeConfig, AnalyzeMode};
use alphazero::{ExperimentConfig, InferencePrecision, RunDir};
use anyhow::Result;
use axum::extract::State;
use axum::http::{header, Method, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use clap::ValueEnum;
use engine_core::game::GameState;
use games::chess::notation;
use games::setup::{ChessSetup, Connect4Setup, GameSetup};
use games::{ChessGame, Connect4};
use search::SearchConfig;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tch::Device;
use tower_http::cors::{Any, CorsLayer};

mod analysis;
mod http;
mod session_view;
mod sessions;

pub use analysis::analyze_request;
use analysis::analyze_request_with_registry;
use http::*;
pub use http::{game_name, open_existing_run, resolve_model, run_dir};
use session_view::*;
use sessions::*;

use crate::evaluation_jobs::{CreateJob, EvaluationService};
use crate::visualization::render_chess_play_page;

const BATCH_TIMEOUT: Duration = Duration::from_millis(2);

fn puct_search(_simulations: usize) -> SearchConfig {
    SearchConfig::Puct(search::PuctConfig {
        leaf_batch_size: 1,
        root_noise: None,
        ..Default::default()
    })
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GameKind {
    Connect4,
    Chess,
}

#[derive(Clone, Debug)]
pub struct ServeConfig {
    pub game: GameKind,
    pub run_dir: PathBuf,
    pub bind: SocketAddr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalyzeRequest {
    pub position: GameSetup,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub mode: Option<AnalyzeMode>,
    #[serde(default = "default_simulations")]
    pub simulations: usize,
    #[serde(default = "default_wait_for_count")]
    pub wait_for_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct CreateSessionRequest {
    #[serde(default)]
    position: Option<GameSetup>,
    #[serde(default = "default_model")]
    model: String,
    #[serde(default)]
    engine_first: bool,
    #[serde(default = "default_simulations")]
    simulations: usize,
    #[serde(default = "default_wait_for_count")]
    wait_for_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct MoveRequest {
    mv: String,
}

#[derive(Clone)]
struct AppState {
    game: GameKind,
    run_dir: PathBuf,
    device: Device,
    sessions: Arc<Mutex<Vec<LiveSession>>>,
    next_session: Arc<AtomicU64>,
    models: Arc<ModelRegistry>,
    evaluations: EvaluationService,
}

struct ModelRegistry {
    models: Mutex<HashMap<ModelKey, Arc<LoadedModel>>>,
}

#[derive(Hash, PartialEq, Eq)]
struct ModelKey {
    checkpoint: PathBuf,
    checkpoint_modified: Option<std::time::SystemTime>,
    checkpoint_size: u64,
    model: alphazero::ModelFingerprint,
    preferred_batch_size: usize,
    max_batch_size: usize,
    wait_milliseconds: u64,
    precision_is_fp16: bool,
    cuda: bool,
}

struct LoadedModel {
    model: alphazero::ModelSpec,
    inference: alphazero::InferenceService,
}

impl ModelRegistry {
    fn new() -> Self {
        Self {
            models: Mutex::new(HashMap::new()),
        }
    }

    fn load(
        &self,
        model: &alphazero::ModelSpec,
        checkpoint: &Path,
        device: Device,
        config: &alphazero::InferenceConfig,
    ) -> Result<Arc<LoadedModel>> {
        let metadata = fs::metadata(checkpoint)?;
        let key = ModelKey {
            checkpoint: checkpoint.to_path_buf(),
            checkpoint_modified: metadata.modified().ok(),
            checkpoint_size: metadata.len(),
            model: model.fingerprint(),
            preferred_batch_size: config.preferred_batch_size,
            max_batch_size: config.max_batch_size,
            wait_milliseconds: config.max_wait.milliseconds,
            precision_is_fp16: config.precision == InferencePrecision::Fp16,
            cuda: device.is_cuda(),
        };
        let mut models = self.models.lock().unwrap();
        if let Some(loaded) = models.get(&key) {
            return Ok(Arc::clone(loaded));
        }

        let inference = alphazero::InferenceService::load(
            model,
            alphazero::InferenceSource::Checkpoint(checkpoint),
            device,
            config,
        )?;
        let loaded = Arc::new(LoadedModel {
            model: model.clone(),
            inference,
        });
        models.insert(key, Arc::clone(&loaded));

        Ok(loaded)
    }
}

fn server_inference_config(
    config: &alphazero::InferenceConfig,
    wait_for_count: usize,
    timeout: Duration,
    device: Device,
) -> alphazero::InferenceConfig {
    let mut config = config.clone();
    // Server model aliases resolve to checkpoint files. Preserve the previous
    // HTTP behavior by using reloadable native inference for those aliases.
    config.engine = alphazero::InferenceEngine::Native;
    config.tensor_rt_module = None;
    config.preferred_batch_size = wait_for_count.max(1);
    config.max_wait = alphazero::DurationConfig {
        milliseconds: timeout.as_millis().try_into().unwrap_or(u64::MAX),
    };
    config.precision = if device.is_cuda() {
        InferencePrecision::Fp16
    }
    else {
        InferencePrecision::Fp32
    };

    config
}

enum LiveSession {
    Chess(Box<SessionState<ChessGame>>),
    Connect4(SessionState<Connect4>),
}

struct SessionState<G: GameState> {
    id: u64,
    game: G,
    moves: Vec<String>,
    san_moves: Vec<String>,
    model: String,
    human_turn: bool,
    simulations: usize,
    wait_for_count: usize,
    chess_history: Option<alphazero::ChessHistory>,
}

pub async fn serve(cfg: ServeConfig) -> Result<()> {
    let evaluations = EvaluationService::new(cfg.run_dir.clone())?;
    let state = AppState {
        game: cfg.game,
        run_dir: cfg.run_dir,
        device: Device::cuda_if_available(),
        sessions: Arc::new(Mutex::new(Vec::new())),
        next_session: Arc::new(AtomicU64::new(1)),
        models: Arc::new(ModelRegistry::new()),
        evaluations,
    };
    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/api/health", get(health))
        .route("/api/games", get(games_http))
        .route("/api/runs", get(runs_http))
        .route("/api/checkpoints", get(checkpoints))
        .route("/api/evaluations/catalog", get(evaluations_catalog))
        .route(
            "/api/evaluations/jobs",
            get(evaluations_jobs).post(create_evaluation_job),
        )
        .route("/api/evaluations/jobs/{id}", get(get_evaluation_job))
        .route(
            "/api/evaluations/jobs/{id}/artifacts/{name}",
            get(get_evaluation_artifact),
        )
        .route("/api/analyze", any(analyze_http))
        .route("/api/sessions", post(create_session))
        .route("/api/sessions/{id}", get(get_session))
        .route("/api/sessions/{id}/move", post(session_move))
        .route("/api/sessions/{id}/engine", post(session_engine_move))
        .route("/runs/status", get(run_status))
        .route("/runs/checkpoints", get(checkpoints))
        .route("/analyze", any(analyze_http))
        .route("/sessions", post(create_session))
        .route("/sessions/{id}", get(get_session))
        .route("/sessions/{id}/move", post(session_move))
        .route("/sessions/{id}/engine", post(session_engine_move))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state);
    println!("engine-zoo serving {} on {}", game_name(cfg.game), cfg.bind);
    let listener = tokio::net::TcpListener::bind(cfg.bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alphazero::AlphaZeroRepresentation;
    use engine_core::notation::GameNotation;

    #[test]
    fn analyze_accepts_query_and_post_only() {
        let query = Method::from_bytes(b"QUERY").unwrap();
        assert!(is_analyze_method(&query));
        assert!(is_analyze_method(&Method::POST));
        assert!(!is_analyze_method(&Method::GET));
    }

    #[test]
    fn chess_analysis_state_replays_request_history_for_all_supported_lengths() {
        let position = ChessSetup {
            fen: None,
            moves: vec!["e2e4".into(), "e7e5".into(), "g1f3".into(), "b8c6".into()],
        };

        let game = ChessGame::from_setup(&position).unwrap();
        assert_eq!(
            alphazero::representation::ChessAzState::<1>::from_game(&game).board(),
            game.board()
        );
        assert_eq!(
            alphazero::representation::ChessAzState::<4>::from_game(&game).board(),
            game.board()
        );
        assert_eq!(
            alphazero::representation::ChessAzState::<8>::from_game(&game).board(),
            game.board()
        );
    }

    #[test]
    fn chess_analysis_state_accepts_a_fen_without_history() {
        let position = ChessSetup {
            fen: Some("8/8/8/8/8/8/8/K6k b - - 0 1".into()),
            moves: Vec::new(),
        };
        let game = ChessGame::from_setup(&position).unwrap();
        assert_eq!(
            alphazero::representation::ChessAzState::<4>::from_game(&game).board(),
            game.board()
        );
    }

    fn assert_chess_session<const HISTORY: usize>() {
        let mut session = SessionState {
            id: 1,
            game: ChessGame::default(),
            moves: Vec::new(),
            san_moves: Vec::new(),
            model: "best".into(),
            human_turn: true,
            simulations: 1,
            wait_for_count: 1,
            chess_history: Some(match HISTORY {
                1 => alphazero::ChessHistory::One,
                4 => alphazero::ChessHistory::Four,
                8 => alphazero::ChessHistory::Eight,
                _ => unreachable!(),
            }),
        };
        let mv = games::chess::ChessUciNotation
            .parse_move(&session.game.position(), "e2e4")
            .unwrap();
        let expected_action = alphazero::representation::ChessAzRepresentation::<HISTORY>
            .move_to_action(
                &alphazero::representation::ChessAzState::from_game(&session.game),
                mv,
            )
            .as_u32();
        let initial_board = *session.game.board();
        let view = session_view_chess(&session);
        let rendered = view["legal_moves"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["move"] == "e2e4")
            .unwrap();
        assert_eq!(rendered["action"], json!(expected_action));

        play_chess_human_turn(&mut session, "e2e4").unwrap();
        assert_ne!(*session.game.board(), initial_board);
        assert_eq!(
            session.game.board(),
            alphazero::representation::ChessAzState::<HISTORY>::from_game(&session.game).board()
        );
        assert_eq!(session.moves, ["e2e4"]);
        assert_eq!(
            session.chess_history,
            Some(match HISTORY {
                1 => alphazero::ChessHistory::One,
                4 => alphazero::ChessHistory::Four,
                8 => alphazero::ChessHistory::Eight,
                _ => unreachable!(),
            })
        );
    }

    #[test]
    fn chess_session_views_and_native_moves_preserve_all_history_lengths() {
        assert_chess_session::<1>();
        assert_chess_session::<4>();
        assert_chess_session::<8>();
    }

    #[test]
    fn opening_a_missing_run_does_not_create_it() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("engine-zoo-missing-run-{unique}"));
        assert!(open_existing_run(&root, "chess").is_err());
        assert!(!root.exists());
    }
}

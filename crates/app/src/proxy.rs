use alphazero::{analyze_game_mcts, analyze_game_net, Analysis, AnalyzeConfig, AnalyzeMode};
use alphazero::{
    Batcher, BatcherConfig, ExperimentConfig, InferencePrecision, Mcts, RunDir, SearchConfig,
};
use anyhow::Result;
use axum::extract::State;
use axum::http::{header, Method, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use clap::ValueEnum;
use engine_core::agent::PolicyMode;
use engine_core::game::GameState;
use games::chess::notation;
use games::setup::{ChessSetup, Connect4Setup, GameSetup};
use games::{ChessGame, Connect4};
use serde::{Deserialize, Serialize};
use serde_json::json;
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
use http::*;
pub use http::{game_name, open_existing_run, resolve_model, run_dir};
use session_view::*;
use sessions::*;

use crate::evaluation_jobs::{CreateJob, EvaluationService};
use crate::visualization::render_chess_play_page;

const BATCH_TIMEOUT: Duration = Duration::from_millis(2);

fn puct_search(simulations: usize) -> SearchConfig {
    let mut search = search::PuctConfig::default();
    search.common.simulations = simulations.max(1);
    search.common.leaf_batch_size = 1;
    search.root_noise = None;
    SearchConfig::Puct(search)
}

fn batcher_config(wait_for_count: usize, timeout: Duration) -> BatcherConfig {
    BatcherConfig {
        preferred_batch_size: wait_for_count.max(1),
        max_batch_size: 256,
        max_wait: timeout,
        max_queued_states: 4096,
    }
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
    evaluations: EvaluationService,
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
    chess_history: Option<alphazero::ChessHistoryLength>,
}

pub async fn serve(cfg: ServeConfig) -> Result<()> {
    let evaluations = EvaluationService::new(cfg.run_dir.clone())?;
    let state = AppState {
        game: cfg.game,
        run_dir: cfg.run_dir,
        device: Device::cuda_if_available(),
        sessions: Arc::new(Mutex::new(Vec::new())),
        next_session: Arc::new(AtomicU64::new(1)),
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
                1 => alphazero::ChessHistoryLength::One,
                4 => alphazero::ChessHistoryLength::Four,
                8 => alphazero::ChessHistoryLength::Eight,
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
                1 => alphazero::ChessHistoryLength::One,
                4 => alphazero::ChessHistoryLength::Four,
                8 => alphazero::ChessHistoryLength::Eight,
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

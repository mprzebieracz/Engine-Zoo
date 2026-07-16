use algorithms::alphazero::{
    Batcher, InferencePrecision, Mcts, MctsConfig, MctsVariant, RunArchitecture, RunConfig, RunDir,
};
use algorithms::analysis::{
    analyze_game, analyze_game_mcts_with_repetitions, analyze_game_net, Analysis, AnalyzeConfig,
    AnalyzeMode,
};
use anyhow::Result;
use axum::extract::State;
use axum::http::{header, Method, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use clap::ValueEnum;
use engine_core::agent::PolicyMode;
use engine_core::game::Game;
use games::chess::notation;
use games::setup::{ChessSetup, Connect4Setup, GameSetup};
use games::{ChessAzGame, ChessGame, Connect4};
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
#[cfg(test)]
use analysis::chess_az_state;
use http::*;
pub use http::{game_name, open_existing_run, resolve_model, run_dir};
use session_view::*;
use sessions::*;

use crate::evaluation_jobs::{CreateJob, EvaluationService};
use crate::visualization::render_chess_play_page;

const BATCH_TIMEOUT: Duration = Duration::from_millis(2);

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
    Chess(SessionState<ChessGame>),
    ChessAzV2(Box<ChessAzV2Session>),
    Connect4(SessionState<Connect4>),
}

/// The authoritative v2 game owns both full-game adjudication and its bounded,
/// allocation-free neural history.
struct ChessAzV2Session {
    id: u64,
    game: ChessAzGameKind,
    moves: Vec<String>,
    san_moves: Vec<String>,
    model: String,
    human_turn: bool,
    simulations: usize,
    wait_for_count: usize,
}

enum ChessAzGameKind {
    H1(Box<ChessAzGame<1>>),
    H4(Box<ChessAzGame<4>>),
    H8(Box<ChessAzGame<8>>),
}

impl ChessAzGameKind {
    fn new(history: usize, initial: games::chess::ChessPosition) -> Result<Self> {
        match history {
            1 => Ok(Self::H1(Box::new(ChessAzGame::new(initial)))),
            4 => Ok(Self::H4(Box::new(ChessAzGame::new(initial)))),
            8 => Ok(Self::H8(Box::new(ChessAzGame::new(initial)))),
            _ => anyhow::bail!("chess-az-v2 history must be 1, 4, or 8, got {history}"),
        }
    }

    fn parse_move(&self, mv: &str) -> Option<engine_core::game::Action> {
        match self {
            Self::H1(state) => state.parse_move(mv),
            Self::H4(state) => state.parse_move(mv),
            Self::H8(state) => state.parse_move(mv),
        }
    }

    fn step(&mut self, action: engine_core::game::Action) {
        match self {
            Self::H1(state) => state.step(action),
            Self::H4(state) => state.step(action),
            Self::H8(state) => state.step(action),
        }
    }

    fn format_action(&self, action: engine_core::game::Action) -> String {
        match self {
            Self::H1(state) => state.format_action(action),
            Self::H4(state) => state.format_action(action),
            Self::H8(state) => state.format_action(action),
        }
    }

    fn san_for_action(&self, action: engine_core::game::Action) -> String {
        match self {
            Self::H1(game) => game.san_for_action(action),
            Self::H4(game) => game.san_for_action(action),
            Self::H8(game) => game.san_for_action(action),
        }
    }

    fn is_terminal(&self) -> bool {
        match self {
            Self::H1(game) => game.is_terminal(),
            Self::H4(game) => game.is_terminal(),
            Self::H8(game) => game.is_terminal(),
        }
    }

    fn reward(&self) -> f32 {
        match self {
            Self::H1(game) => game.reward(),
            Self::H4(game) => game.reward(),
            Self::H8(game) => game.reward(),
        }
    }

    fn board_text(&self) -> String {
        match self {
            Self::H1(game) => game.position().to_string(),
            Self::H4(game) => game.position().to_string(),
            Self::H8(game) => game.position().to_string(),
        }
    }

    fn legal_moves(&self) -> Vec<(engine_core::game::Action, String)> {
        match self {
            Self::H1(game) => legal_moves_for(&game.search_state()),
            Self::H4(game) => legal_moves_for(&game.search_state()),
            Self::H8(game) => legal_moves_for(&game.search_state()),
        }
    }
}

fn legal_moves_for<G: Game>(game: &G) -> Vec<(engine_core::game::Action, String)> {
    game.legal_actions()
        .map(|action| (action, game.format_action(action)))
        .collect()
}

struct SessionState<G: Game> {
    id: u64,
    game: G,
    moves: Vec<String>,
    san_moves: Vec<String>,
    model: String,
    human_turn: bool,
    simulations: usize,
    wait_for_count: usize,
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

    #[test]
    fn analyze_accepts_query_and_post_only() {
        let query = Method::from_bytes(b"QUERY").unwrap();
        assert!(is_analyze_method(&query));
        assert!(is_analyze_method(&Method::POST));
        assert!(!is_analyze_method(&Method::GET));
    }

    #[test]
    fn v2_analysis_state_replays_request_history_for_all_supported_lengths() {
        let position = ChessSetup {
            fen: None,
            moves: vec!["e2e4".into(), "e7e5".into(), "g1f3".into(), "b8c6".into()],
        };

        let h1 = chess_az_state::<1>(&position).unwrap();
        assert_eq!(h1.search_state().board(), h1.board());
        let h4 = chess_az_state::<4>(&position).unwrap();
        assert_eq!(h4.search_state().board(), h4.board());
        let h8 = chess_az_state::<8>(&position).unwrap();
        assert_eq!(h8.search_state().board(), h8.board());
    }

    #[test]
    fn v2_analysis_state_accepts_a_fen_without_history() {
        let position = ChessSetup {
            fen: Some("8/8/8/8/8/8/8/K6k b - - 0 1".into()),
            moves: Vec::new(),
        };
        let game = chess_az_state::<4>(&position).unwrap();
        assert_eq!(game.search_state().board(), game.board());
    }

    #[test]
    fn v2_session_replays_and_steps_for_all_supported_histories() {
        let position = ChessSetup {
            fen: None,
            moves: vec!["e2e4".into(), "e7e5".into(), "g1f3".into()],
        };
        for history in [1, 4, 8] {
            let mut session =
                create_chess_az_v2_session(1, position.clone(), "best".into(), true, 1, 1, history)
                    .unwrap();
            play_chess_az_v2_human_turn(&mut session, "b8c6").unwrap();
            assert_eq!(session.moves.last(), Some(&"b8c6".to_owned()));
        }
    }

    #[test]
    fn opening_a_missing_run_does_not_create_it() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("engine-zoo-missing-run-{unique}"));
        assert!(open_existing_run::<ChessGame>(&root).is_err());
        assert!(!root.exists());
    }
}

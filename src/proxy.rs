use crate::alphazero::{Batcher, Mcts, MctsConfig, RunConfig, RunDir};
use crate::analysis::{analyze_position, Analysis, AnalyzeConfig, AnalyzeMode};
use crate::game::Game;
use crate::games::{ChessGame, Connect4};
use crate::position::{ChessPosition, Connect4Position, PositionGame, PositionSpec};
use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tch::Device;

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

#[derive(Clone, Debug, Deserialize)]
pub struct AnalyzeRequest {
    pub position: PositionSpec,
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
    position: Option<PositionSpec>,
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
}

enum LiveSession {
    Chess(SessionState<ChessGame>),
    Connect4(SessionState<Connect4>),
}

struct SessionState<G: Game> {
    id: u64,
    game: G,
    model: String,
    human_turn: bool,
    simulations: usize,
    wait_for_count: usize,
}

pub async fn serve(cfg: ServeConfig) -> Result<()> {
    let state = AppState {
        game: cfg.game,
        run_dir: cfg.run_dir,
        device: Device::cuda_if_available(),
        sessions: Arc::new(Mutex::new(Vec::new())),
        next_session: Arc::new(AtomicU64::new(1)),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/runs/status", get(run_status))
        .route("/runs/checkpoints", get(checkpoints))
        .route("/analyze", post(analyze_http))
        .route("/sessions", post(create_session))
        .route("/sessions/{id}", get(get_session))
        .route("/sessions/{id}/move", post(session_move))
        .with_state(state);
    println!("engine-zoo serving {} on {}", game_name(cfg.game), cfg.bind);
    let listener = tokio::net::TcpListener::bind(cfg.bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
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
        (GameKind::Chess, PositionSpec::Chess(position)) => {
            let (_, run_cfg) = open_existing_run::<ChessGame>(&run_dir)?;
            let weights = resolve_model(&run_dir, &req.model);
            analyze_position::<ChessGame>(&run_cfg.net, &weights, &position, device, &cfg)
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

pub fn run_dir(game: GameKind, run_dir: Option<PathBuf>) -> PathBuf {
    run_dir.unwrap_or_else(|| PathBuf::from("runs").join(game_name(game)))
}

pub fn resolve_model(run_dir: &Path, model: &str) -> PathBuf {
    match model {
        "best" => run_dir.join("best.safetensors"),
        "candidate" => run_dir.join("candidate.safetensors"),
        other => {
            if let Ok(idx) = other.parse::<u32>() {
                run_dir
                    .join("checkpoints")
                    .join(format!("ckpt_{idx:04}.safetensors"))
            }
            else {
                let path = PathBuf::from(other);
                if path.is_absolute() {
                    path
                }
                else if other.starts_with("ckpt_") {
                    run_dir.join("checkpoints").join(path)
                }
                else {
                    run_dir.join(path)
                }
            }
        }
    }
}

pub fn open_existing_run<G: Game>(root: &Path) -> Result<(RunDir, RunConfig)> {
    let (run, cfg) = RunDir::open_or_create(root, || {
        panic!(
            "no run found at {}; train first or pass --run-dir",
            root.display()
        )
    })?;
    anyhow::ensure!(
        cfg.game == G::NAME,
        "run dir {} holds a {} run, not {}",
        root.display(),
        cfg.game,
        G::NAME
    );
    Ok((run, cfg))
}

pub fn game_name(game: GameKind) -> &'static str {
    match game {
        GameKind::Connect4 => Connect4::NAME,
        GameKind::Chess => ChessGame::NAME,
    }
}

fn default_model() -> String {
    "best".into()
}

fn default_simulations() -> usize {
    800
}

fn default_wait_for_count() -> usize {
    1
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

async fn run_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let metrics = state.run_dir.join("metrics.jsonl");
    let latest = fs::read_to_string(metrics)
        .ok()
        .and_then(|s| s.lines().last().map(str::to_owned));
    Json(json!({
        "game": game_name(state.game),
        "latest_metrics": latest,
    }))
}

async fn checkpoints(State(state): State<AppState>) -> Json<serde_json::Value> {
    let dir = state.run_dir.join("checkpoints");
    let mut paths = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".safetensors") {
                    paths.push(name.to_owned());
                }
            }
        }
    }
    paths.sort();
    Json(json!({ "checkpoints": paths }))
}

async fn analyze_http(
    State(state): State<AppState>,
    Json(req): Json<AnalyzeRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let result = tokio::task::spawn_blocking(move || {
        analyze_request(state.game, state.run_dir, req, state.device)
    })
    .await;
    match result {
        Ok(Ok(analysis)) => (StatusCode::OK, Json(json!(analysis))),
        Ok(Err(err)) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": err.to_string() })),
        ),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": err.to_string() })),
        ),
    }
}

async fn create_session(
    State(state): State<AppState>,
    Json(req): Json<CreateSessionRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    match create_session_inner(&state, req) {
        Ok(value) => (StatusCode::OK, Json(value)),
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": err.to_string() })),
        ),
    }
}

async fn get_session(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<u64>,
) -> (StatusCode, Json<serde_json::Value>) {
    let sessions = state.sessions.lock().unwrap();
    let Some(session) = sessions.iter().find(|s| session_id(s) == id)
    else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "unknown session" })),
        );
    };
    (StatusCode::OK, Json(session_view(session)))
}

async fn session_move(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<u64>,
    Json(req): Json<MoveRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let result = tokio::task::spawn_blocking(move || session_move_inner(state, id, req)).await;
    match result {
        Ok(Ok(value)) => (StatusCode::OK, Json(value)),
        Ok(Err(err)) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": err.to_string() })),
        ),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": err.to_string() })),
        ),
    }
}

fn create_session_inner(state: &AppState, req: CreateSessionRequest) -> Result<serde_json::Value> {
    let id = state.next_session.fetch_add(1, Ordering::Relaxed);
    let mut session = match state.game {
        GameKind::Chess => {
            let position = match req.position {
                Some(PositionSpec::Chess(position)) => position,
                Some(_) => anyhow::bail!("session position does not match chess"),
                None => ChessPosition::default(),
            };
            LiveSession::Chess(SessionState {
                id,
                game: ChessGame::from_position(&position)?,
                model: req.model,
                human_turn: !req.engine_first,
                simulations: req.simulations,
                wait_for_count: req.wait_for_count,
            })
        }
        GameKind::Connect4 => {
            let position = match req.position {
                Some(PositionSpec::Connect4(position)) => position,
                Some(_) => anyhow::bail!("session position does not match connect4"),
                None => Connect4Position::default(),
            };
            LiveSession::Connect4(SessionState {
                id,
                game: Connect4::from_position(&position)?,
                model: req.model,
                human_turn: !req.engine_first,
                simulations: req.simulations,
                wait_for_count: req.wait_for_count,
            })
        }
    };
    if !session_human_turn(&session) {
        play_engine_turn(&state.run_dir, state.device, &mut session)?;
    }
    let value = session_view(&session);
    state.sessions.lock().unwrap().push(session);
    Ok(value)
}

fn session_move_inner(state: AppState, id: u64, req: MoveRequest) -> Result<serde_json::Value> {
    let mut sessions = state.sessions.lock().unwrap();
    let session = sessions
        .iter_mut()
        .find(|s| session_id(s) == id)
        .ok_or_else(|| anyhow::anyhow!("unknown session"))?;
    match session {
        LiveSession::Chess(s) => play_human_turn(&mut s.game, &mut s.human_turn, &req.mv)?,
        LiveSession::Connect4(s) => play_human_turn(&mut s.game, &mut s.human_turn, &req.mv)?,
    }
    if !session_terminal(session) && !session_human_turn(session) {
        play_engine_turn(&state.run_dir, state.device, session)?;
    }
    Ok(session_view(session))
}

fn play_human_turn<G: Game>(game: &mut G, human_turn: &mut bool, mv: &str) -> Result<()> {
    anyhow::ensure!(*human_turn, "not the human's turn");
    let action = game
        .parse_move(mv)
        .ok_or_else(|| anyhow::anyhow!("illegal or unparsable move {mv}"))?;
    game.step(action);
    *human_turn = false;
    Ok(())
}

fn play_engine_turn(run_dir: &Path, device: Device, session: &mut LiveSession) -> Result<()> {
    match session {
        LiveSession::Chess(s) => play_engine_turn_for::<ChessGame>(run_dir, device, s),
        LiveSession::Connect4(s) => play_engine_turn_for::<Connect4>(run_dir, device, s),
    }
}

fn play_engine_turn_for<G: Game>(
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
    let action = mcts.search(&session.game).best_action();
    session.game.step(action);
    session.human_turn = true;
    Ok(())
}

fn session_id(session: &LiveSession) -> u64 {
    match session {
        LiveSession::Chess(s) => s.id,
        LiveSession::Connect4(s) => s.id,
    }
}

fn session_human_turn(session: &LiveSession) -> bool {
    match session {
        LiveSession::Chess(s) => s.human_turn,
        LiveSession::Connect4(s) => s.human_turn,
    }
}

fn session_terminal(session: &LiveSession) -> bool {
    match session {
        LiveSession::Chess(s) => s.game.is_terminal(),
        LiveSession::Connect4(s) => s.game.is_terminal(),
    }
}

fn session_view(session: &LiveSession) -> serde_json::Value {
    match session {
        LiveSession::Chess(s) => session_view_for(s),
        LiveSession::Connect4(s) => session_view_for(s),
    }
}

fn session_view_for<G: Game>(session: &SessionState<G>) -> serde_json::Value {
    let legal_moves: Vec<_> = session
        .game
        .legal_actions()
        .map(|action| {
            json!({
                "action": action,
                "move": session.game.format_action(action),
            })
        })
        .collect();
    json!({
        "id": session.id,
        "board": session.game.to_string(),
        "human_turn": session.human_turn,
        "terminal": session.game.is_terminal(),
        "reward": session.game.reward(),
        "legal_moves": legal_moves,
    })
}

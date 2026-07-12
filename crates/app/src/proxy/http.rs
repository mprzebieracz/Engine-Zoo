use super::*;

pub(super) async fn evaluations_catalog(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::to_value(state.evaluations.catalog()).expect("catalog serializes"))
}

pub(super) async fn evaluations_jobs(
    State(state): State<AppState>,
) -> (StatusCode, Json<serde_json::Value>) {
    match state.evaluations.list() {
        Ok(jobs) => (StatusCode::OK, Json(json!(jobs))),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": err.to_string()})),
        ),
    }
}

pub(super) async fn create_evaluation_job(
    State(state): State<AppState>,
    Json(req): Json<CreateJob>,
) -> (StatusCode, Json<serde_json::Value>) {
    match state.evaluations.create(req) {
        Ok(job) => (StatusCode::ACCEPTED, Json(json!(job))),
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": err.to_string()})),
        ),
    }
}

pub(super) async fn get_evaluation_job(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    match state.evaluations.get(&id) {
        Ok(job) => (StatusCode::OK, Json(json!(job))),
        Err(err) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": err.to_string()})),
        ),
    }
}

pub(super) async fn get_evaluation_artifact(
    State(state): State<AppState>,
    axum::extract::Path((id, name)): axum::extract::Path<(String, String)>,
) -> Response {
    match state
        .evaluations
        .artifact(&id, &name)
        .and_then(|(path, content_type)| Ok((fs::read(path)?, content_type)))
    {
        Ok((bytes, content_type)) => {
            ([(header::CONTENT_TYPE, content_type)], bytes).into_response()
        }
        Err(err) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": err.to_string()})),
        )
            .into_response(),
    }
}

pub(super) async fn index(State(state): State<AppState>) -> Html<String> {
    Html(render_chess_play_page(
        state.game,
        &state.run_dir.display().to_string(),
    ))
}

pub fn run_dir(game: GameKind, run_dir: Option<PathBuf>) -> PathBuf {
    run_dir.unwrap_or_else(|| PathBuf::from("data/runs").join(game_name(game)))
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
    anyhow::ensure!(
        root.join("config.json").is_file(),
        "no run found at {}; train first or pass --run-dir",
        root.display()
    );
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

pub(super) fn default_model() -> String {
    "best".into()
}

pub(super) fn default_simulations() -> usize {
    800
}

pub(super) fn default_wait_for_count() -> usize {
    1
}

pub(super) async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

pub(super) async fn run_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let metrics = state.run_dir.join("metrics.jsonl");
    let latest = fs::read_to_string(metrics)
        .ok()
        .and_then(|s| s.lines().last().map(str::to_owned));
    Json(json!({
        "game": game_name(state.game),
        "latest_metrics": latest,
    }))
}

pub(super) async fn checkpoints(State(state): State<AppState>) -> Json<serde_json::Value> {
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

pub(super) async fn games_http() -> Json<serde_json::Value> {
    Json(json!({
        "games": [
            { "id": "chess", "name": "Chess", "frontend": "implemented" },
            { "id": "connect4", "name": "Connect4", "frontend": "coming_later" }
        ]
    }))
}

pub(super) async fn runs_http(State(state): State<AppState>) -> Json<serde_json::Value> {
    let run = state.run_dir.display().to_string();
    Json(json!({
        "game": game_name(state.game),
        "runs": [run],
    }))
}

pub(super) async fn analyze_http(
    State(state): State<AppState>,
    method: Method,
    Json(req): Json<AnalyzeRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if !is_analyze_method(&method) {
        return (
            StatusCode::METHOD_NOT_ALLOWED,
            Json(json!({ "error": "use QUERY or POST" })),
        );
    }
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

pub(super) fn is_analyze_method(method: &Method) -> bool {
    *method == Method::POST || method.as_str() == "QUERY"
}

pub(super) async fn create_session(
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

pub(super) async fn get_session(
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

pub(super) async fn session_move(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<u64>,
    Json(req): Json<MoveRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    match session_move_inner(&state, id, req) {
        Ok(value) => (StatusCode::OK, Json(value)),
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": err.to_string() })),
        ),
    }
}

pub(super) async fn session_engine_move(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<u64>,
) -> (StatusCode, Json<serde_json::Value>) {
    let result = tokio::task::spawn_blocking(move || session_engine_move_inner(state, id)).await;
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

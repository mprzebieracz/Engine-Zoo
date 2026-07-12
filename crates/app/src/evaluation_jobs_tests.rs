use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_run(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("engine-zoo-{label}-{unique}"));
    fs::create_dir_all(root.join("checkpoints")).unwrap();
    root
}

fn request(suite: Suite) -> CreateJob {
    CreateJob {
        candidate: "best".into(),
        suite,
        baseline: None,
        simulations: 400,
        rounds: 2,
        stockfish_nodes: 3_000,
    }
}

fn job(suite: Suite) -> Job {
    Job {
        id: "job-1".into(),
        candidate: "best".into(),
        suite,
        baseline: (suite == Suite::Arena).then(|| "candidate".into()),
        simulations: 123,
        rounds: 4,
        stockfish_nodes: 5_000,
        status: JobStatus::Queued,
        created_at: 1,
        updated_at: 1,
        error: None,
        artifacts: Vec::new(),
        result: None,
    }
}

fn bins(root: &Path) -> Bins {
    let value = |name: &str| root.join(name);
    Bins {
        puzzle: value("eval-puzzle").display().to_string(),
        puzzle_path: value("eval-puzzle"),
        stockfish: value("eval-stockfish").display().to_string(),
        stockfish_path: value("eval-stockfish"),
        arena: value("eval-arena").display().to_string(),
        arena_path: value("eval-arena"),
        fastchess: value("fastchess").display().to_string(),
        fastchess_path: value("fastchess"),
        uci: value("uci").display().to_string(),
        uci_path: value("uci"),
        stockfish_engine: value("stockfish").display().to_string(),
        stockfish_engine_path: value("stockfish"),
        openings: value("openings.epd"),
        puzzles: value("puzzles.jsonl"),
    }
}

#[test]
fn validation_rejects_unknown_inputs_before_tool_preflight() {
    let root = temp_run("validation");
    fs::write(root.join("best.safetensors"), []).unwrap();
    let service = EvaluationService::new(root.clone()).unwrap();

    let mut req = request(Suite::Puzzle);
    req.candidate = "../best.safetensors".into();
    assert!(service
        .validate(&req)
        .unwrap_err()
        .to_string()
        .contains("candidate"));

    let mut req = request(Suite::Puzzle);
    req.simulations = 0;
    assert!(service
        .validate(&req)
        .unwrap_err()
        .to_string()
        .contains("simulations"));

    let req = request(Suite::Arena);
    assert!(service
        .validate(&req)
        .unwrap_err()
        .to_string()
        .contains("baseline"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn recovery_persists_interrupted_state_and_keeps_completed_jobs() {
    let root = temp_run("recovery");
    let service = EvaluationService::new(root.clone()).unwrap();
    for (id, status) in [
        ("queued", JobStatus::Queued),
        ("done", JobStatus::Succeeded),
    ] {
        let dir = service.jobs_dir.join(id);
        fs::create_dir_all(&dir).unwrap();
        let mut stored = job(Suite::Puzzle);
        stored.id = id.into();
        stored.status = status;
        storage::write_job(&dir, &stored).unwrap();
    }

    service.recover().unwrap();
    assert!(matches!(
        service.get("queued").unwrap().status,
        JobStatus::Interrupted
    ));
    assert!(matches!(
        service.get("done").unwrap().status,
        JobStatus::Succeeded
    ));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn command_construction_is_suite_specific_and_deterministic() {
    let root = temp_run("commands");
    fs::write(root.join("best.safetensors"), []).unwrap();
    fs::write(root.join("candidate.safetensors"), []).unwrap();
    let service = EvaluationService::new(root.clone()).unwrap();
    let output = root.join("out");
    let puzzle = service
        .command_with_bins(&job(Suite::Puzzle), &output, bins(&root))
        .unwrap();
    assert_eq!(puzzle[0], root.join("eval-puzzle").display().to_string());
    assert!(puzzle
        .windows(2)
        .any(|v| v == ["--output", output.join("puzzles.jsonl").to_str().unwrap()]));

    let arena = service
        .command_with_bins(&job(Suite::Arena), &output, bins(&root))
        .unwrap();
    let baseline = arena.iter().position(|arg| arg == "--baseline").unwrap();
    assert_eq!(
        arena[baseline + 1],
        root.join("candidate.safetensors").display().to_string()
    );
    assert_eq!(arena.iter().filter(|arg| *arg == "--device").count(), 1);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn artifact_lookup_cannot_escape_the_job_directory() {
    let root = temp_run("artifacts");
    let service = EvaluationService::new(root.clone()).unwrap();
    let dir = service.jobs_dir.join("job-1");
    fs::create_dir_all(dir.join("artifacts")).unwrap();
    let mut stored = job(Suite::Puzzle);
    stored.artifacts.push("result.json".into());
    storage::write_job(&dir, &stored).unwrap();
    fs::write(dir.join("artifacts/result.json"), "{}").unwrap();

    let (_, content_type) = service.artifact("job-1", "result.json").unwrap();
    assert_eq!(content_type, "application/json");
    assert!(service.artifact("job-1", "../job.json").is_err());
    assert!(service.artifact("../job-1", "result.json").is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn job_directory_allocation_never_reuses_a_same_second_id() {
    let root = temp_run("job-id");
    let service = EvaluationService::new(root.clone()).unwrap();
    let now = storage::unix_now();
    let existing = service.jobs_dir.join(format!("{now}-1"));
    fs::create_dir(&existing).unwrap();
    service.next_id.store(1, Ordering::Relaxed);

    let (id, dir) = service.allocate_job_dir(now).unwrap();
    assert_eq!(id, format!("{now}-2"));
    assert!(dir.is_dir());
    assert!(existing.is_dir());
    fs::remove_dir_all(root).unwrap();
}

use anyhow::{Context, Result};
use serde_json::Value;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

mod model;
mod storage;
mod tools;

pub use model::*;
use storage::{artifact_names, read_json, safe_name, unix_now, validate_id, write_job};
use tools::{read_result, Bins};

const MAX_SIMULATIONS: usize = 10_000;
const MAX_ROUNDS: u32 = 20;
const MAX_STOCKFISH_NODES: usize = 100_000;

#[derive(Clone)]
pub struct EvaluationService {
    root: PathBuf,
    jobs_dir: PathBuf,
    next_id: Arc<AtomicU64>,
}

impl EvaluationService {
    pub fn new(root: PathBuf) -> Result<Self> {
        let jobs_dir = root.join("evaluations").join("jobs");
        fs::create_dir_all(&jobs_dir)?;
        let service = Self {
            root,
            jobs_dir,
            next_id: Arc::new(AtomicU64::new(1)),
        };
        service.recover()?;
        Ok(service)
    }

    pub fn catalog(&self) -> Catalog {
        Catalog {
            checkpoints: self.checkpoints(),
            suites: vec![
                SuiteInfo {
                    id: Suite::Puzzle,
                    label: "Puzzle",
                    defaults: serde_json::json!({"simulations": 400}),
                },
                SuiteInfo {
                    id: Suite::Stockfish,
                    label: "Stockfish",
                    defaults: serde_json::json!({"simulations": 800, "rounds": 2, "stockfish_nodes": 3000}),
                },
                SuiteInfo {
                    id: Suite::Arena,
                    label: "Arena",
                    defaults: serde_json::json!({"simulations": 800, "rounds": 2}),
                },
            ],
            preflight: Preflight {
                puzzle: self.tools_for(Suite::Puzzle),
                stockfish: self.tools_for(Suite::Stockfish),
                arena: self.tools_for(Suite::Arena),
            },
        }
    }

    pub fn list(&self) -> Result<Vec<Job>> {
        let mut jobs: Vec<Job> = Vec::new();
        for entry in fs::read_dir(&self.jobs_dir)? {
            let path = entry?.path().join("job.json");
            if path.is_file() {
                if let Ok(job) = read_json(&path) {
                    jobs.push(job);
                }
            }
        }
        jobs.sort_by_key(|j| std::cmp::Reverse(j.created_at));
        Ok(jobs)
    }

    pub fn get(&self, id: &str) -> Result<Job> {
        validate_id(id)?;
        read_json(&self.jobs_dir.join(id).join("job.json")).context("unknown evaluation job")
    }

    pub fn artifact(&self, id: &str, name: &str) -> Result<(PathBuf, String)> {
        let job = self.get(id)?;
        anyhow::ensure!(
            name == "stdout.log" || name == "stderr.log" || safe_name(name),
            "invalid artifact name"
        );
        anyhow::ensure!(
            job.artifacts.iter().any(|a| a == name) || name.ends_with(".log"),
            "artifact is not available"
        );
        let path = if name == "stdout.log" || name == "stderr.log" {
            self.jobs_dir.join(id).join(name)
        } else {
            self.jobs_dir.join(id).join("artifacts").join(name)
        };
        anyhow::ensure!(path.is_file(), "artifact not found");
        let content_type = if name.ends_with(".pgn") {
            "application/x-chess-pgn"
        } else if name.ends_with(".json") || name.ends_with(".jsonl") {
            "application/json"
        } else {
            "text/plain; charset=utf-8"
        };
        Ok((path, content_type.into()))
    }

    pub fn create(&self, req: CreateJob) -> Result<Job> {
        self.validate(&req)?;
        let now = unix_now();
        let (id, dir) = self.allocate_job_dir(now)?;
        let job = Job {
            id: id.clone(),
            candidate: req.candidate,
            suite: req.suite,
            baseline: req.baseline,
            simulations: req.simulations,
            rounds: req.rounds,
            stockfish_nodes: req.stockfish_nodes,
            status: JobStatus::Queued,
            created_at: now,
            updated_at: now,
            error: None,
            artifacts: vec!["stdout.log".into(), "stderr.log".into()],
            result: None,
        };
        fs::create_dir_all(dir.join("artifacts"))?;
        write_job(&dir, &job)?;
        let worker = self.clone();
        let worker_job = job.clone();
        tokio::spawn(async move {
            let _ = tokio::task::spawn_blocking(move || worker.run(worker_job)).await;
        });
        Ok(job)
    }

    /// Reserves a job directory before writing metadata. The sequence number
    /// resets on server restart, so an existing same-second directory must
    /// never be reused or overwritten.
    fn allocate_job_dir(&self, now: u64) -> Result<(String, PathBuf)> {
        loop {
            let id = format!("{}-{}", now, self.next_id.fetch_add(1, Ordering::Relaxed));
            let dir = self.jobs_dir.join(&id);
            match fs::create_dir(&dir) {
                Ok(()) => return Ok((id, dir)),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
    }

    fn validate(&self, req: &CreateJob) -> Result<()> {
        anyhow::ensure!(
            self.checkpoints().iter().any(|c| c == &req.candidate),
            "candidate is not a catalog checkpoint"
        );
        if req.suite == Suite::Arena {
            anyhow::ensure!(
                req.baseline.is_some(),
                "arena requires a baseline checkpoint"
            );
        }
        if let Some(baseline) = &req.baseline {
            anyhow::ensure!(
                self.checkpoints().iter().any(|c| c == baseline),
                "baseline is not a catalog checkpoint"
            );
        }
        anyhow::ensure!(
            (1..=MAX_SIMULATIONS).contains(&req.simulations),
            "simulations must be between 1 and {MAX_SIMULATIONS}"
        );
        anyhow::ensure!(
            (1..=MAX_ROUNDS).contains(&req.rounds),
            "rounds must be between 1 and {MAX_ROUNDS}"
        );
        anyhow::ensure!(
            (100..=MAX_STOCKFISH_NODES).contains(&req.stockfish_nodes),
            "stockfish_nodes must be between 100 and {MAX_STOCKFISH_NODES}"
        );
        let state = self.tools_for(req.suite);
        anyhow::ensure!(state.available, "{}", state.message);
        Ok(())
    }

    fn run(&self, mut job: Job) -> Result<()> {
        let dir = self.jobs_dir.join(&job.id);
        job.status = JobStatus::Running;
        job.updated_at = unix_now();
        write_job(&dir, &job)?;
        let output = dir.join("artifacts");
        let result = (|| -> Result<Value> {
            let stdout = File::create(dir.join("stdout.log"))?;
            let stderr = File::create(dir.join("stderr.log"))?;
            let args = self.command(&job, &output)?;
            let status = Command::new(&args[0])
                .args(&args[1..])
                .stdout(Stdio::from(stdout))
                .stderr(Stdio::from(stderr))
                .status()
                .context("starting evaluator")?;
            if !status.success() {
                anyhow::bail!("evaluator exited with {status}")
            }
            read_result(&job.suite, &output)
        })();
        if let Ok(artifacts) = artifact_names(&output) {
            job.artifacts.extend(artifacts);
        }
        job.artifacts.sort();
        job.artifacts.dedup();
        match result {
            Ok(result) => {
                job.status = JobStatus::Succeeded;
                job.result = Some(result);
            }
            Err(err) => {
                job.status = JobStatus::Failed;
                job.error = Some(err.to_string());
            }
        }
        job.updated_at = unix_now();
        write_job(&dir, &job)
    }

    fn command(&self, job: &Job, output: &Path) -> Result<Vec<String>> {
        self.command_with_bins(job, output, Bins::discover())
    }

    fn command_with_bins(&self, job: &Job, output: &Path, bins: Bins) -> Result<Vec<String>> {
        let candidate = self.checkpoint_path(&job.candidate)?;
        let run = self.root.display().to_string();
        let mut common = vec![
            "--run-dir".into(),
            run.clone(),
            "--checkpoint".into(),
            candidate.display().to_string(),
            "--simulations".into(),
            job.simulations.to_string(),
            "--device".into(),
            "cpu".into(),
        ];
        Ok(match job.suite {
            Suite::Puzzle => {
                common.splice(0..0, [bins.puzzle]);
                common.extend([
                    "--suite".into(),
                    bins.puzzles.display().to_string(),
                    "--output".into(),
                    output.join("puzzles.jsonl").display().to_string(),
                ]);
                common
            }
            Suite::Stockfish => vec![
                bins.stockfish,
                "--fastchess".into(),
                bins.fastchess,
                "--uci".into(),
                bins.uci,
                "--stockfish".into(),
                bins.stockfish_engine,
                "--run-dir".into(),
                run,
                "--candidate".into(),
                candidate.display().to_string(),
                "--openings".into(),
                bins.openings.display().to_string(),
                "--output-dir".into(),
                output.display().to_string(),
                "--simulations".into(),
                job.simulations.to_string(),
                "--rounds".into(),
                job.rounds.to_string(),
                "--stockfish-nodes".into(),
                job.stockfish_nodes.to_string(),
                "--device".into(),
                "cpu".into(),
            ],
            Suite::Arena => vec![
                bins.arena,
                "--fastchess".into(),
                bins.fastchess,
                "--uci".into(),
                bins.uci,
                "--run-dir".into(),
                run,
                "--candidate".into(),
                candidate.display().to_string(),
                "--baseline".into(),
                self.checkpoint_path(job.baseline.as_deref().unwrap())?
                    .display()
                    .to_string(),
                "--openings".into(),
                bins.openings.display().to_string(),
                "--output-dir".into(),
                output.display().to_string(),
                "--simulations".into(),
                job.simulations.to_string(),
                "--rounds".into(),
                job.rounds.to_string(),
                "--device".into(),
                "cpu".into(),
            ],
        })
    }

    fn checkpoint_path(&self, id: &str) -> Result<PathBuf> {
        let path = if id == "best" {
            self.root.join("best.safetensors")
        } else if id == "candidate" {
            self.root.join("candidate.safetensors")
        } else {
            self.root.join("checkpoints").join(id)
        };
        anyhow::ensure!(path.is_file(), "checkpoint is unavailable");
        Ok(path)
    }
    fn checkpoints(&self) -> Vec<String> {
        let mut out = Vec::new();
        for name in ["best.safetensors", "candidate.safetensors"] {
            if self.root.join(name).is_file() {
                out.push(name.trim_end_matches(".safetensors").into());
            }
        }
        if let Ok(entries) = fs::read_dir(self.root.join("checkpoints")) {
            for e in entries.flatten() {
                let n = e.file_name().to_string_lossy().into_owned();
                if n.starts_with("ckpt_") && n.ends_with(".safetensors") && e.path().is_file() {
                    out.push(n);
                }
            }
        }
        out.sort();
        out
    }
    fn recover(&self) -> Result<()> {
        for mut job in self.list()? {
            if matches!(job.status, JobStatus::Queued | JobStatus::Running) {
                job.status = JobStatus::Interrupted;
                job.error = Some("The server stopped before this job completed.".into());
                job.updated_at = unix_now();
                write_job(&self.jobs_dir.join(&job.id), &job)?;
            }
        }
        Ok(())
    }
    fn tools_for(&self, suite: Suite) -> ToolState {
        let b = Bins::discover();
        let (ok, missing) = match suite {
            Suite::Puzzle => (
                b.puzzle_path.is_file() && b.puzzles.is_file(),
                "eval-puzzle or puzzle suite",
            ),
            Suite::Stockfish => (
                b.stockfish_path.is_file()
                    && b.fastchess_path.is_file()
                    && b.uci_path.is_file()
                    && b.stockfish_engine_path.is_file()
                    && b.openings.is_file(),
                "Fastchess, Stockfish, UCI binary, or opening suite",
            ),
            Suite::Arena => (
                b.arena_path.is_file()
                    && b.fastchess_path.is_file()
                    && b.uci_path.is_file()
                    && b.openings.is_file(),
                "Fastchess, UCI binary, or opening suite",
            ),
        };
        ToolState {
            available: ok,
            message: if ok {
                "Ready".into()
            } else {
                format!("Missing {missing}.")
            },
        }
    }
}

#[cfg(test)]
#[path = "evaluation_jobs_tests.rs"]
mod tests;

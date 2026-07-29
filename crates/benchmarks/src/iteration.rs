//! Complete, reproducible training-iteration benchmarks.
//!
//! The benchmark owns all wall-clock measurement. Production code exposes its
//! existing durable iteration report and remains free of benchmark timers.

use crate::cli::IterationArgs;
use crate::device::{self, Precision};
use crate::environment;
use crate::report::{BenchmarkReport, BenchmarkSample, BenchmarkSummary};
use alphazero::{
    ExperimentConfig, InferencePrecision, IterationReport, RunDir, SearchBudget,
    SearchBudgetSchedule, TrainingRun,
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub fn full_training_iteration(args: &IterationArgs) -> Result<BenchmarkReport> {
    ensure!(args.samples > 0, "--samples must be positive");
    let device = device::select(args.device)?;
    let experiment = effective_experiment(args, device)?;
    let workload = workload(args, &experiment, device);

    for _ in 0..args.warmup_iterations {
        run_once(&experiment, device)?;
    }

    let samples = (0..args.samples)
        .map(|_| run_once(&experiment, device))
        .collect::<Result<Vec<_>>>()?;

    Ok(BenchmarkReport {
        schema_version: 1,
        benchmark: benchmark_name(&experiment).into(),
        started_at: environment::timestamp(),
        git: environment::git(),
        build: environment::build(),
        host: environment::host(),
        workload,
        summary: summarize(&samples),
        samples,
        reports: Vec::new(),
    })
}

fn benchmark_name(experiment: &ExperimentConfig) -> &'static str {
    if experiment.training.train_steps == 0 {
        "iteration.self-play"
    }
    else {
        "iteration.full-training"
    }
}

fn effective_experiment(
    args: &IterationArgs,
    execution_device: tch::Device,
) -> Result<ExperimentConfig> {
    let mut experiment = ExperimentConfig::read_toml(&args.experiment).with_context(|| {
        format!(
            "failed to load iteration experiment {}",
            args.experiment.display()
        )
    })?;

    if let Some(games) = args.games {
        experiment.self_play.num_games = games;
    }
    if let Some(train_steps) = args.train_steps {
        experiment.training.train_steps = train_steps;
    }
    if let Some(batch_size) = args.batch_size {
        experiment.training.batch_size = batch_size;
    }
    if let Some(micro_batch_size) = args.micro_batch_size {
        experiment.training.micro_batch_size = micro_batch_size;
    }
    if let Some(seed) = args.seed {
        experiment.seed = seed;
    }
    if let Some(precision) = args.precision {
        experiment.inference.precision = inference_precision(precision);
    }

    if experiment.inference.precision == InferencePrecision::Fp16 && !execution_device.is_cuda() {
        anyhow::bail!("the effective experiment requests fp16 inference, which requires CUDA; pass --device cuda or --precision fp32")
    }
    experiment.validate()?;
    Ok(experiment)
}

fn inference_precision(precision: Precision) -> InferencePrecision {
    match precision {
        Precision::Fp32 => InferencePrecision::Fp32,
        Precision::Fp16 => InferencePrecision::Fp16,
    }
}

fn workload(args: &IterationArgs, experiment: &ExperimentConfig, device: tch::Device) -> Value {
    json!({
        "artifact_name": args.name,
        "experiment_path": args.experiment,
        "effective_experiment": experiment,
        "device": device::name(device),
        "seed": experiment.seed,
        "measurement": {
            "samples": args.samples,
            "warmup_iterations": args.warmup_iterations,
            "cuda_synchronized_at_iteration_boundary": device.is_cuda(),
            "phase_note": phase_note(experiment)
        }
    })
}

fn phase_note(experiment: &ExperimentConfig) -> &'static str {
    if experiment.training.train_steps == 0 {
        "This self-play-only iteration is measured around TrainingRun::step. It excludes optimizer work while retaining checkpoint persistence, inference reload, and metrics persistence."
    }
    else {
        "The full iteration is measured around TrainingRun::step. Training subphases come from TrainMetrics. The remaining time combines self-play, checkpoint persistence, inference reload, and metrics persistence because production code intentionally has no benchmark timers."
    }
}

fn run_once(experiment: &ExperimentConfig, device: tch::Device) -> Result<BenchmarkSample> {
    let temporary = TemporaryRun::create()?;
    RunDir::initialize(temporary.path(), experiment.clone())?;

    let mut run = TrainingRun::open(temporary.path(), device)?;
    device::synchronize(device);
    let started = Instant::now();
    let report = run.step()?;
    device::synchronize(device);
    let elapsed_ns = started.elapsed().as_nanos();

    Ok(sample(report, elapsed_ns, experiment))
}

fn sample(
    report: IterationReport,
    elapsed_ns: u128,
    experiment: &ExperimentConfig,
) -> BenchmarkSample {
    let elapsed_seconds = elapsed_ns as f64 / 1_000_000_000.0;
    let training_seconds = report.training.as_ref().map_or(0.0, |metrics| {
        training_wall_seconds(metrics, experiment.training.batch_size)
    });
    let non_training_seconds = (elapsed_seconds - training_seconds).max(0.0);
    let positions_per_second = report.moves as f64 / elapsed_seconds.max(f64::EPSILON);
    let configured_simulations = configured_simulations(experiment, report.moves);
    let configured_simulations_per_second =
        configured_simulations.map(|count| count as f64 / elapsed_seconds.max(f64::EPSILON));
    let inference_states_per_second =
        report.inference.total_states as f64 / elapsed_seconds.max(f64::EPSILON);

    BenchmarkSample {
        elapsed_ns,
        operations: report.moves as u64,
        metrics: json!({
            "iteration": report.iteration,
            "games": report.games,
            "positions": report.moves,
            "positions_per_second": positions_per_second,
            "configured_simulations": configured_simulations,
            "configured_simulations_per_second": configured_simulations_per_second,
            "simulation_note": "Configured simulations are exact for this fixed PUCT baseline. The production iteration report does not expose completed simulation counts, so non-fixed schedules report null rather than an estimate.",
            "inference_states_per_second": inference_states_per_second,
            "replay_samples": report.replay_samples,
            "phase_breakdown": {
                "full_iteration_seconds": elapsed_seconds,
                "training_wall_seconds": training_seconds,
                "self_play_checkpoint_reload_and_persist_seconds": non_training_seconds,
                "checkpoint_reload_count": report.inference.reload_count,
                "checkpoint_reload_timing": "combined with self-play and persistence; separate timing would require production instrumentation"
            },
            "training": report.training,
            "batcher": report.inference,
            "batch_stats": batch_stats(&report.inference),
        }),
    }
}

fn training_wall_seconds(metrics: &alphazero::TrainMetrics, batch_size: usize) -> f64 {
    metrics.train_steps as f64 * batch_size as f64 / metrics.samples_per_second.max(f64::EPSILON)
}

fn configured_simulations(experiment: &ExperimentConfig, moves: usize) -> Option<u64> {
    let SearchBudgetSchedule::Fixed(SearchBudget::Puct { simulations }) =
        &experiment.self_play.budget_schedule
    else {
        return None;
    };
    u64::try_from(moves).ok()?.checked_mul(*simulations as u64)
}

fn batch_stats(stats: &alphazero::BatcherStats) -> Value {
    let average_inference_batch = stats.total_states as f64 / stats.inference_batches.max(1) as f64;
    json!({
        "submitted_batches": stats.submitted_batches,
        "submitted_states": stats.submitted_states,
        "inference_batches": stats.inference_batches,
        "average_inference_batch": average_inference_batch,
        "maximum_inference_batch": stats.lifetime_max_inference_batch,
        "partial_batches": stats.partial_batches,
        "coalesced_extra_requests": stats.coalesced_extra_requests,
        "queue_wait_seconds": stats.queue_wait_total.as_secs_f64(),
        "backend_execution_seconds": stats.backend_execution_total.as_secs_f64(),
        "reload_count": stats.reload_count,
    })
}

fn summarize(samples: &[BenchmarkSample]) -> BenchmarkSummary {
    let mut elapsed = samples
        .iter()
        .map(|sample| sample.elapsed_ns)
        .collect::<Vec<_>>();
    elapsed.sort_unstable();
    let repetitions = elapsed.len();
    let mean_ns = elapsed.iter().sum::<u128>() as f64 / repetitions.max(1) as f64;
    let variance = elapsed
        .iter()
        .map(|value| (*value as f64 - mean_ns).powi(2))
        .sum::<f64>()
        / repetitions.max(1) as f64;

    BenchmarkSummary {
        repetitions,
        total_operations: samples.iter().map(|sample| sample.operations).sum(),
        minimum_ns: elapsed.first().copied().unwrap_or(0),
        maximum_ns: elapsed.last().copied().unwrap_or(0),
        mean_ns,
        median_ns: elapsed.get(repetitions / 2).copied().unwrap_or(0),
        stddev_ns: variance.sqrt(),
    }
}

struct TemporaryRun {
    path: PathBuf,
}

impl TemporaryRun {
    fn create() -> Result<Self> {
        let root = std::env::temp_dir().join("engine-zoo-bench");
        fs::create_dir_all(&root)?;

        for sequence in 0..100 {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = root.join(format!(
                "iteration-{}-{nonce}-{sequence}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        anyhow::bail!("could not allocate a unique temporary training run directory")
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryRun {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.path) {
            eprintln!(
                "warning: failed to remove temporary benchmark run {}: {error}",
                self.path.display()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_baseline_config_is_valid() {
        let args = IterationArgs {
            experiment: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../benchmarks/configs/default-chess-h4-cuda-500.toml"),
            games: None,
            train_steps: None,
            batch_size: None,
            micro_batch_size: None,
            seed: None,
            device: crate::device::DeviceKind::Cpu,
            precision: Some(Precision::Fp32),
            warmup_iterations: 0,
            samples: 1,
            name: None,
            output: None,
            human: false,
        };
        let config = effective_experiment(&args, tch::Device::Cpu).unwrap();
        assert_eq!(config.self_play.num_games, 500);
        assert_eq!(config.training.train_steps, 80);
    }

    #[test]
    fn summary_uses_full_iteration_elapsed_time() {
        let summary = summarize(&[
            BenchmarkSample {
                elapsed_ns: 2,
                operations: 1,
                metrics: json!({}),
            },
            BenchmarkSample {
                elapsed_ns: 4,
                operations: 1,
                metrics: json!({}),
            },
        ]);
        assert_eq!(summary.median_ns, 4);
        assert_eq!(summary.total_operations, 2);
    }
}

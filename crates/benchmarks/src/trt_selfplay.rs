//! Fixed-checkpoint chess self-play measurements for native and TensorRT inference.
//!
//! This module deliberately lives in `engine-bench`: TensorRT is immutable for a
//! generation, so it cannot be passed to `TrainingRun`, whose inference service
//! must reload checkpoint weights after an optimizer step.

use crate::cli::{ChessInferenceBackend, ChessSelfPlayArgs};
use crate::report::{BenchmarkReport, BenchmarkSample, BenchmarkSummary};
use crate::{device, environment};
use alphazero::{
    ChessCanonicalSelfPlayDomain, ChessHistory, DomainSelfPlayWorkerFactory, ExperimentConfig,
    InferenceEngine, InferenceService, InferenceSource, ReplayBuffer, SelfPlayCoordinator,
    SelfPlayEpoch,
};
use anyhow::{ensure, Context, Result};
use serde_json::json;
use std::time::Instant;

pub fn chess_fixed_checkpoint(args: &ChessSelfPlayArgs) -> Result<BenchmarkReport> {
    ensure!(args.samples > 0, "--samples must be positive");

    let device = device::select(args.device)?;
    ensure!(device.is_cuda(), "chess-self-play requires CUDA");

    let experiment = effective_experiment(args)?;
    let samples = (0..args.samples)
        .map(|_| run_once(args, &experiment, device))
        .collect::<Result<Vec<_>>>()?;

    Ok(BenchmarkReport {
        schema_version: 1,
        benchmark: "chess.fixed-checkpoint-self-play".into(),
        started_at: environment::timestamp(),
        git: environment::git(),
        build: environment::build(),
        host: environment::host(),
        workload: json!({
            "experiment_path": args.experiment,
            "effective_experiment": experiment,
            "backend": backend_name(args.backend),
            "checkpoint": args.checkpoint,
            "tensor_rt_module": args.tensor_rt_module,
            "measurement": {
                "same_seed_per_backend": true,
                "fresh_service_and_replay_per_sample": true,
                "cuda_synchronized_at_measurement_boundary": true,
                "p95_request_latency": "unavailable: BatcherStats exposes aggregate and maximum queue/backend durations, not a latency histogram"
            }
        }),
        summary: summarize(&samples),
        samples,
        reports: Vec::new(),
    })
}

fn effective_experiment(args: &ChessSelfPlayArgs) -> Result<ExperimentConfig> {
    let mut experiment = ExperimentConfig::read_toml(&args.experiment).with_context(|| {
        format!(
            "failed to load chess self-play experiment {}",
            args.experiment.display()
        )
    })?;

    ensure!(
        experiment.model.chess_history() == Some(ChessHistory::Four),
        "chess-self-play currently requires the canonical H4 chess model"
    );

    if let Some(games) = args.games {
        experiment.self_play.num_games = games;
    }
    if let Some(seed) = args.seed {
        experiment.seed = seed;
    }

    match args.backend {
        ChessInferenceBackend::Native => {
            ensure!(
                args.checkpoint.is_some(),
                "--checkpoint is required for --backend native"
            );
            experiment.inference.engine = InferenceEngine::Native;
            experiment.inference.tensor_rt_module = None;
        }
        ChessInferenceBackend::TensorRt => {
            let module = args
                .tensor_rt_module
                .clone()
                .context("--tensor-rt-module is required for --backend tensor-rt")?;
            experiment.inference.engine = InferenceEngine::TensorRtTorchScript;
            experiment.inference.tensor_rt_module = Some(module);
        }
    }

    experiment.validate()?;
    Ok(experiment)
}

fn run_once(
    args: &ChessSelfPlayArgs,
    experiment: &ExperimentConfig,
    execution_device: tch::Device,
) -> Result<BenchmarkSample> {
    let source = match args.backend {
        ChessInferenceBackend::Native => InferenceSource::Checkpoint(
            args.checkpoint
                .as_deref()
                .expect("validated native checkpoint"),
        ),
        ChessInferenceBackend::TensorRt => InferenceSource::TensorRtModule(
            args.tensor_rt_module
                .as_deref()
                .expect("validated TensorRT module"),
        ),
    };
    let service = InferenceService::load(
        &experiment.model,
        source,
        execution_device,
        &experiment.inference,
    )?;
    let factory = DomainSelfPlayWorkerFactory::<ChessCanonicalSelfPlayDomain<4>>::new(
        &service,
        experiment.self_play.clone(),
    )?;
    let replay = ReplayBuffer::new(experiment.replay.capacity, experiment.model.action_size());
    let coordinator = SelfPlayCoordinator::new(experiment.self_play.clone(), experiment.seed)?;

    device::synchronize(execution_device);
    let started = Instant::now();
    let stats = coordinator.run(
        &factory,
        &replay,
        SelfPlayEpoch {
            model_generation: 0,
            first_game_id: 0,
        },
    )?;
    device::synchronize(execution_device);
    let elapsed_ns = started.elapsed().as_nanos();
    let seconds = elapsed_ns as f64 / 1_000_000_000.0;
    let batcher = service.stats();

    Ok(BenchmarkSample {
        elapsed_ns,
        operations: stats.moves as u64,
        metrics: json!({
            "backend": backend_name(args.backend),
            "games": stats.games,
            "positions": stats.moves,
            "games_per_second": stats.games as f64 / seconds.max(f64::EPSILON),
            "positions_per_second": stats.moves as f64 / seconds.max(f64::EPSILON),
            "backend_evaluations": stats.backend_evaluations,
            "backend_evaluations_per_second": stats.backend_evaluations as f64 / seconds.max(f64::EPSILON),
            "completed_simulations": stats.completed_simulations,
            "replay_samples": replay.len(),
            "batcher": {
                "inference_batches": batcher.inference_batches,
                "total_states": batcher.total_states,
                "states_per_second": batcher.total_states as f64 / seconds.max(f64::EPSILON),
                "average_global_batch": batcher.total_states as f64 / batcher.inference_batches.max(1) as f64,
                "maximum_global_batch": batcher.lifetime_max_inference_batch,
                "queue_wait_seconds": batcher.queue_wait_total.as_secs_f64(),
                "queue_wait_max_seconds": batcher.queue_wait_max.as_secs_f64(),
                "backend_execution_seconds": batcher.backend_execution_total.as_secs_f64(),
                "backend_execution_max_seconds": batcher.backend_execution_max.as_secs_f64(),
                "p95_request_latency_seconds": null,
                "p95_request_latency_note": "unavailable: BatcherStats has no latency histogram; max queue and backend durations are reported instead"
            }
        }),
    })
}

fn backend_name(backend: ChessInferenceBackend) -> &'static str {
    match backend {
        ChessInferenceBackend::Native => "native",
        ChessInferenceBackend::TensorRt => "tensor-rt-torch-script",
    }
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
        .map(|&value| (value as f64 - mean_ns).powi(2))
        .sum::<f64>()
        / repetitions.max(1) as f64;

    BenchmarkSummary {
        repetitions,
        total_operations: samples.iter().map(|sample| sample.operations).sum(),
        minimum_ns: elapsed.first().copied().unwrap_or_default(),
        maximum_ns: elapsed.last().copied().unwrap_or_default(),
        mean_ns,
        median_ns: elapsed[repetitions / 2],
        stddev_ns: variance.sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_uses_the_upper_median() {
        let samples = vec![
            BenchmarkSample {
                elapsed_ns: 4,
                operations: 2,
                metrics: json!({}),
            },
            BenchmarkSample {
                elapsed_ns: 1,
                operations: 3,
                metrics: json!({}),
            },
        ];
        let summary = summarize(&samples);
        assert_eq!(summary.median_ns, 4);
        assert_eq!(summary.total_operations, 5);
    }
}

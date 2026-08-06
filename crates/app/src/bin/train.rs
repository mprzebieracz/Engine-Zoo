use alphazero::{ExperimentConfig, InferenceEngine, NextInference, RunDir, RunLimit, TrainingRun};
use anyhow::{Context, Result};
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use engine_app::tensor_rt::{
    compile_latest_tensor_rt, ensure_latest_checkpoint, TensorRtBatchShapes, TensorRtCompiler,
};
use std::path::PathBuf;
use std::time::Instant;
use tch::Device;

#[derive(Clone, Copy, ValueEnum)]
enum DeviceKind {
    Auto,
    Cuda,
    Cpu,
}

#[derive(Parser)]
#[command(about = "AlphaZero self-play and training")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate an immutable experiment TOML and initialize an empty run.
    Init {
        #[arg(long)]
        experiment: PathBuf,
        #[arg(long)]
        run_dir: PathBuf,
    },
    /// Resume an initialized run. Runtime flags do not change its experiment.
    Run(RunArgs),
    /// Print the immutable experiment and current mutable run state.
    Inspect {
        #[arg(long)]
        run_dir: PathBuf,
    },
}

#[derive(Parser)]
struct RunArgs {
    #[arg(long)]
    run_dir: PathBuf,
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    #[arg(long)]
    forever: bool,
    #[arg(long, value_enum, default_value_t = DeviceKind::Cuda)]
    device: DeviceKind,
    /// Use the raw TensorRT engine path with a persistent builder timing cache.
    ///
    /// Overrides a `tensor-rt-torch-script` experiment at runtime (does not
    /// rewrite `experiment.toml`): compiles via `compile_tensorrt_raw.py` into
    /// `model.raw.engine` and stores timings in `<run-dir>/tensorrt-timing.cache`
    /// unless `--tensor-rt-timing-cache` is set. Requires the `raw-tensorrt`
    /// Cargo feature (on by default for this binary).
    #[arg(long)]
    cache: bool,
    #[command(flatten)]
    tensor_rt: TensorRtRunArgs,
}

#[derive(ClapArgs)]
struct TensorRtRunArgs {
    /// Python executable in the matching Torch-TensorRT / TensorRT environment.
    #[arg(long)]
    tensor_rt_python: Option<PathBuf>,
    /// Existing TensorRT compiler script. Defaults to this checkout's script.
    #[arg(long)]
    tensor_rt_compiler: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    tensor_rt_min_batch_size: usize,
    #[arg(long)]
    tensor_rt_opt_batch_size: Option<usize>,
    #[arg(long)]
    tensor_rt_max_batch_size: Option<usize>,
    /// Builder timing-cache path. Set automatically under `--cache`.
    #[arg(long)]
    tensor_rt_timing_cache: Option<PathBuf>,
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Init {
            experiment,
            run_dir,
        } => initialize(experiment, run_dir),
        Command::Run(args) => run(args),
        Command::Inspect { run_dir } => inspect(run_dir),
    }
}

fn initialize(experiment_path: PathBuf, run_dir: PathBuf) -> Result<()> {
    let experiment = ExperimentConfig::read_toml(&experiment_path)?;
    RunDir::initialize(&run_dir, experiment)?;

    println!("initialized {}", run_dir.display());
    Ok(())
}

fn run(args: RunArgs) -> Result<()> {
    let device = select_device(args.device)?;
    let (_, mut experiment, state) = RunDir::open(&args.run_dir)?;
    let limit = if args.forever {
        RunLimit::Forever
    } else {
        RunLimit::Iterations(args.iterations)
    };
    let mut tensor_rt = args.tensor_rt;

    if args.cache {
        apply_cache_backend(&args.run_dir, &mut experiment, &mut tensor_rt)?;
    }

    if matches!(
        experiment.inference.engine,
        InferenceEngine::TensorRtTorchScript | InferenceEngine::TensorRtRaw
    ) {
        return run_tensor_rt(args.run_dir, device, limit, experiment, tensor_rt);
    }

    anyhow::ensure!(
        !args.cache,
        "--cache requires a TensorRT experiment (engine = tensor-rt-torch-script or tensor-rt-raw)"
    );

    print_startup_summary(&args.run_dir, device, &experiment, &state);
    let mut run = TrainingRun::open(&args.run_dir, device)?;
    run.run(limit)
}

fn apply_cache_backend(
    run_dir: &PathBuf,
    experiment: &mut ExperimentConfig,
    args: &mut TensorRtRunArgs,
) -> Result<()> {
    #[cfg(not(feature = "raw-tensorrt"))]
    {
        let _ = (run_dir, experiment, args);
        anyhow::bail!(
            "--cache requires building train with `--features raw-tensorrt` \
             (enabled by default for engine_app)"
        );
    }

    #[cfg(feature = "raw-tensorrt")]
    {
        anyhow::ensure!(
            matches!(
                experiment.inference.engine,
                InferenceEngine::TensorRtTorchScript | InferenceEngine::TensorRtRaw
            ),
            "--cache requires a TensorRT experiment (engine = tensor-rt-torch-script or tensor-rt-raw)"
        );

        experiment.inference.engine = InferenceEngine::TensorRtRaw;
        experiment.inference.tensor_rt_module = Some(PathBuf::from("model.raw.engine"));
        experiment.validate()?;

        if args.tensor_rt_compiler.is_none() {
            args.tensor_rt_compiler = Some(default_raw_tensor_rt_compiler_script());
        }
        if args.tensor_rt_timing_cache.is_none() {
            args.tensor_rt_timing_cache = Some(run_dir.join("tensorrt-timing.cache"));
        }

        println!(
            "cache backend: raw TensorRT + timing cache ({})",
            args.tensor_rt_timing_cache
                .as_ref()
                .expect("set above")
                .display()
        );
        Ok(())
    }
}

fn print_startup_summary(
    run_dir: &PathBuf,
    device: Device,
    experiment: &ExperimentConfig,
    state: &alphazero::RunState,
) {
    println!("\n===== TRAINING CONFIGURATION =====");
    println!("run directory: {}", run_dir.display());
    println!("device: {device:?}");
    println!(
        "starting model: generation {} | iteration {} | checkpoint {} | resume {:?}",
        state.model_generation,
        state.iteration,
        state.latest_checkpoint.as_deref().unwrap_or("new model"),
        state.resume_kind,
    );
    println!("model: {:?}", experiment.model);
    println!("seed: {}", experiment.seed);

    let self_play = &experiment.self_play;
    println!(
        "self-play: {} games | {} threads | max {} moves",
        self_play.num_games, self_play.threads, self_play.max_moves,
    );
    println!("search: {:?}", self_play.search);
    println!("budget schedule: {:?}", self_play.budget_schedule);

    let training = &experiment.training;
    println!(
        "training: {} steps | batch {} (micro {}) | lr {:.6} | weight decay {:.6}",
        training.train_steps,
        training.batch_size,
        training.micro_batch_size,
        training.lr,
        training.weight_decay,
    );
    println!(
        "loss: value weight {:.3} | replay reuse cap {:?}",
        training.value_loss_weight, training.max_replay_reuse_per_iteration,
    );

    let inference = &experiment.inference;
    println!(
        "inference: {:?} {:?} | batch {}-{} | wait {} ms | queue {}",
        inference.engine,
        inference.precision,
        inference.preferred_batch_size,
        inference.max_batch_size,
        inference.max_wait.milliseconds,
        inference.max_queued_states,
    );
    println!("==================================\n");
}

fn run_tensor_rt(
    run_dir: PathBuf,
    device: Device,
    limit: RunLimit,
    experiment: ExperimentConfig,
    args: TensorRtRunArgs,
) -> Result<()> {
    let compiler = tensor_rt_compiler(&experiment, args)?;
    let (run, _, mut state) = RunDir::open(&run_dir)?;

    ensure_latest_checkpoint(&run, &experiment, &mut state, device)?;
    print_startup_summary(&run_dir, device, &experiment, &state);
    println!("COMPILATION");
    compile_tensor_rt_generation(&run, &experiment, device, &compiler)?;
    let mut training_run = TrainingRun::open_with_experiment(&run_dir, experiment.clone(), device)?;
    let total_started = Instant::now();

    match limit {
        RunLimit::Iterations(iterations) => {
            for iteration in 1..=iterations {
                print_iteration_header(iteration);
                let iteration_started = Instant::now();
                let report = training_run.step()?;
                print_iteration_report(&report, iteration_started.elapsed());
                ensure_tensor_rt_recompile_required(&report.next_inference)?;
                println!("COMPILATION");
                compile_tensor_rt_generation(&run, &experiment, device, &compiler)?;
                training_run.reload_tensor_rt_inference()?;
            }
            println!(
                "TOTAL TRAINING TIME: {:.2}s",
                total_started.elapsed().as_secs_f64()
            );
        }
        RunLimit::Forever => {
            let mut iteration = 1;

            loop {
                print_iteration_header(iteration);
                let iteration_started = Instant::now();
                let report = training_run.step()?;
                print_iteration_report(&report, iteration_started.elapsed());
                ensure_tensor_rt_recompile_required(&report.next_inference)?;
                println!("COMPILATION");
                compile_tensor_rt_generation(&run, &experiment, device, &compiler)?;
                training_run.reload_tensor_rt_inference()?;
                iteration += 1;
            }
        }
    }

    Ok(())
}

fn print_iteration_header(iteration: usize) {
    println!("\n===== ITERATION {iteration} =====");
    println!("SELFPLAY");
}

fn print_iteration_report(report: &alphazero::IterationReport, elapsed: std::time::Duration) {
    println!("TRAINING");

    if let Some(training) = &report.training {
        println!(
            "steps: {}/{} | policy loss: {:.5} | value loss: {:.5} | samples/s: {:.1}",
            training.train_steps,
            training.configured_train_steps,
            training.policy_loss,
            training.value_loss,
            training.samples_per_second,
        );
        println!(
            "replay: {} fresh samples | reuse: {:.2}x | learning rate: {:.6}",
            training.fresh_replay_samples, training.replay_reuse, training.learning_rate,
        );
        println!(
            "timing: replay {:.2}s | H2D {:.2}s | forward/backward {:.2}s | optimizer {:.2}s",
            training.replay_sampling_seconds,
            training.host_to_device_seconds,
            training.forward_backward_seconds,
            training.optimizer_seconds,
        );
    } else {
        println!("no training steps configured");
    }

    println!(
        "iteration total: {} games | {} positions | {} replay samples",
        report.games, report.moves, report.replay_samples,
    );
    println!("iteration elapsed: {:.2}s", elapsed.as_secs_f64());
}

fn tensor_rt_compiler(
    experiment: &ExperimentConfig,
    args: TensorRtRunArgs,
) -> Result<TensorRtCompiler> {
    let python = args.tensor_rt_python.context(
        "TensorRT inference requires --tensor-rt-python pointing to the matching compiler environment",
    )?;
    let script = args.tensor_rt_compiler.unwrap_or_else(|| {
        if experiment.inference.engine == InferenceEngine::TensorRtRaw {
            default_raw_tensor_rt_compiler_script()
        } else {
            default_tensor_rt_compiler_script()
        }
    });
    let optimal = args
        .tensor_rt_opt_batch_size
        .unwrap_or(experiment.inference.preferred_batch_size);
    let max = args
        .tensor_rt_max_batch_size
        .unwrap_or(experiment.inference.max_batch_size);
    let batch_shapes = TensorRtBatchShapes {
        min: args.tensor_rt_min_batch_size,
        optimal,
        max,
    };
    batch_shapes.validate()?;

    Ok(TensorRtCompiler {
        python,
        script,
        batch_shapes,
        precision: experiment.inference.precision,
        timing_cache: args.tensor_rt_timing_cache,
    })
}

fn compile_tensor_rt_generation(
    run_dir: &RunDir,
    experiment: &ExperimentConfig,
    device: Device,
    compiler: &TensorRtCompiler,
) -> Result<()> {
    let started = Instant::now();
    compile_latest_tensor_rt(run_dir, experiment, device, compiler)?;
    println!(
        "compiled TensorRT module in {:.3}s",
        started.elapsed().as_secs_f64()
    );

    Ok(())
}

fn ensure_tensor_rt_recompile_required(next: &NextInference) -> Result<()> {
    anyhow::ensure!(
        matches!(next, NextInference::TensorRtRecompileRequired { .. }),
        "TensorRT iteration did not request compilation of its next fixed-weight module"
    );

    Ok(())
}

fn default_tensor_rt_compiler_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("app manifest is nested under the workspace root")
        .join("scripts/compile_tensorrt.py")
}

fn default_raw_tensor_rt_compiler_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("app manifest is nested under the workspace root")
        .join("scripts/compile_tensorrt_raw.py")
}

fn inspect(run_dir: PathBuf) -> Result<()> {
    let (_, experiment, state) = RunDir::open(&run_dir)?;
    println!("{}", experiment.to_toml()?);
    println!("state:");
    println!("{}", serde_json::to_string_pretty(&state)?);
    Ok(())
}

fn select_device(kind: DeviceKind) -> Result<Device> {
    Ok(match kind {
        DeviceKind::Auto => Device::cuda_if_available(),
        DeviceKind::Cuda => {
            anyhow::ensure!(
                tch::Cuda::is_available(),
                "CUDA was requested but is unavailable"
            );
            Device::Cuda(0)
        }
        DeviceKind::Cpu => Device::Cpu,
    })
}

//! Deterministic workloads intended exclusively for CPU and GPU profilers.
//!
//! This crate deliberately contains no benchmark timing or production profiling hooks.

use alphazero::{
    build_optimizer, train, Action, Batcher, BatcherConfig, DomainSelfPlayWorkerFactory,
    ExperimentConfig, ModelSpec, Network, Outcome, ReplayBuffer, ReplaySample, RunDir,
    SampleMetadata, SelfPlayConfig, SelfPlayCoordinator, SelfPlayEpoch, StandardSelfPlayDomain,
    TrainConfig, TrainingRun, TrainingWeights,
};
use anyhow::{bail, ensure, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use games::Connect4;
use std::env;
use std::fmt::Display;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tch::{nn, Cuda, Device, Kind, Tensor};

#[derive(Parser)]
#[command(
    name = "engine-profile",
    about = "Deterministic profiling workloads and profiler command generation"
)]
struct Command {
    #[command(subcommand)]
    command: ProfileCommand,
}

#[derive(Subcommand)]
enum ProfileCommand {
    /// Execute a workload without collecting timings. Point a profiler at this command.
    Run(WorkloadArgs),
    /// Print a ready-to-run command after checking the requested profiler is installed.
    Command(CommandArgs),
    /// Check CUDA and external profiler availability.
    Requirements,
}

#[derive(Args, Clone)]
struct WorkloadArgs {
    /// Profiling workload to execute.
    #[arg(value_enum)]
    workload: Workload,

    /// Unmeasured workload passes before the captured work begins.
    #[arg(long, default_value_t = 0)]
    warmup: usize,

    /// Captured workload passes. Inference defaults to one pass of a 256-row batch;
    /// use a larger value only for a focused kernel profile.
    #[arg(long, default_value_t = 1)]
    iterations: usize,

    /// Number of games for the standalone self-play fixture and full iteration.
    #[arg(long, default_value_t = 200)]
    games: usize,

    /// Optimizer steps for the training fixture and full iteration.
    #[arg(long, default_value_t = 80)]
    training_steps: usize,

    /// Immutable experiment used by the full-iteration workload.
    #[arg(long, default_value = "experiments/benchmark-chess-canonical-h1.toml")]
    experiment: PathBuf,

    /// Directory for workload state and profiler artifacts. It must be fresh for `iteration`.
    #[arg(long, default_value = "artifacts/profiles")]
    artifact_dir: PathBuf,

    /// Execution device. CUDA is the default because production inference and training use it.
    #[arg(long, value_enum, default_value_t = ExecutionDevice::Cuda)]
    device: ExecutionDevice,
}

#[derive(Args)]
struct CommandArgs {
    #[arg(value_enum)]
    tool: ProfilingTool,

    #[command(flatten)]
    workload: WorkloadArgs,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Workload {
    /// Connect Four network forward passes on a fixed 256-position batch.
    Inference,
    /// Connect Four forward, backward, and optimizer steps over fixed replay data.
    Training,
    /// Connect Four self-play through the production batcher and MCTS worker path.
    #[value(name = "self-play")]
    SelfPlay,
    /// One complete production TrainingRun iteration using the benchmark chess experiment.
    Iteration,
}

impl Display for Workload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Inference => "inference",
            Self::Training => "training",
            Self::SelfPlay => "self-play",
            Self::Iteration => "iteration",
        })
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ExecutionDevice {
    Cuda,
    Cpu,
}

impl Display for ExecutionDevice {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        })
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProfilingTool {
    /// Linux perf record with call stacks for CPU attribution.
    Perf,
    /// cargo-flamegraph, which wraps perf and renders an SVG flamegraph.
    Flamegraph,
    /// NVIDIA Nsight Systems for CPU/GPU timeline and CUDA API tracing.
    Nsys,
    /// NVIDIA Nsight Compute for individual CUDA kernel analysis.
    Ncu,
}

impl ProfilingTool {
    fn program(self) -> &'static str {
        match self {
            Self::Perf => "perf",
            Self::Flamegraph => "cargo-flamegraph",
            Self::Nsys => "nsys",
            Self::Ncu => "ncu",
        }
    }

    fn install_hint(self) -> &'static str {
        match self {
            Self::Perf => "Install the linux-tools package matching the running kernel.",
            Self::Flamegraph => "Install it with `cargo install flamegraph`.",
            Self::Nsys => "Install NVIDIA Nsight Systems and add its `bin` directory to PATH.",
            Self::Ncu => "Install NVIDIA Nsight Compute and add its `bin` directory to PATH.",
        }
    }
}

fn main() -> Result<()> {
    let command = Command::parse();

    match command.command {
        ProfileCommand::Run(args) => run(args),
        ProfileCommand::Command(args) => print_command(args),
        ProfileCommand::Requirements => print_requirements(),
    }
}

fn run(args: WorkloadArgs) -> Result<()> {
    ensure!(args.iterations > 0, "iterations must be positive");
    ensure!(args.games > 0, "games must be positive");
    ensure!(args.training_steps > 0, "training steps must be positive");
    let device = resolve_device(args.device)?;
    prepare_artifact_dir(&args)?;
    tch::manual_seed(0);

    match args.workload {
        Workload::Inference => run_inference(device, args.warmup, args.iterations),
        Workload::Training => {
            run_training(device, args.warmup, args.iterations, args.training_steps)
        }
        Workload::SelfPlay => run_self_play(&args, device)?,
        Workload::Iteration => run_full_iteration(&args, device)?,
    }

    synchronize(device);
    println!(
        "Completed {} profiling workload on {} (warmup {}, iterations {}).",
        args.workload, args.device, args.warmup, args.iterations
    );
    Ok(())
}

fn prepare_artifact_dir(args: &WorkloadArgs) -> Result<()> {
    fs::create_dir_all(&args.artifact_dir)?;
    let manifest = serde_json::json!({
        "workload": args.workload.to_string(),
        "device": args.device.to_string(),
        "warmup_passes": args.warmup,
        "captured_passes": args.iterations,
        "games": args.games,
        "training_steps": args.training_steps,
        "experiment": args.experiment,
    });
    let path = args.artifact_dir.join("workload.json");
    fs::write(
        &path,
        format!("{}\n", serde_json::to_string_pretty(&manifest)?),
    )?;
    println!("Profiling artifacts: {}", args.artifact_dir.display());
    Ok(())
}

fn resolve_device(requested: ExecutionDevice) -> Result<Device> {
    match requested {
        ExecutionDevice::Cpu => Ok(Device::Cpu),
        ExecutionDevice::Cuda if Cuda::is_available() => {
            Cuda::manual_seed_all(0);
            Ok(Device::Cuda(0))
        }
        ExecutionDevice::Cuda => bail!(
            "CUDA is unavailable. Install a CUDA-enabled LibTorch build and NVIDIA driver, or pass `--device cpu` for CPU profiling."
        ),
    }
}

fn run_inference(device: Device, warmup: usize, iterations: usize) {
    let store = nn::VarStore::new(device);
    let network = Network::new(&store.root(), &connect4_model())
        .expect("fixed profiling model specification is valid");
    let states = Tensor::zeros([256, 1, 6, 7], (Kind::Float, device));

    for _ in 0..warmup {
        let _ = network.forward_t(&states, false);
    }
    synchronize(device);

    for _ in 0..iterations {
        let _ = network.forward_t(&states, false);
    }
}

fn run_training(device: Device, warmup: usize, iterations: usize, training_steps: usize) {
    let replay = fixed_replay();
    let store = nn::VarStore::new(device);
    let network = Network::new(&store.root(), &connect4_model())
        .expect("fixed profiling model specification is valid");
    let config = TrainConfig {
        batch_size: 4_096,
        micro_batch_size: 256,
        train_steps: training_steps,
        progress_every: 10,
        ..TrainConfig::default()
    };
    let mut optimizer =
        build_optimizer(&store, &config).expect("fixed profiling training configuration is valid");

    for step in 0..warmup + iterations {
        let metrics = train(
            &network,
            &mut optimizer,
            &replay,
            &alphazero::representation::Connect4AzRepresentation,
            device,
            &config,
            0,
            step as u64,
        );
        assert!(metrics.is_some(), "fixed profiling replay is populated");

        if step + 1 == warmup {
            synchronize(device);
        }
    }
}

fn fixed_replay() -> ReplayBuffer<Connect4> {
    let replay = ReplayBuffer::new(8_192, 7);
    replay.add((0..8_192).map(|game_id| ReplaySample {
        state: Connect4::default(),
        policy: vec![(Action::new(0), 1.0)].into(),
        outcome: Outcome::Draw,
        weights: TrainingWeights::default(),
        metadata: SampleMetadata {
            game_id: game_id as u64,
            ..Default::default()
        },
    }));
    replay
}

fn connect4_model() -> ModelSpec {
    ModelSpec::connect4_basic(4, 64)
}

fn run_self_play(args: &WorkloadArgs, device: Device) -> Result<()> {
    let weights = write_connect4_weights(&args.artifact_dir, device)?;
    let config = SelfPlayConfig {
        num_games: args.games,
        threads: 4,
        max_moves: 256,
        progress_every: 25,
        ..SelfPlayConfig::default()
    };
    run_self_play_pass(&config, &weights, device, args.warmup, 0)?;
    synchronize(device);

    for pass in 0..args.iterations {
        run_self_play_pass(&config, &weights, device, 0, pass as u64)?;
    }
    Ok(())
}

fn run_self_play_pass(
    config: &SelfPlayConfig,
    weights: &Path,
    device: Device,
    warmup_games: usize,
    generation: u64,
) -> Result<()> {
    if warmup_games > 0 {
        let mut warmup = config.clone();
        warmup.num_games = warmup_games;
        run_connect4_self_play(&warmup, weights, device, generation)?;
    }
    run_connect4_self_play(config, weights, device, generation)
}

fn run_connect4_self_play(
    config: &SelfPlayConfig,
    weights: &Path,
    device: Device,
    generation: u64,
) -> Result<()> {
    let batcher = Batcher::new_with_model(
        connect4_model(),
        weights,
        device,
        BatcherConfig {
            preferred_batch_size: 16,
            max_batch_size: 256,
            max_wait: Duration::from_millis(5),
            max_queued_states: 4_096,
        },
    )?;
    let factory = DomainSelfPlayWorkerFactory::<
        StandardSelfPlayDomain<Connect4, alphazero::representation::Connect4AzRepresentation>,
    >::new(&batcher, config.clone())?;
    let replay = ReplayBuffer::new(config.num_games * config.max_moves, 7);
    let coordinator = SelfPlayCoordinator::new(config.clone(), 0)?;
    coordinator.run(
        &factory,
        &replay,
        SelfPlayEpoch {
            model_generation: generation,
            first_game_id: generation * config.num_games as u64,
        },
    )?;
    Ok(())
}

fn write_connect4_weights(artifact_dir: &Path, device: Device) -> Result<PathBuf> {
    let path = artifact_dir.join("connect4-4x64.safetensors");
    let store = nn::VarStore::new(device);
    let _network = Network::new(&store.root(), &connect4_model())?;
    store.save(&path)?;
    Ok(path)
}

fn run_full_iteration(args: &WorkloadArgs, device: Device) -> Result<()> {
    let mut experiment = ExperimentConfig::read_toml(&args.experiment)?;
    experiment.self_play.num_games = args.games;
    experiment.training.train_steps = args.training_steps;
    experiment.validate()?;

    let run_dir = args.artifact_dir.join("training-run");
    RunDir::initialize(&run_dir, experiment)?;
    let mut run = TrainingRun::open(&run_dir, device)?;

    for _ in 0..args.warmup {
        run.step()?;
    }
    synchronize(device);

    for _ in 0..args.iterations {
        run.step()?;
    }
    Ok(())
}

fn synchronize(device: Device) {
    if let Device::Cuda(index) = device {
        Cuda::synchronize(index as i64);
    }
}

fn print_command(args: CommandArgs) -> Result<()> {
    resolve_device(args.workload.device)?;
    require_program(args.tool)?;
    println!("{}", profiler_command(args.tool, &args.workload));
    Ok(())
}

fn profiler_command(tool: ProfilingTool, workload: &WorkloadArgs) -> String {
    let artifacts = shell_quote(&workload.artifact_dir);
    let artifact = |name: String| shell_quote(&workload.artifact_dir.join(name));
    let target = format!(
        "target/profiling/engine-profile run {} --warmup {} --iterations {} --games {} --training-steps {} --experiment {} --artifact-dir {} --device {}",
        workload.workload,
        workload.warmup,
        workload.iterations,
        workload.games,
        workload.training_steps,
        shell_quote(&workload.experiment),
        artifacts,
        workload.device,
    );

    match tool {
        ProfilingTool::Perf => format!(
            "mkdir -p {artifacts} && perf record --output {} --call-graph dwarf -- {target}",
            artifact(format!("perf-{}.data", workload.workload))
        ),
        ProfilingTool::Flamegraph => format!(
            "mkdir -p {artifacts} && cargo flamegraph --profile profiling -p engine-profile --root --output {} -- run {} --warmup {} --iterations {} --games {} --training-steps {} --experiment {} --artifact-dir {} --device {}",
            artifact(format!("flamegraph-{}.svg", workload.workload)),
            workload.workload,
            workload.warmup,
            workload.iterations,
            workload.games,
            workload.training_steps,
            shell_quote(&workload.experiment),
            artifacts,
            workload.device,
        ),
        ProfilingTool::Nsys => format!(
            "mkdir -p {artifacts} && nsys profile --force-overwrite true --trace=cuda,nvtx,osrt --sample=cpu --cpuctxsw=process-tree --output {} -- {target}",
            artifact(format!("nsys-{}", workload.workload))
        ),
        ProfilingTool::Ncu => format!(
            "mkdir -p {artifacts} && ncu --set full --target-processes all --kernel-name-base demangled --launch-count 1 --export {} -- {target}",
            artifact(format!("ncu-{}", workload.workload))
        ),
    }
}

fn shell_quote(path: &Path) -> String {
    let text = path.to_string_lossy();
    format!("'{}'", text.replace('\'', "'\\\"'\\\"'"))
}

fn require_program(tool: ProfilingTool) -> Result<()> {
    if program_on_path(tool.program()) {
        return Ok(());
    }

    bail!(
        "`{}` was not found on PATH. {}",
        tool.program(),
        tool.install_hint()
    )
}

fn program_on_path(program: &str) -> bool {
    let Some(paths) = env::var_os("PATH")
    else {
        return false;
    };

    env::split_paths(&paths).any(|path| path.join(program).is_file())
}

fn print_requirements() -> Result<()> {
    println!(
        "CUDA: {} ({} device{})",
        if Cuda::is_available() {
            "available"
        }
        else {
            "unavailable"
        },
        Cuda::device_count(),
        if Cuda::device_count() == 1 { "" } else { "s" },
    );

    for tool in [
        ProfilingTool::Perf,
        ProfilingTool::Flamegraph,
        ProfilingTool::Nsys,
        ProfilingTool::Ncu,
    ] {
        let status = if program_on_path(tool.program()) {
            "available".to_owned()
        }
        else {
            format!("missing — {}", tool.install_hint())
        };
        println!("{}: {status}", tool.program());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn generated_commands_target_the_profiling_crate() {
        let args = workload_args(Workload::Inference);

        assert!(profiler_command(ProfilingTool::Perf, &args).contains("engine-profile"));
        assert!(!profiler_command(ProfilingTool::Perf, &args).contains("engine-bench"));
    }

    #[test]
    fn defaults_to_cuda_for_workloads() {
        let command = Command::try_parse_from(["engine-profile", "run", "inference"]).unwrap();
        let ProfileCommand::Run(args) = command.command
        else {
            panic!("expected a run command");
        };

        assert!(matches!(args.device, ExecutionDevice::Cuda));
    }

    #[test]
    fn nsys_command_has_cuda_trace() {
        let args = workload_args(Workload::Training);
        let command = profiler_command(ProfilingTool::Nsys, &args);

        assert!(command.contains("--trace=cuda,nvtx,osrt"));
        assert!(command.contains("--cpuctxsw=process-tree"));
    }

    #[test]
    fn full_iteration_uses_the_checked_in_baseline() {
        let args = workload_args(Workload::Iteration);
        assert!(args
            .experiment
            .ends_with("experiments/benchmark-chess-canonical-h1.toml"));
        assert_eq!(args.games, 200);
        assert_eq!(args.training_steps, 80);
    }

    fn workload_args(workload: Workload) -> WorkloadArgs {
        WorkloadArgs {
            workload,
            warmup: 0,
            iterations: 1,
            games: 200,
            training_steps: 80,
            experiment: PathBuf::from("experiments/benchmark-chess-canonical-h1.toml"),
            artifact_dir: PathBuf::from("artifacts/profiles"),
            device: ExecutionDevice::Cuda,
        }
    }
}

//! Deterministic workloads intended exclusively for CPU and GPU profilers.
//!
//! This crate deliberately contains no benchmark timing or production profiling hooks.

use alphazero::{
    build_optimizer, train, Action, ModelSpec, Network, Outcome, ReplayBuffer, ReplaySample,
    SampleMetadata, TrainConfig, TrainingWeights,
};
use anyhow::{bail, ensure, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use games::Connect4;
use std::env;
use std::fmt::Display;
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

    /// Unmeasured iterations before the captured workload begins.
    #[arg(long, default_value_t = 20)]
    warmup: usize,

    /// Number of deterministic workload iterations to execute.
    #[arg(long, default_value_t = 1_000)]
    iterations: usize,

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
    /// Connect Four network forward passes on a fixed input batch.
    Inference,
    /// Connect Four forward, backward, and optimizer steps over fixed replay data.
    Training,
}

impl Display for Workload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Inference => "inference",
            Self::Training => "training",
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
    let device = resolve_device(args.device)?;
    tch::manual_seed(0);

    match args.workload {
        Workload::Inference => run_inference(device, args.warmup, args.iterations),
        Workload::Training => run_training(device, args.warmup, args.iterations),
    }

    synchronize(device);
    println!(
        "Completed {} profiling workload on {} (warmup {}, iterations {}).",
        args.workload, args.device, args.warmup, args.iterations
    );
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
    let network = Network::new(&store.root(), &ModelSpec::connect4_basic(1, 8))
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

fn run_training(device: Device, warmup: usize, iterations: usize) {
    let replay = fixed_replay();
    let store = nn::VarStore::new(device);
    let network = Network::new(&store.root(), &ModelSpec::connect4_basic(1, 8))
        .expect("fixed profiling model specification is valid");
    let config = TrainConfig {
        batch_size: 256,
        micro_batch_size: 256,
        train_steps: 1,
        progress_every: 1,
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
    let replay = ReplayBuffer::new(1_024, 7);
    replay.add((0..1_024).map(|game_id| ReplaySample {
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

fn synchronize(device: Device) {
    if let Device::Cuda(index) = device {
        Cuda::synchronize(index as i64);
    }
}

fn print_command(args: CommandArgs) -> Result<()> {
    require_program(args.tool)?;
    println!("{}", profiler_command(args.tool, &args.workload));
    Ok(())
}

fn profiler_command(tool: ProfilingTool, workload: &WorkloadArgs) -> String {
    let target = format!(
        "cargo run --profile profiling -p engine-profile -- run {} --warmup {} --iterations {} --device {}",
        workload.workload, workload.warmup, workload.iterations, workload.device
    );

    match tool {
        ProfilingTool::Perf => format!("perf record --call-graph dwarf -- {target}"),
        ProfilingTool::Flamegraph => format!(
            "cargo flamegraph --profile profiling -p engine-profile -- run {} --warmup {} --iterations {} --device {}",
            workload.workload, workload.warmup, workload.iterations, workload.device
        ),
        ProfilingTool::Nsys => format!(
            "nsys profile --force-overwrite true --trace=cuda,osrt --sample=cpu --cpuctxsw=process --output engine-profile-{} -- {target}",
            workload.workload
        ),
        ProfilingTool::Ncu => format!(
            "ncu --set full --target-processes all --launch-skip {} --launch-count 1 -- {target}",
            workload.warmup
        ),
    }
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
        let args = WorkloadArgs {
            workload: Workload::Inference,
            warmup: 20,
            iterations: 1_000,
            device: ExecutionDevice::Cuda,
        };

        assert!(profiler_command(ProfilingTool::Perf, &args).contains("-p engine-profile"));
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
        let args = WorkloadArgs {
            workload: Workload::Training,
            warmup: 20,
            iterations: 1_000,
            device: ExecutionDevice::Cuda,
        };

        assert!(profiler_command(ProfilingTool::Nsys, &args).contains("--trace=cuda,osrt"));
    }
}

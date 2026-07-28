use alphazero::{ExperimentConfig, RunDir, RunLimit, TrainingRun};
use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
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
    #[arg(long, value_enum, default_value_t = DeviceKind::Auto)]
    device: DeviceKind,
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
    let mut run = TrainingRun::open(&args.run_dir, select_device(args.device)?)?;
    let limit = if args.forever {
        RunLimit::Forever
    }
    else {
        RunLimit::Iterations(args.iterations)
    };

    run.run(limit)
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

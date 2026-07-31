use alphazero::{ExperimentConfig, RunDir};
use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use engine_app::tensor_rt;
use std::path::PathBuf;
use tch::Device;

#[derive(Clone, Copy, ValueEnum)]
enum DeviceKind {
    Auto,
    Cuda,
    Cpu,
}

#[derive(Parser)]
#[command(about = "AlphaZero model tooling")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    ExportTorchScript {
        #[arg(long)]
        experiment: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_enum, default_value_t = DeviceKind::Auto)]
        device: DeviceKind,
    },
    Inspect {
        #[arg(long)]
        run_dir: PathBuf,
    },
    /// Print the encoded-state channel count for an experiment TOML. Used by
    /// the TensorRT compile helper to size `--channels` without duplicating
    /// the ModelSpec shape math in Python.
    Channels {
        #[arg(long)]
        experiment: PathBuf,
    },
    /// Ensure a run directory has `experiment.toml` (migrating legacy
    /// `config.json` / `experiment.json` when needed) and print its path.
    /// Used by the arena wrapper before TensorRT compilation.
    EnsureExperiment {
        #[arg(long)]
        run_dir: PathBuf,
    },
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::ExportTorchScript {
            experiment,
            checkpoint,
            output,
            device,
        } => export_torchscript(experiment, checkpoint, output, select_device(device)?),
        Command::Inspect { run_dir } => inspect(run_dir),
        Command::Channels { experiment } => channels(experiment),
        Command::EnsureExperiment { run_dir } => ensure_experiment(run_dir),
    }
}

fn ensure_experiment(run_dir: PathBuf) -> Result<()> {
    let (run, _, _) = RunDir::open_or_create(&run_dir, || {
        panic!("no experiment found at {}", run_dir.display())
    })?;
    println!("{}", run.experiment_path().display());
    Ok(())
}

fn channels(experiment_path: PathBuf) -> Result<()> {
    let experiment = ExperimentConfig::read_toml(&experiment_path)?;
    println!("{}", experiment.model.state_shape()[0]);
    Ok(())
}

fn export_torchscript(
    experiment_path: PathBuf,
    checkpoint: PathBuf,
    output: PathBuf,
    device: Device,
) -> Result<()> {
    let experiment = ExperimentConfig::read_toml(&experiment_path)?;
    tensor_rt::export_torchscript(&experiment, &checkpoint, &output, device)?;

    println!("exported {}", output.display());
    Ok(())
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

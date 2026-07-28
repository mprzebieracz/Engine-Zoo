use alphazero::{ExperimentConfig, Network, RunDir};
use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tch::{nn, CModule, Device, Kind, Tensor};

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
    }
}

fn export_torchscript(
    experiment_path: PathBuf,
    checkpoint: PathBuf,
    output: PathBuf,
    device: Device,
) -> Result<()> {
    let experiment = ExperimentConfig::read_toml(&experiment_path)?;
    let mut var_store = nn::VarStore::new(device);
    let network = Network::new(&var_store.root(), &experiment.model)?;
    var_store.load(&checkpoint)?;
    var_store.freeze();

    let [channels, height, width] = experiment.model.state_shape();
    let input = Tensor::zeros([1, channels, height, width], (Kind::Float, device));
    let mut forward = |inputs: &[Tensor]| {
        let output = network.forward_t(&inputs[0], false);
        let scalar = output.value.expected_value();
        vec![Tensor::cat(&[output.policy_logits, scalar], 1)]
    };
    let module = CModule::create_by_tracing("engine_zoo", "forward", &[input], &mut forward)?;
    module.save(&output)?;

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

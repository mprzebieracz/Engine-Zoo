use alphazero::artifact::{CompiledBackendKind, TensorRtBuildPrecision};
use alphazero::{ExperimentConfig, RunDir};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use engine_app::tensor_rt;
use engine_model_runtime::tensor_rt::{TensorRtBatchShapes, TensorRtCompiler};
use engine_model_runtime::{BackendPreference, ModelSelector, ModelStore, RepositoryConfig};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use tch::Device;

#[derive(Clone, Copy, ValueEnum)]
enum DeviceKind {
    Auto,
    Cuda,
    Cpu,
}

#[derive(Clone, Copy, ValueEnum)]
enum CompileBackend {
    Auto,
    Tensorrt,
    RawTensorrt,
    TorchTensorrt,
}

impl From<CompileBackend> for BackendPreference {
    fn from(backend: CompileBackend) -> Self {
        match backend {
            CompileBackend::Auto => Self::Auto,
            CompileBackend::Tensorrt => Self::Tensorrt,
            CompileBackend::RawTensorrt => Self::RawTensorrt,
            CompileBackend::TorchTensorrt => Self::TorchTensorrt,
        }
    }
}

#[derive(Parser)]
#[command(about = "AlphaZero model tooling")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Copy an immutable checkpoint into the permanent CUDA/Torch model store.
    Add {
        name: String,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        experiment: Option<PathBuf>,
    },
    /// List permanent CUDA/Torch models.
    List,
    /// Inspect a permanent model or a checkpoint path.
    Inspect {
        #[arg(required_unless_present = "run_dir", conflicts_with = "run_dir")]
        model: Option<String>,
        /// Required for a standalone checkpoint outside a training run.
        #[arg(long, conflicts_with = "run_dir")]
        experiment: Option<PathBuf>,
        /// Legacy run-directory inspection mode.
        #[arg(long, conflicts_with_all = ["model", "experiment"])]
        run_dir: Option<PathBuf>,
    },
    /// Validate the local CUDA/Torch/TensorRT setup.
    Doctor,
    /// Compile a model into the CUDA/TensorRT artifact cache.
    Compile {
        model: String,
        #[arg(long)]
        experiment: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = CompileBackend::Auto)]
        backend: CompileBackend,
    },
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
    /// Print the encoded-state channel count for an experiment TOML. Used by
    /// the TensorRT compile helper to size `--channels` without duplicating
    /// the ModelSpec shape math in Python.
    Channels {
        #[arg(long)]
        experiment: PathBuf,
    },
    /// Require an initialized run directory and print its experiment path.
    /// Used by the arena wrapper before TensorRT compilation.
    EnsureExperiment {
        #[arg(long)]
        run_dir: PathBuf,
    },
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Add {
            name,
            checkpoint,
            experiment,
        } => add(name, checkpoint, experiment),
        Command::List => list(),
        Command::Inspect {
            model,
            experiment,
            run_dir,
        } => match (model, experiment, run_dir) {
            (Some(model), experiment, None) => inspect_model(model, experiment),
            (None, None, Some(run_dir)) => inspect_run(run_dir),
            _ => unreachable!("clap enforces exactly one inspect target"),
        },
        Command::Doctor => doctor(),
        Command::Compile {
            model,
            experiment,
            backend,
        } => compile_model(model, experiment, backend.into()),
        Command::ExportTorchScript {
            experiment,
            checkpoint,
            output,
            device,
        } => export_torchscript(experiment, checkpoint, output, select_device(device)?),
        Command::Channels { experiment } => channels(experiment),
        Command::EnsureExperiment { run_dir } => ensure_experiment(run_dir),
    }
}

fn repository() -> Result<RepositoryConfig> {
    RepositoryConfig::discover(std::env::current_dir()?)
}

fn add(name: String, checkpoint: PathBuf, experiment: Option<PathBuf>) -> Result<()> {
    let config = repository()?;
    let store = ModelStore::new(config.clone());
    let resolved = store.resolve(&ModelSelector::Checkpoint {
        checkpoint,
        experiment,
    })?;
    let source_run = resolved
        .experiment
        .as_deref()
        .and_then(Path::parent)
        .map(Path::to_path_buf);
    let promoted = store.promote(&name, &resolved.checkpoint, resolved.model, source_run)?;
    println!("added {name}: {}", promoted.checkpoint.display());
    println!("checkpoint sha256: {}", promoted.checkpoint_sha256);
    if let Some(artifact) = compile_resolved_if_available(&config, &promoted)? {
        println!(
            "{} CUDA/TensorRT artifact: {}",
            if artifact.reused {
                "reused"
            }
            else {
                "compiled"
            },
            artifact.artifact_path.display()
        );
    }
    Ok(())
}

fn compile_model(
    input: String,
    experiment: Option<PathBuf>,
    backend: BackendPreference,
) -> Result<()> {
    let config = repository()?;
    let store = ModelStore::new(config.clone());
    let model = store.resolve(&ModelSelector::from_input(input, experiment))?;
    let artifact = compile_resolved(&config, &model, backend, true)?
        .context("explicit CUDA/TensorRT compilation did not select a backend")?;
    println!(
        "{} CUDA/TensorRT artifact: {}",
        if artifact.reused {
            "reused"
        }
        else {
            "compiled"
        },
        artifact.artifact_path.display()
    );
    Ok(())
}

fn compile_resolved_if_available(
    config: &RepositoryConfig,
    model: &engine_model_runtime::ResolvedModel,
) -> Result<Option<engine_model_runtime::tensor_rt::CompiledArtifact>> {
    if config.runtime.default_backend == BackendPreference::Native {
        return Ok(None);
    }
    if !matches!(config.runtime.device.as_str(), "auto" | "cuda" | "cuda:0")
        || !tch::Cuda::is_available()
    {
        return Ok(None);
    }
    compile_resolved(config, model, config.runtime.default_backend, false)
}

fn compile_resolved(
    config: &RepositoryConfig,
    model: &engine_model_runtime::ResolvedModel,
    preference: BackendPreference,
    require_backend: bool,
) -> Result<Option<engine_model_runtime::tensor_rt::CompiledArtifact>> {
    anyhow::ensure!(
        preference != BackendPreference::Native,
        "native models cannot be compiled; select a CUDA/TensorRT backend"
    );
    let Some((backend, python, export_python, script, label)) = select_compiler(config, preference)
    else {
        if require_backend {
            anyhow::bail!(
                "no configured CUDA/TensorRT compiler for requested backend {preference:?}"
            );
        }
        return Ok(None);
    };
    let compiler = TensorRtCompiler::application(
        python,
        export_python,
        script,
        TensorRtBatchShapes {
            min: config.runtime.cuda_tensorrt_batch_min,
            optimal: config.runtime.cuda_tensorrt_batch_optimal,
            max: config.runtime.cuda_tensorrt_batch_max,
        },
        if config.runtime.fp16 {
            TensorRtBuildPrecision::Fp16
        }
        else {
            TensorRtBuildPrecision::Fp32
        },
    );
    let output = config
        .cuda_tensorrt_artifact_cache_dir()
        .join(&model.checkpoint_sha256)
        .join(format!("{label}.engine"));
    Ok(Some(compiler.ensure_artifact(
        &model.checkpoint,
        &model.model,
        backend,
        &output,
        runtime_cuda_device(config)?,
    )?))
}

fn select_compiler(
    config: &RepositoryConfig,
    preference: BackendPreference,
) -> Option<(
    CompiledBackendKind,
    PathBuf,
    Option<PathBuf>,
    PathBuf,
    &'static str,
)> {
    let root = config.root();
    let raw = || {
        let script = root.join("scripts/compile_tensorrt_raw.py");
        let tensorrt_python = configured_path(config, config.toolchain.tensorrt_python.as_deref());
        let cuda_torch_python =
            configured_path(config, config.toolchain.cuda_torch_python.as_deref());
        match (tensorrt_python, cuda_torch_python) {
            (Some(tensorrt_python), Some(cuda_torch_python))
                if script.is_file()
                    && python_imports(&tensorrt_python, "tensorrt")
                    && python_imports(&cuda_torch_python, "torch") =>
            {
                Some((
                    CompiledBackendKind::TensorRtRaw,
                    tensorrt_python,
                    Some(cuda_torch_python),
                    script,
                    "raw-tensorrt",
                ))
            }
            _ => None,
        }
    };
    let torch = || {
        let script = root.join("scripts/compile_tensorrt.py");
        let python = configured_path(config, config.toolchain.cuda_torch_python.as_deref());
        match python {
            Some(python)
                if script.is_file() && python_imports(&python, "torch, torch_tensorrt") =>
            {
                Some((
                    CompiledBackendKind::TensorRtTorchScript,
                    python,
                    None,
                    script,
                    "torch-tensorrt",
                ))
            }
            _ => None,
        }
    };
    match preference {
        BackendPreference::Auto | BackendPreference::Tensorrt => raw().or_else(torch),
        BackendPreference::RawTensorrt => raw(),
        BackendPreference::TorchTensorrt => torch(),
        BackendPreference::Native => None,
    }
}

fn configured_path(config: &RepositoryConfig, path: Option<&Path>) -> Option<PathBuf> {
    path.map(|path| {
        if path.is_absolute() {
            path.to_path_buf()
        }
        else {
            config.root().join(path)
        }
    })
    .filter(|path| path.is_file())
}

fn python_imports(python: &Path, modules: &str) -> bool {
    ProcessCommand::new(python)
        .arg("-c")
        .arg(format!("import {modules}"))
        .output()
        .is_ok_and(|output| output.status.success())
}

fn runtime_cuda_device(config: &RepositoryConfig) -> Result<Device> {
    anyhow::ensure!(
        matches!(config.runtime.device.as_str(), "auto" | "cuda" | "cuda:0"),
        "CUDA/TensorRT compilation requires runtime.device = \"auto\", \"cuda\", or \"cuda:0\""
    );
    anyhow::ensure!(
        tch::Cuda::is_available(),
        "CUDA/TensorRT compilation was selected but CUDA is unavailable"
    );
    Ok(Device::Cuda(0))
}

fn list() -> Result<()> {
    let store = ModelStore::new(repository()?);
    let root = store.root();
    if !root.is_dir() {
        return Ok(());
    }
    let mut names = fs::read_dir(root)?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            entry
                .file_type()
                .ok()
                .filter(|file_type| file_type.is_dir() && entry.path().join("model.toml").is_file())
                .map(|_| entry.file_name().to_string_lossy().into_owned())
        })
        .collect::<Vec<_>>();
    names.sort();
    for name in names {
        let metadata = store.metadata(&name)?;
        println!("{name}\t{}", metadata.checkpoint_sha256);
    }
    Ok(())
}

fn inspect_model(input: String, experiment: Option<PathBuf>) -> Result<()> {
    let store = ModelStore::new(repository()?);
    let model = store.resolve(&ModelSelector::from_input(input, experiment))?;
    println!("checkpoint: {}", model.checkpoint.display());
    println!("checkpoint sha256: {}", model.checkpoint_sha256);
    if let Some(experiment) = model.experiment {
        println!("experiment: {}", experiment.display());
    }
    println!("model:");
    println!("{}", serde_json::to_string_pretty(&model.model)?);
    Ok(())
}

fn doctor() -> Result<()> {
    let config = repository()?;
    let script = config.root().join("scripts/init_cuda_torch_tensorrt.py");
    anyhow::ensure!(
        script.is_file(),
        "CUDA/Torch/TensorRT init helper is missing: {}",
        script.display()
    );
    let status = ProcessCommand::new("python3")
        .arg(script)
        .arg("--repo-root")
        .arg(config.root())
        .arg("--doctor")
        .status()
        .context("starting CUDA/Torch/TensorRT doctor with python3")?;
    anyhow::ensure!(
        status.success(),
        "CUDA/Torch/TensorRT doctor failed with {status}"
    );
    Ok(())
}

fn ensure_experiment(run_dir: PathBuf) -> Result<()> {
    let (run, _, _) = RunDir::open(&run_dir)
        .with_context(|| format!("no experiment found at {}", run_dir.display()))?;

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

fn inspect_run(run_dir: PathBuf) -> Result<()> {
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

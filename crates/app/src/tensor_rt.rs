//! TensorRT artifact creation for fixed-weight self-play generations.

use alphazero::{ExperimentConfig, InferencePrecision, Network, RunDir, RunState};
use anyhow::{bail, ensure, Context, Result};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tch::{nn, CModule, Device, Kind, Tensor};

/// The dynamic batch range baked into a TensorRT module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorRtBatchShapes {
    pub min: usize,
    pub optimal: usize,
    pub max: usize,
}

impl TensorRtBatchShapes {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.min > 0 && self.min <= self.optimal && self.optimal <= self.max,
            "TensorRT batch sizes must satisfy 0 < min <= optimal <= max"
        );
        Ok(())
    }
}

/// External compiler configuration. Python is configurable because the CUDA
/// TensorRT wheel must match the LibTorch runtime used by the Rust process.
#[derive(Clone, Debug)]
pub struct TensorRtCompiler {
    pub python: PathBuf,
    pub script: PathBuf,
    pub batch_shapes: TensorRtBatchShapes,
    /// TensorRT tactic-selection precision. Keep this explicit because the
    /// inference-service precision alone does not configure the external
    /// compiler process.
    pub precision: InferencePrecision,
    /// Optional builder timing-cache request. The compiler validates whether
    /// its selected frontend can honor it without changing the runtime ABI.
    pub timing_cache: Option<PathBuf>,
}

impl TensorRtCompiler {
    pub fn command(&self, input: &Path, output: &Path, channels: i64) -> Result<Command> {
        self.batch_shapes.validate()?;
        ensure!(channels > 0, "TensorRT requires at least one input channel");

        let mut command = Command::new(&self.python);
        command
            // The parent Rust process preloads its matching C++ TensorRT
            // integration. Python must load the libraries packaged in its own
            // virtual environment instead; inheriting either linker override
            // can make `import torch` resolve incompatible symbols.
            .env_remove("LD_PRELOAD")
            .env_remove("LD_LIBRARY_PATH")
            .arg(&self.script)
            .arg("--input")
            .arg(input)
            .arg("--output")
            .arg(output)
            .arg("--channels")
            .arg(channels.to_string())
            .arg("--min-batch-size")
            .arg(self.batch_shapes.min.to_string())
            .arg("--opt-batch-size")
            .arg(self.batch_shapes.optimal.to_string())
            .arg("--max-batch-size")
            .arg(self.batch_shapes.max.to_string())
            .arg("--precision")
            .arg(match self.precision {
                InferencePrecision::Fp16 => "fp16",
                InferencePrecision::Fp32 => "fp32-fp16",
            });
        if let Some(timing_cache) = &self.timing_cache {
            command.arg("--timing-cache").arg(timing_cache);
        }

        Ok(command)
    }
}

/// Exports the latest checkpoint and compiles it into the configured module.
///
/// The final module is only replaced after TensorRT compilation succeeds, so a
/// failed compiler process leaves the previous usable module intact.
pub fn compile_latest_tensor_rt(
    run_dir: &RunDir,
    experiment: &ExperimentConfig,
    device: Device,
    compiler: &TensorRtCompiler,
) -> Result<PathBuf> {
    let module = configured_module_path(run_dir, experiment)?;
    let export = module.with_extension("torchscript.ts");
    let temporary_module = temporary_path(&module, "tmp.trt.ts")?;

    export_torchscript(experiment, &run_dir.latest_path(), &export, device)?;
    let status = compiler
        .command(
            &export,
            &temporary_module,
            experiment.model.state_shape()[0],
        )?
        .status()
        .with_context(|| format!("starting TensorRT compiler {}", compiler.python.display()))?;

    if !status.success() {
        let _ = fs::remove_file(&temporary_module);
        bail!("TensorRT compilation failed with {status}");
    }

    ensure!(
        temporary_module.is_file(),
        "TensorRT compiler succeeded but did not create {}",
        temporary_module.display()
    );

    fs::rename(&temporary_module, &module).with_context(|| {
        format!(
            "installing newly compiled TensorRT module at {}",
            module.display()
        )
    })?;
    let _ = fs::remove_file(export);

    Ok(module)
}

/// Compiles an arbitrary checkpoint into a TensorRT module at *output*.
///
/// This mirrors [`compile_latest_tensor_rt`] but decouples the input checkpoint
/// from the run's `latest.safetensors` and the output path from
/// `inference.tensor_rt_module`. It exists for tooling that caches per-model
/// TensorRT artifacts outside a run directory, such as the multi-opponent
/// arena. The final module is only installed after compilation succeeds.
pub fn compile_checkpoint_tensor_rt(
    experiment: &ExperimentConfig,
    checkpoint: &Path,
    output: &Path,
    device: Device,
    compiler: &TensorRtCompiler,
) -> Result<PathBuf> {
    ensure!(
        checkpoint.is_file(),
        "checkpoint does not exist: {}",
        checkpoint.display()
    );
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("creating TensorRT cache directory {}", parent.display())
            })?;
        }
    }

    let export = temporary_path(output, "torchscript.ts")?;
    let temporary_module = temporary_path(output, "tmp.trt.ts")?;

    export_torchscript(experiment, checkpoint, &export, device)?;
    let status = compiler
        .command(
            &export,
            &temporary_module,
            experiment.model.state_shape()[0],
        )?
        .status()
        .with_context(|| format!("starting TensorRT compiler {}", compiler.python.display()))?;

    if !status.success() {
        let _ = fs::remove_file(&temporary_module);
        let _ = fs::remove_file(&export);
        bail!("TensorRT compilation failed with {status}");
    }

    ensure!(
        temporary_module.is_file(),
        "TensorRT compiler succeeded but did not create {}",
        temporary_module.display()
    );

    fs::rename(&temporary_module, output).with_context(|| {
        format!(
            "installing newly compiled TensorRT module at {}",
            output.display()
        )
    })?;
    let _ = fs::remove_file(export);

    Ok(output.to_path_buf())
}

/// Creates the deterministic initial checkpoint when a newly initialized run
/// has not generated one yet. Native `TrainingRun::open` performs the same
/// initialization; TensorRT needs it one step earlier so its module can be
/// compiled before the inference service is opened.
pub fn ensure_latest_checkpoint(
    run_dir: &RunDir,
    experiment: &ExperimentConfig,
    state: &mut RunState,
    device: Device,
) -> Result<()> {
    if run_dir.latest_path().is_file() {
        return Ok(());
    }

    let var_store = nn::VarStore::new(device);
    let _network = Network::new(&var_store.root(), &experiment.model)?;

    run_dir.write_latest(state, |path| Ok(var_store.save(path)?))
}

/// Exports a checkpoint into the TorchScript ABI consumed by the TensorRT
/// compiler. This is also used by the standalone `engine-zoo-model` CLI.
pub fn export_torchscript(
    experiment: &ExperimentConfig,
    checkpoint: &Path,
    output: &Path,
    device: Device,
) -> Result<()> {
    let mut var_store = nn::VarStore::new(device);
    let network = Network::new(&var_store.root(), &experiment.model)?;
    var_store.load(checkpoint)?;
    var_store.freeze();

    let [channels, height, width] = experiment.model.state_shape();
    let input = Tensor::zeros([1, channels, height, width], (Kind::Float, device));
    let mut forward = |inputs: &[Tensor]| {
        let output = network.forward_t(&inputs[0], false);
        let scalar = output.value.expected_value();

        vec![Tensor::cat(&[output.policy_logits, scalar], 1)]
    };
    let module = CModule::create_by_tracing("engine_zoo", "forward", &[input], &mut forward)?;
    module.save(output)?;

    Ok(())
}

pub fn configured_module_path(run_dir: &RunDir, experiment: &ExperimentConfig) -> Result<PathBuf> {
    let module = experiment
        .inference
        .tensor_rt_module
        .as_ref()
        .context("TensorRT inference requires inference.tensor_rt_module")?;

    ensure!(
        !module.is_absolute(),
        "inference.tensor_rt_module must be relative to the immutable run directory"
    );

    let path = run_dir.root().join(module);
    let parent = path
        .parent()
        .context("TensorRT module path must have a parent directory")?;
    fs::create_dir_all(parent)?;

    Ok(path)
}

fn temporary_path(path: &Path, extension: &str) -> Result<PathBuf> {
    let file_name = path
        .file_name()
        .context("TensorRT module path must have a file name")?;
    let mut temporary = OsString::from(file_name);
    temporary.push(".");
    temporary.push(extension);

    Ok(path.with_file_name(temporary))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn compiler_command_has_all_dynamic_shape_arguments() {
        let compiler = TensorRtCompiler {
            python: "python-with-tensorrt".into(),
            script: "scripts/compile_tensorrt.py".into(),
            batch_shapes: TensorRtBatchShapes {
                min: 1,
                optimal: 32,
                max: 256,
            },
            precision: InferencePrecision::Fp16,
            timing_cache: None,
        };
        let command = compiler
            .command(Path::new("input.ts"), Path::new("output.trt.ts"), 63)
            .unwrap();
        let arguments: Vec<_> = command.get_args().collect();

        assert_eq!(command.get_program(), OsStr::new("python-with-tensorrt"));
        assert_eq!(
            arguments,
            [
                OsStr::new("scripts/compile_tensorrt.py"),
                OsStr::new("--input"),
                OsStr::new("input.ts"),
                OsStr::new("--output"),
                OsStr::new("output.trt.ts"),
                OsStr::new("--channels"),
                OsStr::new("63"),
                OsStr::new("--min-batch-size"),
                OsStr::new("1"),
                OsStr::new("--opt-batch-size"),
                OsStr::new("32"),
                OsStr::new("--max-batch-size"),
                OsStr::new("256"),
                OsStr::new("--precision"),
                OsStr::new("fp16"),
            ]
        );
        assert_linker_overrides_are_removed(&command);
    }

    #[test]
    fn compiler_rejects_an_invalid_batch_range() {
        let compiler = TensorRtCompiler {
            python: "python".into(),
            script: "compile.py".into(),
            batch_shapes: TensorRtBatchShapes {
                min: 32,
                optimal: 16,
                max: 64,
            },
            precision: InferencePrecision::Fp16,
            timing_cache: None,
        };

        assert!(compiler
            .command(Path::new("input.ts"), Path::new("output.ts"), 63)
            .is_err());
    }

    #[test]
    fn compiler_command_passes_an_explicit_timing_cache_request() {
        let compiler = TensorRtCompiler {
            python: "python".into(),
            script: "compile.py".into(),
            batch_shapes: TensorRtBatchShapes {
                min: 1,
                optimal: 32,
                max: 256,
            },
            precision: InferencePrecision::Fp16,
            timing_cache: Some("artifacts/timing.cache".into()),
        };

        let command = compiler
            .command(Path::new("input.ts"), Path::new("output.ts"), 63)
            .unwrap();
        let arguments: Vec<_> = command.get_args().collect();

        assert!(arguments.windows(2).any(|pair| pair
            == [
                OsStr::new("--timing-cache"),
                OsStr::new("artifacts/timing.cache")
            ]));
    }

    fn assert_linker_overrides_are_removed(command: &Command) {
        let removed: Vec<_> = command
            .get_envs()
            .filter_map(|(key, value)| value.is_none().then_some(key))
            .collect();

        assert!(removed.contains(&OsStr::new("LD_PRELOAD")));
        assert!(removed.contains(&OsStr::new("LD_LIBRARY_PATH")));
    }
}

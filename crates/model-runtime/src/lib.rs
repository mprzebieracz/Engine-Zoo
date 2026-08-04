//! Shared CUDA/Torch application-model discovery and permanent model storage.
//!
//! This crate deliberately names CUDA, Torch, and TensorRT-specific settings
//! explicitly. Future engine families should add their own stores and paths
//! rather than inheriting these settings as generic runtime configuration.

pub mod tensor_rt;

use alphazero::{ExperimentConfig, ModelSpec};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MODEL_FORMAT_VERSION: u32 = 1;
pub const REPOSITORY_CONFIG_FILE: &str = "engine-zoo.local.toml";
pub const REPOSITORY_EXAMPLE_CONFIG_FILE: &str = "engine-zoo.example.toml";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct RepositoryConfig {
    #[serde(skip)]
    root: PathBuf,
    pub paths: CudaTorchPaths,
    pub toolchain: CudaTorchToolchain,
    pub runtime: CudaTorchRuntime,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct CudaTorchPaths {
    pub training_runs: PathBuf,
    pub permanent_models: PathBuf,
    pub cuda_tensorrt_artifact_cache: PathBuf,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct CudaTorchToolchain {
    pub cuda_torch_libtorch: Option<PathBuf>,
    pub cuda_root: Option<PathBuf>,
    pub tensorrt_root: Option<PathBuf>,
    pub cuda_torch_python: Option<PathBuf>,
    pub tensorrt_python: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct CudaTorchRuntime {
    pub default_backend: BackendPreference,
    pub device: String,
    pub fp16: bool,
    pub cuda_tensorrt_batch_min: usize,
    pub cuda_tensorrt_batch_optimal: usize,
    pub cuda_tensorrt_batch_max: usize,
}

impl Default for CudaTorchPaths {
    fn default() -> Self {
        Self {
            training_runs: PathBuf::from("runs"),
            permanent_models: PathBuf::from("models"),
            cuda_tensorrt_artifact_cache: PathBuf::from("artifacts/cuda-tensorrt"),
        }
    }
}

impl Default for CudaTorchRuntime {
    fn default() -> Self {
        Self {
            default_backend: BackendPreference::Auto,
            device: "auto".into(),
            fp16: true,
            cuda_tensorrt_batch_min: 1,
            cuda_tensorrt_batch_optimal: 32,
            cuda_tensorrt_batch_max: 256,
        }
    }
}

impl Default for RepositoryConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
            paths: CudaTorchPaths::default(),
            toolchain: CudaTorchToolchain::default(),
            runtime: CudaTorchRuntime::default(),
        }
    }
}

impl RepositoryConfig {
    pub fn discover(start: impl AsRef<Path>) -> Result<Self> {
        let start = start.as_ref().canonicalize().with_context(|| {
            format!(
                "resolving repository search path {}",
                start.as_ref().display()
            )
        })?;
        let root = start
            .ancestors()
            .find(|directory| {
                directory.join(REPOSITORY_CONFIG_FILE).is_file()
                    || directory.join(REPOSITORY_EXAMPLE_CONFIG_FILE).is_file()
            })
            .or_else(|| {
                start.ancestors().find(|directory| {
                    fs::read_to_string(directory.join("Cargo.toml"))
                        .is_ok_and(|manifest| manifest.contains("[workspace]"))
                })
            });
        let root = root.context("could not find an engine-zoo repository")?;
        Self::load(root)
    }

    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().canonicalize()?;
        let local = root.join(REPOSITORY_CONFIG_FILE);
        let example = root.join(REPOSITORY_EXAMPLE_CONFIG_FILE);
        let path = if local.is_file() {
            Some(local)
        }
        else {
            example.is_file().then_some(example)
        };
        let mut config = match path {
            Some(path) => toml::from_str(
                &fs::read_to_string(&path)
                    .with_context(|| format!("reading {}", path.display()))?,
            )
            .with_context(|| format!("parsing {}", path.display()))?,
            None => Self::default(),
        };
        config.root = root;
        config.validate()?;
        Ok(config)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn training_runs_dir(&self) -> PathBuf {
        self.resolve(&self.paths.training_runs)
    }
    pub fn permanent_models_dir(&self) -> PathBuf {
        self.resolve(&self.paths.permanent_models)
    }
    pub fn cuda_tensorrt_artifact_cache_dir(&self) -> PathBuf {
        self.resolve(&self.paths.cuda_tensorrt_artifact_cache)
    }

    fn resolve(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        }
        else {
            self.root.join(path)
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.runtime.cuda_tensorrt_batch_min > 0
                && self.runtime.cuda_tensorrt_batch_min <= self.runtime.cuda_tensorrt_batch_optimal
                && self.runtime.cuda_tensorrt_batch_optimal <= self.runtime.cuda_tensorrt_batch_max,
            "CUDA TensorRT batch profile must satisfy 0 < min <= optimal <= max"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendPreference {
    #[default]
    Auto,
    Tensorrt,
    RawTensorrt,
    TorchTensorrt,
    Native,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeBackend {
    RawTensorrt,
    TorchTensorrt,
    Native,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendSelection {
    pub requested: BackendPreference,
    pub actual: RuntimeBackend,
}

impl BackendPreference {
    pub fn candidates(self) -> &'static [RuntimeBackend] {
        match self {
            Self::Auto => &[
                RuntimeBackend::RawTensorrt,
                RuntimeBackend::TorchTensorrt,
                RuntimeBackend::Native,
            ],
            Self::Tensorrt => &[RuntimeBackend::RawTensorrt, RuntimeBackend::TorchTensorrt],
            Self::RawTensorrt => &[RuntimeBackend::RawTensorrt],
            Self::TorchTensorrt => &[RuntimeBackend::TorchTensorrt],
            Self::Native => &[RuntimeBackend::Native],
        }
    }

    pub fn select_available(
        self,
        mut available: impl FnMut(RuntimeBackend) -> bool,
    ) -> Result<BackendSelection> {
        let actual = self
            .candidates()
            .iter()
            .copied()
            .find(|backend| available(*backend));
        actual
            .map(|actual| BackendSelection {
                requested: self,
                actual,
            })
            .with_context(|| format!("no usable backend for requested preference {self:?}"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelSelector {
    Permanent(String),
    Checkpoint {
        checkpoint: PathBuf,
        experiment: Option<PathBuf>,
    },
    RunAlias {
        run_dir: PathBuf,
        alias: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelOrigin {
    Permanent(String),
    Checkpoint,
    RunAlias { run_dir: PathBuf, alias: String },
}

#[derive(Clone, Debug)]
pub struct ResolvedModel {
    pub origin: ModelOrigin,
    pub checkpoint: PathBuf,
    pub model: ModelSpec,
    pub checkpoint_sha256: String,
    pub experiment: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub format_version: u32,
    pub name: String,
    pub model: ModelSpec,
    pub checkpoint_sha256: String,
    pub source_run: Option<PathBuf>,
    pub source_checkpoint: PathBuf,
}

#[derive(Clone, Debug)]
pub struct ModelStore {
    config: RepositoryConfig,
}

impl ModelStore {
    pub fn new(config: RepositoryConfig) -> Self {
        Self { config }
    }
    pub fn root(&self) -> PathBuf {
        self.config.permanent_models_dir()
    }
    pub fn model_dir(&self, name: &str) -> Result<PathBuf> {
        validate_name(name)?;
        Ok(self.root().join(name))
    }

    pub fn promote(
        &self,
        name: &str,
        checkpoint: &Path,
        model: ModelSpec,
        source_run: Option<PathBuf>,
    ) -> Result<ResolvedModel> {
        validate_name(name)?;
        ensure!(
            checkpoint.is_file(),
            "checkpoint does not exist: {}",
            checkpoint.display()
        );
        model.validate()?;
        let destination = self.model_dir(name)?;
        ensure!(
            !destination.exists(),
            "permanent CUDA/Torch model already exists: {name}"
        );
        fs::create_dir_all(self.root())?;
        let temporary = self
            .root()
            .join(format!(".{name}.promoting-{}", unique_suffix()));
        fs::create_dir(&temporary).with_context(|| format!("reserving permanent model {name}"))?;
        let result = (|| {
            let target_checkpoint = temporary.join("model.safetensors");
            copy_and_sync(checkpoint, &target_checkpoint)?;
            let digest = sha256_file(&target_checkpoint)?;
            let metadata = ModelMetadata {
                format_version: MODEL_FORMAT_VERSION,
                name: name.into(),
                model,
                checkpoint_sha256: digest.clone(),
                source_run,
                source_checkpoint: checkpoint.to_path_buf(),
            };
            write_toml_atomic(&temporary.join("model.toml"), &metadata)?;
            fs::rename(&temporary, &destination)
                .with_context(|| format!("publishing permanent model {name}"))?;
            Ok(ResolvedModel {
                origin: ModelOrigin::Permanent(name.into()),
                checkpoint: destination.join("model.safetensors"),
                model: metadata.model,
                checkpoint_sha256: digest,
                experiment: None,
            })
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&temporary);
        }
        result
    }

    pub fn metadata(&self, name: &str) -> Result<ModelMetadata> {
        let path = self.model_dir(name)?.join("model.toml");
        let metadata: ModelMetadata = toml::from_str(
            &fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?,
        )?;
        ensure!(
            metadata.format_version == MODEL_FORMAT_VERSION,
            "unsupported permanent model metadata version {}",
            metadata.format_version
        );
        ensure!(
            metadata.name == name,
            "permanent model metadata name does not match directory"
        );
        Ok(metadata)
    }

    pub fn resolve(&self, selector: &ModelSelector) -> Result<ResolvedModel> {
        match selector {
            ModelSelector::Permanent(name) => {
                let metadata = self.metadata(name)?;
                let checkpoint = self.model_dir(name)?.join("model.safetensors");
                let actual = sha256_file(&checkpoint)?;
                ensure!(
                    actual == metadata.checkpoint_sha256,
                    "permanent model checkpoint checksum mismatch: {name}"
                );
                Ok(ResolvedModel {
                    origin: ModelOrigin::Permanent(name.clone()),
                    checkpoint,
                    model: metadata.model,
                    checkpoint_sha256: actual,
                    experiment: None,
                })
            }
            ModelSelector::Checkpoint {
                checkpoint,
                experiment,
            } => resolve_checkpoint(checkpoint, experiment.as_deref(), ModelOrigin::Checkpoint),
            ModelSelector::RunAlias { run_dir, alias } => {
                let checkpoint = resolve_run_alias(run_dir, alias)?;
                resolve_checkpoint(
                    &checkpoint,
                    None,
                    ModelOrigin::RunAlias {
                        run_dir: run_dir.clone(),
                        alias: alias.clone(),
                    },
                )
            }
        }
    }
}

impl ModelSelector {
    /// A filesystem path wins over a permanent model name. Run aliases need
    /// their explicit run directory, so they cannot be inferred ambiguously.
    pub fn from_input(value: impl AsRef<str>, experiment: Option<PathBuf>) -> Self {
        let value = value.as_ref();
        let checkpoint = PathBuf::from(value);
        if checkpoint.is_file() {
            Self::Checkpoint {
                checkpoint,
                experiment,
            }
        }
        else {
            Self::Permanent(value.into())
        }
    }
}

pub fn resolve_checkpoint(
    checkpoint: &Path,
    experiment_override: Option<&Path>,
    origin: ModelOrigin,
) -> Result<ResolvedModel> {
    ensure!(
        checkpoint.is_file(),
        "checkpoint does not exist: {}",
        checkpoint.display()
    );
    let experiment = match experiment_override
        .map(Path::to_path_buf)
        .or_else(|| infer_experiment(checkpoint))
    {
        Some(path) if path.is_file() => path,
        Some(path) => bail!("experiment does not exist: {}", path.display()),
        None => bail!(
            "could not infer experiment.toml for {}; pass --experiment",
            checkpoint.display()
        ),
    };
    let config = ExperimentConfig::read_toml(&experiment)?;
    Ok(ResolvedModel {
        origin,
        checkpoint: checkpoint.to_path_buf(),
        model: config.model,
        checkpoint_sha256: sha256_file(checkpoint)?,
        experiment: Some(experiment),
    })
}

pub fn infer_experiment(checkpoint: &Path) -> Option<PathBuf> {
    let parent = checkpoint.parent()?;
    let run_dir = if parent.file_name().is_some_and(|name| name == "checkpoints") {
        parent.parent()?
    }
    else {
        parent
    };
    let experiment = run_dir.join("experiment.toml");
    experiment.is_file().then_some(experiment)
}

pub fn resolve_run_alias(run_dir: &Path, alias: &str) -> Result<PathBuf> {
    let path = match alias {
        "latest" | "best" | "candidate" => {
            let modern = run_dir
                .join("checkpoints")
                .join(format!("{alias}.safetensors"));
            if modern.is_file() {
                modern
            }
            else {
                run_dir.join(format!("{alias}.safetensors"))
            }
        }
        name if name.starts_with("ckpt_") || is_generation_checkpoint_name(name) => {
            run_dir.join("checkpoints").join(name)
        }
        _ => bail!("unsupported run checkpoint alias: {alias}"),
    };
    ensure!(path.is_file(), "checkpoint not found: {}", path.display());
    Ok(path)
}

fn is_generation_checkpoint_name(name: &str) -> bool {
    name.strip_prefix("generation-")
        .and_then(|suffix| suffix.strip_suffix(".safetensors"))
        .is_some_and(|number| {
            !number.is_empty() && number.bytes().all(|byte| byte.is_ascii_digit())
        })
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let size = file.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        hasher.update(&buffer[..size]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\']),
        "invalid permanent model name: {name}"
    );
    Ok(())
}

fn copy_and_sync(source: &Path, destination: &Path) -> Result<()> {
    let mut input = fs::File::open(source)?;
    let mut output = fs::File::create(destination)?;
    std::io::copy(&mut input, &mut output)?;
    output.flush()?;
    output.sync_all()?;
    Ok(())
}

fn write_toml_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension(format!("toml.tmp-{}", unique_suffix()));
    let mut file = fs::File::create(&temporary)?;
    file.write_all(toml::to_string_pretty(value)?.as_bytes())?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn unique_suffix() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

#[cfg(test)]
mod tests;

//! TensorRT artifact creation for fixed-weight self-play generations.

use alphazero::artifact::{
    self, ArtifactBuildConfig, ArtifactDType, ArtifactEnvironment, ArtifactIoContract,
    CompiledArtifactManifest, CompiledBackendKind, CompilerIdentity, TensorRtBuildPrecision,
    ValueLayout, ARTIFACT_MANIFEST_VERSION,
};
use alphazero::{ExperimentConfig, InferenceEngine, ModelSpec, Network, RunDir, RunState};
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
#[cfg(unix)]
use std::fs::OpenOptions;
use std::fs::{self, File};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;
use tch::{nn, CModule, Device, Kind, Tensor};

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TimingCachePolicy {
    Auto,
    Disabled,
    Explicit(PathBuf),
}

#[derive(Clone, Debug)]
pub struct TensorRtCompiler {
    pub python: PathBuf,
    pub export_python: Option<PathBuf>,
    pub script: PathBuf,
    pub batch_shapes: TensorRtBatchShapes,
    pub precision: TensorRtBuildPrecision,
    pub timing_cache: TimingCachePolicy,
    pub keep_intermediates: bool,
}

pub struct CompileRequest<'a> {
    pub checkpoint: &'a Path,
    pub model: &'a ModelSpec,
    pub backend: CompiledBackendKind,
    pub output: &'a Path,
    pub device: Device,
    pub compiler: &'a TensorRtCompiler,
}

#[derive(Debug)]
pub struct CompiledArtifact {
    pub artifact_path: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest: CompiledArtifactManifest,
    pub reused: bool,
}

#[derive(Debug, Deserialize)]
struct CompilerResult {
    schema_version: u32,
    operation: String,
    artifact_kind: Option<String>,
    input_dtype: Option<String>,
    output_dtype: Option<String>,
    internal_precision: Option<String>,
    tensorrt_version: String,
    cuda_runtime_version: String,
    gpu_name: String,
    compute_capability: String,
    python_version: String,
    torch_version: Option<String>,
    torch_tensorrt_version: Option<String>,
    profile: Option<CompilerProfile>,
}

#[derive(Debug, Deserialize)]
struct CompilerProfile {
    min: usize,
    opt: usize,
    max: usize,
}

pub fn compile_checkpoint_to_artifact(request: &CompileRequest<'_>) -> Result<CompiledArtifact> {
    ensure!(
        request.checkpoint.is_file(),
        "checkpoint does not exist: {}",
        request.checkpoint.display()
    );
    request.compiler.batch_shapes.validate()?;

    let parent = request
        .output
        .parent()
        .context("compiled artifact needs a parent directory")?;
    fs::create_dir_all(parent)?;
    let _lock = ArtifactLock::acquire(request.output)?;

    let environment_result = inspect_environment(request.compiler, parent)?;
    let environment = artifact_environment(&environment_result);
    let build = build_config(request);
    let compiler_identity = compiler_identity(request.compiler, &environment_result)?;
    let checkpoint_sha256 = artifact::sha256_file(request.checkpoint)?;
    let model_fingerprint = model_fingerprint(request.model);
    let expected_io = expected_io(request.model, request.compiler.precision);

    let expected = CompiledArtifactManifest {
        schema_version: ARTIFACT_MANIFEST_VERSION,
        backend: request.backend,
        checkpoint_sha256,
        model_fingerprint,
        artifact_sha256: String::new(),
        io: expected_io,
        build,
        environment,
        compiler: compiler_identity,
    };

    if let Ok(existing) = artifact::validate_artifact(request.output) {
        if existing
            .validate_identity(&expected.cache_identity())
            .is_ok()
        {
            return Ok(CompiledArtifact {
                artifact_path: request.output.to_path_buf(),
                manifest_path: artifact::manifest_path(request.output),
                manifest: existing,
                reused: true,
            });
        }
    }

    let temporary_root = artifact::unique_sibling(request.output);
    fs::create_dir(&temporary_root)?;
    let _cleanup =
        CleanupDirectory::new(temporary_root.clone(), request.compiler.keep_intermediates);
    if request.compiler.keep_intermediates {
        eprintln!(
            "keeping TensorRT intermediates at {}",
            temporary_root.display()
        );
    }
    let result = compile_to_temporary(request, &temporary_root, &expected.environment)?;
    validate_compiler_result(request, &result)?;
    ensure!(
        artifact_environment(&result) == expected.environment,
        "TensorRT compiler environment changed between inspection and build"
    );

    let temporary_artifact = temporary_root.join("artifact");
    ensure!(
        temporary_artifact.is_file(),
        "compiler succeeded without producing an artifact"
    );
    File::open(&temporary_artifact)?.sync_all()?;

    let mut manifest = expected;
    manifest.io.input_dtype = parse_dtype(
        result
            .input_dtype
            .as_deref()
            .context("compiler result omitted input_dtype")?,
    )?;
    manifest.io.output_dtype = parse_dtype(
        result
            .output_dtype
            .as_deref()
            .context("compiler result omitted output_dtype")?,
    )?;
    manifest.artifact_sha256 = artifact::sha256_file(&temporary_artifact)?;
    artifact::write_manifest_last(&temporary_artifact, &manifest)?;

    if !request.compiler.keep_intermediates {
        remove_intermediates(&temporary_root)?;
    }

    install_content_addressed(request.output, &temporary_root, &manifest)?;

    Ok(CompiledArtifact {
        artifact_path: request.output.to_path_buf(),
        manifest_path: artifact::manifest_path(request.output),
        manifest,
        reused: false,
    })
}

fn remove_intermediates(directory: &Path) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        let keep = path
            .file_name()
            .is_some_and(|name| name == "artifact" || name == artifact::MANIFEST_FILE_NAME);

        if !keep {
            if path.is_dir() {
                fs::remove_dir_all(path)?;
            }
            else {
                fs::remove_file(path)?;
            }
        }
    }

    Ok(())
}

#[cfg(unix)]
fn install_content_addressed(
    configured_path: &Path,
    temporary_root: &Path,
    manifest: &CompiledArtifactManifest,
) -> Result<()> {
    use std::os::unix::fs::symlink;

    let parent = configured_path
        .parent()
        .context("compiled artifact needs a parent directory")?;
    let store = parent.join(".tensorrt-artifacts");
    fs::create_dir_all(&store)?;

    let key = artifact::semantic_cache_key(manifest)?;
    let installed_directory = store.join(&key);
    if installed_directory.exists() {
        let installed_artifact = installed_directory.join("artifact");
        let installed =
            artifact::validate_artifact(&installed_artifact).map_err(anyhow::Error::new)?;
        ensure!(
            &installed == manifest,
            "content-addressed TensorRT target has conflicting metadata"
        );
        fs::remove_dir_all(temporary_root)?;
    }
    else {
        fs::rename(temporary_root, &installed_directory)?;
        File::open(&store)?.sync_all()?;
    }

    let relative_target = Path::new(".tensorrt-artifacts").join(&key).join("artifact");
    let temporary_link = artifact::unique_sibling(configured_path);
    let mut cleanup = CleanupFile::new(temporary_link.clone());
    symlink(relative_target, &temporary_link)?;
    fs::rename(&temporary_link, configured_path)?;
    cleanup.disarm();
    File::open(parent)?.sync_all()?;

    Ok(())
}

#[cfg(not(unix))]
fn install_content_addressed(
    _configured_path: &Path,
    _temporary_root: &Path,
    _manifest: &CompiledArtifactManifest,
) -> Result<()> {
    bail!("TensorRT artifact installation requires Unix atomic symlinks")
}

fn compile_to_temporary(
    request: &CompileRequest<'_>,
    temporary_root: &Path,
    environment: &ArtifactEnvironment,
) -> Result<CompilerResult> {
    let torchscript = temporary_root.join("model.ts");
    export_torchscript_for_model(
        request.model,
        request.checkpoint,
        &torchscript,
        request.device,
    )?;

    let artifact = temporary_root.join("artifact");
    let result_path = temporary_root.join("build-result.json");
    let channels = request.model.state_shape()[0];

    let mut command = compiler_command(request.compiler);
    match request.backend {
        CompiledBackendKind::TensorRtTorchScript => {
            command
                .arg("--input")
                .arg(&torchscript)
                .arg("--output")
                .arg(&artifact);
        }
        CompiledBackendKind::TensorRtRaw => {
            let onnx = temporary_root.join("model.onnx");
            let export_result = temporary_root.join("export-result.json");
            let export_python = request
                .compiler
                .export_python
                .as_ref()
                .unwrap_or(&request.compiler.python);
            let mut export = Command::new(export_python);
            sanitize_python_command(&mut export);
            export
                .arg(&request.compiler.script)
                .arg("export-onnx")
                .arg("--input")
                .arg(&torchscript)
                .arg("--output")
                .arg(&onnx)
                .arg("--channels")
                .arg(channels.to_string())
                .arg("--precision")
                .arg(precision_name(request.compiler.precision))
                .arg("--result-json")
                .arg(export_result);
            run_command(&mut export, "ONNX exporter")?;

            command
                .arg("build-engine")
                .arg("--input")
                .arg(onnx)
                .arg("--output")
                .arg(&artifact);

            if let Some(cache) = timing_cache_path(request, environment)? {
                command.arg("--timing-cache").arg(cache);
            }
        }
    }

    command
        .arg("--channels")
        .arg(channels.to_string())
        .arg("--min-batch-size")
        .arg(request.compiler.batch_shapes.min.to_string())
        .arg("--opt-batch-size")
        .arg(request.compiler.batch_shapes.optimal.to_string())
        .arg("--max-batch-size")
        .arg(request.compiler.batch_shapes.max.to_string())
        .arg("--precision")
        .arg(precision_name(request.compiler.precision))
        .arg("--result-json")
        .arg(&result_path);
    run_command(&mut command, "TensorRT compiler")?;

    read_compiler_result(&result_path)
}

fn inspect_environment(
    compiler: &TensorRtCompiler,
    output_directory: &Path,
) -> Result<CompilerResult> {
    let result_path = artifact::unique_sibling(&output_directory.join("environment.json"));
    let mut cleanup = CleanupFile::new(result_path.clone());
    let mut command = compiler_command(compiler);
    command
        .arg("inspect-environment")
        .arg("--result-json")
        .arg(&result_path);
    run_command(&mut command, "TensorRT environment inspection")?;

    let result = read_compiler_result(&result_path)?;
    cleanup.remove()?;

    ensure!(
        result.operation == "inspect-environment",
        "unexpected compiler operation {}",
        result.operation
    );

    Ok(result)
}

fn compiler_command(compiler: &TensorRtCompiler) -> Command {
    let mut command = Command::new(&compiler.python);
    sanitize_python_command(&mut command);
    command.arg(&compiler.script);

    command
}

fn sanitize_python_command(command: &mut Command) {
    command.env_remove("LD_PRELOAD");
}

fn run_command(command: &mut Command, description: &str) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("starting {description}"))?;

    if !output.stderr.is_empty() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
    if !output.status.success() {
        bail!(
            "{description} failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout).trim()
        );
    }

    Ok(())
}

fn read_compiler_result(path: &Path) -> Result<CompilerResult> {
    let result: CompilerResult = serde_json::from_slice(&fs::read(path)?)
        .context("invalid TensorRT compiler result JSON")?;
    ensure!(
        result.schema_version == 1,
        "unsupported TensorRT compiler result schema {}",
        result.schema_version
    );

    Ok(result)
}

fn validate_compiler_result(request: &CompileRequest<'_>, result: &CompilerResult) -> Result<()> {
    let expected_kind = match request.backend {
        CompiledBackendKind::TensorRtTorchScript => "tensor-rt-torch-script",
        CompiledBackendKind::TensorRtRaw => "tensor-rt-raw",
    };
    ensure!(
        result.artifact_kind.as_deref() == Some(expected_kind),
        "compiler returned the wrong artifact kind"
    );
    ensure!(
        result.internal_precision.as_deref() == Some(precision_name(request.compiler.precision)),
        "compiler returned the wrong precision policy"
    );

    let profile = result
        .profile
        .as_ref()
        .context("compiler result omitted profile")?;
    let expected = request.compiler.batch_shapes;
    ensure!(
        (profile.min, profile.opt, profile.max) == (expected.min, expected.optimal, expected.max),
        "compiler returned the wrong optimization profile"
    );

    let expected_dtype = expected_dtype(request.compiler.precision);
    ensure!(
        result.input_dtype.as_deref() == Some(expected_dtype),
        "compiler input dtype does not satisfy requested precision"
    );
    ensure!(
        result.output_dtype.as_deref() == Some(expected_dtype),
        "compiler output dtype does not satisfy requested precision"
    );

    Ok(())
}

fn timing_cache_path(
    request: &CompileRequest<'_>,
    environment: &ArtifactEnvironment,
) -> Result<Option<PathBuf>> {
    match &request.compiler.timing_cache {
        TimingCachePolicy::Disabled => Ok(None),
        TimingCachePolicy::Explicit(path) => Ok(Some(path.clone())),
        TimingCachePolicy::Auto => {
            let key = timing_cache_key(request.model, environment, request.compiler)?;
            let parent = request.output.parent().context("artifact needs a parent")?;
            let directory = parent.join("tensorrt").join("timing");
            fs::create_dir_all(&directory)?;

            Ok(Some(directory.join(format!("{key}.cache"))))
        }
    }
}

fn timing_cache_key(
    model: &ModelSpec,
    environment: &ArtifactEnvironment,
    compiler: &TensorRtCompiler,
) -> Result<String> {
    artifact::semantic_cache_key(&serde_json::json!({
        "schema_version": 1,
        "model_fingerprint": model_fingerprint(model),
        "environment": environment,
        "compiler_script_sha256": artifact::sha256_file(&compiler.script)?,
        "precision": compiler.precision,
        "profile": {
            "min": compiler.batch_shapes.min,
            "opt": compiler.batch_shapes.optimal,
            "max": compiler.batch_shapes.max,
        },
        "workspace_bytes": null,
        "optimization_level": null,
    }))
}

fn build_config(request: &CompileRequest<'_>) -> ArtifactBuildConfig {
    let shapes = request.compiler.batch_shapes;

    ArtifactBuildConfig {
        requested_precision: request.compiler.precision,
        min_batch: shapes.min,
        opt_batch: shapes.optimal,
        max_batch: shapes.max,
        workspace_bytes: None,
        optimization_level: None,
        onnx_opset: (request.backend == CompiledBackendKind::TensorRtRaw).then_some(18),
    }
}

fn artifact_environment(result: &CompilerResult) -> ArtifactEnvironment {
    ArtifactEnvironment {
        platform: std::env::consts::OS.to_owned(),
        architecture: std::env::consts::ARCH.to_owned(),
        tensorrt_version: result.tensorrt_version.clone(),
        cuda_runtime_version: result.cuda_runtime_version.clone(),
        cuda_driver_version: None,
        gpu_name: result.gpu_name.clone(),
        compute_capability: result.compute_capability.clone(),
        device_properties: Default::default(),
        torch_version: result.torch_version.clone(),
        torch_tensorrt_version: result.torch_tensorrt_version.clone(),
        onnx_version: None,
        onnx_exporter: None,
    }
}

fn compiler_identity(
    compiler: &TensorRtCompiler,
    result: &CompilerResult,
) -> Result<CompilerIdentity> {
    Ok(CompilerIdentity {
        script_name: compiler
            .script
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        script_sha256: artifact::sha256_file(&compiler.script)?,
        repository_revision: None,
        tool_version: result.tensorrt_version.clone(),
        python_identity: format!("{} {}", compiler.python.display(), result.python_version),
    })
}

fn expected_io(model: &ModelSpec, precision: TensorRtBuildPrecision) -> ArtifactIoContract {
    let [channels, height, width] = model.state_shape();
    let dtype = if precision == TensorRtBuildPrecision::Fp16 {
        ArtifactDType::Fp16
    }
    else {
        ArtifactDType::Fp32
    };

    ArtifactIoContract {
        input_dtype: dtype,
        output_dtype: dtype,
        channels: channels as usize,
        height: height as usize,
        width: width as usize,
        output_columns: model.action_size() + 1,
        value_layout: ValueLayout::Scalar,
    }
}

fn parse_dtype(value: &str) -> Result<ArtifactDType> {
    match value {
        "fp16" => Ok(ArtifactDType::Fp16),
        "fp32" => Ok(ArtifactDType::Fp32),
        _ => bail!("unsupported compiler dtype {value}"),
    }
}

fn expected_dtype(precision: TensorRtBuildPrecision) -> &'static str {
    if precision == TensorRtBuildPrecision::Fp16 {
        "fp16"
    }
    else {
        "fp32"
    }
}

fn precision_name(precision: TensorRtBuildPrecision) -> &'static str {
    match precision {
        TensorRtBuildPrecision::Fp16 => "fp16",
        TensorRtBuildPrecision::Fp32 => "fp32",
        TensorRtBuildPrecision::MixedFp32IoFp16Tactics => "mixed-fp32-io-fp16-tactics",
    }
}

fn model_fingerprint(model: &ModelSpec) -> String {
    model
        .fingerprint()
        .0
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn compile_latest_tensor_rt(
    run_dir: &RunDir,
    state: &RunState,
    experiment: &ExperimentConfig,
    device: Device,
    compiler: &TensorRtCompiler,
) -> Result<CompiledArtifact> {
    let checkpoint = run_dir
        .latest_checkpoint(state)
        .context("run has no current checkpoint")?;
    let output = configured_artifact_path(run_dir, experiment)?;
    let backend = backend_kind(experiment.inference.engine)?;

    compile_checkpoint_to_artifact(&CompileRequest {
        checkpoint: &checkpoint,
        model: &experiment.model,
        backend,
        output: &output,
        device,
        compiler,
    })
}

pub fn compile_checkpoint_tensor_rt(
    experiment: &ExperimentConfig,
    checkpoint: &Path,
    output: &Path,
    device: Device,
    compiler: &TensorRtCompiler,
) -> Result<PathBuf> {
    let artifact = compile_checkpoint_to_artifact(&CompileRequest {
        checkpoint,
        model: &experiment.model,
        backend: backend_kind(experiment.inference.engine)?,
        output,
        device,
        compiler,
    })?;

    Ok(artifact.artifact_path)
}

fn backend_kind(engine: InferenceEngine) -> Result<CompiledBackendKind> {
    match engine {
        InferenceEngine::TensorRtTorchScript => Ok(CompiledBackendKind::TensorRtTorchScript),
        InferenceEngine::TensorRtRaw => Ok(CompiledBackendKind::TensorRtRaw),
        InferenceEngine::Native => bail!("native inference has no TensorRT artifact"),
    }
}

pub fn ensure_latest_checkpoint(
    run_dir: &RunDir,
    experiment: &ExperimentConfig,
    state: &mut RunState,
    device: Device,
) -> Result<()> {
    if run_dir.latest_checkpoint(state).is_some() {
        return Ok(());
    }

    let var_store = nn::VarStore::new(device);
    let _network = Network::new(&var_store.root(), &experiment.model)?;

    run_dir.write_latest(state, |path| Ok(var_store.save(path)?))
}

pub fn export_torchscript(
    experiment: &ExperimentConfig,
    checkpoint: &Path,
    output: &Path,
    device: Device,
) -> Result<()> {
    export_torchscript_for_model(&experiment.model, checkpoint, output, device)
}

fn export_torchscript_for_model(
    model: &ModelSpec,
    checkpoint: &Path,
    output: &Path,
    device: Device,
) -> Result<()> {
    let mut var_store = nn::VarStore::new(device);
    let network = Network::new(&var_store.root(), model)?;
    var_store.load(checkpoint)?;
    var_store.freeze();

    let [channels, height, width] = model.state_shape();
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

pub fn configured_artifact_path(
    run_dir: &RunDir,
    experiment: &ExperimentConfig,
) -> Result<PathBuf> {
    let artifact = experiment
        .inference
        .compiled_artifact
        .as_ref()
        .context("TensorRT inference requires inference.compiled_artifact")?;
    ensure!(
        !artifact.is_absolute(),
        "inference.compiled_artifact must be relative to the run directory"
    );

    Ok(run_dir.root().join(artifact))
}

#[cfg(unix)]
struct ArtifactLock(File);

#[cfg(unix)]
impl ArtifactLock {
    fn acquire(artifact: &Path) -> Result<Self> {
        let path = artifact.with_extension("compile.lock");
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        let status = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        ensure!(status == 0, "failed to lock TensorRT artifact");

        Ok(Self(file))
    }
}

#[cfg(unix)]
impl Drop for ArtifactLock {
    fn drop(&mut self) {
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

#[cfg(not(unix))]
struct ArtifactLock;

#[cfg(not(unix))]
impl ArtifactLock {
    fn acquire(_artifact: &Path) -> Result<Self> {
        bail!("TensorRT compilation locking requires Unix flock support")
    }
}

struct CleanupDirectory {
    path: PathBuf,
    keep: bool,
}

impl CleanupDirectory {
    fn new(path: PathBuf, keep: bool) -> Self {
        Self { path, keep }
    }
}

impl Drop for CleanupDirectory {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

struct CleanupFile(PathBuf);

impl CleanupFile {
    fn new(path: PathBuf) -> Self {
        Self(path)
    }
    fn remove(&mut self) -> Result<()> {
        fs::remove_file(&self.0)?;
        self.0.clear();
        Ok(())
    }

    fn disarm(&mut self) {
        self.0.clear();
    }
}

impl Drop for CleanupFile {
    fn drop(&mut self) {
        if !self.0.as_os_str().is_empty() {
            let _ = fs::remove_file(&self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_shapes_reject_invalid_ranges() {
        assert!(TensorRtBatchShapes {
            min: 32,
            optimal: 16,
            max: 64
        }
        .validate()
        .is_err());
    }

    #[test]
    fn compiler_protocol_rejects_an_unknown_schema() {
        let path = std::env::temp_dir().join(format!(
            "engine-zoo-compiler-result-{}.json",
            std::process::id()
        ));
        fs::write(
            &path,
            r#"{"schema_version":2,"operation":"inspect-environment","tensorrt_version":"1","cuda_runtime_version":"1","gpu_name":"gpu","compute_capability":"1","python_version":"1"}"#,
        )
        .unwrap();

        let error = read_compiler_result(&path).unwrap_err();
        let _ = fs::remove_file(path);

        assert!(error
            .to_string()
            .contains("unsupported TensorRT compiler result schema"));
    }

    #[test]
    fn exact_fp16_requires_fp16_io() {
        assert_eq!(expected_dtype(TensorRtBuildPrecision::Fp16), "fp16");
        assert_eq!(
            expected_dtype(TensorRtBuildPrecision::MixedFp32IoFp16Tactics),
            "fp32"
        );
    }
}

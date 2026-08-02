mod identity;
mod manifest;

pub use identity::{semantic_cache_key, sha256_file};
pub use manifest::{
    ArtifactBuildConfig, ArtifactCacheIdentity, ArtifactDType, ArtifactEnvironment,
    ArtifactIoContract, CompiledArtifactManifest, CompiledBackendKind, CompilerIdentity,
    RecompileReason, RecompileRequired, TensorRtBuildPrecision, ValueLayout,
    ARTIFACT_MANIFEST_VERSION,
};

use anyhow::{Context, Result};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMPORARY_ID: AtomicU64 = AtomicU64::new(0);

pub const MANIFEST_FILE_NAME: &str = "manifest.json";

pub fn read_manifest(
    artifact_path: &Path,
) -> std::result::Result<CompiledArtifactManifest, RecompileRequired> {
    let path = manifest_path(artifact_path);
    let bytes = fs::read(&path).map_err(|_| RecompileRequired {
        reason: RecompileReason::MissingManifest,
    })?;

    let manifest = serde_json::from_slice(&bytes).map_err(|_| RecompileRequired {
        reason: RecompileReason::MalformedManifest,
    })?;

    Ok(manifest)
}

pub fn validate_artifact(
    artifact_path: &Path,
) -> std::result::Result<CompiledArtifactManifest, RecompileRequired> {
    let manifest = read_manifest(artifact_path)?;
    manifest.validate_artifact(artifact_path)?;

    Ok(manifest)
}

pub fn manifest_path(artifact_path: &Path) -> PathBuf {
    let artifact_path = if fs::symlink_metadata(artifact_path)
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        fs::canonicalize(artifact_path).unwrap_or_else(|_| artifact_path.to_path_buf())
    }
    else {
        artifact_path.to_path_buf()
    };

    artifact_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(MANIFEST_FILE_NAME)
}

pub fn write_manifest_last(
    artifact_path: &Path,
    manifest: &CompiledArtifactManifest,
) -> Result<()> {
    manifest.validate_schema()?;

    let path = manifest_path(artifact_path);
    let temporary = unique_sibling(&path);
    let mut cleanup = TemporaryPath::new(temporary.clone());
    let mut file = File::create(&temporary)?;

    serde_json::to_writer_pretty(&mut file, manifest)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&temporary, &path)?;
    cleanup.keep();
    sync_parent(&path)?;

    Ok(())
}

pub fn unique_sibling(path: &Path) -> PathBuf {
    let id = TEMPORARY_ID.fetch_add(1, Ordering::Relaxed);
    let suffix = format!(".tmp-{}-{id}", std::process::id());
    let mut name = OsString::from(".");

    if !path.is_dir() {
        if let (Some(stem), Some(extension)) = (path.file_stem(), path.extension()) {
            name.push(stem);
            name.push(&suffix);
            name.push(".");
            name.push(extension);

            return path.with_file_name(name);
        }
    }

    name.push(path.file_name().unwrap_or_default());
    name.push(suffix);

    path.with_file_name(name)
}

pub(crate) fn sync_parent(path: &Path) -> Result<()> {
    let parent = path.parent().context("installed path must have a parent")?;

    File::open(parent)?.sync_all()?;
    Ok(())
}

pub(crate) struct TemporaryPath {
    path: PathBuf,
    keep: bool,
}

impl TemporaryPath {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self { path, keep: false }
    }

    pub(crate) fn keep(&mut self) {
        self.keep = true;
    }
}

impl Drop for TemporaryPath {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_file(&self.path);
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelSpec;
    use std::fs;

    fn manifest(checkpoint_sha256: &str) -> CompiledArtifactManifest {
        let model = ModelSpec::connect4_basic(1, 4);
        let fingerprint = model
            .fingerprint()
            .0
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();

        CompiledArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_VERSION,
            backend: CompiledBackendKind::TensorRtRaw,
            checkpoint_sha256: checkpoint_sha256.into(),
            model_fingerprint: fingerprint,
            artifact_sha256: String::new(),
            io: ArtifactIoContract {
                input_dtype: ArtifactDType::Fp32,
                output_dtype: ArtifactDType::Fp32,
                channels: 1,
                height: 6,
                width: 7,
                output_columns: 8,
                value_layout: ValueLayout::Scalar,
            },
            build: ArtifactBuildConfig {
                requested_precision: TensorRtBuildPrecision::Fp32,
                min_batch: 1,
                opt_batch: 8,
                max_batch: 16,
                workspace_bytes: None,
                optimization_level: None,
                onnx_opset: Some(17),
            },
            environment: ArtifactEnvironment::default(),
            compiler: CompilerIdentity {
                script_name: "compiler.py".into(),
                script_sha256: "script".into(),
                repository_revision: None,
                tool_version: "1".into(),
                python_identity: "python".into(),
            },
        }
    }

    #[test]
    fn cache_identity_uses_checkpoint_bytes_not_path_metadata() {
        let first = manifest("checkpoint-a");
        let same = manifest("checkpoint-a");
        let changed = manifest("checkpoint-b");

        assert_eq!(
            semantic_cache_key(&first.cache_identity()).unwrap(),
            semantic_cache_key(&same.cache_identity()).unwrap()
        );
        assert_ne!(
            semantic_cache_key(&first.cache_identity()).unwrap(),
            semantic_cache_key(&changed.cache_identity()).unwrap()
        );
    }

    #[test]
    fn unique_sibling_preserves_checkpoint_extension() {
        let checkpoint = Path::new("checkpoints/latest.safetensors");
        let temporary = unique_sibling(checkpoint);

        assert_eq!(temporary.parent(), checkpoint.parent());
        assert_eq!(temporary.extension(), checkpoint.extension());
        assert!(temporary
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with(".latest.tmp-"));

        let extensionless = Path::new("compiled");
        assert!(unique_sibling(extensionless)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with(".compiled.tmp-"));
    }

    #[test]
    fn manifest_is_required_and_artifact_digest_is_checked() {
        let root = std::env::temp_dir().join(format!(
            "engine-zoo-artifact-{}-{}",
            std::process::id(),
            TEMPORARY_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let artifact = root.join("artifact.engine");
        fs::write(&artifact, b"engine").unwrap();

        assert_eq!(
            validate_artifact(&artifact).unwrap_err().reason,
            RecompileReason::MissingManifest
        );

        let mut expected = manifest("checkpoint");
        expected.artifact_sha256 = sha256_file(&artifact).unwrap();
        write_manifest_last(&artifact, &expected).unwrap();
        assert_eq!(validate_artifact(&artifact).unwrap(), expected);

        fs::write(&artifact, b"truncated").unwrap();
        assert_eq!(
            validate_artifact(&artifact).unwrap_err().reason,
            RecompileReason::ArtifactDigest
        );

        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn configured_symlink_uses_target_local_manifest() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!(
            "engine-zoo-artifact-link-{}-{}",
            std::process::id(),
            TEMPORARY_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let target = root.join("targets/one");
        fs::create_dir_all(&target).unwrap();

        let artifact = target.join("artifact");
        fs::write(&artifact, b"engine").unwrap();
        let mut expected = manifest("checkpoint");
        expected.artifact_sha256 = sha256_file(&artifact).unwrap();
        write_manifest_last(&artifact, &expected).unwrap();

        let configured = root.join("model.engine");
        symlink("targets/one/artifact", &configured).unwrap();

        assert_eq!(manifest_path(&configured), target.join(MANIFEST_FILE_NAME));
        assert_eq!(validate_artifact(&configured).unwrap(), expected);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn model_contract_validation_is_exact() {
        let model = ModelSpec::connect4_basic(1, 4);
        let expected = manifest("checkpoint");
        expected.validate_model(&model).unwrap();

        let other = ModelSpec::connect4_basic(1, 8);
        assert!(expected.validate_model(&other).is_err());
    }
}

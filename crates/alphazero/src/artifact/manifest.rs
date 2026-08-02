use super::identity::sha256_file;
use crate::ModelSpec;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

pub const ARTIFACT_MANIFEST_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompiledBackendKind {
    TensorRtTorchScript,
    TensorRtRaw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactDType {
    Fp16,
    Fp32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ValueLayout {
    Scalar,
    WinDrawLoss,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TensorRtBuildPrecision {
    Fp32,
    Fp16,
    MixedFp32IoFp16Tactics,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactIoContract {
    pub input_dtype: ArtifactDType,
    pub output_dtype: ArtifactDType,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub output_columns: usize,
    pub value_layout: ValueLayout,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactBuildConfig {
    pub requested_precision: TensorRtBuildPrecision,
    pub min_batch: usize,
    pub opt_batch: usize,
    pub max_batch: usize,
    pub workspace_bytes: Option<u64>,
    pub optimization_level: Option<u8>,
    pub onnx_opset: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactEnvironment {
    pub platform: String,
    pub architecture: String,
    pub tensorrt_version: String,
    pub cuda_runtime_version: String,
    pub cuda_driver_version: Option<String>,
    pub gpu_name: String,
    pub compute_capability: String,
    #[serde(default)]
    pub device_properties: BTreeMap<String, String>,
    pub torch_version: Option<String>,
    pub torch_tensorrt_version: Option<String>,
    pub onnx_version: Option<String>,
    pub onnx_exporter: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompilerIdentity {
    pub script_name: String,
    pub script_sha256: String,
    pub repository_revision: Option<String>,
    pub tool_version: String,
    pub python_identity: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledArtifactManifest {
    pub schema_version: u32,
    pub backend: CompiledBackendKind,
    pub checkpoint_sha256: String,
    pub model_fingerprint: String,
    pub artifact_sha256: String,
    pub io: ArtifactIoContract,
    pub build: ArtifactBuildConfig,
    pub environment: ArtifactEnvironment,
    pub compiler: CompilerIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ArtifactCacheIdentity<'a> {
    pub schema_version: u32,
    pub backend: CompiledBackendKind,
    pub checkpoint_sha256: &'a str,
    pub model_fingerprint: &'a str,
    pub io: &'a ArtifactIoContract,
    pub build: &'a ArtifactBuildConfig,
    pub environment: &'a ArtifactEnvironment,
    pub compiler: &'a CompilerIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecompileReason {
    MissingManifest,
    LegacyManifest(u32),
    MalformedManifest,
    StaleIdentity(&'static str),
    ArtifactDigest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecompileRequired {
    pub reason: RecompileReason,
}

impl fmt::Display for RecompileRequired {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "compiled artifact requires recompilation: {:?}",
            self.reason
        )
    }
}

impl std::error::Error for RecompileRequired {}

impl CompiledArtifactManifest {
    pub fn cache_identity(&self) -> ArtifactCacheIdentity<'_> {
        ArtifactCacheIdentity {
            schema_version: self.schema_version,
            backend: self.backend,
            checkpoint_sha256: &self.checkpoint_sha256,
            model_fingerprint: &self.model_fingerprint,
            io: &self.io,
            build: &self.build,
            environment: &self.environment,
            compiler: &self.compiler,
        }
    }

    pub fn validate_schema(&self) -> Result<(), RecompileRequired> {
        if self.schema_version != ARTIFACT_MANIFEST_VERSION {
            return Err(RecompileRequired {
                reason: RecompileReason::LegacyManifest(self.schema_version),
            });
        }

        Ok(())
    }

    pub fn validate_artifact(&self, path: &Path) -> Result<(), RecompileRequired> {
        self.validate_schema()?;

        let digest = sha256_file(path).map_err(|_| RecompileRequired {
            reason: RecompileReason::ArtifactDigest,
        })?;
        if digest != self.artifact_sha256 {
            return Err(RecompileRequired {
                reason: RecompileReason::ArtifactDigest,
            });
        }

        Ok(())
    }

    pub fn validate_identity(
        &self,
        expected: &ArtifactCacheIdentity<'_>,
    ) -> Result<(), RecompileRequired> {
        self.validate_schema()?;

        macro_rules! require_equal {
            ($field:ident) => {
                if self.cache_identity().$field != expected.$field {
                    return Err(RecompileRequired {
                        reason: RecompileReason::StaleIdentity(stringify!($field)),
                    });
                }
            };
        }

        require_equal!(backend);
        require_equal!(checkpoint_sha256);
        require_equal!(model_fingerprint);
        require_equal!(io);
        require_equal!(build);
        require_equal!(environment);
        require_equal!(compiler);

        Ok(())
    }

    pub fn validate_model(&self, model: &ModelSpec) -> Result<(), RecompileRequired> {
        let fingerprint = model.fingerprint();
        let fingerprint = fingerprint
            .0
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        if self.model_fingerprint != fingerprint {
            return Err(RecompileRequired {
                reason: RecompileReason::StaleIdentity("model_fingerprint"),
            });
        }

        let [channels, height, width] = model.state_shape();
        let shape_matches = self.io.channels == channels as usize
            && self.io.height == height as usize
            && self.io.width == width as usize
            && self.io.output_columns == model.action_size() + 1
            && self.io.value_layout == ValueLayout::Scalar;
        if !shape_matches {
            return Err(RecompileRequired {
                reason: RecompileReason::StaleIdentity("io"),
            });
        }

        Ok(())
    }
}

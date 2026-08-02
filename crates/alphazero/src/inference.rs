//! Public inference service and its normal model loader.

use crate::batcher::Batcher;
use crate::network::ModelSpec;
use anyhow::Result;
use std::path::Path;
use tch::Device;

pub use crate::batcher::InferencePrecision;
pub use crate::batcher::{Batcher as InferenceService, BatcherClient as InferenceClient};
pub use crate::batcher::{
    BatcherConfig as InferenceBatchConfig, BatcherError as InferenceError,
    BatcherStats as InferenceStats,
};
pub use crate::experiment::{InferenceConfig, InferenceEngine};

/// Where an inference service obtains its immutable executable model.
#[derive(Clone, Copy, Debug)]
pub enum InferenceSource<'a> {
    Checkpoint(&'a Path),
    TensorRtTorchScript(&'a Path),
    TensorRtEngine(&'a Path),
}

impl Batcher {
    /// Loads an inference backend through the single application-facing path.
    ///
    /// Native services are reloadable. TensorRT TorchScript is deliberately
    /// one-generation only because it cannot reload checkpoint weights.
    pub fn load(
        model: &ModelSpec,
        source: InferenceSource<'_>,
        device: Device,
        config: &InferenceConfig,
    ) -> Result<Self> {
        match (config.engine, source) {
            (InferenceEngine::Native, InferenceSource::Checkpoint(checkpoint)) => {
                Self::new_with_model_precision_and_fp16_host_staging(
                    model.clone(),
                    checkpoint,
                    device,
                    config.batcher_config(),
                    config.precision,
                    config.fp16_host_staging,
                )
            }
            (
                InferenceEngine::TensorRtTorchScript,
                InferenceSource::TensorRtTorchScript(module),
            ) => {
                validate_compiled_source(
                    model,
                    module,
                    crate::artifact::CompiledBackendKind::TensorRtTorchScript,
                    config,
                )?;

                Self::new_with_tensor_rt_torchscript(
                    model.clone(),
                    module,
                    device,
                    config.batcher_config(),
                )
            }
            (InferenceEngine::TensorRtRaw, InferenceSource::TensorRtEngine(module)) => {
                validate_compiled_source(
                    model,
                    module,
                    crate::artifact::CompiledBackendKind::TensorRtRaw,
                    config,
                )?;

                #[cfg(feature = "raw-tensorrt")]
                {
                    Self::new_with_raw_tensor_rt(
                        model.clone(),
                        module,
                        device,
                        config.batcher_config(),
                    )
                }
                #[cfg(not(feature = "raw-tensorrt"))]
                {
                    let _ = (model, module, device, config);
                    anyhow::bail!(
                        "raw TensorRT inference requires building with `--features raw-tensorrt`"
                    )
                }
            }
            (
                InferenceEngine::Native,
                InferenceSource::TensorRtTorchScript(_) | InferenceSource::TensorRtEngine(_),
            ) => {
                anyhow::bail!("native inference requires a checkpoint source")
            }
            (
                InferenceEngine::TensorRtTorchScript | InferenceEngine::TensorRtRaw,
                InferenceSource::Checkpoint(_),
            ) => {
                anyhow::bail!("TensorRT inference requires a compiled module/engine source")
            }
            (InferenceEngine::TensorRtTorchScript, InferenceSource::TensorRtEngine(_)) => {
                anyhow::bail!("Torch-TensorRT inference requires a TorchScript artifact")
            }
            (InferenceEngine::TensorRtRaw, InferenceSource::TensorRtTorchScript(_)) => {
                anyhow::bail!("raw TensorRT inference requires an engine-plan artifact")
            }
        }
    }
}

fn validate_compiled_source(
    model: &ModelSpec,
    path: &Path,
    backend: crate::artifact::CompiledBackendKind,
    config: &InferenceConfig,
) -> Result<()> {
    let manifest = crate::artifact::validate_artifact(path)?;
    if manifest.backend != backend {
        return Err(crate::artifact::RecompileRequired {
            reason: crate::artifact::RecompileReason::StaleIdentity("backend"),
        }
        .into());
    }

    manifest.validate_model(model)?;
    validate_build_config(&manifest.build, config)?;
    Ok(())
}

fn validate_build_config(
    build: &crate::artifact::ArtifactBuildConfig,
    config: &InferenceConfig,
) -> Result<()> {
    let precision = match config.precision {
        InferencePrecision::Fp32 => crate::artifact::TensorRtBuildPrecision::Fp32,
        InferencePrecision::Fp16 => crate::artifact::TensorRtBuildPrecision::Fp16,
    };
    let matches = build.requested_precision == precision
        && build.opt_batch == config.preferred_batch_size
        && build.max_batch == config.max_batch_size;
    if !matches {
        return Err(crate::artifact::RecompileRequired {
            reason: crate::artifact::RecompileReason::StaleIdentity("build"),
        }
        .into());
    }

    Ok(())
}

/// Low-level backends are intentionally separate from the normal loader.
pub mod backend {
    pub use crate::batcher::{
        CombinedEncodedBatch as BackendBatch, InferenceBackend, TchInferenceBackend,
    };
}

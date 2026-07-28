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
    TensorRtModule(&'a Path),
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
                Self::new_with_model_precision(
                    model.clone(),
                    checkpoint,
                    device,
                    config.batcher_config(),
                    config.precision,
                )
            }
            (InferenceEngine::TensorRtTorchScript, InferenceSource::TensorRtModule(module)) => {
                Self::new_with_tensor_rt_torchscript(
                    model.clone(),
                    module,
                    device,
                    config.batcher_config(),
                )
            }
            (InferenceEngine::Native, InferenceSource::TensorRtModule(_)) => {
                anyhow::bail!("native inference requires a checkpoint source")
            }
            (InferenceEngine::TensorRtTorchScript, InferenceSource::Checkpoint(_)) => {
                anyhow::bail!("TensorRT inference requires a TorchScript module source")
            }
        }
    }
}

/// Low-level backends are intentionally separate from the normal loader.
pub mod backend {
    pub use crate::batcher::{
        CombinedEncodedBatch as BackendBatch, InferenceBackend, TchInferenceBackend,
    };
}

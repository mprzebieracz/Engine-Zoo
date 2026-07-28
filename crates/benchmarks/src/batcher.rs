use crate::device::{self, Precision};
use crate::harness;
use crate::report::BenchmarkSample;
use alphazero::{
    Action, Batcher, BatcherConfig, CombinedEncodedBatch, InferenceBackend, ModelSpec, Network,
};
use anyhow::{bail, Result};
use search::{Evaluation, PositionValue};
use serde_json::{json, Value};
use std::path::Path;
use std::time::{Duration, Instant};
use tch::{nn, Device, Kind, Tensor};

pub fn connect4(
    warmup: &str,
    samples: usize,
    config: Value,
    execution_device: Device,
    precision: Precision,
) -> Result<crate::report::BenchmarkReport> {
    let batcher = Batcher::with_backend(
        NetworkBackend::new(execution_device, precision)?,
        BatcherConfig {
            preferred_batch_size: 64,
            max_batch_size: 64,
            max_wait: Duration::ZERO,
            max_queued_states: 256,
        },
    )?;
    let mut client = batcher.client();

    harness::measure(
        "batcher.connect4.neural",
        warmup,
        samples,
        json!({ "request_rows": 64, "backend": "native-network", "device": device::name(execution_device), "precision": precision.name(), "config": config }),
        || {
            let mut batch = request(64);
            device::synchronize(execution_device);
            let started = Instant::now();
            let evaluations = client
                .evaluate(&mut batch)
                .expect("network backend is valid");
            device::synchronize(execution_device);
            let elapsed = started.elapsed().as_nanos();
            let stats = batcher.stats();

            BenchmarkSample {
                elapsed_ns: elapsed,
                operations: evaluations.len() as u64,
                metrics: json!({ "inference_batches": stats.inference_batches, "partial_batches": stats.partial_batches, "queue_wait_ns": stats.queue_wait_total.as_nanos() }),
            }
        },
    )
}

fn request(rows: usize) -> alphazero::EncodedEvalBatch {
    alphazero::EncodedEvalBatch {
        states: vec![0.0; rows * 42],
        legal_actions: (0..rows * 7)
            .map(|action| Action::new((action % 7) as u32))
            .collect(),
        offsets: (0..=rows).map(|row| (row * 7) as u32).collect(),
    }
}

pub(crate) struct NetworkBackend {
    _store: nn::VarStore,
    network: Network,
    device: Device,
    input_kind: Kind,
}

impl NetworkBackend {
    pub(crate) fn new(device: Device, precision: Precision) -> Result<Self> {
        if precision == Precision::Fp16 && !device.is_cuda() {
            bail!("FP16 batcher fixture requires CUDA");
        }

        let mut store = nn::VarStore::new(device);
        let network = Network::new(&store.root(), &ModelSpec::connect4_basic(1, 8))?;
        let input_kind = match precision {
            Precision::Fp32 => Kind::Float,
            Precision::Fp16 => {
                store.half();
                Kind::Half
            }
        };

        Ok(Self {
            _store: store,
            network,
            device,
            input_kind,
        })
    }
}

impl InferenceBackend for NetworkBackend {
    fn evaluate(&mut self, batch: &CombinedEncodedBatch) -> Result<Vec<Evaluation>> {
        let rows = batch.len() as i64;
        let states = Tensor::from_slice(&batch.states)
            .view([rows, 1, 6, 7])
            .to_device(self.device)
            .to_kind(self.input_kind);
        let output = self.network.forward_t(&states, false);
        let policy: Vec<f32> = output
            .policy_logits
            .to_device(Device::Cpu)
            .to_kind(Kind::Float)
            .contiguous()
            .view(-1)
            .try_into()?;
        let values: Vec<f32> = output
            .value
            .expected_value()
            .to_device(Device::Cpu)
            .to_kind(Kind::Float)
            .contiguous()
            .view(-1)
            .try_into()?;

        (0..batch.len())
            .map(|row| {
                let begin = batch.offsets[row] as usize;
                let end = batch.offsets[row + 1] as usize;
                let logits = batch.legal_actions[begin..end]
                    .iter()
                    .map(|action| policy[row * 7 + action.index()])
                    .collect();
                Ok(Evaluation {
                    logits,
                    value: PositionValue::new(values[row])?,
                })
            })
            .collect()
    }

    fn reload_weights(&mut self, _path: &Path) -> Result<()> {
        bail!("benchmark network backend does not load weights")
    }
}

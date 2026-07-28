use crate::harness;
use crate::report::BenchmarkSample;
use alphazero::{
    Action, Batcher, BatcherConfig, CombinedEncodedBatch, EncodedEvalBatch, Evaluation,
    InferenceBackend,
};
use anyhow::Result;
use search::PositionValue;
use serde_json::{json, Value};
use std::path::Path;
use std::time::{Duration, Instant};

pub fn connect4(warmup: &str, samples: usize, config: Value) -> crate::report::BenchmarkReport {
    let batcher = Batcher::with_backend(
        UniformBackend,
        BatcherConfig {
            preferred_batch_size: 8,
            max_batch_size: 64,
            max_wait: Duration::ZERO,
            max_queued_states: 256,
        },
    )
    .expect("fixed batcher configuration is valid");
    let mut client = batcher.client();

    harness::measure("batcher.connect4.uniform", warmup, samples, json!({ "request_rows": 8, "backend": "uniform", "config": config }), || {
        let mut batch = request(8);
        let started = Instant::now();
        let evaluations = client.evaluate(&mut batch).expect("uniform backend is infallible");
        let elapsed = started.elapsed().as_nanos();
        let stats = batcher.stats();

        BenchmarkSample { elapsed_ns: elapsed, operations: evaluations.len() as u64, metrics: json!({ "inference_batches": stats.inference_batches, "partial_batches": stats.partial_batches, "queue_wait_ns": stats.queue_wait_total.as_nanos() }) }
    }).expect("fixed benchmark arguments are valid")
}

fn request(rows: usize) -> EncodedEvalBatch {
    EncodedEvalBatch {
        states: vec![0.0; rows * 42],
        legal_actions: (0..rows * 7)
            .map(|action| Action::new((action % 7) as u32))
            .collect(),
        offsets: (0..=rows).map(|row| (row * 7) as u32).collect(),
    }
}

struct UniformBackend;

impl InferenceBackend for UniformBackend {
    fn evaluate(&mut self, batch: &CombinedEncodedBatch) -> Result<Vec<Evaluation>> {
        Ok((0..batch.len())
            .map(|row| {
                let begin = batch.offsets[row] as usize;
                let end = batch.offsets[row + 1] as usize;
                Evaluation {
                    logits: vec![0.0; end - begin],
                    value: PositionValue::DRAW,
                }
            })
            .collect())
    }

    fn reload_weights(&mut self, _path: &Path) -> Result<()> {
        Ok(())
    }
}

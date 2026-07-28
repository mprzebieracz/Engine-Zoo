use crate::device::{self, Precision};
use crate::harness;
use crate::report::BenchmarkSample;
use alphazero::{ModelSpec, Network};
use serde_json::{json, Value};
use std::time::Instant;
use tch::{nn, Device, Kind, Tensor};

pub fn connect4(
    warmup: &str,
    samples: usize,
    config: Value,
    execution_device: Device,
    precision: Precision,
) -> crate::report::BenchmarkReport {
    assert!(
        precision != Precision::Fp16 || execution_device.is_cuda(),
        "FP16 requires CUDA"
    );

    let mut store = nn::VarStore::new(execution_device);
    let spec = ModelSpec::connect4_basic(1, 8);
    let network = Network::new(&store.root(), &spec).expect("fixed model specification is valid");
    if precision == Precision::Fp16 {
        store.half();
    }
    let input_kind = match precision {
        Precision::Fp32 => Kind::Float,
        Precision::Fp16 => Kind::Half,
    };
    let states = Tensor::zeros([128, 1, 6, 7], (input_kind, execution_device));

    harness::measure("inference.connect4.forward", warmup, samples, json!({ "model": "connect4-residual-1x8", "batch_size": 128, "device": device::name(execution_device), "precision": precision.name(), "config": config }), || {
        device::synchronize(execution_device);
        let started = Instant::now();
        let output = network.forward_t(&states, false);
        device::synchronize(execution_device);
        let elapsed = started.elapsed().as_nanos();

        BenchmarkSample { elapsed_ns: elapsed, operations: 128, metrics: json!({ "policy_shape": output.policy_logits.size(), "value_shape": output.value.expected_value().size() }) }
    }).expect("fixed benchmark arguments are valid")
}

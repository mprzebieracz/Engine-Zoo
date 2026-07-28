use crate::harness;
use crate::report::BenchmarkSample;
use alphazero::{ModelSpec, Network};
use serde_json::{json, Value};
use std::time::Instant;
use tch::{nn, Device, Kind, Tensor};

pub fn connect4(warmup: &str, samples: usize, config: Value) -> crate::report::BenchmarkReport {
    let store = nn::VarStore::new(Device::Cpu);
    let spec = ModelSpec::connect4_basic(1, 8);
    let network = Network::new(&store.root(), &spec).expect("fixed model specification is valid");
    let states = Tensor::zeros([32, 1, 6, 7], (Kind::Float, Device::Cpu));

    harness::measure("inference.connect4.forward", warmup, samples, json!({ "model": "connect4-residual-1x8", "batch_size": 32, "device": "cpu", "config": config }), || {
        let started = Instant::now();
        let output = network.forward_t(&states, false);
        let elapsed = started.elapsed().as_nanos();

        BenchmarkSample { elapsed_ns: elapsed, operations: 32, metrics: json!({ "policy_shape": output.policy_logits.size(), "value_shape": output.value.expected_value().size() }) }
    }).expect("fixed benchmark arguments are valid")
}

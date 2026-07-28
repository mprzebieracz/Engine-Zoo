use crate::harness;
use crate::report::BenchmarkSample;
use alphazero::representation::{AlphaZeroRepresentation, Connect4AzRepresentation};
use games::Connect4;
use serde_json::{json, Value};
use std::time::Instant;

pub fn connect4(warmup: &str, samples: usize, config: Value) -> crate::report::BenchmarkReport {
    let representation = Connect4AzRepresentation;
    let game = Connect4::default();
    let state_size = Connect4AzRepresentation::state_size();

    harness::measure("representation.connect4.encode", warmup, samples, json!({ "game": "connect4", "state_size": state_size, "config": config }), || {
        let mut output = vec![0.0; state_size];
        let started = Instant::now();
        for _ in 0..1_000 { representation.encode_state(&game, &mut output); std::hint::black_box(&output); }
        BenchmarkSample { elapsed_ns: started.elapsed().as_nanos(), operations: 1_000, metrics: json!({ "bytes_written": state_size * std::mem::size_of::<f32>() * 1_000 }) }
    }).expect("validated CLI arguments")
}

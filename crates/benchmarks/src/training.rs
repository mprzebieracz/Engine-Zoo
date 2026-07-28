use crate::harness;
use crate::report::BenchmarkSample;
use alphazero::representation::Connect4AzRepresentation;
use alphazero::{
    build_optimizer, train, Action, ModelSpec, Network, Outcome, ReplayBuffer, ReplaySample,
    SampleMetadata, TrainConfig, TrainingWeights,
};
use games::Connect4;
use serde_json::{json, Value};
use std::time::Instant;
use tch::{nn, Device};

pub fn connect4(
    warmup: &str,
    samples: usize,
    workload_config: Value,
) -> crate::report::BenchmarkReport {
    let replay = ReplayBuffer::new(1_024, 7);
    replay.add((0..1_024).map(sample));
    let store = nn::VarStore::new(Device::Cpu);
    let network = Network::new(&store.root(), &ModelSpec::connect4_basic(1, 8))
        .expect("fixed model specification is valid");
    let train_config = TrainConfig {
        batch_size: 64,
        micro_batch_size: 32,
        train_steps: 1,
        progress_every: 1,
        ..TrainConfig::default()
    };
    let mut optimizer =
        build_optimizer(&store, &train_config).expect("fixed training configuration is valid");

    harness::measure("training.connect4.cpu", warmup, samples, json!({ "model": "connect4-residual-1x8", "batch_size": 64, "steps": 1, "train_config": train_config, "config": workload_config }), || {
        let started = Instant::now();
        let metrics = train(&network, &mut optimizer, &replay, &Connect4AzRepresentation, Device::Cpu, &train_config, 1, 0).expect("prefilled replay produces training metrics");

        BenchmarkSample { elapsed_ns: started.elapsed().as_nanos(), operations: 64, metrics: json!({ "samples_per_second": metrics.samples_per_second, "policy_loss": metrics.policy_loss, "value_loss": metrics.value_loss }) }
    }).expect("fixed benchmark arguments are valid")
}

fn sample(game_id: usize) -> ReplaySample<Connect4> {
    ReplaySample {
        state: Connect4::default(),
        policy: vec![(Action::new(0), 1.0)].into(),
        outcome: Outcome::Draw,
        weights: TrainingWeights::default(),
        metadata: SampleMetadata {
            game_id: game_id as u64,
            ..Default::default()
        },
    }
}

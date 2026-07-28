use crate::device::{self, Precision};
use crate::harness;
use crate::report::BenchmarkSample;
use alphazero::representation::Connect4AzRepresentation;
use alphazero::{
    build_optimizer, train, Action, ModelSpec, Network, Outcome, ReplayBuffer, ReplaySample,
    SampleMetadata, TrainConfig, TrainingWeights,
};
use anyhow::{bail, Result};
use games::Connect4;
use serde_json::{json, Value};
use std::time::Instant;
use tch::{nn, Device};

pub fn connect4(
    warmup: &str,
    samples: usize,
    workload_config: Value,
    execution_device: Device,
    precision: Precision,
) -> Result<crate::report::BenchmarkReport> {
    if precision != Precision::Fp32 {
        bail!("training fixtures currently support --precision fp32 only")
    }
    let replay = ReplayBuffer::new(1_024, 7);
    replay.add((0..1_024).map(sample));
    let store = nn::VarStore::new(execution_device);
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

    harness::measure(
        "training.connect4.step",
        warmup,
        samples,
        json!({ "model": "connect4-residual-1x8", "batch_size": 64, "steps": 1, "device": device::name(execution_device), "precision": precision.name(), "train_config": train_config, "config": workload_config }),
        || {
            device::synchronize(execution_device);
            let started = Instant::now();
            let metrics = train(
                &network,
                &mut optimizer,
                &replay,
                &Connect4AzRepresentation,
                execution_device,
                &train_config,
                1,
                0,
            )
            .expect("prefilled replay produces training metrics");
            device::synchronize(execution_device);

            BenchmarkSample {
                elapsed_ns: started.elapsed().as_nanos(),
                operations: 64,
                metrics: json!({ "samples_per_second": metrics.samples_per_second, "policy_loss": metrics.policy_loss, "value_loss": metrics.value_loss }),
            }
        },
    )
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

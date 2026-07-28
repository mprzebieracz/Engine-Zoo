use crate::harness;
use crate::report::BenchmarkSample;
use alphazero::representation::Connect4AzRepresentation;
use alphazero::{Action, Outcome, ReplayBuffer, ReplaySample, SampleMetadata, TrainingWeights};
use games::Connect4;
use rand::{rngs::SmallRng, SeedableRng};
use serde_json::{json, Value};
use std::time::Instant;

pub fn connect4(warmup: &str, samples: usize, config: Value) -> crate::report::BenchmarkReport {
    let replay = ReplayBuffer::new(4_096, 7);
    replay.add((0..4_096).map(sample));

    harness::measure(
        "replay.connect4.sample",
        warmup,
        samples,
        json!({ "positions": replay.len(), "batch_size": 256, "config": config }),
        || {
            let mut rng = SmallRng::seed_from_u64(1);
            let started = Instant::now();
            let batch = replay
                .sample(256, &Connect4AzRepresentation, &mut rng)
                .expect("prefilled replay has enough samples");

            BenchmarkSample {
                elapsed_ns: started.elapsed().as_nanos(),
                operations: 256,
                metrics: json!({ "sparse_policy_entries": batch.policies.actions.size()[0] }),
            }
        },
    )
    .expect("fixed benchmark arguments are valid")
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

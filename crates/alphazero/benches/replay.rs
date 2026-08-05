use alphazero::representation::Connect4AzRepresentation;
use alphazero::{
    Action, Outcome, ReplayBuffer, ReplaySample, ReplaySampler, SampleMetadata, TrainingWeights,
};
use games::Connect4;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    const CAPACITY: usize = 8_192;
    const SAMPLES: usize = 200;
    let replay = ReplayBuffer::new(CAPACITY, 7);
    replay.add((0..CAPACITY).map(|game_id| ReplaySample {
        state: Connect4::default(),
        policy: vec![(Action::new(0), 1.0)].into(),
        outcome: Outcome::Draw,
        weights: TrainingWeights::default(),
        metadata: SampleMetadata {
            game_id: game_id as u64,
            ..Default::default()
        },
    }));

    for batch_size in [256, 1_024, 4_096] {
        let representation = Connect4AzRepresentation;
        let mut sampler = ReplaySampler::new(representation);
        let mut rng = SmallRng::seed_from_u64(1);
        let started = Instant::now();
        for _ in 0..SAMPLES {
            std::hint::black_box(sampler.sample(&replay, batch_size, &mut rng));
        }
        let elapsed = started.elapsed();
        let positions = SAMPLES * batch_size;
        println!(
            "replay batch={batch_size}: {positions} positions in {elapsed:?}; {:.0} positions/s",
            positions as f64 / elapsed.as_secs_f64()
        );
    }
}

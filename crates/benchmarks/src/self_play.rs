use crate::batcher::NetworkBackend;
use crate::device::{self, Precision};
use crate::harness;
use crate::report::BenchmarkSample;
use alphazero::representation::Connect4AzRepresentation;
use alphazero::{Batcher, BatcherConfig, RepresentedEvaluator};
use anyhow::Result;
use engine_core::agent::PolicyMode;
use engine_core::game::GameState;
use games::Connect4;
use search::{Mcts, NoExtraRules, PuctConfig, SearchConfig};
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use tch::Device;

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
            preferred_batch_size: 32,
            max_batch_size: 64,
            max_wait: Duration::ZERO,
            max_queued_states: 256,
        },
    )?;
    let evaluator = RepresentedEvaluator::new(Connect4AzRepresentation, batcher.client());
    let mut worker = Mcts::new(
        evaluator,
        SearchConfig::Puct(PuctConfig {
            leaf_batch_size: 32,
            root_noise: None,
            ..PuctConfig::default()
        }),
        NoExtraRules,
    )
    .with_seed(1);

    harness::measure(
        "self-play.connect4.neural",
        warmup,
        samples,
        json!({ "game": "connect4", "moves_per_game": 12, "model": "connect4-residual-1x8", "device": device::name(execution_device), "precision": precision.name(), "config": config }),
        || {
            let mut game = Connect4::default();
            device::synchronize(execution_device);
            let started = Instant::now();
            let mut simulations = 0;
            let mut moves = 0;

            for _ in 0..12 {
                let result = crate::search::run(&mut worker, &game, 64, PolicyMode::Explore);
                simulations += result.diagnostics.completed_simulations;
                game.play(result.selected_move);
                moves += 1;
                if game.is_terminal() {
                    break;
                }
            }

            device::synchronize(execution_device);
            BenchmarkSample {
                elapsed_ns: started.elapsed().as_nanos(),
                operations: simulations as u64,
                metrics: json!({ "moves": moves }),
            }
        },
    )
}

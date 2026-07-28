use crate::harness;
use crate::report::BenchmarkSample;
use engine_core::agent::PolicyMode;
use engine_core::game::GameState;
use games::Connect4;
use serde_json::{json, Value};
use std::time::Instant;

pub fn connect4(warmup: &str, samples: usize, config: Value) -> crate::report::BenchmarkReport {
    harness::measure(
        "self-play.connect4.uniform",
        warmup,
        samples,
        json!({ "game": "connect4", "moves_per_game": 12, "model": "uniform", "config": config }),
        || {
            let mut worker = crate::search::new_puct(64);
            let mut game = Connect4::default();
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

            BenchmarkSample {
                elapsed_ns: started.elapsed().as_nanos(),
                operations: simulations as u64,
                metrics: json!({ "moves": moves }),
            }
        },
    )
    .expect("fixed benchmark arguments are valid")
}

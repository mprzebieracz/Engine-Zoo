use engine_core::agent::PolicyMode;
use engine_core::game::GameState;
use games::Connect4;
use search::{
    Evaluation, EvaluationError, Mcts, NoExtraRules, PolicyValueEvaluator, PositionValue,
    PuctConfig, SearchConfig,
};
use std::time::Instant;

struct UniformEvaluator;

impl<G: GameState> PolicyValueEvaluator<G> for UniformEvaluator {
    fn evaluate(
        &mut self,
        states: &[G],
        _legal_moves: &[G::Move],
        offsets: &[u32],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        debug_assert_eq!(states.len() + 1, offsets.len());
        Ok(offsets
            .windows(2)
            .map(|range| Evaluation {
                logits: vec![0.0; (range[1] - range[0]) as usize],
                value: PositionValue::DRAW,
            })
            .collect())
    }
}

fn main() {
    const SEARCHES: usize = 200;
    const SIMULATIONS: usize = 256;

    let game = Connect4::default();
    let mut mcts = Mcts::new(
        UniformEvaluator,
        SearchConfig::Puct(PuctConfig {
            leaf_batch_size: 1,
            root_noise: None,
            ..PuctConfig::default()
        }),
        NoExtraRules,
    );

    let request = search::SearchRequest {
        mode: PolicyMode::Deterministic,
        budget: search::SearchBudget::Puct {
            simulations: SIMULATIONS,
        },
    };
    let _ = mcts.search(&game, (), request);
    let started = Instant::now();
    for _ in 0..SEARCHES {
        std::hint::black_box(mcts.search(std::hint::black_box(&game), (), request))
            .expect("uniform benchmark evaluation must succeed");
    }

    let elapsed = started.elapsed();
    let simulations_per_second = (SEARCHES * SIMULATIONS) as f64 / elapsed.as_secs_f64();
    println!("connect4_mcts: {SEARCHES} searches x {SIMULATIONS} simulations in {elapsed:?}; {simulations_per_second:.0} simulations/s");
}

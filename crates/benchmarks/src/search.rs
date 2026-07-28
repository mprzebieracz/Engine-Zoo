use crate::harness;
use crate::report::BenchmarkSample;
use anyhow::Result;
use engine_core::agent::PolicyMode;
use engine_core::game::GameState;
use games::Connect4;
use search::{
    Evaluation, EvaluationError, FullGumbelConfig, Mcts, NoExtraRules, PolicyValueEvaluator,
    PositionValue, PuctConfig, RootGumbelPuctConfig, SearchBudget, SearchConfig, SearchRequest,
};
use serde_json::{json, Value};
use std::time::Instant;

pub fn puct(
    simulations: usize,
    warmup: &str,
    samples: usize,
    config: Value,
) -> Result<crate::report::BenchmarkReport> {
    let mut search = Mcts::new(
        UniformEvaluator,
        SearchConfig::Puct(PuctConfig {
            leaf_batch_size: 1,
            root_noise: None,
            ..PuctConfig::default()
        }),
        NoExtraRules,
    )
    .with_seed(1);
    measure(
        "search.puct.connect4",
        simulations,
        warmup,
        samples,
        config,
        &mut search,
    )
}

pub fn root_gumbel_puct(
    simulations: usize,
    warmup: &str,
    samples: usize,
    config: Value,
) -> Result<crate::report::BenchmarkReport> {
    let mut search = Mcts::new(
        UniformEvaluator,
        SearchConfig::RootGumbelPuct(RootGumbelPuctConfig::default()),
        NoExtraRules,
    )
    .with_seed(1);
    measure(
        "search.root-gumbel-puct.connect4",
        simulations,
        warmup,
        samples,
        config,
        &mut search,
    )
}

pub fn full_gumbel(
    simulations: usize,
    warmup: &str,
    samples: usize,
    config: Value,
) -> Result<crate::report::BenchmarkReport> {
    let mut search = Mcts::new(
        UniformEvaluator,
        SearchConfig::FullGumbel(FullGumbelConfig::default()),
        NoExtraRules,
    )
    .with_seed(1);
    measure(
        "search.full-gumbel.connect4",
        simulations,
        warmup,
        samples,
        config,
        &mut search,
    )
}

pub(crate) fn new_puct(simulations: usize) -> Mcts<Connect4, UniformEvaluator, NoExtraRules> {
    let _ = simulations;
    Mcts::new(
        UniformEvaluator,
        SearchConfig::Puct(PuctConfig {
            leaf_batch_size: 1,
            root_noise: None,
            ..PuctConfig::default()
        }),
        NoExtraRules,
    )
    .with_seed(1)
}

pub(crate) fn run(
    search: &mut Mcts<Connect4, UniformEvaluator, NoExtraRules>,
    game: &Connect4,
    simulations: usize,
    mode: PolicyMode,
) -> search::SearchResult<games::Connect4Move> {
    search
        .search(
            game,
            (),
            SearchRequest {
                mode,
                budget: SearchBudget::Puct { simulations },
            },
        )
        .expect("uniform evaluator is infallible")
}

fn measure<E>(
    name: &str,
    simulations: usize,
    warmup: &str,
    samples: usize,
    config: Value,
    search: &mut Mcts<Connect4, E, NoExtraRules>,
) -> Result<crate::report::BenchmarkReport>
where
    E: PolicyValueEvaluator<Connect4>,
{
    let game = Connect4::default();
    let budget = match search.algorithm() {
        search::SearchAlgorithm::Puct => SearchBudget::Puct { simulations },
        search::SearchAlgorithm::RootGumbelPuct | search::SearchAlgorithm::FullGumbel => {
            SearchBudget::Gumbel {
                simulations,
                max_considered_actions: 7,
            }
        }
    };

    harness::measure(
        name,
        warmup,
        samples,
        json!({ "game": "connect4", "simulations": simulations, "config": config }),
        || {
            let started = Instant::now();
            let result = search
                .search(
                    &game,
                    (),
                    SearchRequest {
                        mode: PolicyMode::Deterministic,
                        budget,
                    },
                )
                .expect("uniform evaluator is infallible");
            let elapsed = started.elapsed().as_nanos();

            BenchmarkSample {
                elapsed_ns: elapsed,
                operations: result.diagnostics.completed_simulations as u64,
                metrics: json!({ "searches_per_second": 1_000_000_000.0 / elapsed.max(1) as f64, "simulations_per_second": result.diagnostics.completed_simulations as f64 * 1_000_000_000.0 / elapsed.max(1) as f64, "backend_evaluations": result.diagnostics.backend_evaluations, "duplicate_leaves": result.diagnostics.duplicate_leaves, "max_depth": result.diagnostics.max_depth }),
            }
        },
    )
}

pub(crate) struct UniformEvaluator;

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

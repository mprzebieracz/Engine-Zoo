use super::*;
use crate::{Evaluation, EvaluationError, NoExtraRules, PolicyValueEvaluator, PositionValue};
use engine_core::agent::PolicyMode;
use engine_core::game::{GameState, TerminalValue};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Move {
    Win,
    Draw,
    Loss,
}

#[derive(Clone, Copy, Default)]
struct OnePly {
    outcome: Option<TerminalValue>,
}

impl GameState for OnePly {
    type Move = Move;
    fn initial() -> Self {
        Self::default()
    }
    fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
        [Move::Win, Move::Draw, Move::Loss]
            .into_iter()
            .filter(|_| self.outcome.is_none())
    }
    fn play(&mut self, action: Self::Move) {
        self.outcome = Some(match action {
            Move::Win => TerminalValue::Loss,
            Move::Draw => TerminalValue::Draw,
            Move::Loss => TerminalValue::Win,
        });
    }
    fn terminal_value(&self) -> Option<TerminalValue> {
        self.outcome
    }
}

#[derive(Default)]
struct Uniform;

impl PolicyValueEvaluator<OnePly> for Uniform {
    fn evaluate(
        &mut self,
        states: &[OnePly],
        _moves: &[Move],
        _offsets: &[u32],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        Ok(states
            .iter()
            .map(|_| Evaluation {
                logits: vec![0.0, 0.0, 0.0],
                value: PositionValue::DRAW,
            })
            .collect())
    }
}

#[derive(Default)]
struct Failing;

impl PolicyValueEvaluator<OnePly> for Failing {
    fn evaluate(
        &mut self,
        _states: &[OnePly],
        _moves: &[Move],
        _offsets: &[u32],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        Err(EvaluationError::new("network unavailable"))
    }
}

#[derive(Default)]
struct WrongCardinality;

impl PolicyValueEvaluator<OnePly> for WrongCardinality {
    fn evaluate(
        &mut self,
        _states: &[OnePly],
        _moves: &[Move],
        _offsets: &[u32],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        Ok(Vec::new())
    }
}

#[derive(Clone)]
struct Counting {
    calls: Arc<AtomicUsize>,
    namespace: Arc<AtomicU64>,
}

impl PolicyValueEvaluator<OnePly> for Counting {
    fn evaluate(
        &mut self,
        states: &[OnePly],
        _moves: &[Move],
        _offsets: &[u32],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        self.calls.fetch_add(states.len(), Ordering::Relaxed);
        Uniform.evaluate(states, &[], &[])
    }

    fn evaluation_key(&self, _state: &OnePly) -> Option<crate::EvaluationKey> {
        Some(crate::EvaluationKey {
            state: 7,
            namespace: self.namespace.load(Ordering::Acquire),
        })
    }
}

#[derive(Default)]
struct NonFiniteLogit;

impl PolicyValueEvaluator<OnePly> for NonFiniteLogit {
    fn evaluate(
        &mut self,
        states: &[OnePly],
        _moves: &[Move],
        _offsets: &[u32],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        Ok(states
            .iter()
            .map(|_| Evaluation {
                logits: vec![0.0, f32::NAN, 0.0],
                value: PositionValue::DRAW,
            })
            .collect())
    }
}

#[derive(Default)]
struct WrongLogitCount;

impl PolicyValueEvaluator<OnePly> for WrongLogitCount {
    fn evaluate(
        &mut self,
        states: &[OnePly],
        _moves: &[Move],
        _offsets: &[u32],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        Ok(states
            .iter()
            .map(|_| Evaluation {
                logits: vec![0.0],
                value: PositionValue::DRAW,
            })
            .collect())
    }
}

#[derive(Clone, Copy, Default)]
struct NoMoves;

impl GameState for NoMoves {
    type Move = Move;

    fn initial() -> Self {
        Self
    }

    fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
        std::iter::empty()
    }

    fn play(&mut self, _: Self::Move) {}

    fn terminal_value(&self) -> Option<TerminalValue> {
        None
    }
}

impl PolicyValueEvaluator<NoMoves> for Counting {
    fn evaluate(
        &mut self,
        states: &[NoMoves],
        _moves: &[Move],
        _offsets: &[u32],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        self.calls.fetch_add(states.len(), Ordering::Relaxed);
        Ok(Vec::new())
    }
}

#[derive(Clone, Copy)]
struct TerminalRule;

impl crate::SearchRules<OnePly> for TerminalRule {
    type Context<'a>
        = ()
    where
        OnePly: 'a;
    type PathState = ();
    type NodeMeta = ();

    fn reset_path<'a>(&self, _: (), _: &OnePly, _: &mut ()) {}

    fn enter_state<'a>(&self, _: (), _: &mut OnePly, _: &mut (), _: &mut ()) -> crate::RuleResult {
        crate::RuleResult::Terminal(PositionValue::DRAW)
    }
}

fn puct() -> SearchConfig {
    SearchConfig::Puct(PuctConfig {
        root_noise: None,
        ..Default::default()
    })
}

fn full_gumbel() -> SearchConfig {
    SearchConfig::FullGumbel(FullGumbelConfig {
        root: GumbelRootConfig {
            gumbel_scale: 0.0,
            ..Default::default()
        },
    })
}

fn root_gumbel_puct() -> SearchConfig {
    SearchConfig::RootGumbelPuct(RootGumbelPuctConfig {
        root: GumbelRootConfig {
            gumbel_scale: 0.0,
            ..Default::default()
        },
        ..Default::default()
    })
}

fn puct_request(simulations: usize, mode: PolicyMode) -> SearchRequest {
    SearchRequest {
        mode,
        budget: SearchBudget::Puct { simulations },
    }
}

fn gumbel_request(simulations: usize, mode: PolicyMode) -> SearchRequest {
    SearchRequest {
        mode,
        budget: SearchBudget::Gumbel {
            simulations,
            max_considered_actions: 2,
        },
    }
}

fn request_for(config: &SearchConfig, mode: PolicyMode) -> SearchRequest {
    match config {
        SearchConfig::Puct(_) => puct_request(1, mode),
        SearchConfig::RootGumbelPuct(_) | SearchConfig::FullGumbel(_) => gumbel_request(1, mode),
    }
}

#[test]
fn puct_finds_terminal_win_and_returns_visit_policy() {
    let mut search = Mcts::new(Uniform, puct(), NoExtraRules);
    let result = search
        .search(
            &OnePly::default(),
            (),
            puct_request(12, PolicyMode::Deterministic),
        )
        .unwrap();
    assert_eq!(result.best_move(), Move::Win);
    assert!(
        (result
            .policy
            .iter()
            .map(|(_, probability)| probability)
            .sum::<f32>()
            - 1.0)
            .abs()
            < 1e-6
    );
    assert_eq!(result.diagnostics.completed_simulations, 12);
}

#[test]
fn puct_tactical_fixture_keeps_the_forced_win_after_backup() {
    let mut search = Mcts::new(Uniform, puct(), NoExtraRules);
    let result = search
        .search(
            &OnePly::default(),
            (),
            puct_request(8, PolicyMode::Deterministic),
        )
        .unwrap();

    assert_eq!(result.best_move(), Move::Win);
    assert!(result.probability(Move::Win) > result.probability(Move::Draw));
    assert!(result.probability(Move::Win) > result.probability(Move::Loss));
    assert!(result.root_value.as_f32() > 0.0);
}

#[test]
fn full_gumbel_is_sequential_and_policy_covers_all_legal_actions() {
    let mut search = Mcts::new(Uniform, full_gumbel(), NoExtraRules);
    let result = search
        .search(
            &OnePly::default(),
            (),
            gumbel_request(8, PolicyMode::Explore),
        )
        .unwrap();
    assert_eq!(result.diagnostics.completed_simulations, 8);
    assert_eq!(result.diagnostics.backend_evaluations, 1);
    assert_eq!(result.diagnostics.evaluation_cache_hits, 0);
    assert_eq!(result.policy.len(), 3);
    assert!(
        (result
            .policy
            .iter()
            .map(|(_, probability)| probability)
            .sum::<f32>()
            - 1.0)
            .abs()
            < 1e-6
    );
}

#[test]
fn root_gumbel_puct_uses_the_gumbel_budget_and_keeps_the_forced_win() {
    let mut search = Mcts::new(Uniform, root_gumbel_puct(), NoExtraRules);
    let result = search
        .search(
            &OnePly::default(),
            (),
            gumbel_request(8, PolicyMode::Deterministic),
        )
        .unwrap();

    assert_eq!(result.best_move(), Move::Win);
    assert_eq!(result.diagnostics.completed_simulations, 8);
    assert_eq!(result.policy.len(), 3);
}

#[test]
fn evaluation_failure_is_reported_without_fabricating_a_value() {
    let mut search = Mcts::new(Failing, puct(), NoExtraRules);
    let error = search
        .search(
            &OnePly::default(),
            (),
            puct_request(1, PolicyMode::Deterministic),
        )
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "search evaluation failed: network unavailable"
    );
}

#[test]
fn malformed_evaluator_cardinality_is_a_typed_error() {
    let mut search = Mcts::new(WrongCardinality, puct(), NoExtraRules);
    let error = search
        .search(
            &OnePly::default(),
            (),
            puct_request(1, PolicyMode::Deterministic),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        SearchError::Evaluation(EvaluationError::ResultCardinality {
            expected: 1,
            actual: 0,
        })
    ));
}

#[test]
fn invalid_algorithm_configs_are_rejected() {
    let puct = PuctConfig {
        leaf_batch_size: 0,
        ..Default::default()
    };
    assert!(puct.validate().is_err());
    let gumbel = RootGumbelPuctConfig {
        leaf_batch_size: 0,
        ..Default::default()
    };
    assert!(gumbel.validate().is_err());
}

#[test]
fn effective_leaf_batch_cap_respects_the_current_search_budget() {
    assert_eq!(effective_leaf_batch_size(16, 800), 16);
    assert_eq!(effective_leaf_batch_size(16, 64), 16);
    assert_eq!(effective_leaf_batch_size(16, 32), 8);
    assert_eq!(effective_leaf_batch_size(16, 1), 1);
}

#[test]
fn effective_leaf_batch_cap_is_nonzero_for_invalid_direct_inputs() {
    assert_eq!(effective_leaf_batch_size(0, 0), 1);
}

#[test]
fn native_terminal_roots_skip_evaluation_for_both_searches() {
    for config in [puct(), full_gumbel(), root_gumbel_puct()] {
        let calls = Arc::new(AtomicUsize::new(0));
        let evaluator = Counting {
            calls: Arc::clone(&calls),
            namespace: Arc::new(AtomicU64::new(0)),
        };
        let request = request_for(&config, PolicyMode::Deterministic);
        let mut search = Mcts::new(evaluator, config, NoExtraRules);
        let error = search
            .search(
                &OnePly {
                    outcome: Some(TerminalValue::Draw),
                },
                (),
                request,
            )
            .unwrap_err();
        assert!(matches!(error, SearchError::TerminalRoot { .. }));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn rule_terminal_roots_skip_evaluation() {
    for config in [puct(), full_gumbel(), root_gumbel_puct()] {
        let calls = Arc::new(AtomicUsize::new(0));
        let evaluator = Counting {
            calls: Arc::clone(&calls),
            namespace: Arc::new(AtomicU64::new(0)),
        };
        let request = request_for(&config, PolicyMode::Deterministic);
        let mut search = Mcts::new(evaluator, config, TerminalRule);
        assert!(matches!(
            search.search(&OnePly::default(), (), request),
            Err(SearchError::TerminalRoot {
                value: PositionValue::DRAW
            })
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn non_finite_logits_are_typed_evaluator_errors() {
    let mut search = Mcts::new(NonFiniteLogit, puct(), NoExtraRules);
    assert!(matches!(
        search.search(
            &OnePly::default(),
            (),
            puct_request(1, PolicyMode::Deterministic)
        ),
        Err(SearchError::Evaluation(EvaluationError::NonFiniteLogit {
            row: 0,
            index: 1,
        }))
    ));
}

#[test]
fn malformed_logit_counts_are_typed_evaluator_errors() {
    let mut search = Mcts::new(WrongLogitCount, puct(), NoExtraRules);
    assert!(matches!(
        search.search(
            &OnePly::default(),
            (),
            puct_request(1, PolicyMode::Deterministic)
        ),
        Err(SearchError::Evaluation(EvaluationError::LogitCardinality {
            row: 0,
            expected: 3,
            actual: 1,
        }))
    ));
}

#[test]
fn nonterminal_roots_without_legal_moves_skip_evaluation() {
    for config in [puct(), full_gumbel(), root_gumbel_puct()] {
        let calls = Arc::new(AtomicUsize::new(0));
        let evaluator = Counting {
            calls: Arc::clone(&calls),
            namespace: Arc::new(AtomicU64::new(0)),
        };
        let request = request_for(&config, PolicyMode::Deterministic);
        let mut search = Mcts::new(evaluator, config, NoExtraRules);
        assert!(matches!(
            search.search(&NoMoves, (), request),
            Err(SearchError::NoLegalMoves)
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn cache_namespace_invalidates_previous_model_results() {
    let calls = Arc::new(AtomicUsize::new(0));
    let namespace = Arc::new(AtomicU64::new(0));
    let evaluator = Counting {
        calls: Arc::clone(&calls),
        namespace: Arc::clone(&namespace),
    };
    let cache = Arc::new(EvalTable::new(64));
    let mut search = Mcts::new(evaluator, puct(), NoExtraRules).with_eval_cache(Arc::clone(&cache));

    let first = search
        .search(
            &OnePly::default(),
            (),
            puct_request(1, PolicyMode::Deterministic),
        )
        .unwrap();
    assert_eq!(first.diagnostics.backend_evaluations, 1);
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    let hit = search
        .search(
            &OnePly::default(),
            (),
            puct_request(1, PolicyMode::Deterministic),
        )
        .unwrap();
    assert_eq!(hit.diagnostics.backend_evaluations, 0);
    assert_eq!(hit.diagnostics.evaluation_cache_hits, 1);

    namespace.store(1, Ordering::Release);
    let reloaded = search
        .search(
            &OnePly::default(),
            (),
            puct_request(1, PolicyMode::Deterministic),
        )
        .unwrap();
    assert_eq!(reloaded.diagnostics.backend_evaluations, 1);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.stats().misses, 2);
    assert_eq!(cache.stats().inserts, 2);
}

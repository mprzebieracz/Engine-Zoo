use super::*;
use crate::{Evaluation, EvaluationError, NoExtraRules, PolicyValueEvaluator, PositionValue};
use engine_core::agent::PolicyMode;
use engine_core::game::{GameState, TerminalValue};

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

fn puct(simulations: usize) -> SearchConfig {
    let mut config = PuctConfig::default();
    config.common.simulations = simulations;
    config.root_noise = None;
    SearchConfig::Puct(config)
}

#[test]
fn puct_finds_terminal_win_and_returns_visit_policy() {
    let mut search = Mcts::new(Uniform, puct(12), NoExtraRules);
    let result = search
        .search(&OnePly::default(), (), PolicyMode::Deterministic)
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
fn full_gumbel_is_sequential_and_policy_covers_all_legal_actions() {
    let config = SearchConfig::Gumbel(GumbelConfig {
        simulations: 8,
        max_considered_actions: 2,
        gumbel_scale: 0.0,
        ..GumbelConfig::default()
    });
    let mut search = Mcts::new(Uniform, config, NoExtraRules);
    let result = search
        .search(&OnePly::default(), (), PolicyMode::Explore)
        .unwrap();
    assert_eq!(result.diagnostics.completed_simulations, 8);
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
fn evaluation_failure_is_reported_without_fabricating_a_value() {
    let mut search = Mcts::new(Failing, puct(1), NoExtraRules);
    let error = search
        .search(&OnePly::default(), (), PolicyMode::Deterministic)
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "search evaluation failed: network unavailable"
    );
}

#[test]
fn malformed_evaluator_cardinality_is_a_typed_error() {
    let mut search = Mcts::new(WrongCardinality, puct(1), NoExtraRules);
    let error = search
        .search(&OnePly::default(), (), PolicyMode::Deterministic)
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
    let mut puct = PuctConfig::default();
    puct.common.leaf_batch_size = 0;
    assert!(puct.validate().is_err());
    let gumbel = GumbelConfig {
        max_considered_actions: 0,
        ..Default::default()
    };
    assert!(gumbel.validate().is_err());
}

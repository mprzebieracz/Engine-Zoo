use crate::representation::AlphaZeroRepresentation;
use crate::ChessRepetitionRules;
use crate::RepresentedEvaluator;
use crate::{EncodedEvaluator, Mcts, SearchConfig};
use anyhow::Result;
use engine_core::agent::PolicyMode;
use engine_core::game::GameState;
use engine_core::notation::GameNotation;
use games::ChessRepetitionContext;
use games::ChessRepetitionState;
use search::{PolicyValueEvaluator, SearchRules};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnalyzeMode {
    Net,
    Mcts,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MoveScore {
    pub action: u32,
    pub mv: String,
    pub p: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Analysis {
    pub value: f32,
    pub network_value: f32,
    pub mcts_value: Option<f32>,
    pub best_action: Option<u32>,
    pub best_move: Option<String>,
    pub policy: Vec<MoveScore>,
    pub network_policy: Vec<MoveScore>,
    pub mcts_policy: Vec<MoveScore>,
}

pub struct AnalyzeConfig {
    pub mode: AnalyzeMode,
    pub mcts: SearchConfig,
    pub wait_for_count: usize,
    pub timeout: Duration,
}

pub fn analyze_game_net<G, Rep, N, Notation>(
    game: G,
    evaluator: N,
    rep: &Rep,
    notation: &Notation,
) -> Result<Analysis>
where
    G: GameState + Clone,
    Rep: AlphaZeroRepresentation<G>,
    N: EncodedEvaluator,
    Notation: GameNotation<G>,
{
    let mut evaluator = RepresentedEvaluator::new(rep.clone(), evaluator);
    let legal: Vec<_> = game.legal_moves().collect();
    let eval = evaluator.evaluate(
        std::slice::from_ref(&game),
        &legal,
        &[0, legal.len() as u32],
    )?;
    anyhow::ensure!(eval.len() == 1, "evaluator returned {} results", eval.len());
    let probs = softmax(&eval[0].logits);
    let policy: Vec<MoveScore> = legal
        .iter()
        .zip(probs)
        .map(|(&mv, p)| MoveScore {
            action: rep.move_to_action(&game, mv).as_u32(),
            mv: notation.format_move(&game, mv),
            p,
        })
        .collect();
    let mut analysis = with_best(eval[0].value.as_f32(), policy.clone());
    analysis.network_policy = policy;
    Ok(analysis)
}

/// Runs ordinary deterministic MCTS for an already-constructed game.
pub fn analyze_game_mcts<G, R, E, Rep, Notation>(
    game: G,
    mcts: &mut Mcts<G, E, R>,
    context: R::Context<'_>,
    network: Analysis,
    rep: &Rep,
    notation: &Notation,
) -> Result<Analysis>
where
    G: GameState + Clone,
    R: SearchRules<G>,
    E: PolicyValueEvaluator<G>,
    Rep: AlphaZeroRepresentation<G>,
    Notation: GameNotation<G>,
{
    let result = mcts.search(&game, context, PolicyMode::Deterministic)?;
    let legal: Vec<_> = game.legal_moves().collect();
    let policy: Vec<MoveScore> = legal
        .into_iter()
        .map(|action| MoveScore {
            action: rep.move_to_action(&game, action).as_u32(),
            mv: notation.format_move(&game, action),
            p: result.probability(action),
        })
        .collect();
    let mut analysis = with_best(result.root_value.as_f32(), policy.clone());
    analysis.network_value = network.value;
    analysis.network_policy = network.policy;
    analysis.mcts_value = Some(result.root_value.as_f32());
    analysis.mcts_policy = policy;
    Ok(analysis)
}

/// Runs deterministic repetition-aware MCTS for an already-constructed game.
/// The callback belongs to the authoritative rules owner, rather than the
/// search state, because it may retain history beyond the encoded frames.
pub fn analyze_game_mcts_with_repetitions<G, E, Rep, Notation>(
    game: G,
    mcts: &mut Mcts<G, E, ChessRepetitionRules>,
    context: ChessRepetitionContext<'_>,
    network: Analysis,
    rep: &Rep,
    notation: &Notation,
) -> Result<Analysis>
where
    G: GameState + Clone + ChessRepetitionState,
    E: PolicyValueEvaluator<G>,
    for<'a> ChessRepetitionRules: SearchRules<G, Context<'a> = ChessRepetitionContext<'a>>,
    Rep: AlphaZeroRepresentation<G>,
    Notation: GameNotation<G>,
{
    analyze_game_mcts(game, mcts, context, network, rep, notation)
}

fn with_best(value: f32, policy: Vec<MoveScore>) -> Analysis {
    let best = policy
        .iter()
        .max_by(|a, b| a.p.total_cmp(&b.p))
        .map(|m| (m.action, m.mv.clone()));
    Analysis {
        value,
        network_value: value,
        mcts_value: None,
        best_action: best.as_ref().map(|(action, _)| *action),
        best_move: best.map(|(_, mv)| mv),
        policy,
        network_policy: Vec::new(),
        mcts_policy: Vec::new(),
    }
}

fn softmax(xs: &[f32]) -> Vec<f32> {
    if xs.is_empty() {
        return Vec::new();
    }
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<_> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        for x in &mut out {
            *x /= sum;
        }
    }
    out
}

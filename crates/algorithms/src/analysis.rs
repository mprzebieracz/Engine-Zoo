use crate::alphazero::{Batcher, EvalBatch, Evaluator, Mcts, MctsConfig, NetConfig};
use anyhow::Result;
use engine_core::agent::PolicyMode;
use engine_core::game::{Action, Game};
use engine_core::rules::{PositionCodec, RepetitionGame};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use tch::Device;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnalyzeMode {
    Net,
    Mcts,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MoveScore {
    pub action: Action,
    pub mv: String,
    pub p: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Analysis {
    pub value: f32,
    pub network_value: f32,
    pub mcts_value: Option<f32>,
    pub best_action: Option<Action>,
    pub best_move: Option<String>,
    pub policy: Vec<MoveScore>,
    pub network_policy: Vec<MoveScore>,
    pub mcts_policy: Vec<MoveScore>,
}

pub struct AnalyzeConfig {
    pub mode: AnalyzeMode,
    pub mcts: MctsConfig,
    pub wait_for_count: usize,
    pub timeout: Duration,
}

pub fn analyze_position<G: PositionCodec>(
    net_cfg: &NetConfig,
    weights: &Path,
    position: &G::Position,
    device: Device,
    cfg: &AnalyzeConfig,
) -> Result<Analysis> {
    let game = G::from_position(position)?;
    if game.is_terminal() {
        return Ok(Analysis {
            value: game.reward(),
            network_value: game.reward(),
            mcts_value: None,
            best_action: None,
            best_move: None,
            policy: Vec::new(),
            network_policy: Vec::new(),
            mcts_policy: Vec::new(),
        });
    }

    let batcher = Batcher::new(net_cfg, weights, device, cfg.wait_for_count, cfg.timeout)?;
    match cfg.mode {
        AnalyzeMode::Net => analyze_game_net::<G>(game, batcher.client()),
        AnalyzeMode::Mcts => {
            let mut mcts_cfg = cfg.mcts;
            mcts_cfg.eps = 0.0;
            let network = analyze_game_net::<G>(game.clone(), batcher.client())?;
            let mut mcts = Mcts::new(batcher.client(), mcts_cfg);
            analyze_game_mcts(game, &mut mcts, network)
        }
    }
}

/// Evaluates an already-constructed game. This keeps position construction in
/// the owning integration crate while allowing alternate game state wrappers
/// (such as history-aware chess) to share the API analysis schema.
pub fn analyze_game_net<G: Game>(game: G, mut evaluator: impl Evaluator) -> Result<Analysis> {
    let legal: Vec<_> = game.legal_actions().collect();
    let mut state = vec![0.0f32; G::state_size()];
    game.encode_state(&mut state);
    let batch = EvalBatch {
        states: state,
        legal: legal.clone(),
        offsets: vec![0, legal.len() as u32],
    };
    let eval = evaluator.evaluate(&batch);
    anyhow::ensure!(eval.len() == 1, "evaluator returned {} results", eval.len());
    let probs = softmax(&eval[0].logits);
    let policy: Vec<MoveScore> = legal
        .iter()
        .zip(probs)
        .map(|(&action, p)| MoveScore {
            action,
            mv: game.format_action(action),
            p,
        })
        .collect();
    let mut analysis = with_best(eval[0].value, policy.clone());
    analysis.network_policy = policy;
    Ok(analysis)
}

/// Runs ordinary deterministic MCTS for an already-constructed game.
pub fn analyze_game_mcts<G: Game, E: Evaluator>(
    game: G,
    mcts: &mut Mcts<E>,
    network: Analysis,
) -> Result<Analysis> {
    let result = mcts.search_with_mode(&game, PolicyMode::Deterministic);
    let legal: Vec<_> = game.legal_actions().collect();
    let policy: Vec<MoveScore> = legal
        .into_iter()
        .map(|action| MoveScore {
            action,
            mv: game.format_action(action),
            p: result.policy[action as usize],
        })
        .collect();
    let mut analysis = with_best(result.value, policy.clone());
    analysis.network_value = network.value;
    analysis.network_policy = network.policy;
    analysis.mcts_value = Some(result.value);
    analysis.mcts_policy = policy;
    Ok(analysis)
}

/// Runs deterministic repetition-aware MCTS for an already-constructed game.
/// The callback belongs to the authoritative rules owner, rather than the
/// search state, because it may retain history beyond the encoded frames.
pub fn analyze_game_mcts_with_repetitions<G, E, F>(
    game: G,
    mcts: &mut Mcts<E>,
    network: Analysis,
    repetitions_before_current: F,
) -> Result<Analysis>
where
    G: RepetitionGame,
    E: Evaluator,
    F: Fn(u64) -> u8 + Copy,
{
    let result = mcts.search_with_repetitions_mode(
        &game,
        repetitions_before_current,
        PolicyMode::Deterministic,
    );
    let legal: Vec<_> = game.legal_actions().collect();
    let policy: Vec<MoveScore> = legal
        .into_iter()
        .map(|action| MoveScore {
            action,
            mv: game.format_action(action),
            p: result.policy[action as usize],
        })
        .collect();
    let mut analysis = with_best(result.value, policy.clone());
    analysis.network_value = network.value;
    analysis.network_policy = network.policy;
    analysis.mcts_value = Some(result.value);
    analysis.mcts_policy = policy;
    Ok(analysis)
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

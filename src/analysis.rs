use crate::agent::PolicyMode;
use crate::alphazero::{Batcher, EvalBatch, Evaluator, Mcts, MctsConfig, NetConfig};
use crate::game::{Action, Game};
use crate::position::PositionGame;
use anyhow::Result;
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
    pub best_action: Option<Action>,
    pub best_move: Option<String>,
    pub policy: Vec<MoveScore>,
}

pub struct AnalyzeConfig {
    pub mode: AnalyzeMode,
    pub mcts: MctsConfig,
    pub wait_for_count: usize,
    pub timeout: Duration,
}

pub fn analyze_position<G: PositionGame>(
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
            best_action: None,
            best_move: None,
            policy: Vec::new(),
        });
    }

    let batcher = Batcher::new(net_cfg, weights, device, cfg.wait_for_count, cfg.timeout)?;
    match cfg.mode {
        AnalyzeMode::Net => analyze_net::<G>(game, batcher.client()),
        AnalyzeMode::Mcts => {
            let mut mcts_cfg = cfg.mcts;
            mcts_cfg.eps = 0.0;
            let mut mcts = Mcts::new(batcher.client(), mcts_cfg);
            analyze_mcts(game, &mut mcts)
        }
    }
}

fn analyze_net<G: Game>(game: G, mut evaluator: impl Evaluator) -> Result<Analysis> {
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
    let policy = legal
        .iter()
        .zip(probs)
        .map(|(&action, p)| MoveScore {
            action,
            mv: game.format_action(action),
            p,
        })
        .collect();
    Ok(with_best(eval[0].value, policy))
}

fn analyze_mcts<G: Game, E: Evaluator>(game: G, mcts: &mut Mcts<E>) -> Result<Analysis> {
    let result = mcts.search_with_mode(&game, PolicyMode::Deterministic);
    let legal: Vec<_> = game.legal_actions().collect();
    let policy = legal
        .into_iter()
        .map(|action| MoveScore {
            action,
            mv: game.format_action(action),
            p: result.policy[action as usize],
        })
        .collect();
    Ok(with_best(result.value, policy))
}

fn with_best(value: f32, policy: Vec<MoveScore>) -> Analysis {
    let best = policy
        .iter()
        .max_by(|a, b| a.p.total_cmp(&b.p))
        .map(|m| (m.action, m.mv.clone()));
    Analysis {
        value,
        best_action: best.as_ref().map(|(action, _)| *action),
        best_move: best.map(|(_, mv)| mv),
        policy,
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

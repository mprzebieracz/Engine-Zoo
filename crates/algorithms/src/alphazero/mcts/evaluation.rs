use super::cache::CachedEvaluation;
use super::core::MctsCore;
use crate::search::Evaluation;
use engine_core::GameState;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::Gamma;

pub(super) fn build_policy<M: Copy>(
    buf: &mut Vec<(M, f32, f32)>,
    legal: &[M],
    eval: &Evaluation,
    noise: bool,
    eps: f32,
    alpha: f32,
    rng: &mut SmallRng,
) {
    buf.clear();
    let max = eval
        .logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for (&m, &logit) in legal.iter().zip(&eval.logits) {
        let prior = (logit - max).exp();
        sum += prior;
        buf.push((m, prior, logit));
    }
    if sum.is_finite() && sum > 0.0 {
        for (_, p, _) in &mut *buf {
            *p /= sum;
        }
    }
    else if !buf.is_empty() {
        let p = 1.0 / buf.len() as f32;
        for (_, prior, _) in &mut *buf {
            *prior = p;
        }
    }
    if noise && eps > 0.0 && !buf.is_empty() {
        let gamma = Gamma::new(alpha, 1.0).expect("alpha > 0");
        let mut samples: Vec<f32> = (0..buf.len()).map(|_| gamma.sample(rng)).collect();
        let total: f32 = samples.iter().sum();
        if total > 0.0 {
            for sample in &mut samples {
                *sample /= total;
            }
        }
        for ((_, prior, _), sample) in buf.iter_mut().zip(samples) {
            *prior = (1.0 - eps) * *prior + eps * sample;
        }
    }
}

impl<G, E, R, V> MctsCore<G, E, R, V>
where
    G: GameState + Clone,
    E: crate::search::PolicyValueEvaluator<G>,
    R: crate::search::SearchRules<G>,
    V: 'static,
{
    pub(super) fn evaluate_position(&mut self, state: &G) -> (Vec<G::Move>, Evaluation) {
        self.evaluate_positions(std::slice::from_ref(state))
            .pop()
            .unwrap()
    }
    pub(super) fn build_policy_from(&mut self, legal: &[G::Move], eval: &Evaluation, noise: bool) {
        build_policy(
            &mut self.policy_buf,
            legal,
            eval,
            noise,
            self.cfg.eps,
            self.cfg.alpha,
            &mut self.rng,
        );
    }
    pub(super) fn expand(&mut self, node: u32) {
        let first = self.nodes.len() as u32;
        for &(m, p, l) in &self.policy_buf {
            self.nodes
                .push(super::Node::new(Some(node), Some(m), p, l, false, 0.0));
        }
        let n = &mut self.nodes[node as usize];
        n.first_child = first;
        n.num_children = self.policy_buf.len() as u16;
        n.expanded = true;
    }
    pub(super) fn evaluate_positions(&mut self, states: &[G]) -> Vec<(Vec<G::Move>, Evaluation)> {
        if states.is_empty() {
            return Vec::new();
        }
        let mut out: Vec<Option<(Vec<G::Move>, Evaluation)>> = vec![None; states.len()];
        let mut misses = Vec::new();
        let mut keys: Vec<Option<u64>> = Vec::new();
        let mut destinations = Vec::new();
        for (i, state) in states.iter().enumerate() {
            let Some(key) = self.evaluator.evaluation_key(state)
            else {
                let j = misses.len();
                misses.push(state.clone());
                keys.push(None);
                destinations.push((i, Some(j)));
                continue;
            };
            if let Some(cache) = &self.eval_cache {
                if let Some(value) = cache.get(key) {
                    out[i] = Some((value.legal, value.eval));
                    continue;
                }
            }
            if let Some(j) = keys.iter().position(|&k| k == Some(key)) {
                destinations.push((i, Some(j)));
            }
            else {
                keys.push(Some(key));
                destinations.push((i, Some(misses.len())));
                misses.push(state.clone());
            }
        }
        self.legal_moves.clear();
        self.offsets.clear();
        self.offsets.push(0);
        for state in &misses {
            self.legal_moves.extend(state.legal_moves());
            self.offsets.push(self.legal_moves.len() as u32);
        }
        let evaluations = self
            .evaluator
            .evaluate(&misses, &self.legal_moves, &self.offsets);
        let mut unique = Vec::with_capacity(misses.len());
        for (i, eval) in evaluations.into_iter().enumerate() {
            let begin = self.offsets[i] as usize;
            let end = self.offsets[i + 1] as usize;
            let value = (self.legal_moves[begin..end].to_vec(), eval);
            if let (Some(cache), Some(key)) = (&self.eval_cache, keys[i]) {
                cache.insert(
                    key,
                    CachedEvaluation {
                        legal: value.0.clone(),
                        eval: value.1.clone(),
                    },
                );
            }
            unique.push(value);
        }
        for (i, destination) in destinations {
            out[i] = Some(unique[destination.unwrap()].clone());
        }
        out.into_iter().map(Option::unwrap).collect()
    }
}

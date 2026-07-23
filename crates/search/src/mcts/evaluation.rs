use super::cache::CachedEvaluation;
use super::core::MctsCore;
use crate::{EvaluationError, EvaluationKey, PositionValue};
use engine_core::GameState;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::Gamma;

pub(super) fn build_policy<M: Copy>(
    buf: &mut Vec<(M, f32, f32)>,
    legal: &[M],
    logits: &[f32],
    noise: Option<super::DirichletConfig>,
    rng: &mut SmallRng,
) {
    buf.clear();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for (&m, &logit) in legal.iter().zip(logits) {
        let prior = (logit - max).exp();
        sum += prior;
        buf.push((m, prior, logit));
    }
    debug_assert!(sum.is_finite() && sum > 0.0);
    for (_, prior, _) in &mut *buf {
        *prior /= sum;
    }
    if let Some(noise) = noise.filter(|noise| noise.epsilon > 0.0 && !buf.is_empty()) {
        let gamma = Gamma::new(noise.alpha, 1.0).expect("validated Dirichlet alpha");
        let mut samples: Vec<f32> = (0..buf.len()).map(|_| gamma.sample(rng)).collect();
        let total: f32 = samples.iter().sum();
        if total > 0.0 {
            for sample in &mut samples {
                *sample /= total;
            }
        }
        for ((_, prior, _), sample) in buf.iter_mut().zip(samples) {
            *prior = (1.0 - noise.epsilon) * *prior + noise.epsilon * sample;
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct EvaluationBatchStats {
    pub(super) requested_states: usize,
    pub(super) duplicate_requests: usize,
    pub(super) cache_hits: usize,
    pub(super) backend_evaluations: usize,
}

type EvaluationBatch<M> = (Vec<CachedEvaluation<M>>, EvaluationBatchStats);

impl<G, E, R, V> MctsCore<G, E, R, V>
where
    G: GameState + Clone,
    E: crate::PolicyValueEvaluator<G>,
    R: crate::SearchRules<G>,
    V: 'static,
{
    pub(super) fn evaluate_position(
        &mut self,
        state: &G,
    ) -> Result<(CachedEvaluation<G::Move>, EvaluationBatchStats), crate::EvaluationError> {
        self.evaluate_positions(std::slice::from_ref(state))
            .map(|(mut evaluations, stats)| {
                (
                    evaluations
                        .pop()
                        .expect("one requested evaluation must produce one result"),
                    stats,
                )
            })
    }
    pub(super) fn build_policy_from(
        &mut self,
        legal: &[G::Move],
        logits: &[f32],
        exploratory_root: bool,
        root_noise: Option<super::DirichletConfig>,
    ) {
        build_policy(
            &mut self.policy_buf,
            legal,
            logits,
            if exploratory_root { root_noise } else { None },
            &mut self.rng,
        );
    }
    pub(super) fn expand(&mut self, node: u32) {
        let first = self.nodes.len() as u32;
        for &(m, p, l) in &self.policy_buf {
            self.nodes.push(super::Node::new(
                Some(node),
                Some(m),
                p,
                l,
                false,
                PositionValue::DRAW,
            ));
        }
        let n = &mut self.nodes[node as usize];
        n.first_child = first;
        n.num_children = self.policy_buf.len() as u16;
        n.expanded = true;
    }
    pub(super) fn evaluate_positions(
        &mut self,
        states: &[G],
    ) -> Result<EvaluationBatch<G::Move>, crate::EvaluationError> {
        if states.is_empty() {
            return Ok((Vec::new(), EvaluationBatchStats::default()));
        }
        let mut stats = EvaluationBatchStats {
            requested_states: states.len(),
            ..EvaluationBatchStats::default()
        };
        let mut out: Vec<Option<CachedEvaluation<G::Move>>> = vec![None; states.len()];
        let mut misses = Vec::new();
        let mut keys: Vec<Option<EvaluationKey>> = Vec::new();
        let mut destinations = Vec::new();
        for (i, state) in states.iter().enumerate() {
            let Some(key) = self.evaluator.evaluation_key(state) else {
                let j = misses.len();
                misses.push(state.clone());
                keys.push(None);
                destinations.push((i, Some(j)));
                continue;
            };
            if let Some(cache) = &self.eval_cache {
                if let Some(value) = cache.get(key) {
                    out[i] = Some(value);
                    stats.cache_hits += 1;
                    continue;
                }
            }
            if let Some(j) = keys.iter().position(|&k| k == Some(key)) {
                destinations.push((i, Some(j)));
                stats.duplicate_requests += 1;
            } else {
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
            .evaluate(&misses, &self.legal_moves, &self.offsets)?;
        if evaluations.len() != misses.len() {
            return Err(crate::EvaluationError::result_cardinality(
                misses.len(),
                evaluations.len(),
            ));
        }
        stats.backend_evaluations = misses.len();
        let mut unique = Vec::with_capacity(misses.len());
        for (i, eval) in evaluations.into_iter().enumerate() {
            let begin = self.offsets[i] as usize;
            let end = self.offsets[i + 1] as usize;
            let expected = end - begin;
            if eval.logits.len() != expected {
                return Err(EvaluationError::LogitCardinality {
                    row: i,
                    expected,
                    actual: eval.logits.len(),
                });
            }
            if let Some(index) = eval.logits.iter().position(|logit| !logit.is_finite()) {
                return Err(EvaluationError::NonFiniteLogit { row: i, index });
            }
            let value = eval.value.as_f32();
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err(EvaluationError::InvalidValue { row: i, value });
            }
            let value = CachedEvaluation::new(
                self.legal_moves[begin..end].to_vec(),
                eval.logits,
                eval.value,
            );
            if let (Some(cache), Some(key)) = (&self.eval_cache, keys[i]) {
                cache.insert(key, value.clone());
            }
            unique.push(value);
        }
        for (i, destination) in destinations {
            out[i] = Some(unique[destination.unwrap()].clone());
        }
        let evaluations = out.into_iter().map(Option::unwrap).collect::<Vec<_>>();
        debug_assert_eq!(stats.requested_states, evaluations.len());
        Ok((evaluations, stats))
    }
}

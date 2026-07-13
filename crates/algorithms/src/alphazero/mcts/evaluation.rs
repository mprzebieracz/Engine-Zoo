use super::super::evaluator::{Evaluation, Evaluator};
use super::cache::CachedEvaluation;
use super::core::MctsCore;
use super::Node;
use engine_core::game::{Action, Game};
use engine_core::rules::RepetitionGame;
use rand::prelude::*;
use rand_distr::Gamma;

impl<E: Evaluator, V> MctsCore<E, V> {
    /// Encodes `game` and its legal actions as the next entry of `self.batch`.
    fn enqueue_state<G: Game>(&mut self, game: &G) {
        let start = self.batch.states.len();
        self.batch.states.resize(start + G::state_size(), 0.0);
        game.encode_state(&mut self.batch.states[start..]);

        self.batch.legal.extend(game.legal_actions());
        self.batch.offsets.push(self.batch.legal.len() as u32);
    }

    pub(super) fn evaluate_position<G: Game>(&mut self, game: &G) -> (Vec<Action>, Evaluation) {
        self.evaluate_positions(std::slice::from_ref(game))
            .pop()
            .expect("single game evaluation")
    }

    pub(super) fn evaluate_positions<G: Game>(
        &mut self,
        games: &[G],
    ) -> Vec<(Vec<Action>, Evaluation)> {
        if games.is_empty() {
            return Vec::new();
        }
        self.batch.clear();
        let expected = games.len();
        let mut legal_per_state = Vec::with_capacity(games.len());
        for game in games {
            let begin = self.batch.legal.len();
            self.enqueue_state(game);
            legal_per_state.push(self.batch.legal[begin..].to_vec());
        }
        let results = self.evaluator.evaluate(&mut self.batch);
        debug_assert_eq!(
            results.len(),
            expected,
            "evaluator must return one result per input state"
        );
        legal_per_state.into_iter().zip(results).collect()
    }

    pub(super) fn evaluate_repetition_position<G: RepetitionGame>(
        &mut self,
        game: &G,
    ) -> (Vec<Action>, Evaluation) {
        let key = game.evaluation_cache_key();
        if let Some(cache) = &self.eval_cache {
            if let Some(cached) = cache.get(key) {
                return (cached.legal, cached.eval);
            }
        }

        let (legal, eval) = self.evaluate_position(game);
        if let Some(cache) = &self.eval_cache {
            cache.insert(
                key,
                CachedEvaluation {
                    legal: legal.clone(),
                    eval: eval.clone(),
                },
            );
        }
        (legal, eval)
    }

    pub(super) fn evaluate_repetition_positions<G: RepetitionGame>(
        &mut self,
        games: &[G],
    ) -> Vec<(Vec<Action>, Evaluation)> {
        if games.is_empty() {
            return Vec::new();
        }

        let Some(cache) = self.eval_cache.clone()
        else {
            return self.evaluate_positions(games);
        };
        let mut out = vec![None; games.len()];
        let mut miss_destinations = Vec::new();
        let mut miss_keys = Vec::new();
        let mut miss_games = Vec::new();

        for (i, game) in games.iter().enumerate() {
            let key = game.evaluation_cache_key();
            // Distinct tree nodes can transpose to the same encoded state in
            // one leaf batch. The shared table cannot contain the first miss
            // until inference completes, so deduplicate those misses locally.
            if let Some(unique) = miss_keys.iter().position(|&candidate| candidate == key) {
                miss_destinations.push((i, unique));
                continue;
            }
            if let Some(cached) = cache.get(key) {
                out[i] = Some((cached.legal, cached.eval));
                continue;
            }
            let unique = miss_games.len();
            miss_destinations.push((i, unique));
            miss_keys.push(key);
            miss_games.push(*game);
        }

        let miss_results = self.evaluate_positions(&miss_games);
        for (&key, (legal, eval)) in miss_keys.iter().zip(&miss_results) {
            cache.insert(
                key,
                CachedEvaluation {
                    legal: legal.clone(),
                    eval: eval.clone(),
                },
            );
        }
        for (i, unique) in miss_destinations {
            out[i] = Some(miss_results[unique].clone());
        }

        out.into_iter()
            .map(|value| value.expect("all repetition eval slots filled"))
            .collect()
    }

    /// Softmax over legal-action logits (optionally mixed with root Dirichlet
    /// noise) into `self.policy_buf`. Raw logits are retained for Gumbel search.
    pub(super) fn build_policy_from(
        &mut self,
        legal: &[Action],
        res: &Evaluation,
        root_noise: bool,
    ) {
        debug_assert_eq!(
            legal.len(),
            res.logits.len(),
            "evaluator logits must match the legal actions for each state"
        );
        debug_assert!(
            res.logits.iter().all(|logit| logit.is_finite()),
            "evaluator must return finite logits for legal actions"
        );

        self.policy_buf.clear();
        let max_logit = res.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (&action, &logit) in legal.iter().zip(&res.logits) {
            let prior = (logit - max_logit).exp();
            self.policy_buf.push((action, prior, logit));
            sum += prior;
        }
        if sum.is_finite() && sum > 0.0 {
            for (_, prior, _) in &mut self.policy_buf {
                *prior /= sum;
            }
        }
        else if !self.policy_buf.is_empty() {
            let uniform = 1.0 / self.policy_buf.len() as f32;
            for (_, prior, _) in &mut self.policy_buf {
                *prior = uniform;
            }
        }

        if root_noise && self.cfg.eps > 0.0 && !self.policy_buf.is_empty() {
            let gamma = Gamma::new(self.cfg.alpha, 1.0).expect("alpha > 0");
            let mut noise: Vec<f32> = (0..self.policy_buf.len())
                .map(|_| gamma.sample(&mut self.rng))
                .collect();
            let noise_sum: f32 = noise.iter().sum();
            if noise_sum > 0.0 {
                for x in &mut noise {
                    *x /= noise_sum;
                }
            }
            for ((_, prior, _), noise) in self.policy_buf.iter_mut().zip(&noise) {
                *prior = (1.0 - self.cfg.eps) * *prior + self.cfg.eps * noise;
            }
        }
    }

    /// Creates all legal children of `node`, contiguous in the arena.
    pub(super) fn expand(&mut self, node: u32) {
        debug_assert!(self.policy_buf.len() <= u16::MAX as usize);
        let first_child = self.nodes.len() as u32;
        for &(action, prior, logit) in &self.policy_buf {
            self.nodes
                .push(Node::new(Some(node), action, 0, prior, logit, false, 0.0));
        }
        let n = &mut self.nodes[node as usize];
        n.first_child = first_child;
        n.num_children = self.policy_buf.len() as u16;
        n.expanded = true;
    }
}

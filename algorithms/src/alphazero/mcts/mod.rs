mod cache;
mod core;
mod gumbel;
mod puct;

#[cfg(test)]
mod tests;

use super::evaluator::Evaluator;
#[cfg(test)]
use super::evaluator::{EvalBatch, Evaluation};
use engine_core::agent::{Agent, PolicyMode};
use engine_core::game::{Action, Game};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;

#[cfg(test)]
pub(crate) use cache::CachedEvaluation;
pub(crate) use cache::{EvalTable, EvalTableStats};
pub use core::Mcts;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MctsVariant {
    /// Standard AlphaZero PUCT search.
    Puct,
    /// Gumbel AlphaZero-style root sampling with sequential halving.
    Gumbel {
        /// Root actions considered before sequential halving.
        sampled_actions: usize,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub c_init: f32,
    pub c_base: f32,
    pub variant: MctsVariant,
    /// Simulations per search.
    pub simulations: usize,
    /// Dirichlet noise weight at the root (0 disables, e.g. for match play).
    pub eps: f32,
    pub alpha: f32,
    /// First Play Urgency reduction. Unvisited children are valued as the
    /// parent's current value minus this amount, from the parent's perspective.
    pub fpu_reduction: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            c_init: 1.25,
            c_base: 19652.0,
            variant: MctsVariant::Puct,
            simulations: 800,
            eps: 0.25,
            alpha: 0.3,
            fpu_reduction: 0.1,
        }
    }
}

/// Outcome of one search.
///
/// For PUCT, `policy` is the normalized root visit-count distribution. For
/// Gumbel AlphaZero, it is the search-improved policy
/// `softmax(root_logits + transformed_completed_q)` used as the policy target.
pub struct SearchResult {
    pub policy: Vec<f32>,
    pub selected_action: Action,
    pub value: f32,
}

impl SearchResult {
    /// The action proposed by the search.
    pub fn best_action(&self) -> Action {
        self.selected_action
    }

    /// Samples from the returned policy distribution.
    pub fn sample_action<R: Rng + ?Sized>(&self, rng: &mut R) -> Action {
        WeightedIndex::new(&self.policy)
            .expect("search of a non-terminal position returns a non-empty policy")
            .sample(rng) as Action
    }
}

pub(crate) fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

#[derive(Clone, Copy, Debug)]
struct RootAction {
    node: u32,
    /// Raw network logit plus the one Gumbel sample reused throughout search.
    gumbel_logit: f32,
}

#[derive(Clone, Copy)]
struct RootQTransform {
    completed_value: f32,
    min_value: f32,
    inv_range: f32,
    scale: f32,
}

impl RootQTransform {
    fn apply(self, q: f32) -> f32 {
        let normalized = if self.inv_range == 0.0 {
            0.0
        } else {
            (q - self.min_value) * self.inv_range
        };
        self.scale * normalized
    }
}

/// Tree node. Children are stored sparsely (one node per legal action) and
/// contiguously, so a node only needs the range
/// `first_child .. first_child + num_children`. All nodes live in one `Vec`
/// arena, indexed by `u32` and cleared (not freed) between searches.
struct Node {
    parent: Option<u32>,
    hash: u64,
    first_child: u32,
    action_from_parent: Action,
    visits: u32,
    value_sum: f32,
    prior: f32,
    logit: f32,
    reward: f32,
    num_children: u16,
    expanded: bool,
    terminal: bool,
}

impl Node {
    fn new(
        parent: Option<u32>,
        action_from_parent: Action,
        hash: u64,
        prior: f32,
        logit: f32,
        terminal: bool,
        reward: f32,
    ) -> Self {
        Node {
            parent,
            hash,
            first_child: 0,
            action_from_parent,
            visits: 0,
            value_sum: 0.0,
            prior,
            logit,
            reward,
            num_children: 0,
            expanded: false,
            terminal,
        }
    }

    /// Mean value from this node's own player-to-move perspective.
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

impl<G: Game, E: Evaluator> Agent<G> for Mcts<E> {
    fn act_with_mode(&mut self, game: &G, mode: PolicyMode) -> Action {
        let variant = self.config().variant;
        let explore = mode == PolicyMode::Explore;
        let result = self.search_with_mode(game, mode);
        if matches!(variant, MctsVariant::Puct) && explore {
            let mut rng = rand::rng();
            result.sample_action(&mut rng)
        } else {
            // Gumbel exploration is already supplied by the root Gumbel sample.
            result.best_action()
        }
    }
}

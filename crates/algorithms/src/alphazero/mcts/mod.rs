mod cache;
mod core;
mod evaluation;
mod gumbel;
mod puct;
mod traversal;

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
pub use cache::{EvalTable, EvalTableStats};
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

/// The two parameters that must change together when varying a Gumbel search
/// budget. Keeping them in one value prevents a lower simulation budget from
/// accidentally retaining an oversized sequential-halving root set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GumbelSearchProfile {
    pub simulations: usize,
    pub root_candidates: usize,
}

impl GumbelSearchProfile {
    pub const fn new(simulations: usize, root_candidates: usize) -> Self {
        Self {
            simulations,
            root_candidates,
        }
    }

    pub fn validate(self) -> Result<(), &'static str> {
        if self.simulations == 0 {
            return Err("Gumbel profile simulations must be positive");
        }
        if self.root_candidates == 0 {
            return Err("Gumbel profile root_candidates must be positive");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub c_init: f32,
    pub c_base: f32,
    pub variant: MctsVariant,
    /// Simulations per search.
    pub simulations: usize,
    /// Leaf evaluations collected inside one tree search before calling the evaluator.
    /// `1` preserves the original single-leaf simulation behavior.
    pub leaf_batch_size: usize,
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
            leaf_batch_size: 1,
            eps: 0.25,
            alpha: 0.3,
            fpu_reduction: 0.1,
        }
    }
}

impl MctsConfig {
    /// Checks invariants required by every MCTS variant.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.c_init.is_finite() || self.c_init < 0.0 {
            return Err("c_init must be finite and non-negative");
        }
        if !self.c_base.is_finite() || self.c_base <= 0.0 {
            return Err("c_base must be finite and positive");
        }
        if self.simulations == 0 {
            return Err("simulations must be positive");
        }
        if self.leaf_batch_size == 0 {
            return Err("leaf_batch_size must be positive");
        }
        if !self.eps.is_finite() || !(0.0..=1.0).contains(&self.eps) {
            return Err("eps must be finite and in [0, 1]");
        }
        if !self.alpha.is_finite() || (self.eps > 0.0 && self.alpha <= 0.0) {
            return Err("alpha must be finite and positive when root noise is enabled");
        }
        if !self.fpu_reduction.is_finite() || self.fpu_reduction < 0.0 {
            return Err("fpu_reduction must be finite and non-negative");
        }
        if matches!(self.variant, MctsVariant::Gumbel { sampled_actions: 0 }) {
            return Err("Gumbel sampled_actions must be positive");
        }
        Ok(())
    }
}

/// Outcome of one search.
///
/// For PUCT, `policy` is the normalized root visit-count distribution. For
/// Gumbel AlphaZero, it is the search-improved policy
/// `softmax(root_logits + transformed_completed_q)` used as the policy target.
pub struct SearchResult {
    /// Normalized probability mass for legal root actions only.
    pub policy: Vec<(Action, f32)>,
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
        let index = WeightedIndex::new(self.policy.iter().map(|&(_, probability)| probability))
            .expect("search of a non-terminal position returns a non-empty policy")
            .sample(rng);
        self.policy[index].0
    }

    /// Probability assigned to `action`, or zero when the action is illegal.
    pub fn probability(&self, action: Action) -> f32 {
        self.policy
            .iter()
            .find_map(|&(candidate, probability)| (candidate == action).then_some(probability))
            .unwrap_or(0.0)
    }

    /// Expands the sparse legal-action policy for compatibility consumers.
    pub fn dense_policy(&self, action_size: usize) -> Vec<f32> {
        let mut dense = vec![0.0; action_size];
        for &(action, probability) in &self.policy {
            dense[action as usize] = probability;
        }
        dense
    }
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
        }
        else {
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
    repetitions_before_current: u8,
    repetition_cached: bool,
    first_child: u32,
    action_from_parent: Action,
    visits: u32,
    virtual_loss_count: u32,
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
            repetitions_before_current: 0,
            repetition_cached: false,
            first_child: 0,
            action_from_parent,
            visits: 0,
            virtual_loss_count: 0,
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
        }
        else {
            self.value_sum / self.visits as f32
        }
    }

    fn effective_visits(&self) -> u32 {
        self.visits + self.virtual_loss_count
    }

    fn effective_q(&self) -> f32 {
        let visits = self.effective_visits();
        if visits == 0 {
            0.0
        }
        else {
            (self.value_sum + self.virtual_loss_count as f32) / visits as f32
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
        }
        else {
            // Gumbel exploration is already supplied by the root Gumbel sample.
            result.best_action()
        }
    }
}

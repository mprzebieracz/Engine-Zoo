mod cache;
mod core;
mod evaluation;
mod gumbel;
mod puct;
mod root_gumbel_puct;

mod traversal;

#[cfg(test)]
mod tests;

use crate::PositionValue;
use engine_core::agent::PolicyMode;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

pub use cache::{EvalTable, EvalTableStats};
pub use core::Mcts;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "algorithm", rename_all = "kebab-case")]
pub enum SearchConfig {
    Puct(PuctConfig),
    RootGumbelPuct(RootGumbelPuctConfig),
    FullGumbel(FullGumbelConfig),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SearchAlgorithm {
    Puct,
    RootGumbelPuct,
    FullGumbel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SearchBudget {
    Puct {
        simulations: usize,
    },
    Gumbel {
        simulations: usize,
        max_considered_actions: usize,
    },
}

impl SearchBudget {
    fn validate(self) -> Result<(), &'static str> {
        match self {
            Self::Puct { simulations } if simulations > 0 => Ok(()),
            Self::Gumbel {
                simulations,
                max_considered_actions,
            } if simulations > 0 && max_considered_actions > 0 => Ok(()),
            Self::Puct { .. } => Err("PUCT simulations must be positive"),
            Self::Gumbel { simulations: 0, .. } => Err("Gumbel simulations must be positive"),
            Self::Gumbel { .. } => Err("Gumbel max_considered_actions must be positive"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SearchRequest {
    pub mode: PolicyMode,
    pub budget: SearchBudget,
}

impl SearchRequest {
    pub const fn deterministic_puct(simulations: usize) -> Self {
        Self {
            mode: PolicyMode::Deterministic,
            budget: SearchBudget::Puct { simulations },
        }
    }
}

/// Limits parallel leaf selection so fast searches do not spend most of their
/// budget on one stale selection round.
///
/// Search budgets are validated before this is called, but the final `max`
/// keeps the helper safe for direct callers and documents its non-zero result.
pub(super) fn effective_leaf_batch_size(configured_leaf_batch: usize, simulations: usize) -> usize {
    configured_leaf_batch.min(simulations.div_ceil(4)).max(1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchConfigError(&'static str);

impl fmt::Display for SearchConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl Error for SearchConfigError {}

impl SearchConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        match self {
            Self::Puct(config) => config.validate(),
            Self::RootGumbelPuct(config) => config.validate(),
            Self::FullGumbel(config) => config.validate(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self::Puct(PuctConfig::default())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PuctConfig {
    pub leaf_batch_size: usize,
    pub selection: PuctSelectionConfig,
    pub fpu: FpuConfig,
    pub in_flight: InFlightConfig,
    pub root_noise: Option<DirichletConfig>,
}

impl Default for PuctConfig {
    fn default() -> Self {
        Self {
            leaf_batch_size: 1,
            selection: PuctSelectionConfig::default(),
            fpu: FpuConfig::default(),
            in_flight: InFlightConfig::UnscoredVirtualVisits,
            root_noise: Some(DirichletConfig::default()),
        }
    }
}

impl PuctConfig {
    /// Single-position application analysis defaults.
    pub fn analysis_default(leaf_batch_size: usize) -> Self {
        Self {
            leaf_batch_size,
            root_noise: None,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.leaf_batch_size == 0 {
            return Err("PUCT leaf_batch_size must be positive");
        }
        self.selection.validate()?;
        self.fpu.validate()?;
        self.in_flight.validate()?;
        if let Some(noise) = self.root_noise {
            noise.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PuctTreeConfig {
    pub selection: PuctSelectionConfig,
    pub fpu: FpuConfig,
    pub in_flight: InFlightConfig,
}

impl From<&PuctConfig> for PuctTreeConfig {
    fn from(config: &PuctConfig) -> Self {
        Self {
            selection: config.selection,
            fpu: config.fpu,
            in_flight: config.in_flight,
        }
    }
}

impl PuctTreeConfig {
    fn validate(self) -> Result<(), &'static str> {
        self.selection.validate()?;
        self.fpu.validate()?;
        self.in_flight.validate()
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PuctSelectionConfig {
    pub pb_c_init: f32,
    pub pb_c_base: f32,
    pub pb_c_factor: f32,
}

impl Default for PuctSelectionConfig {
    fn default() -> Self {
        Self {
            pb_c_init: 1.25,
            pb_c_base: 19_652.0,
            pb_c_factor: 1.0,
        }
    }
}

impl PuctSelectionConfig {
    fn validate(self) -> Result<(), &'static str> {
        if !self.pb_c_init.is_finite() || self.pb_c_init < 0.0 {
            return Err("pb_c_init must be finite and non-negative");
        }
        if !self.pb_c_base.is_finite() || self.pb_c_base <= 0.0 {
            return Err("pb_c_base must be finite and positive");
        }
        if !self.pb_c_factor.is_finite() || self.pb_c_factor < 0.0 {
            return Err("pb_c_factor must be finite and non-negative");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum FpuConfig {
    Reduction {
        reduction: f32,
        root_reduction: Option<f32>,
    },
    Absolute {
        value: PositionValue,
        root_value: Option<PositionValue>,
    },
}

impl Default for FpuConfig {
    fn default() -> Self {
        Self::Reduction {
            reduction: 0.33,
            root_reduction: None,
        }
    }
}

impl FpuConfig {
    fn validate(self) -> Result<(), &'static str> {
        match self {
            Self::Reduction {
                reduction,
                root_reduction,
            } if reduction.is_finite()
                && reduction >= 0.0
                && root_reduction.is_none_or(|value| value.is_finite() && value >= 0.0) =>
            {
                Ok(())
            }
            Self::Absolute { .. } => Ok(()),
            _ => Err("FPU reduction must be finite and non-negative"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum InFlightConfig {
    UnscoredVirtualVisits,
    VirtualLoss { value: PositionValue },
}

impl InFlightConfig {
    fn validate(self) -> Result<(), &'static str> {
        match self {
            Self::UnscoredVirtualVisits => Ok(()),
            Self::VirtualLoss { value } if value.as_f32().is_finite() => Ok(()),
            Self::VirtualLoss { .. } => Err("virtual-loss value must be finite"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DirichletConfig {
    pub epsilon: f32,
    pub alpha: f32,
}

impl Default for DirichletConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.25,
            alpha: 0.3,
        }
    }
}

impl DirichletConfig {
    fn validate(self) -> Result<(), &'static str> {
        if !self.epsilon.is_finite() || !(0.0..=1.0).contains(&self.epsilon) {
            return Err("Dirichlet epsilon must be finite and in [0, 1]");
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err("Dirichlet alpha must be finite and positive");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GumbelRootConfig {
    pub gumbel_scale: f32,
    pub completed_q: CompletedQConfig,
}

impl Default for GumbelRootConfig {
    fn default() -> Self {
        Self {
            gumbel_scale: 1.0,
            completed_q: CompletedQConfig::default(),
        }
    }
}

impl GumbelRootConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.gumbel_scale.is_finite() || self.gumbel_scale < 0.0 {
            return Err("Gumbel scale must be finite and non-negative");
        }
        self.completed_q.validate()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RootGumbelPuctConfig {
    pub leaf_batch_size: usize,
    pub puct: PuctTreeConfig,
    pub root: GumbelRootConfig,
}

impl Default for RootGumbelPuctConfig {
    fn default() -> Self {
        Self {
            leaf_batch_size: 1,
            puct: PuctTreeConfig::from(&PuctConfig::default()),
            root: GumbelRootConfig::default(),
        }
    }
}

impl RootGumbelPuctConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.leaf_batch_size == 0 {
            return Err("root-Gumbel-PUCT leaf_batch_size must be positive");
        }

        self.puct.validate()?;
        self.root.validate()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FullGumbelConfig {
    pub root: GumbelRootConfig,
}

impl FullGumbelConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.root.validate()
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CompletedQConfig {
    pub value_scale: f32,
    pub maxvisit_init: f32,
    pub rescale_values: bool,
    pub use_mixed_value: bool,
    pub epsilon: f32,
}

impl Default for CompletedQConfig {
    fn default() -> Self {
        Self {
            value_scale: 0.1,
            maxvisit_init: 50.0,
            rescale_values: true,
            use_mixed_value: true,
            epsilon: 1e-8,
        }
    }
}

impl CompletedQConfig {
    fn validate(self) -> Result<(), &'static str> {
        if !self.value_scale.is_finite() || self.value_scale < 0.0 {
            return Err("completed-Q value_scale must be finite and non-negative");
        }
        if !self.maxvisit_init.is_finite() || self.maxvisit_init < 0.0 {
            return Err("completed-Q maxvisit_init must be finite and non-negative");
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err("completed-Q epsilon must be finite and positive");
        }
        Ok(())
    }
}

/// Outcome of one search.
///
/// For PUCT, `policy` is the normalized root visit-count distribution. For
/// Gumbel AlphaZero, it is the search-improved policy
/// `softmax(root_logits + transformed_completed_q)` used as the policy target.
#[derive(Debug)]
pub struct SearchResult<M> {
    /// Normalized probability mass for legal root actions only.
    pub policy: Vec<(M, f32)>,
    pub selected_move: M,
    /// Value from the root state's player-to-move perspective.
    pub root_value: PositionValue,
    pub diagnostics: SearchDiagnostics,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchDiagnostics {
    pub completed_simulations: usize,
    pub backend_evaluations: usize,
    pub evaluation_cache_hits: usize,
    pub evaluation_cache_misses: usize,
    pub duplicate_leaves: usize,
    pub nodes_created: usize,
    pub max_depth: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SearchError {
    InvalidBudget {
        message: &'static str,
    },
    BudgetAlgorithmMismatch {
        algorithm: SearchAlgorithm,
        budget: SearchBudget,
    },
    TerminalRoot {
        value: PositionValue,
    },
    NoLegalMoves,
    Evaluation(crate::EvaluationError),
}

impl fmt::Display for SearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBudget { message } => formatter.write_str(message),
            Self::BudgetAlgorithmMismatch { algorithm, budget } => {
                write!(
                    formatter,
                    "{algorithm:?} search cannot use {budget:?} budget"
                )
            }
            Self::TerminalRoot { value } => {
                write!(
                    formatter,
                    "cannot search a terminal root (value {})",
                    value.as_f32()
                )
            }
            Self::NoLegalMoves => formatter.write_str("cannot search a root with no legal moves"),
            Self::Evaluation(error) => write!(formatter, "search evaluation failed: {error}"),
        }
    }
}

impl Error for SearchError {}

impl<M: Copy> SearchResult<M> {
    /// The move proposed by the search.
    pub fn best_move(&self) -> M {
        self.selected_move
    }

    /// Samples from the returned policy distribution.
    pub fn sample_move<R: Rng + ?Sized>(&self, rng: &mut R) -> M {
        let index = WeightedIndex::new(self.policy.iter().map(|&(_, probability)| probability))
            .expect("search of a non-terminal position returns a non-empty policy")
            .sample(rng);
        self.policy[index].0
    }
}

impl<M: Copy + Eq> SearchResult<M> {
    /// Probability assigned to `move_`, or zero when the move is illegal.
    pub fn probability(&self, move_: M) -> f32 {
        self.policy
            .iter()
            .find_map(|&(candidate, probability)| (candidate == move_).then_some(probability))
            .unwrap_or(0.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RootAction {
    node: u32,
    /// Raw network logit plus the one Gumbel sample reused throughout search.
    gumbel_logit: f32,
}

#[derive(Clone, Copy)]

/// Tree node. Children are stored sparsely (one node per legal action) and
/// contiguously, so a node only needs the range
/// `first_child .. first_child + num_children`. All nodes live in one `Vec`
/// arena, indexed by `u32` and cleared (not freed) between searches.
pub(super) struct Node<M, Meta = ()> {
    parent: Option<u32>,
    meta: Meta,
    first_child: u32,
    move_from_parent: Option<M>,
    completed_visits: u32,
    in_flight_visits: u32,
    value_sum_from_node_pov: f32,
    /// Sum of priors for direct children that have completed at least one visit.
    /// This lets reduction FPU avoid scanning every child during selection.
    visited_child_prior_mass: f32,
    prior: f32,
    logit: f32,
    reward: PositionValue,
    raw_value: PositionValue,
    num_children: u16,
    expanded: bool,
    terminal: bool,
}

impl<M, Meta: Default> Node<M, Meta> {
    fn new(
        parent: Option<u32>,
        move_from_parent: Option<M>,
        prior: f32,
        logit: f32,
        terminal: bool,
        reward: PositionValue,
    ) -> Self {
        Node {
            parent,
            meta: Meta::default(),
            first_child: 0,
            move_from_parent,
            completed_visits: 0,
            in_flight_visits: 0,
            value_sum_from_node_pov: 0.0,
            visited_child_prior_mass: 0.0,
            prior,
            logit,
            reward,
            raw_value: PositionValue::DRAW,
            num_children: 0,
            expanded: false,
            terminal,
        }
    }

    /// Mean value from this node's own player-to-move perspective.
    fn completed_q(&self) -> Option<PositionValue> {
        (self.completed_visits > 0).then(|| {
            PositionValue::from_finite_clamped(
                self.value_sum_from_node_pov / self.completed_visits as f32,
            )
            .expect("completed finite values must have a finite average")
        })
    }

    fn selection_visits(&self) -> u32 {
        self.completed_visits + self.in_flight_visits
    }

    fn reserve_visit(&mut self) {
        self.in_flight_visits += 1;
    }

    fn complete_reserved_visit(&mut self, value: PositionValue) {
        assert!(self.in_flight_visits > 0, "in-flight visit underflow");
        self.in_flight_visits -= 1;
        self.completed_visits += 1;
        self.value_sum_from_node_pov += value.as_f32();
    }

    fn cancel_reserved_visit(&mut self) {
        assert!(self.in_flight_visits > 0, "in-flight visit underflow");
        self.in_flight_visits -= 1;
    }
}

use super::core::{LeafBatch, MctsCore};
use super::{
    FpuConfig, InFlightConfig, Node, PuctConfig, PuctSelectionConfig, PuctTreeConfig,
    SearchDiagnostics, SearchResult,
};
use crate::PositionValue;
use engine_core::agent::PolicyMode;
use engine_core::game::GameState;

#[derive(Clone, Debug)]
pub(super) struct Puct {
    pub(super) leaf_batch_size: usize,
    selection: PuctSelectionConfig,
    fpu: FpuConfig,
    in_flight: InFlightConfig,
    root_noise: Option<super::DirichletConfig>,
}

impl From<PuctConfig> for Puct {
    fn from(config: PuctConfig) -> Self {
        Self {
            leaf_batch_size: config.leaf_batch_size,
            selection: config.selection,
            fpu: config.fpu,
            in_flight: config.in_flight,
            root_noise: config.root_noise,
        }
    }
}

impl Puct {
    pub(super) fn config(&self) -> PuctConfig {
        PuctConfig {
            leaf_batch_size: self.leaf_batch_size,
            selection: self.selection,
            fpu: self.fpu,
            in_flight: self.in_flight,
            root_noise: self.root_noise,
        }
    }
}

pub(super) fn dynamic_c(config: PuctSelectionConfig, parent_visits: u32) -> f32 {
    config.pb_c_init
        + config.pb_c_factor
            * (((parent_visits as f32 + config.pb_c_base + 1.0) / config.pb_c_base).ln())
}

impl<G, E, R> MctsCore<G, E, R, Puct>
where
    G: GameState + Clone,
    E: crate::PolicyValueEvaluator<G>,
    R: crate::SearchRules<G>,
{
    pub(super) fn search_inner(
        &mut self,
        game: &G,
        context: R::Context<'_>,
        mode: PolicyMode,
        simulations: usize,
    ) -> Result<SearchResult<G::Move>, super::SearchError> {
        self.clear_tree_common();
        self.nodes
            .push(Node::new(None, None, 0.0, 0.0, false, PositionValue::DRAW));

        let mut root = game.clone();
        self.rules.reset_path(context, game, &mut self.path_state);
        let root_rule = self.rules.enter_state(
            context,
            &mut root,
            &mut self.path_state,
            &mut self.nodes[0].meta,
        );
        if let Some(value) = self.terminal_value(&root, root_rule) {
            return Err(super::SearchError::TerminalRoot { value });
        }
        if root.legal_moves().next().is_none() {
            return Err(super::SearchError::NoLegalMoves);
        }
        let (root_eval, root_stats) = self
            .evaluate_position(&root)
            .map_err(super::SearchError::Evaluation)?;
        if root_eval.legal().is_empty() {
            return Err(super::SearchError::NoLegalMoves);
        }
        self.nodes[0].raw_value = root_eval.value;
        self.build_policy_from(
            root_eval.legal(),
            root_eval.logits(),
            mode == PolicyMode::Explore,
            self.variant.root_noise,
        );
        self.expand(0);

        let mut batch = LeafBatch::with_capacity(self.variant.leaf_batch_size);
        let mut diagnostics = SearchDiagnostics {
            backend_evaluations: root_stats.backend_evaluations,
            evaluation_cache_hits: root_stats.cache_hits,
            evaluation_cache_misses: root_stats.backend_evaluations,
            ..SearchDiagnostics::default()
        };
        while diagnostics.completed_simulations < simulations {
            let remaining = simulations - diagnostics.completed_simulations;
            self.collect_puct_leaves(
                game,
                remaining,
                context,
                &mut batch.leaves,
                &mut diagnostics,
            );
            self.finish_leaf_batch(&mut batch, &mut diagnostics)
                .map_err(super::SearchError::Evaluation)?;
        }
        let policy = self.root_visit_policy();
        let selected_move = policy
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("non-terminal root must contain a legal move")
            .0;
        let root_value = self.nodes[0].completed_q().unwrap_or(root_eval.value);
        diagnostics.nodes_created = self.nodes.len();

        Ok(SearchResult {
            policy,
            selected_move,
            root_value,
            diagnostics,
        })
    }

    fn collect_puct_leaves(
        &mut self,
        game: &G,
        budget: usize,
        context: R::Context<'_>,
        leaves: &mut Vec<super::core::PendingLeaf<G>>,
        diagnostics: &mut SearchDiagnostics,
    ) {
        leaves.clear();
        for _ in 0..budget.min(self.variant.leaf_batch_size) {
            let (node, state, depth) = self.descend(game, 0, context, |core, node| {
                puct_child(core, node, core.tree_config())
            });
            self.reserve_path(node);
            leaves.push(super::core::PendingLeaf { node, game: state });
            diagnostics.max_depth = diagnostics.max_depth.max(depth);
        }
    }

    fn tree_config(&self) -> PuctTreeConfig {
        PuctTreeConfig {
            selection: self.variant.selection,
            fpu: self.variant.fpu,
            in_flight: self.variant.in_flight,
        }
    }
}

pub(super) fn puct_child<G, E, R, V>(
    core: &MctsCore<G, E, R, V>,
    node: u32,
    config: PuctTreeConfig,
) -> Option<u32>
where
    G: GameState + Clone,
    E: crate::PolicyValueEvaluator<G>,
    R: crate::SearchRules<G>,
    V: 'static,
{
    let parent = &core.nodes[node as usize];
    let parent_visits = parent.selection_visits();
    let c = dynamic_c(config.selection, parent_visits);
    let sqrt_parent = (parent_visits as f32 + 1.0).sqrt();
    let visited_prior_mass: f32 = child_indices(core, node)
        .map(|child| &core.nodes[child as usize])
        .filter(|child| child.completed_visits > 0)
        .map(|child| child.prior)
        .sum();
    let parent_q = parent.completed_q().unwrap_or(parent.raw_value).as_f32();
    let fpu = match config.fpu {
        FpuConfig::Absolute { value, root_value } => {
            if node == 0 {
                root_value.unwrap_or(value).as_f32()
            }
            else {
                value.as_f32()
            }
        }
        FpuConfig::Reduction {
            reduction,
            root_reduction,
        } => {
            let reduction = if node == 0 {
                root_reduction.unwrap_or(reduction)
            }
            else {
                reduction
            };
            parent_q - reduction * visited_prior_mass.sqrt()
        }
    };

    child_indices(core, node).max_by(|&left, &right| {
        puct_score(core, left, c, sqrt_parent, fpu, config.in_flight).total_cmp(&puct_score(
            core,
            right,
            c,
            sqrt_parent,
            fpu,
            config.in_flight,
        ))
    })
}

fn puct_score<G, E, R, V>(
    core: &MctsCore<G, E, R, V>,
    child: u32,
    c: f32,
    sqrt_parent: f32,
    fpu: f32,
    in_flight: InFlightConfig,
) -> f32
where
    G: GameState + Clone,
    E: crate::PolicyValueEvaluator<G>,
    R: crate::SearchRules<G>,
    V: 'static,
{
    let child = &core.nodes[child as usize];
    let mut q = child
        .completed_q()
        .map_or(fpu, |value| value.flipped().as_f32());
    if let InFlightConfig::VirtualLoss { value } = in_flight {
        if child.in_flight_visits > 0 {
            q = (q * child.completed_visits as f32
                - value.as_f32() * child.in_flight_visits as f32)
                / child.selection_visits() as f32;
        }
    }
    q + c * child.prior * sqrt_parent / (1 + child.selection_visits()) as f32
}

fn child_indices<G, E, R, V>(core: &MctsCore<G, E, R, V>, node: u32) -> std::ops::Range<u32>
where
    G: GameState + Clone,
    E: crate::PolicyValueEvaluator<G>,
    R: crate::SearchRules<G>,
    V: 'static,
{
    let node = &core.nodes[node as usize];
    node.first_child..node.first_child + u32::from(node.num_children)
}

impl<G, E, R> MctsCore<G, E, R, Puct>
where
    G: GameState + Clone,
    E: crate::PolicyValueEvaluator<G>,
    R: crate::SearchRules<G>,
{
    fn child_indices(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        let node = &self.nodes[node as usize];
        node.first_child..node.first_child + u32::from(node.num_children)
    }

    fn root_visit_policy(&self) -> Vec<(G::Move, f32)> {
        let mut policy: Vec<_> = self
            .child_indices(0)
            .map(|child| {
                let node = &self.nodes[child as usize];
                (
                    node.move_from_parent.expect("root child must store a move"),
                    node.completed_visits as f32,
                )
            })
            .collect();
        normalize_policy(&mut policy);
        policy
    }
}

fn normalize_policy<M>(policy: &mut [(M, f32)]) {
    let total: f32 = policy.iter().map(|(_, probability)| *probability).sum();
    if total.is_finite() && total > 0.0 {
        for (_, probability) in policy {
            *probability /= total;
        }
    }
    else if !policy.is_empty() {
        let uniform = 1.0 / policy.len() as f32;
        for (_, probability) in policy {
            *probability = uniform;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_c_matches_formula() {
        let config = PuctSelectionConfig {
            pb_c_init: 1.25,
            pb_c_base: 19_652.0,
            pb_c_factor: 1.0,
        };
        let expected = 1.25_f32 + ((101.0_f32 + 19_652.0) / 19_652.0).ln();
        assert!((dynamic_c(config, 100) - expected).abs() < 1e-6);
    }
}

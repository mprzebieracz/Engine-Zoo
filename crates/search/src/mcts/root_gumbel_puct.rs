use super::core::{LeafBatch, MctsCore, PendingLeaf};
use super::gumbel::{sample_gumbel, softmax, transform_completed_q};
use super::puct::puct_child;
use super::{Node, RootAction, RootGumbelPuctConfig, SearchDiagnostics, SearchResult};
use crate::PositionValue;
use engine_core::agent::PolicyMode;
use engine_core::game::GameState;

fn sequential_halving_round_visits(remaining: usize, active: usize) -> usize {
    if active == 1 {
        return remaining;
    }

    let rounds = active.ilog2() as usize;
    let per_action = (remaining / (rounds.max(1) * active)).max(1);
    (per_action * active).min(remaining)
}

#[cfg(test)]
mod tests {
    use super::sequential_halving_round_visits;

    #[test]
    fn round_budget_never_exceeds_remaining_work() {
        assert_eq!(sequential_halving_round_visits(3, 8), 3);
        assert_eq!(sequential_halving_round_visits(16, 4), 8);
    }

    #[test]
    fn lone_survivor_receives_all_remaining_work() {
        assert_eq!(sequential_halving_round_visits(11, 1), 11);
    }
}

/// Hybrid root-Gumbel search. Root actions receive Gumbel sequential-halving
/// allocations; descendants use the ordinary PUCT scoring implementation.
#[derive(Debug)]
pub(super) struct RootGumbelPuct {
    pub(super) config: RootGumbelPuctConfig,
    root_actions: Vec<RootAction>,
}

impl From<RootGumbelPuctConfig> for RootGumbelPuct {
    fn from(config: RootGumbelPuctConfig) -> Self {
        Self {
            config,
            root_actions: Vec::new(),
        }
    }
}

impl<G, E, R> MctsCore<G, E, R, RootGumbelPuct>
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
        max_considered_actions: usize,
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

        let (evaluation, root_stats) = self
            .evaluate_position(&root)
            .map_err(super::SearchError::Evaluation)?;
        if evaluation.legal().is_empty() {
            return Err(super::SearchError::NoLegalMoves);
        }

        let root_value = evaluation.value;
        self.nodes[0].raw_value = root_value;
        self.build_policy_from(
            evaluation.legal(),
            evaluation.logits(),
            mode == PolicyMode::Explore,
            None,
        );
        self.expand(0);
        self.init_root_actions(simulations, max_considered_actions);

        let mut diagnostics = SearchDiagnostics {
            backend_evaluations: root_stats.backend_evaluations,
            evaluation_cache_hits: root_stats.cache_hits,
            evaluation_cache_misses: root_stats.backend_evaluations,
            ..SearchDiagnostics::default()
        };
        let mut active = self.variant.root_actions.clone();
        let mut allocated = 0usize;
        let mut batch = LeafBatch::with_capacity(self.variant.config.leaf_batch_size);

        while allocated < simulations {
            let round_visits =
                sequential_halving_round_visits(simulations - allocated, active.len());
            let targets = self.round_targets(&active, round_visits);

            while targets
                .iter()
                .any(|&(node, target)| self.nodes[node as usize].completed_visits < target)
            {
                batch.leaves.clear();
                while batch.leaves.len() < self.variant.config.leaf_batch_size {
                    let Some(root_child) = self.best_round_action(&active, &targets)
                    else {
                        break;
                    };
                    let (leaf, state, depth) =
                        self.descend(game, root_child, context, |core, node| {
                            puct_child(core, node, core.variant.config.puct)
                        });
                    self.reserve_path(leaf);
                    batch.leaves.push(PendingLeaf {
                        node: leaf,
                        game: state,
                    });
                    diagnostics.max_depth = diagnostics.max_depth.max(depth);
                }
                self.finish_leaf_batch(&mut batch, &mut diagnostics)
                    .map_err(super::SearchError::Evaluation)?;
            }

            allocated += round_visits;
            if active.len() > 1 && allocated < simulations {
                active = self.round_survivors(active);
            }
        }

        let selected = self.select_root_winner();
        let selected_move = self.nodes[selected as usize]
            .move_from_parent
            .expect("root action must have a move");
        let policy = self.root_improved_policy();
        let root_value = self.nodes[0].completed_q().unwrap_or(root_value);
        diagnostics.nodes_created = self.nodes.len();

        Ok(SearchResult {
            policy,
            selected_move,
            root_value,
            diagnostics,
        })
    }

    fn init_root_actions(&mut self, simulations: usize, max_considered_actions: usize) {
        self.variant.root_actions = self
            .child_indices(0)
            .map(|node| RootAction {
                node,
                gumbel_logit: self.nodes[node as usize].logit
                    + self.variant.config.root.gumbel_scale * sample_gumbel(&mut self.rng),
            })
            .collect();
        self.variant
            .root_actions
            .sort_unstable_by(|left, right| right.gumbel_logit.total_cmp(&left.gumbel_logit));
        self.variant.root_actions.truncate(
            max_considered_actions
                .min(simulations)
                .min(self.variant.root_actions.len()),
        );
    }

    fn best_round_action(&self, active: &[RootAction], targets: &[(u32, u32)]) -> Option<u32> {
        let transformed_q = self.child_transformed_q(0);
        active
            .iter()
            .filter(|action| {
                targets
                    .iter()
                    .find(|(node, _)| *node == action.node)
                    .is_some_and(|(_, target)| {
                        self.nodes[action.node as usize].selection_visits() < *target
                    })
            })
            .max_by(|left, right| {
                self.root_score(**left, &transformed_q)
                    .total_cmp(&self.root_score(**right, &transformed_q))
            })
            .map(|action| action.node)
    }

    fn round_survivors(&self, mut active: Vec<RootAction>) -> Vec<RootAction> {
        let transformed_q = self.child_transformed_q(0);
        active.sort_unstable_by(|left, right| {
            self.root_score(*right, &transformed_q)
                .total_cmp(&self.root_score(*left, &transformed_q))
        });
        active.truncate((active.len() / 2).max(1));
        active
    }

    fn round_targets(&self, active: &[RootAction], round_visits: usize) -> Vec<(u32, u32)> {
        let base = round_visits / active.len();
        let remainder = round_visits % active.len();
        active
            .iter()
            .enumerate()
            .map(|(index, action)| {
                let target = self.nodes[action.node as usize].completed_visits
                    + (base + usize::from(index < remainder)) as u32;
                (action.node, target)
            })
            .collect()
    }

    fn root_score(&self, action: RootAction, transformed_q: &[f32]) -> f32 {
        let index = (action.node - self.nodes[0].first_child) as usize;
        action.gumbel_logit + transformed_q[index]
    }

    fn select_root_winner(&self) -> u32 {
        let visits = self
            .variant
            .root_actions
            .iter()
            .map(|action| self.nodes[action.node as usize].completed_visits)
            .max()
            .expect("root candidates");
        self.variant
            .root_actions
            .iter()
            .filter(|action| self.nodes[action.node as usize].completed_visits == visits)
            .max_by(|left, right| {
                self.root_score(**left, &self.child_transformed_q(0))
                    .total_cmp(&self.root_score(**right, &self.child_transformed_q(0)))
            })
            .expect("root candidate at maximum visits")
            .node
    }

    fn child_transformed_q(&self, node: u32) -> Vec<f32> {
        let q_values = self
            .child_indices(node)
            .map(|child| {
                self.nodes[child as usize]
                    .completed_q()
                    .map(PositionValue::flipped)
            })
            .collect::<Vec<_>>();
        let visits = self
            .child_indices(node)
            .map(|child| self.nodes[child as usize].completed_visits)
            .collect::<Vec<_>>();
        let priors = self
            .child_indices(node)
            .map(|child| self.nodes[child as usize].prior)
            .collect::<Vec<_>>();
        transform_completed_q(
            self.nodes[node as usize].raw_value,
            &q_values,
            &visits,
            &priors,
            self.variant.config.root.completed_q,
        )
    }

    fn root_improved_policy(&self) -> Vec<(G::Move, f32)> {
        let transformed_q = self.child_transformed_q(0);
        let logits = self
            .child_indices(0)
            .enumerate()
            .map(|(index, child)| self.nodes[child as usize].logit + transformed_q[index])
            .collect::<Vec<_>>();
        self.child_indices(0)
            .zip(softmax(&logits))
            .map(|(child, probability)| {
                (
                    self.nodes[child as usize]
                        .move_from_parent
                        .expect("root child must have move"),
                    probability,
                )
            })
            .collect()
    }

    fn child_indices(&self, node: u32) -> std::ops::Range<u32> {
        let node = &self.nodes[node as usize];
        node.first_child..node.first_child + u32::from(node.num_children)
    }
}

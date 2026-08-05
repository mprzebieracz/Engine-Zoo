use super::core::{LeafBatch, MctsCore, PendingLeaf};
use super::gumbel::{prepare_transformed_q, sample_gumbel, softmax_into, GumbelScratch};
use super::puct::puct_child;
use super::{
    effective_leaf_batch_size, Node, RootAction, RootGumbelPuctConfig, SearchDiagnostics,
    SearchResult,
};
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
    active: Vec<RootAction>,
    round_candidates: Vec<RoundCandidate>,
    scratch: GumbelScratch,
}

/// A root action and the completed-visit target assigned for the current
/// sequential-halving round.
///
/// Keeping these together avoids repeatedly searching a separate target list
/// while the local leaf batch is collected.
#[derive(Clone, Copy, Debug)]
struct RoundCandidate {
    action: RootAction,
    target_visits: u32,
}

impl From<RootGumbelPuctConfig> for RootGumbelPuct {
    fn from(config: RootGumbelPuctConfig) -> Self {
        Self {
            config,
            root_actions: Vec::new(),
            active: Vec::new(),
            round_candidates: Vec::new(),
            scratch: GumbelScratch::default(),
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
        let mut active = std::mem::take(&mut self.variant.active);
        active.clear();
        active.extend_from_slice(&self.variant.root_actions);
        let mut allocated = 0usize;
        let leaf_batch_size =
            effective_leaf_batch_size(self.variant.config.leaf_batch_size, simulations);
        let mut batch = LeafBatch::with_capacity(leaf_batch_size);

        while allocated < simulations {
            let round_visits =
                sequential_halving_round_visits(simulations - allocated, active.len());
            self.prepare_round_candidates(&active, round_visits);

            while self.variant.round_candidates.iter().any(|candidate| {
                self.nodes[candidate.action.node as usize].completed_visits
                    < candidate.target_visits
            }) {
                // Completed visits and Q values cannot change until this local batch
                // is evaluated and backed up. Only selection visits change while
                // leaves are collected, so reuse the transformed root Q values.
                prepare_transformed_q(
                    &self.nodes,
                    0,
                    self.variant.config.root.completed_q,
                    &mut self.variant.scratch,
                );

                batch.leaves.clear();
                while batch.leaves.len() < leaf_batch_size {
                    let Some(root_child) = self.best_round_action(
                        &self.variant.round_candidates,
                        &self.variant.scratch.transformed_q,
                    )
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
                self.retain_round_survivors(&mut active);
            }
        }

        let selected = self.select_root_winner();
        let selected_move = self.nodes[selected as usize]
            .move_from_parent
            .expect("root action must have a move");
        let policy = self.root_improved_policy();
        let root_value = self.nodes[0].completed_q().unwrap_or(root_value);
        diagnostics.nodes_created = self.nodes.len();
        self.variant.active = active;

        Ok(SearchResult {
            policy,
            selected_move,
            root_value,
            diagnostics,
        })
    }

    fn init_root_actions(&mut self, simulations: usize, max_considered_actions: usize) {
        self.variant.root_actions.clear();
        let root = &self.nodes[0];
        let children = root.first_child..root.first_child + u32::from(root.num_children);
        for node in children {
            self.variant.root_actions.push(RootAction {
                node,
                gumbel_logit: self.nodes[node as usize].logit
                    + self.variant.config.root.gumbel_scale * sample_gumbel(&mut self.rng),
            });
        }
        self.variant
            .root_actions
            .sort_unstable_by(|left, right| right.gumbel_logit.total_cmp(&left.gumbel_logit));
        self.variant.root_actions.truncate(
            max_considered_actions
                .min(simulations)
                .min(self.variant.root_actions.len()),
        );
    }

    fn best_round_action(
        &self,
        candidates: &[RoundCandidate],
        transformed_q: &[f32],
    ) -> Option<u32> {
        candidates
            .iter()
            .filter(|candidate| {
                self.nodes[candidate.action.node as usize].selection_visits()
                    < candidate.target_visits
            })
            .max_by(|left, right| {
                self.root_score(left.action, transformed_q)
                    .total_cmp(&self.root_score(right.action, transformed_q))
            })
            .map(|candidate| candidate.action.node)
    }

    fn retain_round_survivors(&mut self, active: &mut Vec<RootAction>) {
        prepare_transformed_q(
            &self.nodes,
            0,
            self.variant.config.root.completed_q,
            &mut self.variant.scratch,
        );
        active.sort_unstable_by(|left, right| {
            self.root_score(*right, &self.variant.scratch.transformed_q)
                .total_cmp(&self.root_score(*left, &self.variant.scratch.transformed_q))
        });
        active.truncate((active.len() / 2).max(1));
    }

    fn prepare_round_candidates(&mut self, active: &[RootAction], round_visits: usize) {
        let base = round_visits / active.len();
        let remainder = round_visits % active.len();
        self.variant.round_candidates.clear();
        for (index, action) in active.iter().enumerate() {
            let target = self.nodes[action.node as usize].completed_visits
                + (base + usize::from(index < remainder)) as u32;
            self.variant.round_candidates.push(RoundCandidate {
                action: *action,
                target_visits: target,
            });
        }
    }

    fn root_score(&self, action: RootAction, transformed_q: &[f32]) -> f32 {
        let index = (action.node - self.nodes[0].first_child) as usize;
        action.gumbel_logit + transformed_q[index]
    }

    fn select_root_winner(&mut self) -> u32 {
        prepare_transformed_q(
            &self.nodes,
            0,
            self.variant.config.root.completed_q,
            &mut self.variant.scratch,
        );
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
                self.root_score(**left, &self.variant.scratch.transformed_q)
                    .total_cmp(&self.root_score(**right, &self.variant.scratch.transformed_q))
            })
            .expect("root candidate at maximum visits")
            .node
    }

    fn root_improved_policy(&mut self) -> Vec<(G::Move, f32)> {
        prepare_transformed_q(
            &self.nodes,
            0,
            self.variant.config.root.completed_q,
            &mut self.variant.scratch,
        );
        self.variant.scratch.logits.clear();
        let root = &self.nodes[0];
        let children = root.first_child..root.first_child + u32::from(root.num_children);
        for (index, child) in children.enumerate() {
            self.variant
                .scratch
                .logits
                .push(self.nodes[child as usize].logit + self.variant.scratch.transformed_q[index]);
        }
        softmax_into(
            &self.variant.scratch.logits,
            &mut self.variant.scratch.probabilities,
        );
        self.child_indices(0)
            .zip(self.variant.scratch.probabilities.iter().copied())
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

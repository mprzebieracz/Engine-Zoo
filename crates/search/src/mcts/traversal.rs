use super::core::{LeafBatch, MctsCore, PendingBackup, PendingResult};
use super::{Node, SearchDiagnostics};
use crate::{PositionValue, RuleResult, SearchRules};
use engine_core::game::GameState;

fn find_leaf_result(result_by_node: &[(u32, usize)], node: u32) -> Option<usize> {
    result_by_node
        .iter()
        .find_map(|&(candidate, index)| (candidate == node).then_some(index))
}

/// Completes one reservation along a leaf-to-root path. Values always stay in
/// the node's player-to-move perspective, hence the flip at each edge.
fn complete_path<M, Meta: Default>(
    nodes: &mut [Node<M, Meta>],
    mut node: u32,
    mut value: PositionValue,
) {
    loop {
        let current = &mut nodes[node as usize];
        let first_completed_visit = current.completed_visits == 0;
        let prior = current.prior;
        let parent = current.parent;

        current.complete_reserved_visit(value);

        if first_completed_visit {
            if let Some(parent) = parent {
                nodes[parent as usize].visited_child_prior_mass += prior;
            }
        }

        value = value.flipped();
        let Some(parent) = parent
        else {
            break;
        };
        node = parent;
    }
}

fn cancel_path<M, Meta: Default>(nodes: &mut [Node<M, Meta>], mut node: u32) {
    loop {
        let current = &mut nodes[node as usize];
        current.cancel_reserved_visit();
        let Some(parent) = current.parent
        else {
            break;
        };
        node = parent;
    }
}

impl<G, E, R, V> MctsCore<G, E, R, V>
where
    G: GameState + Clone,
    E: crate::PolicyValueEvaluator<G>,
    R: SearchRules<G>,
    V: 'static,
{
    pub(super) fn descend<'a, F>(
        &mut self,
        game: &G,
        mut node: u32,
        context: R::Context<'a>,
        mut select_child: F,
    ) -> (u32, G, usize)
    where
        G: 'a,
        F: FnMut(&Self, u32) -> Option<u32>,
    {
        let mut current = game.clone();
        let mut depth = 0;
        self.rules.reset_path(context, game, &mut self.path_state);
        let root_rule = self.rules.enter_state(
            context,
            &mut current,
            &mut self.path_state,
            &mut self.nodes[0].meta,
        );
        self.cache_terminal(0, &current, root_rule);
        if node != 0 {
            current.play(
                self.nodes[node as usize]
                    .move_from_parent
                    .expect("non-root node must store a move"),
            );
            depth = 1;
        }
        loop {
            if node != 0 {
                let rule = {
                    let current_node = &mut self.nodes[node as usize];
                    self.rules.enter_state(
                        context,
                        &mut current,
                        &mut self.path_state,
                        &mut current_node.meta,
                    )
                };
                self.cache_terminal(node, &current, rule);
            }
            if !self.nodes[node as usize].expanded || self.nodes[node as usize].terminal {
                break;
            }
            let Some(next) = select_child(self, node)
            else {
                break;
            };
            node = next;
            current.play(
                self.nodes[node as usize]
                    .move_from_parent
                    .expect("selected child must store a move"),
            );
            depth += 1;
        }
        (node, current, depth)
    }

    fn cache_terminal(&mut self, node: u32, game: &G, rule: RuleResult) {
        let reward = self.terminal_value(game, rule);
        if let Some(reward) = reward {
            let current = &mut self.nodes[node as usize];
            current.terminal = true;
            current.reward = reward;
            current.raw_value = reward;
        }
    }

    pub(super) fn terminal_value(&self, game: &G, rule: RuleResult) -> Option<PositionValue> {
        match rule {
            RuleResult::Terminal(value) => Some(value),
            RuleResult::Continue => game.terminal_value().map(PositionValue::from),
        }
    }

    pub(super) fn reserve_path(&mut self, mut node: u32) {
        loop {
            self.nodes[node as usize].reserve_visit();
            let Some(parent) = self.nodes[node as usize].parent
            else {
                break;
            };
            node = parent;
        }
    }

    pub(super) fn finish_leaf_batch(
        &mut self,
        batch: &mut LeafBatch<G>,
        diagnostics: &mut SearchDiagnostics,
    ) -> Result<(), crate::EvaluationError> {
        if batch.leaves.is_empty() {
            return Ok(());
        }
        batch.pending.clear();
        batch.unique_games.clear();
        batch.result_by_node.clear();
        for leaf in batch.leaves.drain(..) {
            if self.nodes[leaf.node as usize].terminal {
                batch.pending.push(PendingBackup {
                    node: leaf.node,
                    result: PendingResult::Terminal,
                });
            }
            else if let Some(index) = find_leaf_result(&batch.result_by_node, leaf.node) {
                diagnostics.duplicate_leaves += 1;
                batch.pending.push(PendingBackup {
                    node: leaf.node,
                    result: PendingResult::Evaluation(index),
                });
            }
            else {
                let index = batch.unique_games.len();
                batch.result_by_node.push((leaf.node, index));
                batch.unique_games.push(leaf.game);
                batch.pending.push(PendingBackup {
                    node: leaf.node,
                    result: PendingResult::Evaluation(index),
                });
            }
        }
        let (evaluations, evaluation_stats) = match self.evaluate_positions(&batch.unique_games) {
            Ok(result) => result,
            Err(error) => {
                for pending in batch.pending.drain(..) {
                    cancel_path(&mut self.nodes, pending.node);
                }
                return Err(error);
            }
        };
        diagnostics.backend_evaluations += evaluation_stats.backend_evaluations;
        diagnostics.evaluation_cache_hits += evaluation_stats.cache_hits;
        diagnostics.evaluation_cache_misses += evaluation_stats.backend_evaluations;
        diagnostics.duplicate_leaves += evaluation_stats.duplicate_requests;
        debug_assert_eq!(
            evaluations.len(),
            batch.unique_games.len(),
            "driver must return one result per unique non-terminal leaf"
        );
        for backup in batch.pending.drain(..) {
            match backup.result {
                PendingResult::Terminal => {
                    let reward = self.nodes[backup.node as usize].reward;
                    complete_path(&mut self.nodes, backup.node, reward);
                }
                PendingResult::Evaluation(index) => {
                    let evaluation = &evaluations[index];
                    if !self.nodes[backup.node as usize].expanded {
                        self.nodes[backup.node as usize].raw_value = evaluation.value;
                        self.build_policy_from(
                            evaluation.legal(),
                            evaluation.logits(),
                            false,
                            None,
                        );
                        self.expand(backup.node);
                    }
                    complete_path(&mut self.nodes, backup.node, evaluation.value);
                }
            }
            diagnostics.completed_simulations += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reserved_path(length: usize) -> Vec<Node<u8>> {
        (0..=length)
            .map(|index| {
                let mut node = Node::new(
                    index.checked_sub(1).map(|parent| parent as u32),
                    None,
                    0.0,
                    0.0,
                    false,
                    PositionValue::DRAW,
                );
                node.reserve_visit();
                node
            })
            .collect()
    }

    #[test]
    fn backup_flips_value_and_removes_every_reservation() {
        for length in 0..=8 {
            let mut nodes = reserved_path(length);
            complete_path(&mut nodes, length as u32, PositionValue::WIN);
            for (index, node) in nodes.iter().enumerate() {
                let expected = if (length - index) % 2 == 0 {
                    PositionValue::WIN
                }
                else {
                    PositionValue::LOSS
                };
                assert_eq!(node.completed_q(), Some(expected));
                assert_eq!(node.in_flight_visits, 0);
            }
        }
    }

    #[test]
    fn cancellation_removes_every_reservation() {
        let mut nodes = reserved_path(3);
        cancel_path(&mut nodes, 3);
        assert!(nodes
            .iter()
            .all(|node| node.in_flight_visits == 0 && node.completed_visits == 0));
    }

    #[test]
    fn backup_tracks_each_visited_child_prior_once() {
        let mut nodes: Vec<Node<u8>> = vec![
            Node::new(None, None, 0.0, 0.0, false, PositionValue::DRAW),
            Node::new(Some(0), None, 0.25, 0.0, false, PositionValue::DRAW),
            Node::new(Some(0), None, 0.75, 0.0, false, PositionValue::DRAW),
        ];

        for child in [1_u32, 1, 2] {
            nodes[0].reserve_visit();
            nodes[child as usize].reserve_visit();

            complete_path(&mut nodes, child, PositionValue::DRAW);
        }

        assert_eq!(nodes[0].visited_child_prior_mass, 1.0);
        assert_eq!(nodes[1].completed_visits, 2);
        assert_eq!(nodes[2].completed_visits, 1);
    }
}

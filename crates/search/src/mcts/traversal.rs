use super::core::{LeafBatch, MctsCore, PendingBackup, PendingLeaf, PendingResult};
use crate::{RuleResult, SearchRules};
use engine_core::game::{GameState, TerminalValue};

fn find_leaf_result(result_by_node: &[(u32, usize)], node: u32) -> Option<usize> {
    result_by_node
        .iter()
        .find_map(|&(candidate, idx)| (candidate == node).then_some(idx))
}

impl<G, E, R, V> MctsCore<G, E, R, V>
where
    G: GameState + Clone,
    E: crate::PolicyValueEvaluator<G>,
    R: SearchRules<G>,
    V: 'static,
{
    pub(super) fn descend<'a>(
        &mut self,
        game: &G,
        mut node: u32,
        context: R::Context<'a>,
    ) -> (u32, G)
    where
        G: 'a,
    {
        let mut current = game.clone();
        self.rules.reset_path(context, game, &mut self.path_state);
        {
            let root = &mut self.nodes[0];
            let rule = self
                .rules
                .enter_state(context, &mut current, &mut self.path_state, &mut root.meta);
            self.cache_terminal(0, &current, rule);
        }
        if node != 0 {
            current.play(
                self.nodes[node as usize]
                    .move_from_parent
                    .expect("non-root node must store a move"),
            );
        }
        loop {
            if node != 0 {
                let rule = {
                    let n = &mut self.nodes[node as usize];
                    self.rules
                        .enter_state(context, &mut current, &mut self.path_state, &mut n.meta)
                };
                self.cache_terminal(node, &current, rule);
            }
            if !self.nodes[node as usize].expanded || self.nodes[node as usize].terminal {
                break;
            }
            let n = &self.nodes[node as usize];
            let c_puct = ((1.0 + n.effective_visits() as f32 + self.cfg.c_base) / self.cfg.c_base)
                .ln()
                + self.cfg.c_init;
            let Some(best) = self.select_child(node, c_puct)
            else {
                break;
            };
            node = best;
            current.play(
                self.nodes[node as usize]
                    .move_from_parent
                    .expect("selected child must store a move"),
            );
        }
        (node, current)
    }

    fn cache_terminal(&mut self, node: u32, game: &G, rule: RuleResult) {
        let reward = match rule {
            RuleResult::Terminal(v) => Some(v.as_f32()),
            RuleResult::Continue => game.terminal_value().map(TerminalValue::as_f32),
        };
        if self.nodes[node as usize].visits == 0 {
            if let Some(reward) = reward {
                self.nodes[node as usize].terminal = true;
                self.nodes[node as usize].reward = reward;
            }
        }
    }

    pub(super) fn collect_leaf_batch(
        &mut self,
        game: &G,
        budget: usize,
        context: R::Context<'_>,
        leaves: &mut Vec<PendingLeaf<G>>,
    ) -> usize {
        let target = budget.min(self.leaf_batch_size());
        leaves.clear();
        for _ in 0..target {
            leaves.push(self.collect_leaf(game, None, context));
        }
        target
    }

    pub(super) fn collect_leaf(
        &mut self,
        game: &G,
        child: Option<u32>,
        context: R::Context<'_>,
    ) -> PendingLeaf<G> {
        let (node, current) = self.descend(game, child.unwrap_or(0), context);
        self.add_virtual_loss(node);
        PendingLeaf {
            node,
            game: current,
        }
    }

    pub(super) fn finish_leaf_batch(&mut self, batch: &mut LeafBatch<G>) {
        if batch.leaves.is_empty() {
            return;
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
                continue;
            }
            let idx = match find_leaf_result(&batch.result_by_node, leaf.node) {
                Some(idx) => idx,
                None => {
                    let idx = batch.unique_games.len();
                    batch.result_by_node.push((leaf.node, idx));
                    batch.unique_games.push(leaf.game);
                    idx
                }
            };
            batch.pending.push(PendingBackup {
                node: leaf.node,
                result: PendingResult::Evaluation(idx),
            });
        }

        let evaluations = self.evaluate_positions(&batch.unique_games);
        debug_assert_eq!(
            evaluations.len(),
            batch.unique_games.len(),
            "driver must return one result per unique non-terminal leaf"
        );

        for backup in batch.pending.drain(..) {
            match backup.result {
                PendingResult::Terminal => {
                    let reward = self.nodes[backup.node as usize].reward;
                    self.backpropagate_after_virtual_loss(backup.node, reward);
                }
                PendingResult::Evaluation(idx) => {
                    let (legal, res) = &evaluations[idx];
                    if !self.nodes[backup.node as usize].expanded {
                        self.build_policy_from(legal, res, false);
                        self.expand(backup.node);
                    }
                    self.backpropagate_after_virtual_loss(backup.node, res.value);
                }
            }
        }
    }

    fn add_virtual_loss(&mut self, mut node: u32) {
        loop {
            self.nodes[node as usize].virtual_loss_count += 1;
            let Some(parent) = self.nodes[node as usize].parent
            else {
                break;
            };
            node = parent;
        }
    }

    fn select_child(&self, node: u32, c_puct: f32) -> Option<u32> {
        let n = &self.nodes[node as usize];
        let sqrt_parent = ((n.effective_visits() + 1) as f32).sqrt();

        let mut best = None;
        let mut best_ucb = f32::NEG_INFINITY;
        for c in n.first_child..n.first_child + u32::from(n.num_children) {
            let child = &self.nodes[c as usize];
            let q = if child.effective_visits() == 0 {
                n.effective_q() - self.cfg.fpu_reduction
            }
            else {
                -child.effective_q()
            };
            let ucb =
                q + c_puct * child.prior * sqrt_parent / (1 + child.effective_visits()) as f32;
            if ucb > best_ucb {
                best_ucb = ucb;
                best = Some(c);
            }
        }
        best
    }

    fn backpropagate_after_virtual_loss(&mut self, mut node: u32, mut value: f32) {
        loop {
            let n = &mut self.nodes[node as usize];
            debug_assert!(n.virtual_loss_count > 0, "virtual loss underflow");
            n.virtual_loss_count = n.virtual_loss_count.saturating_sub(1);
            n.visits += 1;
            n.value_sum += value;
            value = -value;
            let Some(parent) = n.parent
            else {
                break;
            };

            node = parent;
        }
    }
}

use super::super::evaluator::Evaluator;
use super::core::{MctsCore, PendingBackup, PendingLeaf, PendingResult, SearchDriver};
use engine_core::game::Game;
use engine_core::rules::RepetitionGame;

fn find_leaf_result(result_by_node: &[(u32, usize)], node: u32) -> Option<usize> {
    result_by_node
        .iter()
        .find_map(|&(candidate, idx)| (candidate == node).then_some(idx))
}

impl<E: Evaluator, V> MctsCore<E, V> {
    pub(super) fn descend<G: Game>(&mut self, game: &G, mut node: u32) -> (u32, G) {
        let mut current = game.clone();

        if node != 0 {
            current.step(self.nodes[node as usize].action_from_parent);
            self.cache_terminal(node, &current);
        }

        loop {
            let n = &self.nodes[node as usize];
            if !n.expanded || n.terminal {
                break;
            }
            let c_init = self.cfg.c_init;
            let c_base = self.cfg.c_base;
            let c_puct = ((1.0 + n.effective_visits() as f32 + c_base) / c_base).ln() + c_init;
            let Some(best) = self.select_child(node, c_puct)
            else {
                break;
            };
            node = best;
            current.step(self.nodes[best as usize].action_from_parent);
            self.cache_terminal(best, &current);
        }

        (node, current)
    }

    pub(super) fn descend_repetition<G, F>(
        &mut self,
        game: &G,
        mut node: u32,
        root_repetitions: F,
    ) -> (u32, G)
    where
        G: RepetitionGame,
        F: Fn(u64) -> u8 + Copy,
    {
        let mut current = *game;
        // The callback reports occurrences before the root, so retain the
        // root itself here. Otherwise a simulated return to the root omits
        // one occurrence and fails to recognize a threefold repetition.
        let mut history_stack = Vec::with_capacity(64);
        history_stack.push(current.repetition_hash());

        if node != 0 {
            current.step(self.nodes[node as usize].action_from_parent);
            history_stack.push(current.repetition_hash());
            self.cache_repetition_state(node, &mut current, root_repetitions, &history_stack);
        }

        loop {
            let n = &self.nodes[node as usize];
            if !n.expanded || n.terminal {
                break;
            }
            let c_init = self.cfg.c_init;
            let c_base = self.cfg.c_base;
            let c_puct = ((1.0 + n.effective_visits() as f32 + c_base) / c_base).ln() + c_init;
            let Some(best) = self.select_child(node, c_puct)
            else {
                break;
            };
            node = best;
            current.step(self.nodes[best as usize].action_from_parent);
            history_stack.push(current.repetition_hash());
            self.cache_repetition_state(best, &mut current, root_repetitions, &history_stack);
        }

        (node, current)
    }

    fn cache_repetition_state<G, F>(
        &mut self,
        node: u32,
        game: &mut G,
        root_repetitions: F,
        history_stack: &[u64],
    ) where
        G: RepetitionGame,
        F: Fn(u64) -> u8 + Copy,
    {
        if self.nodes[node as usize].visits != 0 {
            return;
        }
        let hash = game.repetition_hash();
        self.nodes[node as usize].hash = hash;
        let repetitions_before = self.repetitions_before_current(
            hash,
            game.halfmove_clock(),
            root_repetitions,
            history_stack,
        );
        game.set_repetitions_before_current(repetitions_before);
        if !game.is_terminal() && repetitions_before >= 2 {
            game.set_repetition_draw();
        }
        self.cache_terminal(node, game);
    }

    fn repetitions_before_current<F>(
        &self,
        hash: u64,
        halfmove_clock: usize,
        root_repetitions: F,
        history_stack: &[u64],
    ) -> u8
    where
        F: Fn(u64) -> u8 + Copy,
    {
        let mut count = root_repetitions(hash);

        for seen in history_stack.iter().rev().skip(1).take(halfmove_clock) {
            if *seen == hash {
                count = count.saturating_add(1);
            }
        }
        count
    }

    fn cache_terminal<G: Game>(&mut self, node: u32, game: &G) {
        if self.nodes[node as usize].visits == 0 && game.is_terminal() {
            self.nodes[node as usize].terminal = true;
            self.nodes[node as usize].reward = game.reward();
        }
    }

    pub(super) fn collect_leaf_batch<G, D>(
        &mut self,
        game: &G,
        budget: usize,
        driver: D,
        leaves: &mut Vec<PendingLeaf<G>>,
    ) -> usize
    where
        G: Game,
        D: SearchDriver<G>,
    {
        let target = budget.min(self.leaf_batch_size());
        leaves.clear();
        for _ in 0..target {
            leaves.push(self.collect_leaf(game, None, driver));
        }
        target
    }

    pub(super) fn collect_leaf<G, D>(
        &mut self,
        game: &G,
        child: Option<u32>,
        driver: D,
    ) -> PendingLeaf<G>
    where
        G: Game,
        D: SearchDriver<G>,
    {
        let (node, current) = driver.descend(self, game, child);
        self.add_virtual_loss(node);
        PendingLeaf {
            node,
            game: current,
        }
    }

    pub(super) fn finish_leaf_batch<G, D>(&mut self, leaves: &mut Vec<PendingLeaf<G>>, driver: D)
    where
        G: Game,
        D: SearchDriver<G>,
    {
        if leaves.is_empty() {
            return;
        }

        let mut pending = Vec::<PendingBackup>::with_capacity(leaves.len());
        let mut unique_games = Vec::<G>::with_capacity(leaves.len());
        let mut result_by_node = Vec::<(u32, usize)>::with_capacity(leaves.len());

        for leaf in leaves.drain(..) {
            if self.nodes[leaf.node as usize].terminal {
                pending.push(PendingBackup {
                    node: leaf.node,
                    result: PendingResult::Terminal,
                });
                continue;
            }
            let idx = match find_leaf_result(&result_by_node, leaf.node) {
                Some(idx) => idx,
                None => {
                    let idx = unique_games.len();
                    result_by_node.push((leaf.node, idx));
                    unique_games.push(leaf.game);
                    idx
                }
            };
            pending.push(PendingBackup {
                node: leaf.node,
                result: PendingResult::Evaluation(idx),
            });
        }

        let evaluations = driver.evaluate_many(self, &unique_games);
        debug_assert_eq!(
            evaluations.len(),
            unique_games.len(),
            "driver must return one result per unique non-terminal leaf"
        );

        for backup in pending {
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

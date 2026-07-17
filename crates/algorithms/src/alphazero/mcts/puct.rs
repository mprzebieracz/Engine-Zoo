use super::super::evaluator::EncodedEvaluator;
use super::core::{LeafBatch, MctsCore, SearchDriver};
use super::{Node, SearchResult};
use engine_core::agent::PolicyMode;
use engine_core::game::{Action, Game};

#[derive(Clone, Copy, Debug)]
pub(super) struct Puct;

impl<E: EncodedEvaluator> MctsCore<E, Puct, Action, Vec<u64>> {
    pub(super) fn search_inner<G, D>(
        &mut self,
        game: &G,
        mode: PolicyMode,
        driver: D,
    ) -> SearchResult<Action>
    where
        G: Game,
        D: SearchDriver<G>,
    {
        if game.is_terminal() {
            return SearchResult {
                policy: Vec::new(),
                selected_move: 0,
                value: game.reward(),
            };
        }
        self.clear_tree_common();
        self.nodes.push(Node::new(
            None,
            None,
            driver.root_hash(game),
            0.0,
            0.0,
            game.is_terminal(),
            game.reward(),
        ));

        let (root_legal, root_eval) = driver.evaluate(self, game);
        let root_value = root_eval.value;
        self.build_policy_from(&root_legal, &root_eval, mode == PolicyMode::Explore);
        self.expand(0);

        let mut batch = LeafBatch::with_capacity(self.leaf_batch_size());
        let mut simulations_done = 0usize;
        while simulations_done < self.cfg.simulations {
            simulations_done += self.collect_leaf_batch(
                game,
                self.cfg.simulations - simulations_done,
                driver,
                &mut batch.leaves,
            );
            self.finish_leaf_batch(&mut batch, driver);
        }
        let policy = self.root_visit_policy();
        let selected_move = policy
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(0, |&(action, _)| action);

        SearchResult {
            policy,
            selected_move,
            value: if self.nodes[0].visits > 0 {
                self.nodes[0].q()
            }
            else {
                root_value
            },
        }
    }

    fn root_visit_policy(&self) -> Vec<(Action, f32)> {
        let root = &self.nodes[0];
        let mut policy = Vec::with_capacity(usize::from(root.num_children));

        for c in root.first_child..root.first_child + root.num_children as u32 {
            let child = &self.nodes[c as usize];
            policy.push((
                child
                    .move_from_parent
                    .expect("root child must store a move"),
                child.visits as f32,
            ));
        }

        let sum: f32 = policy.iter().map(|&(_, probability)| probability).sum();
        if sum > 0.0 {
            for (_, probability) in &mut policy {
                *probability /= sum;
            }
        }
        else if !policy.is_empty() {
            let uniform = 1.0 / policy.len() as f32;
            for (_, probability) in &mut policy {
                *probability = uniform;
            }
        }
        policy
    }
}

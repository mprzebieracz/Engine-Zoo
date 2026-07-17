use super::core::{LeafBatch, MctsCore};
use super::{Node, SearchResult};
use engine_core::agent::PolicyMode;
use engine_core::game::GameState;

#[derive(Clone, Copy, Debug)]
pub(super) struct Puct;

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
    ) -> SearchResult<G::Move> {
        assert!(!game.is_terminal(), "terminal roots must be rejected");
        self.clear_tree_common();
        self.nodes.push(Node::new(None, None, 0.0, 0.0, false, 0.0));

        let mut root = game.clone();
        let _ = self.rules.enter_state(
            context,
            &mut root,
            &mut self.path_state,
            &mut self.nodes[0].meta,
        );
        let (root_legal, root_eval) = self.evaluate_position(&root);
        let root_value = root_eval.value;
        self.build_policy_from(&root_legal, &root_eval, mode == PolicyMode::Explore);
        self.expand(0);

        let mut batch = LeafBatch::with_capacity(self.leaf_batch_size());
        let mut simulations_done = 0usize;
        while simulations_done < self.cfg.simulations {
            simulations_done += self.collect_leaf_batch(
                game,
                self.cfg.simulations - simulations_done,
                context,
                &mut batch.leaves,
            );
            self.finish_leaf_batch(&mut batch);
        }
        let policy = self.root_visit_policy();
        let selected_move = policy
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("non-terminal root must contain a legal move")
            .0;

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

    fn root_visit_policy(&self) -> Vec<(G::Move, f32)> {
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

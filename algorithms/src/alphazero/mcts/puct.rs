use super::super::evaluator::Evaluator;
use super::core::{MctsCore, SearchDriver};
use super::{argmax, Node, SearchResult};
use engine_core::agent::PolicyMode;
use engine_core::game::{Action, Game};

#[derive(Clone, Copy, Debug)]
pub(super) struct Puct;

impl<E: Evaluator> MctsCore<E, Puct> {
    pub(super) fn search_inner<G, D>(
        &mut self,
        game: &G,
        mode: PolicyMode,
        driver: D,
    ) -> SearchResult
    where
        G: Game,
        D: SearchDriver<G>,
    {
        if game.is_terminal() {
            return SearchResult {
                policy: vec![0.0; G::ACTION_SIZE],
                selected_action: 0,
                value: game.reward(),
            };
        }
        self.clear_tree_common();
        self.nodes.push(Node::new(
            None,
            0,
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

        for _ in 0..self.cfg.simulations {
            driver.simulate(self, game);
        }
        let policy = self.root_visit_policy::<G>();
        let selected_action = argmax(&policy) as Action;

        SearchResult {
            policy,
            selected_action,
            value: if self.nodes[0].visits > 0 {
                self.nodes[0].q()
            } else {
                root_value
            },
        }
    }

    fn root_visit_policy<G: Game>(&self) -> Vec<f32> {
        let root = &self.nodes[0];
        let mut policy = vec![0.0f32; G::ACTION_SIZE];

        for c in root.first_child..root.first_child + root.num_children as u32 {
            let child = &self.nodes[c as usize];
            policy[child.action_from_parent as usize] = child.visits as f32;
        }

        let sum: f32 = policy.iter().sum();
        if sum > 0.0 {
            for x in &mut policy {
                *x /= sum;
            }
        }
        policy
    }
}

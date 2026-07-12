use super::super::evaluator::Evaluator;
use super::core::{MctsCore, SearchDriver};
use super::{Node, RootAction, RootQTransform, SearchResult};
use engine_core::agent::PolicyMode;
use engine_core::game::Game;
use rand::prelude::*;

#[derive(Debug)]
pub(super) struct Gumbel {
    pub(super) sampled_actions: usize,
    pub(super) root_actions: Vec<RootAction>,
}

impl Gumbel {
    /// Parameters of the completed-Q transformation. The transformed value is
    /// `(C_VISIT + max_child_visits) * C_SCALE * normalized_q`.
    const C_VISIT: f32 = 50.0;
    const C_SCALE: f32 = 1.0;
    const Q_EPSILON: f32 = 1e-8;
    const EXPLORATION_SCALE: f32 = 1.0;
}

impl<E: Evaluator> MctsCore<E, Gumbel> {
    fn clear_tree(&mut self) {
        self.nodes.clear();
        self.variant.root_actions.clear();
        self.policy_buf.clear();
        self.batch.clear();
    }

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
        self.clear_tree();
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
        self.build_policy_from(&root_legal, &root_eval, false);
        self.expand(0);

        let gumbel_scale = if mode == PolicyMode::Explore {
            Gumbel::EXPLORATION_SCALE
        }
        else {
            0.0
        };
        let winner = self.run_gumbel(game, root_value, gumbel_scale, driver);
        let selected_action = self.nodes[winner as usize].action_from_parent;
        let policy = self.root_gumbel_policy::<G>(root_value);

        SearchResult {
            policy,
            selected_action,
            value: if self.nodes[0].visits > 0 {
                self.nodes[0].q()
            }
            else {
                root_value
            },
        }
    }

    fn root_gumbel_policy<G: Game>(&self, root_value: f32) -> Vec<f32> {
        let root = &self.nodes[0];
        let transform = self.root_q_transform(root_value);
        let mut policy = vec![0.0f32; G::ACTION_SIZE];
        let mut max_search_logit = f32::NEG_INFINITY;

        for c in root.first_child..root.first_child + u32::from(root.num_children) {
            let child = &self.nodes[c as usize];
            let q = self.completed_root_child_q(c, transform);
            max_search_logit = max_search_logit.max(child.logit + transform.apply(q));
        }

        let mut sum = 0.0f32;
        for c in root.first_child..root.first_child + u32::from(root.num_children) {
            let child = &self.nodes[c as usize];
            let q = self.completed_root_child_q(c, transform);
            let p = (child.logit + transform.apply(q) - max_search_logit).exp();
            policy[child.action_from_parent as usize] = p;
            sum += p;
        }

        if sum.is_finite() && sum > 0.0 {
            for p in &mut policy {
                *p /= sum;
            }
        }
        else {
            let uniform = 1.0 / root.num_children as f32;
            for c in root.first_child..root.first_child + u32::from(root.num_children) {
                let action = self.nodes[c as usize].action_from_parent as usize;
                policy[action] = uniform;
            }
        }
        policy
    }

    fn run_gumbel<G, D>(&mut self, game: &G, root_value: f32, gumbel_scale: f32, driver: D) -> u32
    where
        G: Game,
        D: SearchDriver<G>,
    {
        self.init_gumbel_root_actions(gumbel_scale);
        debug_assert!(!self.variant.root_actions.is_empty());

        let schedule = gumbel_visit_schedule(self.variant.root_actions.len(), self.cfg.simulations);
        let mut leaves = Vec::with_capacity(self.leaf_batch_size());
        for considered_visit in schedule {
            let transform = self.root_q_transform(root_value);
            let child = self
                .best_gumbel_action_with_visits(considered_visit, transform)
                .expect("Gumbel visit schedule must always have a considered action");
            leaves.push(self.collect_leaf(game, Some(child), driver));
            if leaves.len() >= self.leaf_batch_size() {
                self.finish_leaf_batch(&mut leaves, driver);
            }
        }
        if !leaves.is_empty() {
            self.finish_leaf_batch(&mut leaves, driver);
        }

        self.select_gumbel_winner(root_value)
    }

    fn init_gumbel_root_actions(&mut self, gumbel_scale: f32) {
        self.variant.root_actions.clear();
        let root = &self.nodes[0];
        for c in root.first_child..root.first_child + u32::from(root.num_children) {
            let child = &self.nodes[c as usize];
            self.variant.root_actions.push(RootAction {
                node: c,
                gumbel_logit: child.logit + gumbel_scale * sample_gumbel(&mut self.rng),
            });
        }

        self.variant
            .root_actions
            .sort_unstable_by(|a, b| b.gumbel_logit.total_cmp(&a.gumbel_logit));
        let limit = self
            .variant
            .sampled_actions
            .max(1)
            .min(self.cfg.simulations)
            .min(self.variant.root_actions.len());
        self.variant.root_actions.truncate(limit);
    }

    fn best_gumbel_action_with_visits(
        &self,
        considered_visit: u32,
        transform: RootQTransform,
    ) -> Option<u32> {
        let mut best = None;
        let mut best_score = f32::NEG_INFINITY;
        for &action in &self.variant.root_actions {
            if self.nodes[action.node as usize].effective_visits() != considered_visit {
                continue;
            }
            let score = self.root_action_score(action, transform);
            if score > best_score {
                best_score = score;
                best = Some(action.node);
            }
        }
        best
    }

    fn select_gumbel_winner(&self, root_value: f32) -> u32 {
        let considered_visit = self
            .variant
            .root_actions
            .iter()
            .map(|action| self.nodes[action.node as usize].visits)
            .max()
            .expect("non-terminal root must contain a legal action");
        let transform = self.root_q_transform(root_value);
        self.best_gumbel_action_with_visits(considered_visit, transform)
            .expect("at least one Gumbel action must have the maximum visit count")
    }

    fn root_action_score(&self, action: RootAction, transform: RootQTransform) -> f32 {
        let q = self.completed_root_child_q(action.node, transform);
        action.gumbel_logit + transform.apply(q)
    }

    fn completed_root_child_q(&self, child: u32, transform: RootQTransform) -> f32 {
        if self.nodes[child as usize].effective_visits() == 0 {
            transform.completed_value
        }
        else {
            -self.nodes[child as usize].effective_q()
        }
    }

    fn root_q_transform(&self, root_value: f32) -> RootQTransform {
        let root = &self.nodes[0];
        let mut total_visits = 0u32;
        let mut visited_prior_sum = 0.0f32;
        let mut prior_weighted_q = 0.0f32;
        let mut max_visits = 0u32;

        for c in root.first_child..root.first_child + u32::from(root.num_children) {
            let child = &self.nodes[c as usize];
            max_visits = max_visits.max(child.effective_visits());
            if child.effective_visits() > 0 {
                let prior = child.prior.max(f32::MIN_POSITIVE);
                total_visits += child.effective_visits();
                visited_prior_sum += prior;
                prior_weighted_q += prior * -child.effective_q();
            }
        }

        let weighted_q = if visited_prior_sum > 0.0 {
            prior_weighted_q / visited_prior_sum
        }
        else {
            root_value
        };
        let completed_value =
            (root_value + total_visits as f32 * weighted_q) / (total_visits as f32 + 1.0);

        let mut min_value = completed_value;
        let mut max_value = completed_value;
        for c in root.first_child..root.first_child + u32::from(root.num_children) {
            let child = &self.nodes[c as usize];
            let q = if child.effective_visits() == 0 {
                completed_value
            }
            else {
                -child.effective_q()
            };
            min_value = min_value.min(q);
            max_value = max_value.max(q);
        }

        let range = max_value - min_value;
        RootQTransform {
            completed_value,
            min_value,
            inv_range: if range > Gumbel::Q_EPSILON {
                range.recip()
            }
            else {
                0.0
            },
            scale: (Gumbel::C_VISIT + max_visits as f32) * Gumbel::C_SCALE,
        }
    }
}

fn ceil_log2(n: usize) -> usize {
    usize::BITS as usize - (n.saturating_sub(1)).leading_zeros() as usize
}

/// Visit-count schedule used by Sequential Halving. This is equivalent to the
/// schedule in DeepMind's Mctx implementation and handles budgets that do not
/// divide evenly between phases without wasting simulations.
fn gumbel_visit_schedule(num_considered: usize, simulations: usize) -> Vec<u32> {
    debug_assert!(num_considered > 0);
    let mut sequence = Vec::with_capacity(simulations);
    if num_considered == 1 {
        sequence.extend((0..simulations).map(|visit| visit as u32));
        return sequence;
    }

    let phases = ceil_log2(num_considered).max(1);
    let mut visits = vec![0u32; num_considered];
    let mut active = num_considered;

    while sequence.len() < simulations {
        let extra_visits = (simulations / (phases * active)).max(1);
        for _ in 0..extra_visits {
            sequence.extend_from_slice(&visits[..active]);
            for visit in &mut visits[..active] {
                *visit += 1;
            }
            if sequence.len() >= simulations {
                break;
            }
        }
        active = (active / 2).max(2);
    }

    sequence.truncate(simulations);
    sequence
}

fn sample_gumbel<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u = rng.random_range(f32::MIN_POSITIVE..1.0);
    -(-u.ln()).ln()
}

use super::evaluator::{EvalBatch, Evaluation, Evaluator};
use crate::agent::{Agent, PolicyMode};
use crate::game::{Action, Game};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::Gamma;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MctsVariant {
    /// Standard AlphaZero PUCT search.
    Puct,
    /// Gumbel AlphaZero-style root sampling with sequential halving.
    Gumbel {
        /// Root actions considered before sequential halving.
        sampled_actions: usize,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub c_init: f32,
    pub c_base: f32,
    pub variant: MctsVariant,
    /// Simulations per search.
    pub simulations: usize,
    /// Leaves evaluated per network call within one search.
    pub batch_size: usize,
    /// Dirichlet noise weight at the root (0 disables, e.g. for match play).
    pub eps: f32,
    pub alpha: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            c_init: 1.25,
            c_base: 19652.0,
            variant: MctsVariant::Puct,
            simulations: 800,
            batch_size: 32,
            eps: 0.25,
            alpha: 0.3,
        }
    }
}

/// Outcome of one search: the visit-count distribution over the full action
/// space and the network's value estimate of the root.
pub struct SearchResult {
    pub policy: Vec<f32>,
    pub value: f32,
}

impl SearchResult {
    /// The most-visited action.
    pub fn best_action(&self) -> Action {
        argmax(&self.policy) as Action
    }

    /// Samples an action proportionally to visit counts.
    pub fn sample_action<R: Rng + ?Sized>(&self, rng: &mut R) -> Action {
        WeightedIndex::new(&self.policy)
            .expect("search of a non-terminal position visits at least one action")
            .sample(rng) as Action
    }
}

pub(crate) fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

/// Virtual loss for concurrent simulations to spread out.
const VIRTUAL_LOSS: f32 = 1.0;

/// Scale for Gumbel-style root action selection.
const GUMBEL_Q_SCALE: f32 = 2.0;

#[derive(Clone, Copy)]
struct RootAction {
    node: u32,
    gumbel_log_prior: f32,
}

/// Tree node. Children are stored sparsely (one node per legal action with a
/// nonzero prior) and contiguously, so a node only needs the range
/// `first_child .. first_child + num_children`. All nodes live in one `Vec`
/// arena, indexed by `u32` and cleared (not freed) between searches.
struct Node {
    parent: Option<u32>,
    first_child: u32,
    action_from_parent: Action,
    // MCTS stats
    visits: u32,
    vloss: u32,
    value_sum: f32,
    prior: f32,
    reward: f32,
    num_children: u16,
    expanded: bool,
    terminal: bool,
}

impl Node {
    fn new(
        parent: Option<u32>,
        action_from_parent: Action,
        prior: f32,
        terminal: bool,
        reward: f32,
    ) -> Self {
        Node {
            parent,
            first_child: 0,
            action_from_parent,
            visits: 0,
            vloss: 0,
            value_sum: 0.0,
            prior,
            reward,
            num_children: 0,
            expanded: false,
            terminal,
        }
    }

    /// Mean value from this node's own (player-to-move) perspective; the
    /// parent selects on -Q. Virtual loss is added so an in-flight node
    /// looks worse to its parent and concurrent simulations spread out.
    fn q(&self) -> f32 {
        let visits = self.visits + self.vloss;
        if visits == 0 {
            return 0.0;
        }
        (self.value_sum + self.vloss as f32 * VIRTUAL_LOSS) / visits as f32
    }
}

/// PUCT search with virtual loss: simulations descend in groups of
/// `batch_size`, and their leaves are evaluated in a single network call.
pub struct Mcts<E: Evaluator> {
    evaluator: E,
    cfg: MctsConfig,
    nodes: Vec<Node>,
    // Reused per-search scratch:
    batch: EvalBatch,
    /// (leaf node id, index into the batch's results) in simulation order.
    leaves: Vec<(u32, usize)>,
    /// Deduplicates leaves reached by several simulations in one batch.
    pending: HashMap<u32, usize>,
    policy_buf: Vec<(Action, f32)>,
    root_actions: Vec<RootAction>,
    rng: SmallRng,
}

impl<E: Evaluator> Mcts<E> {
    pub fn new(evaluator: E, cfg: MctsConfig) -> Self {
        debug_assert!(cfg.simulations > 0, "MCTS simulations must be positive");
        debug_assert!(cfg.batch_size > 0, "MCTS batch_size must be positive");
        debug_assert!(cfg.c_base > 0.0, "MCTS c_base must be positive");
        debug_assert!((0.0..=1.0).contains(&cfg.eps), "MCTS eps must be in [0, 1]");
        debug_assert!(
            cfg.eps == 0.0 || cfg.alpha > 0.0,
            "MCTS alpha must be positive when root noise is enabled"
        );

        Mcts {
            evaluator,
            cfg,
            nodes: Vec::new(),
            batch: EvalBatch::new(),
            leaves: Vec::new(),
            pending: HashMap::new(),
            policy_buf: Vec::new(),
            root_actions: Vec::new(),
            rng: SmallRng::from_os_rng(),
        }
    }

    pub fn config(&self) -> &MctsConfig {
        &self.cfg
    }

    pub fn search<G: Game>(&mut self, game: &G) -> SearchResult {
        if game.is_terminal() {
            return SearchResult {
                policy: vec![0.0; G::ACTION_SIZE],
                value: game.reward(),
            };
        }

        self.nodes.clear();
        self.nodes
            .push(Node::new(None, 0, 0.0, game.is_terminal(), game.reward()));

        // Root evaluation (with exploration noise) and expansion.
        self.batch.clear();
        self.enqueue_state(game);

        let root_eval = self.evaluator.evaluate(&self.batch);
        debug_assert_eq!(
            root_eval.len(),
            self.batch.len(),
            "evaluator must return one result per input state"
        );
        let root_value = root_eval[0].value;

        self.build_policy(0, &root_eval[0], true);
        self.expand(0);

        match self.cfg.variant {
            MctsVariant::Puct => self.run_puct(game),
            MctsVariant::Gumbel { sampled_actions } => self.run_gumbel(game, sampled_actions),
        }

        SearchResult {
            policy: self.root_policy::<G>(),
            value: root_value,
        }
    }

    fn root_policy<G: Game>(&self) -> Vec<f32> {
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

    fn run_puct<G: Game>(&mut self, game: &G) {
        let batch_size = self.cfg.batch_size.max(1);
        let mut simulations_done = 0;
        while simulations_done < self.cfg.simulations {
            self.batch.clear();
            self.leaves.clear();
            self.pending.clear();

            let mut in_flight = 0;
            while in_flight < batch_size && simulations_done < self.cfg.simulations {
                self.simulate(game);
                in_flight += 1;
                simulations_done += 1;
            }

            if !self.batch.is_empty() {
                let results = self.evaluator.evaluate(&self.batch);
                debug_assert_eq!(
                    results.len(),
                    self.batch.len(),
                    "evaluator must return one result per input state"
                );
                self.apply_results(&results);
            }
        }
    }

    fn run_gumbel<G: Game>(&mut self, game: &G, sampled_actions: usize) {
        self.init_gumbel_root_actions(sampled_actions);
        if self.root_actions.is_empty() {
            return;
        }

        let mut active = self.root_actions.clone();
        let mut remaining = self.cfg.simulations;

        while remaining > 0 {
            let rounds_left = ceil_log2(active.len()).max(1);
            let visits_per_action = if active.len() == 1 {
                remaining
            }
            else {
                (remaining / (rounds_left * active.len())).max(1)
            };
            self.run_gumbel_round(game, &active, visits_per_action, &mut remaining);

            if active.len() == 1 {
                continue;
            }
            active.sort_unstable_by(|a, b| {
                self.root_action_score(*b)
                    .total_cmp(&self.root_action_score(*a))
            });
            active.truncate(active.len().div_ceil(2));
        }
    }

    fn init_gumbel_root_actions(&mut self, sampled_actions: usize) {
        self.root_actions.clear();
        let root = &self.nodes[0];
        for c in root.first_child..root.first_child + u32::from(root.num_children) {
            let child = &self.nodes[c as usize];
            let prior = child.prior.max(f32::MIN_POSITIVE);
            self.root_actions.push(RootAction {
                node: c,
                gumbel_log_prior: prior.ln() + sample_gumbel(&mut self.rng),
            });
        }
        self.root_actions
            .sort_unstable_by(|a, b| b.gumbel_log_prior.total_cmp(&a.gumbel_log_prior));
        let limit = sampled_actions.max(1);
        if self.root_actions.len() > limit {
            self.root_actions.truncate(limit);
        }
    }

    fn run_gumbel_round<G: Game>(
        &mut self,
        game: &G,
        active: &[RootAction],
        visits_per_action: usize,
        remaining: &mut usize,
    ) {
        self.batch.clear();
        self.leaves.clear();
        self.pending.clear();

        let batch_size = self.cfg.batch_size.max(1);
        let mut in_flight = 0usize;
        for _ in 0..visits_per_action {
            for action in active {
                if *remaining == 0 {
                    break;
                }
                self.simulate_from_root_child(game, action.node);
                *remaining -= 1;
                in_flight += 1;

                if in_flight == batch_size {
                    self.flush_pending();
                    in_flight = 0;
                }
            }
        }
        if in_flight > 0 {
            self.flush_pending();
        }
    }

    fn flush_pending(&mut self) {
        if !self.batch.is_empty() {
            let results = self.evaluator.evaluate(&self.batch);
            debug_assert_eq!(
                results.len(),
                self.batch.len(),
                "evaluator must return one result per input state"
            );
            self.apply_results(&results);
        }
        self.batch.clear();
        self.leaves.clear();
        self.pending.clear();
    }

    fn root_action_score(&self, action: RootAction) -> f32 {
        action.gumbel_log_prior + GUMBEL_Q_SCALE * self.root_child_q(action.node)
    }

    fn root_child_q(&self, child: u32) -> f32 {
        let child = &self.nodes[child as usize];
        if child.visits == 0 {
            0.0
        }
        else {
            -child.q()
        }
    }

    /// One descent to a leaf: applies virtual loss along the path, then either
    /// backpropagates a terminal reward immediately or queues the leaf for
    /// batched evaluation.
    fn simulate<G: Game>(&mut self, game: &G) {
        let (node, current) = self.descend(game, 0);
        self.finish_simulation(node, &current);
    }

    fn simulate_from_root_child<G: Game>(&mut self, game: &G, child: u32) {
        self.nodes[0].vloss += 1;

        let (node, current) = self.descend(game, child);
        self.finish_simulation(node, &current);
    }

    fn descend<G: Game>(&mut self, game: &G, mut node: u32) -> (u32, G) {
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
            let c_puct = ((1.0 + n.visits as f32 + c_base) / c_base).ln() + c_init;
            let Some(best) = self.select_child(node, c_puct)
            else {
                break;
            };
            self.nodes[node as usize].vloss += 1;
            node = best;
            current.step(self.nodes[best as usize].action_from_parent);
            self.cache_terminal(best, &current);
        }

        self.nodes[node as usize].vloss += 1;
        (node, current)
    }

    fn cache_terminal<G: Game>(&mut self, node: u32, game: &G) {
        if self.nodes[node as usize].visits == 0 && game.is_terminal() {
            self.nodes[node as usize].terminal = true;
            self.nodes[node as usize].reward = game.reward();
        }
    }

    fn finish_simulation<G: Game>(&mut self, node: u32, game: &G) {
        if self.nodes[node as usize].terminal {
            self.backpropagate(node, self.nodes[node as usize].reward);
        }
        else {
            match self.pending.get(&node) {
                Some(&idx) => self.leaves.push((node, idx)),
                None => {
                    let idx = self.batch.len();
                    self.enqueue_state(game);
                    self.pending.insert(node, idx);
                    self.leaves.push((node, idx));
                }
            }
        }
    }

    fn apply_results(&mut self, results: &[Evaluation]) {
        // A node can appear several times when concurrent simulations collide:
        // the first occurrence expands it, the rest only backpropagate.
        for i in 0..self.leaves.len() {
            let (node, idx) = self.leaves[i];
            let res = &results[idx];
            if !self.nodes[node as usize].expanded {
                self.build_policy(idx, res, false);
                self.expand(node);
            }
            self.backpropagate(node, res.value);
        }
    }

    fn select_child(&self, node: u32, c_puct: f32) -> Option<u32> {
        let n = &self.nodes[node as usize];
        let sqrt_parent = ((n.visits + n.vloss + 1) as f32).sqrt();

        let mut best = None;
        let mut best_ucb = f32::NEG_INFINITY;
        for c in n.first_child..n.first_child + u32::from(n.num_children) {
            let child = &self.nodes[c as usize];
            let ucb = -child.q()
                + c_puct * child.prior * sqrt_parent / (1 + child.visits + child.vloss) as f32;
            if ucb > best_ucb {
                best_ucb = ucb;
                best = Some(c);
            }
        }
        best
    }

    fn backpropagate(&mut self, mut node: u32, mut value: f32) {
        loop {
            let n = &mut self.nodes[node as usize];
            n.visits += 1;
            n.value_sum += value;
            n.vloss -= 1;
            value = -value;
            let Some(parent) = n.parent
            else {
                break;
            };
            node = parent;
        }
    }

    /// Encodes `game` and its legal actions as the next entry of `self.batch`.
    fn enqueue_state<G: Game>(&mut self, game: &G) {
        let start = self.batch.states.len();
        self.batch.states.resize(start + G::state_size(), 0.0);
        game.encode_state(&mut self.batch.states[start..]);

        self.batch.legal.extend(game.legal_actions());
        self.batch.offsets.push(self.batch.legal.len() as u32);
    }

    /// Softmax over the legal-action logits of batch entry `idx` (optionally
    /// mixed with root Dirichlet noise) into `self.policy_buf`.
    fn build_policy(&mut self, idx: usize, res: &Evaluation, root_noise: bool) {
        let legal = &self.batch.legal
            [self.batch.offsets[idx] as usize..self.batch.offsets[idx + 1] as usize];
        debug_assert_eq!(
            legal.len(),
            res.logits.len(),
            "evaluator logits must match the legal actions for each state"
        );

        self.policy_buf.clear();
        let max_logit = res.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (&a, &logit) in legal.iter().zip(&res.logits) {
            let p = (logit - max_logit).exp();
            self.policy_buf.push((a, p));
            sum += p;
        }
        if sum > 0.0 {
            for (_, p) in &mut self.policy_buf {
                *p /= sum;
            }
        }

        if root_noise && self.cfg.eps > 0.0 && !self.policy_buf.is_empty() {
            let gamma = Gamma::new(self.cfg.alpha, 1.0).expect("alpha > 0");
            let mut noise: Vec<f32> = (0..self.policy_buf.len())
                .map(|_| gamma.sample(&mut self.rng))
                .collect();
            let noise_sum: f32 = noise.iter().sum();
            if noise_sum > 0.0 {
                for x in &mut noise {
                    *x /= noise_sum;
                }
            }
            for ((_, p), n) in self.policy_buf.iter_mut().zip(&noise) {
                *p = (1.0 - self.cfg.eps) * *p + self.cfg.eps * n;
            }
        }
    }

    /// Creates children of `node` from `self.policy_buf`, contiguous in the arena.
    fn expand(&mut self, node: u32) {
        let first_child = self.nodes.len() as u32;
        let mut count = 0u16;
        for i in 0..self.policy_buf.len() {
            let (action, prior) = self.policy_buf[i];
            if prior == 0.0 {
                continue;
            }
            self.nodes
                .push(Node::new(Some(node), action, prior, false, 0.0));
            count += 1;
        }
        let n = &mut self.nodes[node as usize];
        n.first_child = first_child;
        n.num_children = count;
        n.expanded = true;
    }
}

fn ceil_log2(n: usize) -> usize {
    usize::BITS as usize - (n.saturating_sub(1)).leading_zeros() as usize
}

fn sample_gumbel<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u = rng.random_range(f32::MIN_POSITIVE..1.0);
    -(-u.ln()).ln()
}

impl<G: Game, E: Evaluator> Agent<G> for Mcts<E> {
    fn act_with_mode(&mut self, game: &G, mode: PolicyMode) -> Action {
        let result = self.search(game);
        if mode == PolicyMode::Explore {
            let mut rng = rand::rng();
            result.sample_action(&mut rng)
        }
        else {
            result.best_action()
        }
    }
}

#[cfg(test)]
mod tests;

use super::evaluator::{EvalBatch, Evaluation, Evaluator};
use crate::agent::{Agent, PolicyMode};
use crate::game::{Action, Game};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::Gamma;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

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
    /// Dirichlet noise weight at the root (0 disables, e.g. for match play).
    pub eps: f32,
    pub alpha: f32,
    /// First Play Urgency reduction. Unvisited children are valued as the
    /// parent's current value minus this amount, from the parent's perspective.
    pub fpu_reduction: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            c_init: 1.25,
            c_base: 19652.0,
            variant: MctsVariant::Puct,
            simulations: 800,
            eps: 0.25,
            alpha: 0.3,
            fpu_reduction: 0.1,
        }
    }
}

/// Outcome of one search: the visit-count distribution over the full action
/// space and the network's value estimate of the root.
pub struct SearchResult {
    pub policy: Vec<f32>,
    pub value: f32,
}

pub trait RepetitionGame: Game + Copy {
    fn repetition_hash(&self) -> u64;
    fn halfmove_clock(&self) -> usize;
    fn set_repetition_draw(&mut self);
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
    hash: u64,
    first_child: u32,
    action_from_parent: Action,
    visits: u32,
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
        hash: u64,
        prior: f32,
        terminal: bool,
        reward: f32,
    ) -> Self {
        Node {
            parent,
            hash,
            first_child: 0,
            action_from_parent,
            visits: 0,
            value_sum: 0.0,
            prior,
            reward,
            num_children: 0,
            expanded: false,
            terminal,
        }
    }

    /// Mean value from this node's own player-to-move perspective.
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        }
        else {
            self.value_sum / self.visits as f32
        }
    }
}

pub(crate) struct CachedEvaluation {
    legal: Vec<Action>,
    eval: Evaluation,
}

struct EvalTableEntry {
    hash: u64,
    cached: CachedEvaluation,
}

pub(crate) struct EvalTable {
    slots: Box<[RwLock<Option<EvalTableEntry>>]>,
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
}

impl EvalTable {
    pub(crate) fn new(entries: usize) -> Self {
        let slots = (0..entries.max(1)).map(|_| RwLock::new(None)).collect();
        EvalTable {
            slots,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
        }
    }

    fn get(&self, hash: u64) -> Option<CachedEvaluation> {
        let slot = &self.slots[hash as usize % self.slots.len()];
        let guard = slot.read().unwrap();
        let hit = guard
            .as_ref()
            .filter(|entry| entry.hash == hash)
            .map(|entry| entry.cached.clone());
        if hit.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    fn insert(&self, hash: u64, cached: CachedEvaluation) {
        let slot = &self.slots[hash as usize % self.slots.len()];
        *slot.write().unwrap() = Some(EvalTableEntry { hash, cached });
        self.inserts.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn stats(&self) -> EvalTableStats {
        EvalTableStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EvalTableStats {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
}

impl Clone for CachedEvaluation {
    fn clone(&self) -> Self {
        CachedEvaluation {
            legal: self.legal.clone(),
            eval: self.eval.clone(),
        }
    }
}

/// PUCT search. Each simulation descends to one leaf and submits at most one
/// network request. GPU batching happens across many blocked search threads in
/// the shared `Batcher`, not inside one MCTS tree.
pub struct Mcts<E: Evaluator> {
    evaluator: E,
    cfg: MctsConfig,
    nodes: Vec<Node>,
    batch: EvalBatch,
    policy_buf: Vec<(Action, f32)>,
    root_actions: Vec<RootAction>,
    eval_cache: Option<Arc<EvalTable>>,
    rng: SmallRng,
}

impl<E: Evaluator> Mcts<E> {
    pub fn new(evaluator: E, cfg: MctsConfig) -> Self {
        debug_assert!(cfg.simulations > 0, "MCTS simulations must be positive");
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
            policy_buf: Vec::new(),
            root_actions: Vec::new(),
            eval_cache: None,
            rng: SmallRng::from_os_rng(),
        }
    }

    pub(crate) fn with_eval_cache(mut self, cache: Arc<EvalTable>) -> Self {
        self.eval_cache = Some(cache);
        self
    }

    pub fn config(&self) -> &MctsConfig {
        &self.cfg
    }

    pub fn set_simulations(&mut self, simulations: usize) {
        debug_assert!(simulations > 0, "MCTS simulations must be positive");
        self.cfg.simulations = simulations.max(1);
    }

    fn clear_tree(&mut self) {
        self.nodes.clear();
        self.root_actions.clear();
        self.policy_buf.clear();
        self.batch.clear();
    }

    pub fn search<G: Game>(&mut self, game: &G) -> SearchResult {
        if game.is_terminal() {
            return SearchResult {
                policy: vec![0.0; G::ACTION_SIZE],
                value: game.reward(),
            };
        }

        self.clear_tree();
        self.nodes.push(Node::new(
            None,
            0,
            0,
            0.0,
            game.is_terminal(),
            game.reward(),
        ));

        let (root_legal, root_eval) = self.evaluate_position(game);
        let root_value = root_eval.value;
        self.build_policy_from(&root_legal, &root_eval, true);
        self.expand(0);

        match self.cfg.variant {
            MctsVariant::Puct => self.run_puct(game),
            MctsVariant::Gumbel { sampled_actions } => self.run_gumbel(game, sampled_actions),
        }

        SearchResult {
            policy: self.root_policy::<G>(),
            value: if self.nodes[0].visits > 0 {
                self.nodes[0].q()
            }
            else {
                root_value
            },
        }
    }

    pub fn search_with_repetitions<G, F>(&mut self, game: &G, root_repetitions: F) -> SearchResult
    where
        G: RepetitionGame,
        F: Fn(u64) -> u8 + Copy,
    {
        if game.is_terminal() {
            return SearchResult {
                policy: vec![0.0; G::ACTION_SIZE],
                value: game.reward(),
            };
        }

        self.clear_tree();
        self.nodes.push(Node::new(
            None,
            0,
            game.repetition_hash(),
            0.0,
            game.is_terminal(),
            game.reward(),
        ));

        let (root_legal, root_eval) = self.evaluate_repetition_position(game);
        let root_value = root_eval.value;
        self.build_policy_from(&root_legal, &root_eval, true);
        self.expand(0);

        match self.cfg.variant {
            MctsVariant::Puct => self.run_puct_repetition(game, root_repetitions),
            MctsVariant::Gumbel { sampled_actions } => {
                self.run_gumbel_repetition(game, sampled_actions, root_repetitions)
            }
        }

        SearchResult {
            policy: self.root_policy::<G>(),
            value: if self.nodes[0].visits > 0 {
                self.nodes[0].q()
            }
            else {
                root_value
            },
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
        for _ in 0..self.cfg.simulations {
            self.simulate(game);
        }
    }

    fn run_puct_repetition<G, F>(&mut self, game: &G, root_repetitions: F)
    where
        G: RepetitionGame,
        F: Fn(u64) -> u8 + Copy,
    {
        for _ in 0..self.cfg.simulations {
            self.simulate_repetition(game, root_repetitions);
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

    fn run_gumbel_repetition<G, F>(&mut self, game: &G, sampled_actions: usize, root_repetitions: F)
    where
        G: RepetitionGame,
        F: Fn(u64) -> u8 + Copy,
    {
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
            self.run_gumbel_round_repetition(
                game,
                &active,
                visits_per_action,
                &mut remaining,
                root_repetitions,
            );

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
        for _ in 0..visits_per_action {
            for action in active {
                if *remaining == 0 {
                    break;
                }
                self.simulate_from_root_child(game, action.node);
                *remaining -= 1;
            }
        }
    }

    fn run_gumbel_round_repetition<G, F>(
        &mut self,
        game: &G,
        active: &[RootAction],
        visits_per_action: usize,
        remaining: &mut usize,
        root_repetitions: F,
    ) where
        G: RepetitionGame,
        F: Fn(u64) -> u8 + Copy,
    {
        for _ in 0..visits_per_action {
            for action in active {
                if *remaining == 0 {
                    break;
                }
                self.simulate_from_root_child_repetition(game, action.node, root_repetitions);
                *remaining -= 1;
            }
        }
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

    fn simulate<G: Game>(&mut self, game: &G) {
        let (node, current) = self.descend(game, 0);
        self.finish_simulation(node, &current);
    }

    fn simulate_repetition<G, F>(&mut self, game: &G, root_repetitions: F)
    where
        G: RepetitionGame,
        F: Fn(u64) -> u8 + Copy,
    {
        let (node, current) = self.descend_repetition(game, 0, root_repetitions);
        self.finish_repetition_simulation(node, &current);
    }

    fn simulate_from_root_child<G: Game>(&mut self, game: &G, child: u32) {
        let (node, current) = self.descend(game, child);
        self.finish_simulation(node, &current);
    }

    fn simulate_from_root_child_repetition<G, F>(
        &mut self,
        game: &G,
        child: u32,
        root_repetitions: F,
    ) where
        G: RepetitionGame,
        F: Fn(u64) -> u8 + Copy,
    {
        let (node, current) = self.descend_repetition(game, child, root_repetitions);
        self.finish_repetition_simulation(node, &current);
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
            node = best;
            current.step(self.nodes[best as usize].action_from_parent);
            self.cache_terminal(best, &current);
        }

        (node, current)
    }

    fn descend_repetition<G, F>(&mut self, game: &G, mut node: u32, root_repetitions: F) -> (u32, G)
    where
        G: RepetitionGame,
        F: Fn(u64) -> u8 + Copy,
    {
        let mut current = *game;
        let mut history_stack = Vec::with_capacity(64);

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
            let c_puct = ((1.0 + n.visits as f32 + c_base) / c_base).ln() + c_init;
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
        if !game.is_terminal()
            && self.is_repetition(hash, game.halfmove_clock(), root_repetitions, history_stack)
        {
            game.set_repetition_draw();
        }
        self.cache_terminal(node, game);
    }

    fn is_repetition<F>(
        &self,
        hash: u64,
        halfmove_clock: usize,
        root_repetitions: F,
        history_stack: &[u64],
    ) -> bool
    where
        F: Fn(u64) -> u8 + Copy,
    {
        let mut count = 1u8;
        count = count.saturating_add(root_repetitions(hash));
        if count >= 3 {
            return true;
        }

        for seen in history_stack.iter().rev().skip(1).take(halfmove_clock) {
            if *seen == hash {
                count += 1;
                if count >= 3 {
                    return true;
                }
            }
        }
        false
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
            return;
        }

        let (legal, res) = self.evaluate_position(game);
        if !self.nodes[node as usize].expanded {
            self.build_policy_from(&legal, &res, false);
            self.expand(node);
        }
        self.backpropagate(node, res.value);
    }

    fn finish_repetition_simulation<G: RepetitionGame>(&mut self, node: u32, game: &G) {
        if self.nodes[node as usize].terminal {
            self.backpropagate(node, self.nodes[node as usize].reward);
            return;
        }

        let (legal, res) = self.evaluate_repetition_position(game);
        if !self.nodes[node as usize].expanded {
            self.build_policy_from(&legal, &res, false);
            self.expand(node);
        }
        self.backpropagate(node, res.value);
    }

    fn select_child(&self, node: u32, c_puct: f32) -> Option<u32> {
        let n = &self.nodes[node as usize];
        let sqrt_parent = ((n.visits + 1) as f32).sqrt();

        let mut best = None;
        let mut best_ucb = f32::NEG_INFINITY;
        for c in n.first_child..n.first_child + u32::from(n.num_children) {
            let child = &self.nodes[c as usize];
            let q = if child.visits == 0 {
                n.q() - self.cfg.fpu_reduction
            }
            else {
                -child.q()
            };
            let ucb = q + c_puct * child.prior * sqrt_parent / (1 + child.visits) as f32;
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

    fn evaluate_position<G: Game>(&mut self, game: &G) -> (Vec<Action>, Evaluation) {
        self.batch.clear();
        self.enqueue_state(game);
        let legal = self.batch.legal.clone();
        let mut results = self.evaluator.evaluate(&self.batch);
        debug_assert_eq!(
            results.len(),
            self.batch.len(),
            "evaluator must return one result per input state"
        );
        (legal, results.remove(0))
    }

    fn evaluate_repetition_position<G: RepetitionGame>(
        &mut self,
        game: &G,
    ) -> (Vec<Action>, Evaluation) {
        let hash = game.repetition_hash();
        if let Some(cache) = &self.eval_cache {
            if let Some(cached) = cache.get(hash) {
                return (cached.legal, cached.eval);
            }
        }

        let (legal, eval) = self.evaluate_position(game);
        if let Some(cache) = &self.eval_cache {
            cache.insert(
                hash,
                CachedEvaluation {
                    legal: legal.clone(),
                    eval: eval.clone(),
                },
            );
        }
        (legal, eval)
    }

    /// Softmax over legal-action logits (optionally mixed with root Dirichlet
    /// noise) into `self.policy_buf`.
    fn build_policy_from(&mut self, legal: &[Action], res: &Evaluation, root_noise: bool) {
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
                .push(Node::new(Some(node), action, 0, prior, false, 0.0));
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

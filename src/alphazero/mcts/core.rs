use super::super::evaluator::{EvalBatch, Evaluation, Evaluator};
use super::cache::{CachedEvaluation, EvalTable};
use super::gumbel::Gumbel;
use super::puct::Puct;
use super::{MctsConfig, MctsVariant, Node, RepetitionGame, SearchResult};
use crate::agent::PolicyMode;
use crate::game::{Action, Game};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::Gamma;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(super) struct CoreConfig {
    pub(super) c_init: f32,
    pub(super) c_base: f32,
    pub(super) simulations: usize,
    pub(super) eps: f32,
    pub(super) alpha: f32,
    pub(super) fpu_reduction: f32,
}

impl CoreConfig {
    fn from_mcts(cfg: MctsConfig) -> Self {
        CoreConfig {
            c_init: cfg.c_init,
            c_base: cfg.c_base,
            simulations: cfg.simulations,
            eps: cfg.eps,
            alpha: cfg.alpha,
            fpu_reduction: cfg.fpu_reduction,
        }
    }

    fn into_mcts(self, variant: MctsVariant) -> MctsConfig {
        MctsConfig {
            c_init: self.c_init,
            c_base: self.c_base,
            variant,
            simulations: self.simulations,
            eps: self.eps,
            alpha: self.alpha,
            fpu_reduction: self.fpu_reduction,
        }
    }
}

/// Runtime facade over the concrete MCTS variants. Callers can choose the
/// variant from config at runtime; internally each variant has typed state.
pub struct Mcts<E: Evaluator> {
    pub(super) inner: MctsKind<E>,
}

pub(super) enum MctsKind<E: Evaluator> {
    Puct(MctsCore<E, Puct>),
    Gumbel(MctsCore<E, Gumbel>),
}

/// Shared tree/evaluator state for one concrete root search algorithm. Each
/// simulation descends to one leaf and submits at most one network request.
/// GPU batching happens across blocked search threads in the shared `Batcher`,
/// not inside one MCTS tree.
pub(super) struct MctsCore<E: Evaluator, V> {
    evaluator: E,
    pub(super) cfg: CoreConfig,
    pub(super) nodes: Vec<Node>,
    pub(super) batch: EvalBatch,
    pub(super) policy_buf: Vec<(Action, f32, f32)>,
    eval_cache: Option<Arc<EvalTable>>,
    pub(super) rng: SmallRng,
    pub(super) variant: V,
}

#[derive(Clone, Copy)]
struct StandardSearch;

#[derive(Clone, Copy)]
struct RepetitionSearch<F> {
    root_repetitions: F,
}

pub(super) trait SearchDriver<G: Game>: Copy {
    fn root_hash(self, game: &G) -> u64;
    fn evaluate<E: Evaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        game: &G,
    ) -> (Vec<Action>, Evaluation);
    fn simulate<E: Evaluator, V>(self, mcts: &mut MctsCore<E, V>, game: &G);
    fn simulate_from_child<E: Evaluator, V>(self, mcts: &mut MctsCore<E, V>, game: &G, child: u32);
}

impl<G: Game> SearchDriver<G> for StandardSearch {
    fn root_hash(self, _game: &G) -> u64 {
        0
    }

    fn evaluate<E: Evaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        game: &G,
    ) -> (Vec<Action>, Evaluation) {
        mcts.evaluate_position(game)
    }

    fn simulate<E: Evaluator, V>(self, mcts: &mut MctsCore<E, V>, game: &G) {
        let (node, current) = mcts.descend(game, 0);
        mcts.finish_simulation(node, &current);
    }

    fn simulate_from_child<E: Evaluator, V>(self, mcts: &mut MctsCore<E, V>, game: &G, child: u32) {
        let (node, current) = mcts.descend(game, child);
        mcts.finish_simulation(node, &current);
    }
}

impl<G, F> SearchDriver<G> for RepetitionSearch<F>
where
    G: RepetitionGame,
    F: Fn(u64) -> u8 + Copy,
{
    fn root_hash(self, game: &G) -> u64 {
        game.repetition_hash()
    }

    fn evaluate<E: Evaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        game: &G,
    ) -> (Vec<Action>, Evaluation) {
        mcts.evaluate_repetition_position(game)
    }

    fn simulate<E: Evaluator, V>(self, mcts: &mut MctsCore<E, V>, game: &G) {
        let (node, current) = mcts.descend_repetition(game, 0, self.root_repetitions);
        mcts.finish_repetition_simulation(node, &current);
    }

    fn simulate_from_child<E: Evaluator, V>(self, mcts: &mut MctsCore<E, V>, game: &G, child: u32) {
        let (node, current) = mcts.descend_repetition(game, child, self.root_repetitions);
        mcts.finish_repetition_simulation(node, &current);
    }
}

impl<E: Evaluator> Mcts<E> {
    pub fn new(evaluator: E, cfg: MctsConfig) -> Self {
        let core_cfg = CoreConfig::from_mcts(cfg);
        let inner = match cfg.variant {
            MctsVariant::Puct => MctsKind::Puct(MctsCore::new(evaluator, core_cfg, Puct)),
            MctsVariant::Gumbel { sampled_actions } => MctsKind::Gumbel(MctsCore::new(
                evaluator,
                core_cfg,
                Gumbel {
                    sampled_actions,
                    root_actions: Vec::new(),
                },
            )),
        };
        Mcts { inner }
    }

    pub(crate) fn with_eval_cache(mut self, cache: Arc<EvalTable>) -> Self {
        match &mut self.inner {
            MctsKind::Puct(core) => core.set_eval_cache(cache),
            MctsKind::Gumbel(core) => core.set_eval_cache(cache),
        }
        self
    }

    pub fn config(&self) -> MctsConfig {
        match &self.inner {
            MctsKind::Puct(core) => core.config(MctsVariant::Puct),
            MctsKind::Gumbel(core) => core.config(MctsVariant::Gumbel {
                sampled_actions: core.variant.sampled_actions,
            }),
        }
    }

    pub fn set_simulations(&mut self, simulations: usize) {
        match &mut self.inner {
            MctsKind::Puct(core) => core.set_simulations(simulations),
            MctsKind::Gumbel(core) => core.set_simulations(simulations),
        }
    }

    pub fn search<G: Game>(&mut self, game: &G) -> SearchResult {
        self.search_with_mode(game, PolicyMode::Explore)
    }

    /// Runs a search with root exploration enabled or disabled according to
    /// `mode`. PUCT uses Dirichlet noise only in `Explore`; Gumbel search uses
    /// Gumbel noise only in `Explore`.
    pub fn search_with_mode<G: Game>(&mut self, game: &G, mode: PolicyMode) -> SearchResult {
        match &mut self.inner {
            MctsKind::Puct(core) => core.search_inner(game, mode, StandardSearch),
            MctsKind::Gumbel(core) => core.search_inner(game, mode, StandardSearch),
        }
    }

    pub fn search_with_repetitions<G, F>(&mut self, game: &G, root_repetitions: F) -> SearchResult
    where
        G: RepetitionGame,
        F: Fn(u64) -> u8 + Copy,
    {
        self.search_with_repetitions_mode(game, root_repetitions, PolicyMode::Explore)
    }

    pub fn search_with_repetitions_mode<G, F>(
        &mut self,
        game: &G,
        root_repetitions: F,
        mode: PolicyMode,
    ) -> SearchResult
    where
        G: RepetitionGame,
        F: Fn(u64) -> u8 + Copy,
    {
        match &mut self.inner {
            MctsKind::Puct(core) => {
                core.search_inner(game, mode, RepetitionSearch { root_repetitions })
            }
            MctsKind::Gumbel(core) => {
                core.search_inner(game, mode, RepetitionSearch { root_repetitions })
            }
        }
    }
}

impl<E: Evaluator, V> MctsCore<E, V> {
    fn new(evaluator: E, cfg: CoreConfig, variant: V) -> Self {
        debug_assert!(cfg.simulations > 0, "MCTS simulations must be positive");
        debug_assert!(cfg.c_base > 0.0, "MCTS c_base must be positive");
        debug_assert!((0.0..=1.0).contains(&cfg.eps), "MCTS eps must be in [0, 1]");
        debug_assert!(
            cfg.eps == 0.0 || cfg.alpha > 0.0,
            "MCTS alpha must be positive when root noise is enabled"
        );

        MctsCore {
            evaluator,
            cfg,
            nodes: Vec::new(),
            batch: EvalBatch::new(),
            policy_buf: Vec::new(),
            eval_cache: None,
            rng: SmallRng::from_os_rng(),
            variant,
        }
    }

    fn config(&self, variant: MctsVariant) -> MctsConfig {
        self.cfg.into_mcts(variant)
    }

    fn set_eval_cache(&mut self, cache: Arc<EvalTable>) {
        self.eval_cache = Some(cache);
    }

    fn set_simulations(&mut self, simulations: usize) {
        debug_assert!(simulations > 0, "MCTS simulations must be positive");
        self.cfg.simulations = simulations.max(1);
    }

    pub(super) fn clear_tree_common(&mut self) {
        self.nodes.clear();
        self.policy_buf.clear();
        self.batch.clear();
    }
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

    pub(super) fn finish_simulation<G: Game>(&mut self, node: u32, game: &G) {
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

    pub(super) fn finish_repetition_simulation<G: RepetitionGame>(&mut self, node: u32, game: &G) {
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

    pub(super) fn evaluate_position<G: Game>(&mut self, game: &G) -> (Vec<Action>, Evaluation) {
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

    pub(super) fn evaluate_repetition_position<G: RepetitionGame>(
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
    /// noise) into `self.policy_buf`. Raw logits are retained for Gumbel search.
    pub(super) fn build_policy_from(
        &mut self,
        legal: &[Action],
        res: &Evaluation,
        root_noise: bool,
    ) {
        debug_assert_eq!(
            legal.len(),
            res.logits.len(),
            "evaluator logits must match the legal actions for each state"
        );
        debug_assert!(
            res.logits.iter().all(|logit| logit.is_finite()),
            "evaluator must return finite logits for legal actions"
        );

        self.policy_buf.clear();
        let max_logit = res.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (&action, &logit) in legal.iter().zip(&res.logits) {
            let prior = (logit - max_logit).exp();
            self.policy_buf.push((action, prior, logit));
            sum += prior;
        }
        if sum.is_finite() && sum > 0.0 {
            for (_, prior, _) in &mut self.policy_buf {
                *prior /= sum;
            }
        }
        else if !self.policy_buf.is_empty() {
            let uniform = 1.0 / self.policy_buf.len() as f32;
            for (_, prior, _) in &mut self.policy_buf {
                *prior = uniform;
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
            for ((_, prior, _), noise) in self.policy_buf.iter_mut().zip(&noise) {
                *prior = (1.0 - self.cfg.eps) * *prior + self.cfg.eps * noise;
            }
        }
    }

    /// Creates all legal children of `node`, contiguous in the arena.
    pub(super) fn expand(&mut self, node: u32) {
        debug_assert!(self.policy_buf.len() <= u16::MAX as usize);
        let first_child = self.nodes.len() as u32;
        for &(action, prior, logit) in &self.policy_buf {
            self.nodes
                .push(Node::new(Some(node), action, 0, prior, logit, false, 0.0));
        }
        let n = &mut self.nodes[node as usize];
        n.first_child = first_child;
        n.num_children = self.policy_buf.len() as u16;
        n.expanded = true;
    }
}

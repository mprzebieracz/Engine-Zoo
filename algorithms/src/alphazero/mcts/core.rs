use super::super::evaluator::{EvalBatch, Evaluation, Evaluator};
use super::cache::{CachedEvaluation, EvalTable};
use super::gumbel::Gumbel;
use super::puct::Puct;
use super::{MctsConfig, MctsVariant, Node, SearchResult};
use engine_core::agent::PolicyMode;
use engine_core::game::{Action, Game};
use engine_core::rules::RepetitionGame;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::Gamma;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(super) struct CoreConfig {
    pub(super) c_init: f32,
    pub(super) c_base: f32,
    pub(super) simulations: usize,
    pub(super) leaf_batch_size: usize,
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
            leaf_batch_size: cfg.leaf_batch_size.max(1),
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
            leaf_batch_size: self.leaf_batch_size,
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

/// Shared tree/evaluator state for one concrete root search algorithm.
///
/// Search collects a small batch of leaves inside one tree, applies virtual
/// loss while the batch is open, and sends the unique non-terminal leaves to
/// the evaluator together. The shared `Batcher` can still coalesce requests
/// from many self-play threads into larger GPU batches.
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

pub(super) struct PendingLeaf<G> {
    node: u32,
    game: G,
}

struct PendingBackup {
    node: u32,
    result: PendingResult,
}

enum PendingResult {
    Terminal,
    Evaluation(usize),
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
    fn descend<E: Evaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        game: &G,
        child: Option<u32>,
    ) -> (u32, G);
    fn evaluate_many<E: Evaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        games: &[G],
    ) -> Vec<(Vec<Action>, Evaluation)>;
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

    fn descend<E: Evaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        game: &G,
        child: Option<u32>,
    ) -> (u32, G) {
        mcts.descend(game, child.unwrap_or(0))
    }

    fn evaluate_many<E: Evaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        games: &[G],
    ) -> Vec<(Vec<Action>, Evaluation)> {
        mcts.evaluate_positions(games)
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

    fn descend<E: Evaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        game: &G,
        child: Option<u32>,
    ) -> (u32, G) {
        mcts.descend_repetition(game, child.unwrap_or(0), self.root_repetitions)
    }

    fn evaluate_many<E: Evaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        games: &[G],
    ) -> Vec<(Vec<Action>, Evaluation)> {
        mcts.evaluate_repetition_positions(games)
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
        debug_assert!(
            cfg.leaf_batch_size > 0,
            "MCTS leaf batch size must be positive"
        );
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

    pub(super) fn leaf_batch_size(&self) -> usize {
        self.cfg.leaf_batch_size.max(1)
    }

    pub(super) fn clear_tree_common(&mut self) {
        self.nodes.clear();
        self.policy_buf.clear();
        self.batch.clear();
    }
}

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
            let Some(best) = self.select_child(node, c_puct) else {
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
            let c_puct = ((1.0 + n.effective_visits() as f32 + c_base) / c_base).ln() + c_init;
            let Some(best) = self.select_child(node, c_puct) else {
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
            let Some(parent) = self.nodes[node as usize].parent else {
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
            } else {
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
            let Some(parent) = n.parent else {
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
        self.evaluate_positions(std::slice::from_ref(game))
            .pop()
            .expect("single game evaluation")
    }

    pub(super) fn evaluate_positions<G: Game>(
        &mut self,
        games: &[G],
    ) -> Vec<(Vec<Action>, Evaluation)> {
        if games.is_empty() {
            return Vec::new();
        }
        self.batch.clear();
        let mut legal_per_state = Vec::with_capacity(games.len());
        for game in games {
            let begin = self.batch.legal.len();
            self.enqueue_state(game);
            legal_per_state.push(self.batch.legal[begin..].to_vec());
        }
        let results = self.evaluator.evaluate(&self.batch);
        debug_assert_eq!(
            results.len(),
            self.batch.len(),
            "evaluator must return one result per input state"
        );
        legal_per_state.into_iter().zip(results).collect()
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

    pub(super) fn evaluate_repetition_positions<G: RepetitionGame>(
        &mut self,
        games: &[G],
    ) -> Vec<(Vec<Action>, Evaluation)> {
        if games.is_empty() {
            return Vec::new();
        }

        let cache = self.eval_cache.clone();
        let mut out = vec![None; games.len()];
        let mut miss_indexes = Vec::new();
        let mut miss_games = Vec::new();

        for (i, game) in games.iter().enumerate() {
            let hash = game.repetition_hash();
            if let Some(cache) = &cache {
                if let Some(cached) = cache.get(hash) {
                    out[i] = Some((cached.legal, cached.eval));
                    continue;
                }
            }
            miss_indexes.push(i);
            miss_games.push(*game);
        }

        let miss_results = self.evaluate_positions(&miss_games);
        for ((i, game), (legal, eval)) in miss_indexes.into_iter().zip(miss_games).zip(miss_results)
        {
            if let Some(cache) = &cache {
                cache.insert(
                    game.repetition_hash(),
                    CachedEvaluation {
                        legal: legal.clone(),
                        eval: eval.clone(),
                    },
                );
            }
            out[i] = Some((legal, eval));
        }

        out.into_iter()
            .map(|value| value.expect("all repetition eval slots filled"))
            .collect()
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
        } else if !self.policy_buf.is_empty() {
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

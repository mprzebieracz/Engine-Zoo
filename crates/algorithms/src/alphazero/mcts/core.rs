use super::super::evaluator::{EncodedEvalBatch, EncodedEvaluator, Evaluation};
use super::cache::EvalTable;
use super::gumbel::Gumbel;
use super::puct::Puct;
use super::{GumbelSearchProfile, MctsConfig, MctsVariant, Node, SearchResult};
use engine_core::agent::PolicyMode;
use engine_core::game::{Action, Game};
use engine_core::rules::RepetitionGame;
use rand::prelude::*;
use rand::rngs::SmallRng;
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
pub struct Mcts<E: EncodedEvaluator> {
    pub(super) inner: MctsKind<E>,
}

pub(super) enum MctsKind<E: EncodedEvaluator> {
    Puct(MctsCore<E, Puct>),
    Gumbel(MctsCore<E, Gumbel>),
}

/// Shared tree/evaluator state for one concrete root search algorithm.
///
/// Search collects a small batch of leaves inside one tree, applies virtual
/// loss while the batch is open, and sends the unique non-terminal leaves to
/// the evaluator together. The shared `Batcher` can still coalesce requests
/// from many self-play threads into larger GPU batches.
pub(super) struct MctsCore<E: EncodedEvaluator, V> {
    pub(super) evaluator: E,
    pub(super) cfg: CoreConfig,
    pub(super) nodes: Vec<Node<Action>>,
    pub(super) batch: EncodedEvalBatch,
    pub(super) policy_buf: Vec<(Action, f32, f32)>,
    pub(super) repetition_path: Vec<u64>,
    pub(super) eval_cache: Option<Arc<EvalTable<Action>>>,
    pub(super) rng: SmallRng,
    pub(super) variant: V,
}

pub(super) struct PendingLeaf<G> {
    pub(super) node: u32,
    pub(super) game: G,
}

pub(super) struct LeafBatch<G> {
    pub(super) leaves: Vec<PendingLeaf<G>>,
    pub(super) pending: Vec<PendingBackup>,
    pub(super) unique_games: Vec<G>,
    pub(super) result_by_node: Vec<(u32, usize)>,
}

impl<G> LeafBatch<G> {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            leaves: Vec::with_capacity(capacity),
            pending: Vec::with_capacity(capacity),
            unique_games: Vec::with_capacity(capacity),
            result_by_node: Vec::with_capacity(capacity),
        }
    }
}

pub(super) struct PendingBackup {
    pub(super) node: u32,
    pub(super) result: PendingResult,
}

pub(super) enum PendingResult {
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
    fn evaluate<E: EncodedEvaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        game: &G,
    ) -> (Vec<Action>, Evaluation);
    fn descend<E: EncodedEvaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        game: &G,
        child: Option<u32>,
    ) -> (u32, G);
    fn evaluate_many<E: EncodedEvaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        games: &[G],
    ) -> Vec<(Vec<Action>, Evaluation)>;
}

impl<G: Game> SearchDriver<G> for StandardSearch {
    fn root_hash(self, _game: &G) -> u64 {
        0
    }

    fn evaluate<E: EncodedEvaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        game: &G,
    ) -> (Vec<Action>, Evaluation) {
        mcts.evaluate_position(game)
    }

    fn descend<E: EncodedEvaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        game: &G,
        child: Option<u32>,
    ) -> (u32, G) {
        mcts.descend(game, child.unwrap_or(0))
    }

    fn evaluate_many<E: EncodedEvaluator, V>(
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

    fn evaluate<E: EncodedEvaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        game: &G,
    ) -> (Vec<Action>, Evaluation) {
        mcts.evaluate_repetition_position(game)
    }

    fn descend<E: EncodedEvaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        game: &G,
        child: Option<u32>,
    ) -> (u32, G) {
        mcts.descend_repetition(game, child.unwrap_or(0), self.root_repetitions)
    }

    fn evaluate_many<E: EncodedEvaluator, V>(
        self,
        mcts: &mut MctsCore<E, V>,
        games: &[G],
    ) -> Vec<(Vec<Action>, Evaluation)> {
        mcts.evaluate_repetition_positions(games)
    }
}

impl<E: EncodedEvaluator> Mcts<E> {
    pub fn new(evaluator: E, cfg: MctsConfig) -> Self {
        cfg.validate().expect("invalid MCTS configuration");
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

    pub fn with_eval_cache(mut self, cache: Arc<EvalTable<Action>>) -> Self {
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
        assert!(simulations > 0, "MCTS simulations must be positive");
        match &mut self.inner {
            MctsKind::Puct(core) => core.set_simulations(simulations),
            MctsKind::Gumbel(core) => core.set_simulations(simulations),
        }
    }

    /// Atomically applies a Gumbel simulation budget and sequential-halving
    /// root-candidate count. This intentionally cannot change a PUCT search
    /// into a Gumbel search: doing that would require rebuilding variant state.
    pub fn set_gumbel_profile(&mut self, profile: GumbelSearchProfile) {
        profile.validate().expect("invalid Gumbel search profile");
        match &mut self.inner {
            MctsKind::Gumbel(core) => {
                core.cfg.simulations = profile.simulations;
                core.variant.sampled_actions = profile.root_candidates;
            }
            MctsKind::Puct(_) => panic!("cannot apply a Gumbel profile to PUCT MCTS"),
        }
    }

    pub fn search<G: Game>(&mut self, game: &G) -> SearchResult<Action> {
        self.search_with_mode(game, PolicyMode::Explore)
    }

    /// Runs a search with root exploration enabled or disabled according to
    /// `mode`. PUCT uses Dirichlet noise only in `Explore`; Gumbel search uses
    /// Gumbel noise only in `Explore`.
    pub fn search_with_mode<G: Game>(
        &mut self,
        game: &G,
        mode: PolicyMode,
    ) -> SearchResult<Action> {
        match &mut self.inner {
            MctsKind::Puct(core) => core.search_inner(game, mode, StandardSearch),
            MctsKind::Gumbel(core) => core.search_inner(game, mode, StandardSearch),
        }
    }

    pub fn search_with_repetitions<G, F>(
        &mut self,
        game: &G,
        root_repetitions: F,
    ) -> SearchResult<Action>
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
    ) -> SearchResult<Action>
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

impl<E: EncodedEvaluator, V> MctsCore<E, V> {
    fn new(evaluator: E, cfg: CoreConfig, variant: V) -> Self {
        MctsCore {
            evaluator,
            cfg,
            nodes: Vec::new(),
            batch: EncodedEvalBatch::new(),
            policy_buf: Vec::new(),
            repetition_path: Vec::with_capacity(128),
            eval_cache: None,
            rng: SmallRng::from_os_rng(),
            variant,
        }
    }

    fn config(&self, variant: MctsVariant) -> MctsConfig {
        self.cfg.into_mcts(variant)
    }

    fn set_eval_cache(&mut self, cache: Arc<EvalTable<Action>>) {
        self.eval_cache = Some(cache);
    }

    fn set_simulations(&mut self, simulations: usize) {
        self.cfg.simulations = simulations;
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

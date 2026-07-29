use super::cache::{CachedEvaluation, EvalTable};
use super::{Node, SearchAlgorithm, SearchBudget, SearchConfig, SearchRequest};
use crate::{EvaluationKey, PolicyValueEvaluator, SearchRules};
use engine_core::game::GameState;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::marker::PhantomData;
use std::sync::Arc;

pub struct Mcts<G, E, R>
where
    G: GameState + Clone,
    E: PolicyValueEvaluator<G>,
    R: SearchRules<G>,
{
    pub(super) inner: MctsKind<G, E, R>,
}

pub(super) enum MctsKind<G, E, R>
where
    G: GameState + Clone,
    E: PolicyValueEvaluator<G>,
    R: SearchRules<G>,
{
    Puct(MctsCore<G, E, R, super::puct::Puct>),
    RootGumbelPuct(MctsCore<G, E, R, super::root_gumbel_puct::RootGumbelPuct>),
    FullGumbel(MctsCore<G, E, R, super::gumbel::FullGumbel>),
}

pub(super) struct MctsCore<G, E, R, V>
where
    G: GameState + Clone,
    E: PolicyValueEvaluator<G>,
    R: SearchRules<G>,
    V: 'static,
{
    pub(super) evaluator: E,
    pub(super) rules: R,
    pub(super) nodes: Vec<Node<G::Move, R::NodeMeta>>,
    pub(super) legal_moves: Vec<G::Move>,
    pub(super) offsets: Vec<u32>,
    pub(super) evaluation_workspace: EvaluationWorkspace<G>,
    pub(super) policy_buf: Vec<(G::Move, f32, f32)>,
    pub(super) path_state: R::PathState,
    pub(super) eval_cache: Option<Arc<EvalTable<G::Move>>>,
    pub(super) rng: SmallRng,
    pub(super) variant: V,
    pub(super) marker: PhantomData<fn() -> G>,
}

/// Reused scratch space for one batched evaluator call.
///
/// The returned flat evaluation vector remains owned by the caller. Everything
/// used only to route cache hits and backend results stays here so successive
/// leaf batches retain their allocations.
pub(super) struct EvaluationWorkspace<G: GameState> {
    pub(super) outputs: Vec<Option<CachedEvaluation<G::Move>>>,
    pub(super) misses: Vec<G>,
    pub(super) keys: Vec<Option<EvaluationKey>>,
    pub(super) destinations: Vec<(usize, usize)>,
    pub(super) unique: Vec<CachedEvaluation<G::Move>>,
}

impl<G: GameState> Default for EvaluationWorkspace<G> {
    fn default() -> Self {
        Self {
            outputs: Vec::new(),
            misses: Vec::new(),
            keys: Vec::new(),
            destinations: Vec::new(),
            unique: Vec::new(),
        }
    }
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

pub(super) struct PendingBackup {
    pub(super) node: u32,
    pub(super) result: PendingResult,
}
pub(super) enum PendingResult {
    Terminal,
    Evaluation(usize),
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

impl<G, E, R, V> MctsCore<G, E, R, V>
where
    G: GameState + Clone,
    E: PolicyValueEvaluator<G>,
    R: SearchRules<G>,
    V: 'static,
{
    pub(super) fn clear_tree_common(&mut self) {
        self.nodes.clear();
        self.policy_buf.clear();
    }
}

impl<G, E, R> Mcts<G, E, R>
where
    G: GameState + Clone,
    E: PolicyValueEvaluator<G>,
    R: SearchRules<G>,
{
    /// Creates a searcher, panicking when `config` is invalid.
    /// Prefer [`Self::try_new`] at recoverable application boundaries.
    pub fn new(evaluator: E, config: SearchConfig, rules: R) -> Self {
        Self::try_new(evaluator, config, rules).expect("invalid search configuration")
    }

    pub fn try_new(
        evaluator: E,
        config: SearchConfig,
        rules: R,
    ) -> Result<Self, super::SearchConfigError> {
        config.validate().map_err(super::SearchConfigError)?;
        fn make_core<G, E, R, V>(evaluator: E, rules: R, variant: V) -> MctsCore<G, E, R, V>
        where
            G: GameState + Clone,
            E: PolicyValueEvaluator<G>,
            R: SearchRules<G>,
            V: 'static,
        {
            MctsCore {
                evaluator,
                rules,
                nodes: Vec::new(),
                legal_moves: Vec::new(),
                offsets: Vec::new(),
                evaluation_workspace: EvaluationWorkspace::default(),
                policy_buf: Vec::new(),
                path_state: R::PathState::default(),
                eval_cache: None,
                rng: SmallRng::from_os_rng(),
                variant,
                marker: PhantomData,
            }
        }
        let inner = match config {
            SearchConfig::Puct(config) => {
                MctsKind::Puct(make_core(evaluator, rules, super::puct::Puct::from(config)))
            }
            SearchConfig::RootGumbelPuct(config) => MctsKind::RootGumbelPuct(make_core(
                evaluator,
                rules,
                super::root_gumbel_puct::RootGumbelPuct::from(config),
            )),
            SearchConfig::FullGumbel(config) => MctsKind::FullGumbel(make_core(
                evaluator,
                rules,
                super::gumbel::FullGumbel::from(config),
            )),
        };
        Ok(Self { inner })
    }

    pub fn with_eval_cache(mut self, cache: Arc<EvalTable<G::Move>>) -> Self {
        match &mut self.inner {
            MctsKind::Puct(core) => core.eval_cache = Some(cache.clone()),
            MctsKind::RootGumbelPuct(core) => core.eval_cache = Some(cache.clone()),
            MctsKind::FullGumbel(core) => core.eval_cache = Some(cache),
        }
        self
    }

    /// Replaces the default OS-seeded RNG for reproducible search experiments.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.reseed(seed);
        self
    }

    /// Resets the search RNG without rebuilding its evaluator or cache.
    pub fn reseed(&mut self, seed: u64) {
        match &mut self.inner {
            MctsKind::Puct(core) => core.rng = SmallRng::seed_from_u64(seed),
            MctsKind::RootGumbelPuct(core) => core.rng = SmallRng::seed_from_u64(seed),
            MctsKind::FullGumbel(core) => core.rng = SmallRng::seed_from_u64(seed),
        }
    }

    pub fn config(&self) -> SearchConfig {
        match &self.inner {
            MctsKind::Puct(core) => SearchConfig::Puct(core.variant.config()),
            MctsKind::RootGumbelPuct(core) => {
                SearchConfig::RootGumbelPuct(core.variant.config.clone())
            }
            MctsKind::FullGumbel(core) => SearchConfig::FullGumbel(core.variant.config.clone()),
        }
    }

    pub fn algorithm(&self) -> SearchAlgorithm {
        match &self.inner {
            MctsKind::Puct(_) => SearchAlgorithm::Puct,
            MctsKind::RootGumbelPuct(_) => SearchAlgorithm::RootGumbelPuct,
            MctsKind::FullGumbel(_) => SearchAlgorithm::FullGumbel,
        }
    }

    pub fn search(
        &mut self,
        game: &G,
        context: R::Context<'_>,
        request: SearchRequest,
    ) -> Result<super::SearchResult<G::Move>, super::SearchError> {
        request
            .budget
            .validate()
            .map_err(|message| super::SearchError::InvalidBudget { message })?;
        match &mut self.inner {
            MctsKind::Puct(core) => match request.budget {
                SearchBudget::Puct { simulations } => {
                    core.search_inner(game, context, request.mode, simulations)
                }
                budget => Err(super::SearchError::BudgetAlgorithmMismatch {
                    algorithm: SearchAlgorithm::Puct,
                    budget,
                }),
            },
            MctsKind::RootGumbelPuct(core) => match request.budget {
                SearchBudget::Gumbel {
                    simulations,
                    max_considered_actions,
                } => core.search_inner(
                    game,
                    context,
                    request.mode,
                    simulations,
                    max_considered_actions,
                ),
                budget => Err(super::SearchError::BudgetAlgorithmMismatch {
                    algorithm: SearchAlgorithm::RootGumbelPuct,
                    budget,
                }),
            },
            MctsKind::FullGumbel(core) => match request.budget {
                SearchBudget::Gumbel {
                    simulations,
                    max_considered_actions,
                } => core.search_inner(
                    game,
                    context,
                    request.mode,
                    simulations,
                    max_considered_actions,
                ),
                budget => Err(super::SearchError::BudgetAlgorithmMismatch {
                    algorithm: SearchAlgorithm::FullGumbel,
                    budget,
                }),
            },
        }
    }
}

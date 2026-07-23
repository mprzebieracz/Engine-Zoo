use super::cache::EvalTable;
use super::{CommonSearchConfig, GumbelConfig, Node, SearchConfig};
use crate::{PolicyValueEvaluator, SearchRules};
use engine_core::game::GameState;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(super) struct CoreConfig {
    pub(super) simulations: usize,
    pub(super) leaf_batch_size: usize,
}

impl From<CommonSearchConfig> for CoreConfig {
    fn from(config: CommonSearchConfig) -> Self {
        Self {
            simulations: config.simulations,
            leaf_batch_size: config.leaf_batch_size,
        }
    }
}

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
    Gumbel(MctsCore<G, E, R, super::gumbel::FullGumbel>),
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
    pub(super) cfg: CoreConfig,
    pub(super) nodes: Vec<Node<G::Move, R::NodeMeta>>,
    pub(super) legal_moves: Vec<G::Move>,
    pub(super) offsets: Vec<u32>,
    pub(super) policy_buf: Vec<(G::Move, f32, f32)>,
    pub(super) path_state: R::PathState,
    pub(super) eval_cache: Option<Arc<EvalTable<G::Move>>>,
    pub(super) rng: SmallRng,
    pub(super) variant: V,
    pub(super) marker: PhantomData<fn() -> G>,
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
        fn make_core<G, E, R, V>(
            evaluator: E,
            rules: R,
            cfg: CoreConfig,
            variant: V,
        ) -> MctsCore<G, E, R, V>
        where
            G: GameState + Clone,
            E: PolicyValueEvaluator<G>,
            R: SearchRules<G>,
            V: 'static,
        {
            MctsCore {
                evaluator,
                rules,
                cfg,
                nodes: Vec::new(),
                legal_moves: Vec::new(),
                offsets: Vec::new(),
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
                let core = CoreConfig::from(config.common);
                MctsKind::Puct(make_core(
                    evaluator,
                    rules,
                    core,
                    super::puct::Puct::from(config),
                ))
            }
            SearchConfig::Gumbel(config) => {
                let core = CoreConfig {
                    simulations: config.simulations,
                    leaf_batch_size: 1,
                };
                MctsKind::Gumbel(make_core(
                    evaluator,
                    rules,
                    core,
                    super::gumbel::FullGumbel::from(config),
                ))
            }
        };
        Ok(Self { inner })
    }

    pub fn with_eval_cache(mut self, cache: Arc<EvalTable<G::Move>>) -> Self {
        match &mut self.inner {
            MctsKind::Puct(core) => core.eval_cache = Some(cache.clone()),
            MctsKind::Gumbel(core) => core.eval_cache = Some(cache),
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
            MctsKind::Gumbel(core) => core.rng = SmallRng::seed_from_u64(seed),
        }
    }

    pub fn config(&self) -> SearchConfig {
        match &self.inner {
            MctsKind::Puct(core) => SearchConfig::Puct(core.variant.config(core.cfg)),
            MctsKind::Gumbel(core) => SearchConfig::Gumbel(core.variant.config.clone()),
        }
    }

    pub fn set_simulations(&mut self, simulations: usize) {
        assert!(simulations > 0, "search simulations must be positive");
        match &mut self.inner {
            MctsKind::Puct(core) => core.cfg.simulations = simulations,
            MctsKind::Gumbel(core) => {
                core.cfg.simulations = simulations;
                core.variant.config.simulations = simulations;
            }
        }
    }

    pub fn set_gumbel_config(&mut self, config: GumbelConfig) -> Result<(), &'static str> {
        config.validate()?;
        match &mut self.inner {
            MctsKind::Gumbel(core) => {
                core.cfg.simulations = config.simulations;
                core.variant.config = config;
                Ok(())
            }
            MctsKind::Puct(_) => Err("Gumbel configuration requires Full Gumbel search"),
        }
    }

    pub fn search(
        &mut self,
        game: &G,
        context: R::Context<'_>,
        mode: engine_core::agent::PolicyMode,
    ) -> Result<super::SearchResult<G::Move>, super::SearchError> {
        match &mut self.inner {
            MctsKind::Puct(core) => core.search_inner(game, context, mode),
            MctsKind::Gumbel(core) => core.search_inner(game, context, mode),
        }
    }
}

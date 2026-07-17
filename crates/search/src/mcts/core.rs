use super::cache::EvalTable;
use super::{MctsConfig, MctsVariant, Node};
use crate::{PolicyValueEvaluator, SearchRules};
use engine_core::game::GameState;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::marker::PhantomData;
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
    pub(super) fn from_mcts(c: MctsConfig) -> Self {
        Self {
            c_init: c.c_init,
            c_base: c.c_base,
            simulations: c.simulations,
            leaf_batch_size: c.leaf_batch_size,
            eps: c.eps,
            alpha: c.alpha,
            fpu_reduction: c.fpu_reduction,
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
    Gumbel(MctsCore<G, E, R, super::gumbel::Gumbel>),
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
    pub(super) fn with_capacity(n: usize) -> Self {
        Self {
            leaves: Vec::with_capacity(n),
            pending: Vec::with_capacity(n),
            unique_games: Vec::with_capacity(n),
            result_by_node: Vec::with_capacity(n),
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
    pub(super) fn leaf_batch_size(&self) -> usize {
        self.cfg.leaf_batch_size
    }
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
    pub fn new(evaluator: E, cfg: MctsConfig, rules: R) -> Self {
        cfg.validate().expect("invalid MCTS configuration");
        fn make_core<G, E, R, V>(
            evaluator: E,
            rules: R,
            cfg: MctsConfig,
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
                cfg: CoreConfig::from_mcts(cfg),
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
        Self {
            inner: match cfg.variant {
                MctsVariant::Puct => {
                    MctsKind::Puct(make_core(evaluator, rules, cfg, super::puct::Puct))
                }
                MctsVariant::Gumbel { sampled_actions } => MctsKind::Gumbel(make_core(
                    evaluator,
                    rules,
                    cfg,
                    super::gumbel::Gumbel {
                        sampled_actions,
                        root_actions: Vec::new(),
                    },
                )),
            },
        }
    }

    pub fn with_eval_cache(mut self, cache: Arc<EvalTable<G::Move>>) -> Self {
        match &mut self.inner {
            MctsKind::Puct(core) => core.eval_cache = Some(cache.clone()),
            MctsKind::Gumbel(core) => core.eval_cache = Some(cache),
        }
        self
    }

    pub fn config(&self) -> MctsConfig {
        match &self.inner {
            MctsKind::Puct(core) => core.cfg.to_mcts(MctsVariant::Puct),
            MctsKind::Gumbel(core) => core.cfg.to_mcts(MctsVariant::Gumbel {
                sampled_actions: core.variant.sampled_actions,
            }),
        }
    }

    pub fn set_simulations(&mut self, simulations: usize) {
        assert!(simulations > 0, "MCTS simulations must be positive");
        match &mut self.inner {
            MctsKind::Puct(core) => core.cfg.simulations = simulations,
            MctsKind::Gumbel(core) => core.cfg.simulations = simulations,
        }
    }

    pub fn set_gumbel_profile(&mut self, profile: super::GumbelSearchProfile) {
        profile.validate().expect("invalid Gumbel search profile");
        match &mut self.inner {
            MctsKind::Gumbel(core) => {
                core.cfg.simulations = profile.simulations;
                core.variant.sampled_actions = profile.root_candidates;
            }
            MctsKind::Puct(_) => panic!("Gumbel profile requires Gumbel MCTS"),
        }
    }

    pub fn search(
        &mut self,
        game: &G,
        context: R::Context<'_>,
        mode: engine_core::agent::PolicyMode,
    ) -> super::SearchResult<G::Move> {
        match &mut self.inner {
            MctsKind::Puct(core) => core.search_inner(game, context, mode),
            MctsKind::Gumbel(core) => core.search_inner(game, context, mode),
        }
    }
}

impl CoreConfig {
    fn to_mcts(self, variant: MctsVariant) -> MctsConfig {
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

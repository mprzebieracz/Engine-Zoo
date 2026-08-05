use super::core::{LeafBatch, MctsCore, PendingLeaf};
use super::{
    CompletedQConfig, FullGumbelConfig, Node, RootAction, SearchDiagnostics, SearchResult,
};
use crate::PositionValue;
use engine_core::agent::PolicyMode;
use engine_core::game::GameState;
use rand::Rng;

/// Strict Full Gumbel search: root Gumbel sequential halving and Mctx's
/// improved-policy-deficit selector at interior nodes. Every leaf is completed
/// before another one is selected, so it never shares PUCT batching semantics.
#[derive(Debug)]
pub(super) struct FullGumbel {
    pub(super) config: FullGumbelConfig,
    pub(super) root_actions: Vec<RootAction>,
    scratch: GumbelScratch,
}

impl From<FullGumbelConfig> for FullGumbel {
    fn from(config: FullGumbelConfig) -> Self {
        Self {
            config,
            root_actions: Vec::new(),
            scratch: GumbelScratch::default(),
        }
    }
}

impl<G, E, R> MctsCore<G, E, R, FullGumbel>
where
    G: GameState + Clone,
    E: crate::PolicyValueEvaluator<G>,
    R: crate::SearchRules<G>,
{
    pub(super) fn search_inner(
        &mut self,
        game: &G,
        context: R::Context<'_>,
        _mode: PolicyMode,
        simulations: usize,
        max_considered_actions: usize,
    ) -> Result<SearchResult<G::Move>, super::SearchError> {
        self.clear_tree_common();
        self.nodes
            .push(Node::new(None, None, 0.0, 0.0, false, PositionValue::DRAW));
        let mut root = game.clone();
        self.rules.reset_path(context, game, &mut self.path_state);
        let root_rule = self.rules.enter_state(
            context,
            &mut root,
            &mut self.path_state,
            &mut self.nodes[0].meta,
        );
        if let Some(value) = self.terminal_value(&root, root_rule) {
            return Err(super::SearchError::TerminalRoot { value });
        }
        if root.legal_moves().next().is_none() {
            return Err(super::SearchError::NoLegalMoves);
        }
        let (evaluation, root_stats) = self
            .evaluate_position(&root)
            .map_err(super::SearchError::Evaluation)?;
        if evaluation.legal().is_empty() {
            return Err(super::SearchError::NoLegalMoves);
        }
        let root_value = evaluation.value;
        self.nodes[0].raw_value = root_value;
        self.build_policy_from(evaluation.legal(), evaluation.logits(), false, None);
        self.expand(0);
        self.init_root_actions(simulations, max_considered_actions);

        let mut diagnostics = SearchDiagnostics {
            backend_evaluations: root_stats.backend_evaluations,
            evaluation_cache_hits: root_stats.cache_hits,
            evaluation_cache_misses: root_stats.backend_evaluations,
            ..SearchDiagnostics::default()
        };
        gumbel_visit_schedule_into(
            self.variant.root_actions.len(),
            simulations,
            &mut self.variant.scratch.schedule_visits,
            &mut self.variant.scratch.visit_schedule,
        );
        let mut batch = LeafBatch::with_capacity(1);
        for schedule_index in 0..self.variant.scratch.visit_schedule.len() {
            let considered_visits = self.variant.scratch.visit_schedule[schedule_index];
            let child = self
                .best_root_action(considered_visits)
                .expect("sequential-halving schedule must select an action");
            let (leaf, state, depth) = self.descend(game, child, context, |core, node| {
                core.select_interior_action(node)
            });
            self.reserve_path(leaf);
            batch.leaves.push(PendingLeaf {
                node: leaf,
                game: state,
            });
            diagnostics.max_depth = diagnostics.max_depth.max(depth);
            self.finish_leaf_batch(&mut batch, &mut diagnostics)
                .map_err(super::SearchError::Evaluation)?;
        }
        let selected = self.select_root_winner();
        let selected_move = self.nodes[selected as usize]
            .move_from_parent
            .expect("root action must be a child move");
        let policy = self.root_improved_policy();
        let root_value = self.nodes[0].completed_q().unwrap_or(root_value);
        diagnostics.nodes_created = self.nodes.len();

        Ok(SearchResult {
            policy,
            selected_move,
            root_value,
            diagnostics,
        })
    }

    fn init_root_actions(&mut self, simulations: usize, max_considered_actions: usize) {
        self.variant.root_actions.clear();
        let root = &self.nodes[0];
        let children = root.first_child..root.first_child + u32::from(root.num_children);
        for node in children {
            let logit = self.nodes[node as usize].logit;
            self.variant.root_actions.push(RootAction {
                node,
                gumbel_logit: logit
                    + self.variant.config.root.gumbel_scale * sample_gumbel(&mut self.rng),
            });
        }
        self.variant
            .root_actions
            .sort_unstable_by(|left, right| right.gumbel_logit.total_cmp(&left.gumbel_logit));
        let limit = max_considered_actions
            .min(simulations)
            .min(self.variant.root_actions.len());
        self.variant.root_actions.truncate(limit);
    }

    fn best_root_action(&mut self, considered_visits: u32) -> Option<u32> {
        prepare_transformed_q(
            &self.nodes,
            0,
            self.variant.config.root.completed_q,
            &mut self.variant.scratch,
        );
        let q = &self.variant.scratch.transformed_q;
        self.variant
            .root_actions
            .iter()
            .filter(|action| self.nodes[action.node as usize].completed_visits == considered_visits)
            .max_by(|left, right| {
                let left_q = q[(left.node - self.nodes[0].first_child) as usize];
                let right_q = q[(right.node - self.nodes[0].first_child) as usize];
                (left.gumbel_logit + left_q).total_cmp(&(right.gumbel_logit + right_q))
            })
            .map(|action| action.node)
    }

    fn select_root_winner(&mut self) -> u32 {
        let visits = self
            .variant
            .root_actions
            .iter()
            .map(|action| self.nodes[action.node as usize].completed_visits)
            .max()
            .expect("root must have candidates");
        self.best_root_action(visits)
            .expect("root candidate at maximum completed visits")
    }

    fn select_interior_action(&mut self, node: u32) -> Option<u32> {
        prepare_transformed_q(
            &self.nodes,
            node,
            self.variant.config.root.completed_q,
            &mut self.variant.scratch,
        );
        self.variant.scratch.logits.clear();
        let parent = &self.nodes[node as usize];
        let children = parent.first_child..parent.first_child + u32::from(parent.num_children);
        for (index, child) in children.enumerate() {
            self.variant
                .scratch
                .logits
                .push(self.nodes[child as usize].logit + self.variant.scratch.transformed_q[index]);
        }
        softmax_into(
            &self.variant.scratch.logits,
            &mut self.variant.scratch.probabilities,
        );
        let total_visits: u32 = self
            .child_indices(node)
            .map(|child| self.nodes[child as usize].completed_visits)
            .sum();
        self.child_indices(node)
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                let left_deficit = self.variant.scratch.probabilities[*left_index]
                    - self.nodes[*left as usize].completed_visits as f32
                        / (1 + total_visits) as f32;
                let right_deficit = self.variant.scratch.probabilities[*right_index]
                    - self.nodes[*right as usize].completed_visits as f32
                        / (1 + total_visits) as f32;
                left_deficit.total_cmp(&right_deficit)
            })
            .map(|(_, child)| child)
    }

    fn root_improved_policy(&mut self) -> Vec<(G::Move, f32)> {
        prepare_transformed_q(
            &self.nodes,
            0,
            self.variant.config.root.completed_q,
            &mut self.variant.scratch,
        );
        self.variant.scratch.logits.clear();
        let root = &self.nodes[0];
        let children = root.first_child..root.first_child + u32::from(root.num_children);
        for (index, child) in children.enumerate() {
            self.variant
                .scratch
                .logits
                .push(self.nodes[child as usize].logit + self.variant.scratch.transformed_q[index]);
        }
        softmax_into(
            &self.variant.scratch.logits,
            &mut self.variant.scratch.probabilities,
        );
        self.child_indices(0)
            .zip(self.variant.scratch.probabilities.iter().copied())
            .map(|(child, probability)| {
                (
                    self.nodes[child as usize]
                        .move_from_parent
                        .expect("root child must have move"),
                    probability,
                )
            })
            .collect()
    }

    fn child_indices(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        let node = &self.nodes[node as usize];
        node.first_child..node.first_child + u32::from(node.num_children)
    }
}

#[derive(Debug, Default)]
pub(super) struct GumbelScratch {
    q_values: Vec<Option<PositionValue>>,
    visits: Vec<u32>,
    priors: Vec<f32>,
    pub(super) transformed_q: Vec<f32>,
    pub(super) logits: Vec<f32>,
    pub(super) probabilities: Vec<f32>,
    schedule_visits: Vec<u32>,
    visit_schedule: Vec<u32>,
}

pub(super) fn prepare_transformed_q<M, Meta>(
    nodes: &[Node<M, Meta>],
    node: u32,
    config: CompletedQConfig,
    scratch: &mut GumbelScratch,
) where
    Meta: Default,
{
    scratch.q_values.clear();
    scratch.visits.clear();
    scratch.priors.clear();
    let parent = &nodes[node as usize];
    let children = parent.first_child..parent.first_child + u32::from(parent.num_children);
    for child in children {
        let child = &nodes[child as usize];
        scratch
            .q_values
            .push(child.completed_q().map(PositionValue::flipped));
        scratch.visits.push(child.completed_visits);
        scratch.priors.push(child.prior);
    }
    transform_completed_q_into(
        parent.raw_value,
        &scratch.q_values,
        &scratch.visits,
        &scratch.priors,
        config,
        &mut scratch.transformed_q,
    );
}

pub(super) fn mixed_value(
    raw_value: PositionValue,
    q_values: &[Option<PositionValue>],
    completed_visits: &[u32],
    prior_probs: &[f32],
) -> PositionValue {
    debug_assert_eq!(q_values.len(), completed_visits.len());
    debug_assert_eq!(q_values.len(), prior_probs.len());
    let mut weighted_q = 0.0;
    let mut prior_mass = 0.0;
    let mut visits = 0u32;
    for ((q, &count), &prior) in q_values.iter().zip(completed_visits).zip(prior_probs) {
        if let Some(q) = q.filter(|_| count > 0) {
            // Mctx weights each *visited action* by its prior, then uses the
            // total visit count only for the raw-value interpolation below.
            // Multiplying this weight by `count` would bias the completed
            // value toward frequently selected actions twice.
            let weight = prior.max(f32::MIN_POSITIVE);
            weighted_q += weight * q.as_f32();
            prior_mass += weight;
            visits += count;
        }
    }
    if visits == 0 || prior_mass <= 0.0 {
        return raw_value;
    }
    PositionValue::from_finite_clamped(
        (raw_value.as_f32() + visits as f32 * weighted_q / prior_mass) / (visits as f32 + 1.0),
    )
    .expect("finite search values must produce a finite mixed value")
}

fn completed_q_values_into(
    raw_value: PositionValue,
    q_values: &[Option<PositionValue>],
    completed_visits: &[u32],
    prior_probs: &[f32],
    use_mixed_value: bool,
    values: &mut Vec<f32>,
) {
    let mixed = if use_mixed_value {
        mixed_value(raw_value, q_values, completed_visits, prior_probs)
    }
    else {
        raw_value
    };
    values.clear();
    values.extend(
        q_values
            .iter()
            .zip(completed_visits)
            .map(|(&q, &visits)| q.filter(|_| visits > 0).unwrap_or(mixed).as_f32()),
    );
}

pub(super) fn rescale_q_values(values: &mut [f32], epsilon: f32) {
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range > epsilon {
        for value in values {
            *value = (*value - min) / range;
        }
    }
    else {
        values.fill(0.0);
    }
}

fn transform_completed_q_into(
    raw_value: PositionValue,
    q_values: &[Option<PositionValue>],
    completed_visits: &[u32],
    prior_probs: &[f32],
    config: CompletedQConfig,
    values: &mut Vec<f32>,
) {
    completed_q_values_into(
        raw_value,
        q_values,
        completed_visits,
        prior_probs,
        config.use_mixed_value,
        values,
    );
    if config.rescale_values {
        rescale_q_values(values, config.epsilon);
    }
    let max_visits = completed_visits.iter().copied().max().unwrap_or(0) as f32;
    let scale = (config.maxvisit_init + max_visits) * config.value_scale;
    for value in values.iter_mut() {
        *value *= scale;
    }
}

pub(super) fn softmax_into(logits: &[f32], probabilities: &mut Vec<f32>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    probabilities.clear();
    probabilities.extend(logits.iter().map(|logit| (*logit - max).exp()));
    let total: f32 = probabilities.iter().sum();
    debug_assert!(total.is_finite() && total > 0.0);
    for probability in probabilities.iter_mut() {
        *probability /= total;
    }
}

fn ceil_log2(n: usize) -> usize {
    usize::BITS as usize - (n.saturating_sub(1).leading_zeros() as usize)
}

#[cfg(test)]
pub(super) fn gumbel_visit_schedule(num_considered: usize, simulations: usize) -> Vec<u32> {
    let mut visits = Vec::with_capacity(num_considered);
    let mut sequence = Vec::with_capacity(simulations);
    gumbel_visit_schedule_into(num_considered, simulations, &mut visits, &mut sequence);
    sequence
}

fn gumbel_visit_schedule_into(
    num_considered: usize,
    simulations: usize,
    visits: &mut Vec<u32>,
    sequence: &mut Vec<u32>,
) {
    assert!(
        num_considered > 0,
        "Gumbel requires at least one considered action"
    );
    if num_considered == 1 {
        sequence.clear();
        sequence.extend((0..simulations).map(|visit| visit as u32));
        return;
    }
    let phases = ceil_log2(num_considered).max(1);
    visits.clear();
    visits.resize(num_considered, 0);
    let mut active = num_considered;
    sequence.clear();
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
}

pub(super) fn sample_gumbel<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let uniform = rng.random_range(f32::MIN_POSITIVE..1.0);
    -(-uniform.ln()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mctx_mixed_value_oracle(
        raw: f32,
        q_values: &[Option<f32>],
        visits: &[u32],
        priors: &[f32],
    ) -> f32 {
        let total_visits: u32 = visits.iter().sum();
        if total_visits == 0 {
            return raw;
        }
        let prior_mass: f32 = priors
            .iter()
            .zip(visits)
            .filter(|(_, &count)| count > 0)
            .map(|(&prior, _)| prior.max(f32::MIN_POSITIVE))
            .sum();
        let weighted_q: f32 = q_values
            .iter()
            .zip(visits)
            .zip(priors)
            .filter(|&((&_, &count), &_)| count > 0)
            .map(|((&q, &_), &prior)| {
                q.expect("visited actions have q values") * prior.max(f32::MIN_POSITIVE)
                    / prior_mass
            })
            .sum();
        (raw + total_visits as f32 * weighted_q) / (total_visits + 1) as f32
    }

    #[test]
    fn completed_q_uses_raw_value_without_visits() {
        let raw = PositionValue::new(0.4).unwrap();
        assert_eq!(mixed_value(raw, &[None, None], &[0, 0], &[0.5, 0.5]), raw);
    }

    #[test]
    fn mixed_value_matches_independent_mctx_formula() {
        let raw = 0.2;
        let q_values = [Some(0.9), Some(-0.5), None];
        let visits = [7, 1, 0];
        let priors = [0.2, 0.8, 0.0];
        let expected = mctx_mixed_value_oracle(raw, &q_values, &visits, &priors);
        let actual = mixed_value(
            PositionValue::new(raw).unwrap(),
            &q_values.map(|value| value.map(|value| PositionValue::new(value).unwrap())),
            &visits,
            &priors,
        )
        .as_f32();
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }

    #[test]
    fn rescaling_equal_values_is_finite() {
        let mut values = [0.3, 0.3];
        rescale_q_values(&mut values, 1e-8);
        assert_eq!(values, [0.0, 0.0]);
    }

    #[test]
    fn reusable_math_scratch_matches_completed_q_and_softmax_formulas() {
        let raw = PositionValue::new(0.2).unwrap();
        let q_values = [Some(PositionValue::new(0.5).unwrap()), None];
        let visits = [2, 0];
        let priors = [0.25, 0.75];
        let config = CompletedQConfig {
            rescale_values: false,
            ..Default::default()
        };
        let mixed = mixed_value(raw, &q_values, &visits, &priors).as_f32();
        let scale = (config.maxvisit_init + 2.0) * config.value_scale;
        let mut values = vec![f32::NAN; 8];

        transform_completed_q_into(raw, &q_values, &visits, &priors, config, &mut values);

        assert_eq!(values, [0.5 * scale, mixed * scale]);

        let mut probabilities = vec![f32::NAN; 8];
        softmax_into(&[0.0, 2.0f32.ln()], &mut probabilities);
        assert!((probabilities[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((probabilities[1] - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(probabilities.len(), 2);
    }

    #[test]
    fn one_action_schedule_is_consecutive() {
        assert_eq!(gumbel_visit_schedule(1, 8), (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn sequential_halving_matches_checked_in_reference_schedules() {
        for line in include_str!("../../fixtures/full_gumbel_schedule.csv").lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut fields = line.split(',');
            let actions = fields.next().unwrap().parse().unwrap();
            let simulations = fields.next().unwrap().parse().unwrap();
            let expected = fields
                .next()
                .unwrap()
                .split_whitespace()
                .map(str::parse)
                .collect::<Result<Vec<u32>, _>>()
                .unwrap();
            assert!(fields.next().is_none(), "invalid fixture row: {line}");
            assert_eq!(
                gumbel_visit_schedule(actions, simulations),
                expected,
                "{line}"
            );
        }
    }
}

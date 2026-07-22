use super::representation::{Action, AlphaZeroRepresentation};
use engine_core::GameState;
use rand::seq::index;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::{Mutex, RwLock};
use tch::Tensor;

const POLICY_EPSILON: f32 = 1e-12;

/// Canonical sparse probability target. Entries are sorted by action, merged,
/// stripped of tiny mass, and normalized once at the replay boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct SparsePolicy(Vec<(Action, f32)>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparsePolicyError(&'static str);

impl std::fmt::Display for SparsePolicyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for SparsePolicyError {}

impl SparsePolicy {
    pub fn new(
        entries: impl IntoIterator<Item = (Action, f32)>,
    ) -> Result<Self, SparsePolicyError> {
        let mut entries: Vec<_> = entries.into_iter().collect();
        if entries
            .iter()
            .any(|(_, probability)| !probability.is_finite() || *probability < 0.0)
        {
            return Err(SparsePolicyError(
                "replay policy probabilities must be finite and non-negative",
            ));
        }
        entries.sort_unstable_by(
            |(left_action, left_probability), (right_action, right_probability)| {
                left_action
                    .as_u32()
                    .cmp(&right_action.as_u32())
                    .then_with(|| left_probability.total_cmp(right_probability))
            },
        );
        let mut canonical = Vec::with_capacity(entries.len());
        for (action, probability) in entries {
            if let Some((stored_action, stored_probability)) = canonical.last_mut() {
                if *stored_action == action {
                    *stored_probability += probability;
                    continue;
                }
            }
            canonical.push((action, probability));
        }
        canonical.retain(|(_, probability)| *probability > POLICY_EPSILON);
        let total: f32 = canonical.iter().map(|(_, probability)| probability).sum();
        if !canonical.is_empty() && (!total.is_finite() || total <= POLICY_EPSILON) {
            return Err(SparsePolicyError(
                "replay policy must have positive finite mass",
            ));
        }
        if total > 0.0 {
            for (_, probability) in &mut canonical {
                *probability /= total;
            }
        }
        Ok(Self(canonical))
    }

    fn validate_for(
        &self,
        action_size: usize,
        policy_weight: f32,
    ) -> Result<(), SparsePolicyError> {
        if self.iter().any(|(action, _)| action.index() >= action_size) {
            return Err(SparsePolicyError("replay policy action is out of range"));
        }
        if policy_weight > 0.0 && self.is_empty() {
            return Err(SparsePolicyError(
                "non-zero policy weight requires a non-empty replay policy",
            ));
        }
        Ok(())
    }
}

impl std::ops::Deref for SparsePolicy {
    type Target = [(Action, f32)];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> IntoIterator for &'a SparsePolicy {
    type Item = &'a (Action, f32);
    type IntoIter = std::slice::Iter<'a, (Action, f32)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl FromIterator<(Action, f32)> for SparsePolicy {
    fn from_iter<T: IntoIterator<Item = (Action, f32)>>(entries: T) -> Self {
        Self::new(entries).expect("self-play must create a valid sparse policy")
    }
}

impl From<Vec<(Action, f32)>> for SparsePolicy {
    fn from(entries: Vec<(Action, f32)>) -> Self {
        Self::from_iter(entries)
    }
}

impl Serialize for SparsePolicy {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SparsePolicy {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let entries = Vec::<(Action, f32)>::deserialize(deserializer)?;
        Self::new(entries).map_err(serde::de::Error::custom)
    }
}

/// A game result from the player-to-move perspective of a replay state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Win,
    Draw,
    Loss,
}

impl Outcome {
    pub const fn flipped(self) -> Self {
        match self {
            Self::Win => Self::Loss,
            Self::Draw => Self::Draw,
            Self::Loss => Self::Win,
        }
    }

    pub const fn scalar(self) -> f32 {
        match self {
            Self::Win => 1.0,
            Self::Draw => 0.0,
            Self::Loss => -1.0,
        }
    }

    pub const fn wdl_index(self) -> i64 {
        match self {
            Self::Win => 0,
            Self::Draw => 1,
            Self::Loss => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingWeights {
    pub policy: f32,
    pub value: f32,
}

impl Default for TrainingWeights {
    fn default() -> Self {
        Self {
            policy: 1.0,
            value: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchKind {
    Full,
    Fast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleMetadata {
    pub search_kind: SearchKind,
    pub simulations: u32,
    pub model_generation: u64,
    pub game_id: u64,
    pub ply: u16,
}

impl Default for SampleMetadata {
    fn default() -> Self {
        Self {
            search_kind: SearchKind::Full,
            simulations: 0,
            model_generation: 0,
            game_id: 0,
            ply: 0,
        }
    }
}

/// Compact self-play data. The state remains in the game representation until
/// the trainer samples it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplaySample<S> {
    pub state: S,
    pub policy: SparsePolicy,
    pub outcome: Outcome,
    pub weights: TrainingWeights,
    pub metadata: SampleMetadata,
}

struct Inner<S> {
    entries: Vec<Option<ReplaySample<S>>>,
    ptr: usize,
    len: usize,
}

struct SampleScratch {
    states: Vec<f32>,
    policy_actions: Vec<i64>,
    policy_probabilities: Vec<f32>,
    policy_rows: Vec<i64>,
    policy_weights: Vec<f32>,
    value_weights: Vec<f32>,
    outcomes: Vec<i64>,
}

#[derive(Debug)]
pub struct SparsePolicyBatch {
    pub actions: Tensor,
    pub probabilities: Tensor,
    pub rows: Tensor,
    pub offsets: Vec<i64>,
}

#[derive(Debug)]
pub struct ReplayBatch {
    pub states: Tensor,
    pub policies: SparsePolicyBatch,
    pub outcomes: Tensor,
    pub policy_weights: Tensor,
    pub value_weights: Tensor,
}

/// A fixed-capacity, compact ring buffer. It deliberately does not know how a
/// state is encoded; the representation performs that work only for sampled
/// entries.
pub struct ReplayBuffer<S> {
    capacity: usize,
    action_size: usize,
    inner: RwLock<Inner<S>>,
    sample_scratch: Mutex<SampleScratch>,
}

impl<S> ReplayBuffer<S> {
    pub fn new(capacity: usize, action_size: usize) -> Self {
        assert!(capacity > 0, "replay capacity must be positive");
        Self {
            capacity,
            action_size,
            inner: RwLock::new(Inner {
                entries: (0..capacity).map(|_| None).collect(),
                ptr: 0,
                len: 0,
            }),
            sample_scratch: Mutex::new(SampleScratch {
                states: Vec::new(),
                policy_actions: Vec::new(),
                policy_probabilities: Vec::new(),
                policy_rows: Vec::new(),
                policy_weights: Vec::new(),
                value_weights: Vec::new(),
                outcomes: Vec::new(),
            }),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap().len
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn add(&self, samples: impl IntoIterator<Item = ReplaySample<S>>) {
        let mut inner = self.inner.write().unwrap();
        for mut sample in samples {
            validate_sample(&mut sample, self.action_size);
            let slot = inner.ptr;
            inner.entries[slot] = Some(sample);
            inner.ptr = (slot + 1) % self.capacity;
            inner.len = (inner.len + 1).min(self.capacity);
        }
    }
}

impl<S: Clone> ReplayBuffer<S> {
    /// Kept primarily for diagnostics and tests; entries are returned oldest first.
    pub fn export_filled(&self) -> Vec<ReplaySample<S>> {
        let inner = self.inner.read().unwrap();
        let start = if inner.len == self.capacity {
            inner.ptr
        } else {
            0
        };
        (0..inner.len)
            .map(|offset| {
                inner.entries[(start + offset) % self.capacity]
                    .as_ref()
                    .unwrap()
                    .clone()
            })
            .collect()
    }
}

impl<S: GameState + Clone> ReplayBuffer<S> {
    pub fn sample<R: AlphaZeroRepresentation<S>>(
        &self,
        batch_size: usize,
        representation: &R,
    ) -> Option<ReplayBatch> {
        let inner = self.inner.read().unwrap();
        let batch_size = batch_size.min(inner.len);
        if batch_size == 0 {
            return None;
        }
        let mut scratch = self.sample_scratch.lock().unwrap();
        let state_size = R::state_size();
        scratch.states.resize(batch_size * state_size, 0.0);
        scratch.policy_actions.clear();
        scratch.policy_probabilities.clear();
        scratch.policy_rows.clear();
        scratch.policy_weights.clear();
        scratch.value_weights.clear();
        scratch.outcomes.clear();
        let mut offsets = Vec::with_capacity(batch_size + 1);
        offsets.push(0);
        for (row, index) in index::sample(&mut rand::rng(), inner.len, batch_size)
            .into_iter()
            .enumerate()
        {
            let sample = inner.entries[index].as_ref().expect("filled replay slot");
            representation.encode_state(
                &sample.state,
                &mut scratch.states[row * state_size..(row + 1) * state_size],
            );
            for &(action, probability) in &sample.policy {
                scratch.policy_actions.push(action.as_u32() as i64);
                scratch.policy_probabilities.push(probability);
                scratch.policy_rows.push(row as i64);
            }
            scratch.policy_weights.push(sample.weights.policy);
            scratch.value_weights.push(sample.weights.value);
            scratch.outcomes.push(sample.outcome.wdl_index());
            offsets.push(scratch.policy_actions.len() as i64);
        }
        drop(inner);
        Some(ReplayBatch {
            states: Tensor::from_slice(&scratch.states)
                .view([batch_size as i64, state_size as i64]),
            policies: SparsePolicyBatch {
                actions: Tensor::from_slice(&scratch.policy_actions),
                probabilities: Tensor::from_slice(&scratch.policy_probabilities),
                rows: Tensor::from_slice(&scratch.policy_rows),
                offsets,
            },
            outcomes: Tensor::from_slice(&scratch.outcomes),
            policy_weights: Tensor::from_slice(&scratch.policy_weights),
            value_weights: Tensor::from_slice(&scratch.value_weights),
        })
    }
}

fn validate_sample<S>(sample: &mut ReplaySample<S>, action_size: usize) {
    assert!(
        sample.weights.policy.is_finite() && sample.weights.policy >= 0.0,
        "policy weight must be finite and non-negative"
    );
    assert!(
        sample.weights.value.is_finite() && sample.weights.value >= 0.0,
        "value weight must be finite and non-negative"
    );
    sample
        .policy
        .validate_for(action_size, sample.weights.policy)
        .expect("invalid replay policy");
}

#[cfg(test)]
mod tests;

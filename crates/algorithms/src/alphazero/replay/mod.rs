use engine_core::game::Action;
use std::sync::{Mutex, RwLock};
use tch::Tensor;

/// One training example.
#[derive(Clone)]
pub struct Transition {
    pub state: Vec<f32>,
    pub policy: Vec<(Action, f32)>,
    pub reward: f32,
}

struct Inner {
    states: Vec<f32>,
    policies: Vec<Vec<(Action, f32)>>,
    rewards: Vec<f32>,
    ptr: usize,
    len: usize,
}

struct SampleScratch {
    states: Vec<f32>,
    policy_actions: Vec<i64>,
    policy_probabilities: Vec<f32>,
    policy_rows: Vec<i64>,
    policy_offsets: Vec<i64>,
    rewards: Vec<f32>,
}

/// Sparse policy targets for a sampled replay batch.
///
/// Entries are packed by row. `offsets[row]..offsets[row + 1]` addresses one
/// position's targets, while `rows` permits a single gather from flattened
/// network logits without constructing a dense action-space tensor.
#[derive(Debug)]
pub struct SparsePolicyBatch {
    pub actions: Tensor,
    pub probabilities: Tensor,
    pub rows: Tensor,
    pub offsets: Vec<i64>,
}

/// A sampled replay minibatch. States and rewards remain dense; policies stay
/// sparse from self-play storage through the training loss.
#[derive(Debug)]
pub struct ReplayBatch {
    pub states: Tensor,
    pub policies: SparsePolicyBatch,
    pub rewards: Tensor,
}

/// Fixed-capacity ring buffer shared by the self-play threads (writers) and
/// the trainer (reader).
pub struct ReplayBuffer {
    capacity: usize,
    state_size: usize,
    action_size: usize,
    inner: RwLock<Inner>,
    sample_scratch: Mutex<SampleScratch>,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, state_size: usize, action_size: usize) -> Self {
        assert!(capacity > 0);
        ReplayBuffer {
            capacity,
            state_size,
            action_size,
            inner: RwLock::new(Inner {
                states: vec![0.0; capacity * state_size],
                policies: (0..capacity).map(|_| Vec::new()).collect(),
                rewards: vec![0.0; capacity],
                ptr: 0,
                len: 0,
            }),
            sample_scratch: Mutex::new(SampleScratch {
                states: Vec::new(),
                policy_actions: Vec::new(),
                policy_probabilities: Vec::new(),
                policy_rows: Vec::new(),
                policy_offsets: Vec::new(),
                rewards: Vec::new(),
            }),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap().len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn add(&self, transitions: Vec<Transition>) {
        let mut inner = self.inner.write().unwrap();
        for mut t in transitions {
            assert_eq!(t.state.len(), self.state_size, "invalid replay state size");
            let mut canonical_policy = Vec::with_capacity(t.policy.len());
            for (action, probability) in t.policy.drain(..) {
                assert!(
                    (action as usize) < self.action_size,
                    "replay policy action is out of range"
                );
                assert!(
                    probability.is_finite() && probability >= 0.0,
                    "replay policy probability must be finite and non-negative"
                );
                if let Some((_, stored)) = canonical_policy
                    .iter_mut()
                    .find(|(stored_action, _)| *stored_action == action)
                {
                    // Preserve the dense replay representation's historical
                    // last-write-wins behavior for duplicate action entries.
                    *stored = probability;
                }
                else {
                    canonical_policy.push((action, probability));
                }
            }
            let ptr = inner.ptr;
            let state_start = ptr * self.state_size;
            inner.states[state_start..state_start + self.state_size].copy_from_slice(&t.state);
            inner.policies[ptr].clear();
            inner.policies[ptr].extend(canonical_policy);
            inner.rewards[ptr] = t.reward;
            inner.ptr = (ptr + 1) % self.capacity;
            if inner.len < self.capacity {
                inner.len += 1;
            }
        }
    }

    /// Returns a snapshot of all transitions currently in the buffer (sparse policies).
    pub fn export_filled(&self) -> Vec<Transition> {
        let inner = self.inner.read().unwrap();
        let n = inner.len;
        if n == 0 {
            return Vec::new();
        }

        let start = if n < self.capacity { 0 } else { inner.ptr };
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let idx = (start + i) % self.capacity;
            let state_start = idx * self.state_size;
            out.push(Transition {
                state: inner.states[state_start..state_start + self.state_size].to_vec(),
                policy: inner.policies[idx].clone(),
                reward: inner.rewards[idx],
            });
        }
        out
    }

    /// Samples up to `batch_size` distinct transitions. Policies remain packed
    /// sparse targets instead of being expanded to `[batch, action_size]`.
    /// Returns `None` if the buffer is empty; otherwise `b = min(batch_size, len)`.
    pub fn sample(&self, batch_size: usize) -> Option<ReplayBatch> {
        let inner = self.inner.read().unwrap();
        let n = inner.len;
        let b = batch_size.min(n);
        if b == 0 {
            return None;
        }

        let state_bytes = b * self.state_size;
        let mut scratch = self.sample_scratch.lock().unwrap();
        scratch.states.resize(state_bytes, 0.0);
        scratch.rewards.resize(b, 0.0);
        scratch.policy_actions.clear();
        scratch.policy_probabilities.clear();
        scratch.policy_rows.clear();
        scratch.policy_offsets.clear();
        scratch.policy_offsets.push(0);

        let indices = rand::seq::index::sample(&mut rand::rng(), n, b);
        for (row, idx) in indices.into_iter().enumerate() {
            let src_state = idx * self.state_size;
            let dst_state = row * self.state_size;
            scratch.states[dst_state..dst_state + self.state_size]
                .copy_from_slice(&inner.states[src_state..src_state + self.state_size]);
            for &(a, p) in &inner.policies[idx] {
                scratch.policy_actions.push(a as i64);
                scratch.policy_probabilities.push(p);
                scratch.policy_rows.push(row as i64);
            }
            let policy_end = scratch.policy_actions.len() as i64;
            scratch.policy_offsets.push(policy_end);
            scratch.rewards[row] = inner.rewards[idx];
        }
        drop(inner);

        Some(ReplayBatch {
            states: Tensor::from_slice(&scratch.states).view([b as i64, self.state_size as i64]),
            policies: SparsePolicyBatch {
                actions: Tensor::from_slice(&scratch.policy_actions),
                probabilities: Tensor::from_slice(&scratch.policy_probabilities),
                rows: Tensor::from_slice(&scratch.policy_rows),
                offsets: scratch.policy_offsets.clone(),
            },
            rewards: Tensor::from_slice(&scratch.rewards),
        })
    }
}

#[cfg(test)]
mod tests;

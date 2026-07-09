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
    policies: Vec<f32>,
    rewards: Vec<f32>,
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
                policies: Vec::new(),
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
        for t in transitions {
            debug_assert_eq!(t.state.len(), self.state_size);
            let ptr = inner.ptr;
            let state_start = ptr * self.state_size;
            inner.states[state_start..state_start + self.state_size].copy_from_slice(&t.state);
            inner.policies[ptr].clear();
            inner.policies[ptr].extend_from_slice(&t.policy);
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

    /// Samples up to `batch_size` distinct transitions and densifies them into
    /// (states [b, state_size], policies [b, action_size], rewards [b]) CPU
    /// tensors. Returns `None` if the buffer is empty; otherwise `b = min(batch_size, len)`.
    pub fn sample(&self, batch_size: usize) -> Option<(Tensor, Tensor, Tensor)> {
        let inner = self.inner.read().unwrap();
        let n = inner.len;
        let b = batch_size.min(n);
        if b == 0 {
            return None;
        }

        let state_bytes = b * self.state_size;
        let policy_bytes = b * self.action_size;
        let mut scratch = self.sample_scratch.lock().unwrap();
        scratch.states.resize(state_bytes, 0.0);
        scratch.policies.resize(policy_bytes, 0.0);
        scratch.rewards.resize(b, 0.0);
        scratch.policies[..policy_bytes].fill(0.0);

        let indices = rand::seq::index::sample(&mut rand::rng(), n, b);
        for (row, idx) in indices.into_iter().enumerate() {
            let src_state = idx * self.state_size;
            let dst_state = row * self.state_size;
            scratch.states[dst_state..dst_state + self.state_size]
                .copy_from_slice(&inner.states[src_state..src_state + self.state_size]);
            for &(a, p) in &inner.policies[idx] {
                scratch.policies[row * self.action_size + a as usize] = p;
            }
            scratch.rewards[row] = inner.rewards[idx];
        }

        Some((
            Tensor::from_slice(&scratch.states).view([b as i64, self.state_size as i64]),
            Tensor::from_slice(&scratch.policies).view([b as i64, self.action_size as i64]),
            Tensor::from_slice(&scratch.rewards),
        ))
    }
}

#[cfg(test)]
mod tests;

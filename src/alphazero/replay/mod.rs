use crate::game::Action;
use std::sync::{Mutex, RwLock};
use tch::Tensor;

/// One training example.
pub struct Transition {
    pub state: Vec<f32>,
    pub policy: Vec<(Action, f32)>,
    pub reward: f32,
}

struct Inner {
    buf: Vec<Option<Transition>>,
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
                buf: (0..capacity).map(|_| None).collect(),
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
            inner.buf[ptr] = Some(t);
            inner.ptr = (ptr + 1) % self.capacity;
            if inner.len < self.capacity {
                inner.len += 1;
            }
        }
    }

    /// Samples up to `batch_size` distinct transitions and densifies them into
    /// (states [b, state_size], policies [b, action_size], rewards [b]) CPU
    /// tensors. Returns fewer than `batch_size` rows if the buffer is small.
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
            let t = inner.buf[idx]
                .as_ref()
                .expect("sampled slot must be filled");
            scratch.states[row * self.state_size..(row + 1) * self.state_size]
                .copy_from_slice(&t.state);
            for &(a, p) in &t.policy {
                scratch.policies[row * self.action_size + a as usize] = p;
            }
            scratch.rewards[row] = t.reward;
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

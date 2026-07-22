use crate::PositionValue;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Immutable network output shared by cache hits and tree expansion.
pub(crate) struct CachedEvaluation<M> {
    legal: Arc<[M]>,
    logits: Arc<[f32]>,
    pub(super) value: PositionValue,
}

impl<M> CachedEvaluation<M> {
    pub(super) fn new(legal: Vec<M>, logits: Vec<f32>, value: PositionValue) -> Self {
        Self {
            legal: legal.into(),
            logits: logits.into(),
            value,
        }
    }

    pub(super) fn legal(&self) -> &[M] {
        &self.legal
    }

    pub(super) fn logits(&self) -> &[f32] {
        &self.logits
    }
}

impl<M> Clone for CachedEvaluation<M> {
    fn clone(&self) -> Self {
        Self {
            legal: Arc::clone(&self.legal),
            logits: Arc::clone(&self.logits),
            value: self.value,
        }
    }
}

struct EvalTableEntry<M> {
    hash: u64,
    cached: CachedEvaluation<M>,
}

struct CacheShard<M> {
    slots: Box<[Option<EvalTableEntry<M>>]>,
}

/// Fixed-capacity direct-mapped cache, sharded to avoid a lock per entry.
pub struct EvalTable<M> {
    shards: Box<[RwLock<CacheShard<M>>]>,
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
}

impl<M> EvalTable<M> {
    pub fn new(entries: usize) -> Self {
        let entries = entries.max(1);
        let shard_count = entries.min(64).next_power_of_two();
        let base_capacity = entries / shard_count;
        let remainder = entries % shard_count;
        let shards = (0..shard_count)
            .map(|shard| {
                let capacity = base_capacity + usize::from(shard < remainder);
                RwLock::new(CacheShard {
                    slots: (0..capacity.max(1)).map(|_| None).collect(),
                })
            })
            .collect();
        Self {
            shards,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
        }
    }

    fn shard_index(&self, hash: u64) -> usize {
        hash as usize & (self.shards.len() - 1)
    }

    pub(super) fn insert(&self, hash: u64, cached: CachedEvaluation<M>) {
        let shard = &self.shards[self.shard_index(hash)];
        let mut shard = shard.write().unwrap();
        let slot = hash as usize % shard.slots.len();
        shard.slots[slot] = Some(EvalTableEntry { hash, cached });
        self.inserts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn stats(&self) -> EvalTableStats {
        EvalTableStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
        }
    }
}

impl<M> EvalTable<M> {
    pub(super) fn get(&self, hash: u64) -> Option<CachedEvaluation<M>> {
        let shard = &self.shards[self.shard_index(hash)];
        let shard = shard.read().unwrap();
        let slot = hash as usize % shard.slots.len();
        let hit = shard.slots[slot]
            .as_ref()
            .filter(|entry| entry.hash == hash)
            .map(|entry| entry.cached.clone());
        if hit.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EvalTableStats {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
}

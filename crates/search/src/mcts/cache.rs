use crate::{EvaluationKey, PositionValue};
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
    key: EvaluationKey,
    cached: CachedEvaluation<M>,
}

struct CacheShard<M> {
    slots: Box<[Option<EvalTableEntry<M>>]>,
}

/// Fixed-capacity direct-mapped cache, sharded to avoid a lock per entry.
///
/// Production statistics use relaxed atomic increments. They are intentionally
/// always enabled: callers use them for lifetime observability, not a
/// transactionally consistent snapshot.
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

    fn mix(mut value: u64) -> u64 {
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn key_hash(key: EvaluationKey) -> u64 {
        Self::mix(key.state) ^ Self::mix(key.namespace.wrapping_add(0x9e37_79b9_7f4a_7c15))
    }

    fn shard_index(&self, key: EvaluationKey) -> usize {
        (Self::mix(key.namespace ^ key.state.rotate_left(17)) as usize) & (self.shards.len() - 1)
    }

    fn slot_index(hash: u64, slots: usize) -> usize {
        Self::mix(hash.rotate_left(29)) as usize % slots
    }

    pub(super) fn insert(&self, key: EvaluationKey, cached: CachedEvaluation<M>) {
        let hash = Self::key_hash(key);
        let shard = &self.shards[self.shard_index(key)];
        let mut shard = shard.write().unwrap();
        let slot = Self::slot_index(hash, shard.slots.len());
        shard.slots[slot] = Some(EvalTableEntry { key, cached });
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
    pub(super) fn get(&self, key: EvaluationKey) -> Option<CachedEvaluation<M>> {
        let hash = Self::key_hash(key);
        let shard = &self.shards[self.shard_index(key)];
        let shard = shard.read().unwrap();
        let slot = Self::slot_index(hash, shard.slots.len());
        let hit = shard.slots[slot]
            .as_ref()
            .filter(|entry| entry.key == key)
            .map(|entry| entry.cached.clone());
        if hit.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        else {
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

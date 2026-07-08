use super::super::evaluator::Evaluation;
use crate::game::Action;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

pub(crate) struct CachedEvaluation {
    pub(super) legal: Vec<Action>,
    pub(super) eval: Evaluation,
}

struct EvalTableEntry {
    hash: u64,
    cached: CachedEvaluation,
}

pub(crate) struct EvalTable {
    slots: Box<[RwLock<Option<EvalTableEntry>>]>,
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
}

impl EvalTable {
    pub(crate) fn new(entries: usize) -> Self {
        let slots = (0..entries.max(1)).map(|_| RwLock::new(None)).collect();
        EvalTable {
            slots,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
        }
    }

    pub(super) fn get(&self, hash: u64) -> Option<CachedEvaluation> {
        let slot = &self.slots[hash as usize % self.slots.len()];
        let guard = slot.read().unwrap();
        let hit = guard
            .as_ref()
            .filter(|entry| entry.hash == hash)
            .map(|entry| entry.cached.clone());
        if hit.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    pub(super) fn insert(&self, hash: u64, cached: CachedEvaluation) {
        let slot = &self.slots[hash as usize % self.slots.len()];
        *slot.write().unwrap() = Some(EvalTableEntry { hash, cached });
        self.inserts.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn stats(&self) -> EvalTableStats {
        EvalTableStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EvalTableStats {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
}

impl Clone for CachedEvaluation {
    fn clone(&self) -> Self {
        CachedEvaluation {
            legal: self.legal.clone(),
            eval: self.eval.clone(),
        }
    }
}

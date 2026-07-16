use super::zobrist::ZobristBuildHasher;
use std::collections::HashMap;

#[derive(Clone)]
pub struct RepetitionTracker {
    counts: HashMap<u64, u8, ZobristBuildHasher>,
}

impl RepetitionTracker {
    pub(crate) fn new(root: u64) -> Self {
        let mut tracker = Self {
            counts: HashMap::with_capacity_and_hasher(16, ZobristBuildHasher::default()),
        };
        tracker.record(root);
        tracker
    }

    pub(crate) fn record(&mut self, hash: u64) -> u8 {
        let count = self.counts.entry(hash).or_insert(0);
        *count += 1;
        *count
    }

    pub(crate) fn reset(&mut self, root: u64) {
        self.counts.clear();
        self.record(root);
    }

    pub(crate) fn current_count(&self, hash: u64) -> u8 {
        self.counts.get(&hash).copied().unwrap_or(0)
    }

    pub(crate) fn root_count(&self, hash: u64, root: u64) -> u8 {
        self.current_count(hash)
            .saturating_sub(u8::from(hash == root))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_non_root_and_reset_semantics() {
        let mut tracker = RepetitionTracker::new(10);
        assert_eq!(tracker.root_count(10, 10), 0);
        assert_eq!(tracker.root_count(20, 10), 0);

        tracker.record(20);
        tracker.record(10);
        assert_eq!(tracker.root_count(10, 10), 1);
        assert_eq!(tracker.root_count(20, 10), 1);

        tracker.reset(30);
        assert_eq!(tracker.root_count(10, 30), 0);
        assert_eq!(tracker.root_count(20, 30), 0);
        assert_eq!(tracker.root_count(30, 30), 0);
        assert_eq!(tracker.current_count(30), 1);
    }
}

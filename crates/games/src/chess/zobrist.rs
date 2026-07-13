use std::hash::{BuildHasherDefault, Hasher};

/// Hashes an already-random Zobrist key without running it through a second
/// general-purpose hash function.
#[derive(Default)]
pub(super) struct ZobristHasher(u64);

impl Hasher for ZobristHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        assert_eq!(bytes.len(), size_of::<u64>(), "Zobrist keys must be u64");
        self.0 = u64::from_ne_bytes(bytes.try_into().unwrap());
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

pub(super) type ZobristBuildHasher = BuildHasherDefault<ZobristHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::BuildHasher;

    #[test]
    fn preserves_zobrist_key() {
        let mut hasher = ZobristBuildHasher::default().build_hasher();
        hasher.write_u64(0xdead_beef_cafe_babe);
        assert_eq!(hasher.finish(), 0xdead_beef_cafe_babe);
    }
}

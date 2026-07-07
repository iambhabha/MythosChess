//! Transposition table — a large hash table that remembers the results of
//! previously searched positions, keyed by their Zobrist [`key`].
//!
//! During search the same position is reached through many different move
//! orders (transpositions). Caching a position's best move, score, search
//! depth and score-[`Bound`] lets alpha-beta reuse that work instead of
//! re-searching the subtree.
//!
//! The table is sized to a power-of-two number of entries, so the slot for a
//! key is a fast bitmask (`key & (len - 1)`) rather than a modulo. Because two
//! different keys can land in the same slot (an *index collision*), each entry
//! stores the full key it was written with and [`probe`] returns a hit only if
//! that stored key matches — the low bits alone are not enough.
//!
//! [`key`]: crate::position::Position::key
//! [`probe`]: TranspositionTable::probe

use crate::types::Move;

/// The kind of bound a stored score represents, from the point of view of the
/// alpha-beta window it was searched in.
///
/// - `Exact` — the score is the true value (a PV node, `alpha < score < beta`).
/// - `Lower` — a fail-high / beta cutoff: the true score is *at least* this
///   (a lower bound). Search stopped early because it was already good enough.
/// - `Upper` — a fail-low: the true score is *at most* this (an upper bound).
///   No move beat `alpha`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

/// One cached search result.
///
/// The `key` is the full Zobrist hash of the position, used to distinguish
/// entries that collide into the same slot. Scores are stored verbatim — the
/// search layer is responsible for any mate-distance (ply) adjustment on store
/// and probe; the table never touches them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TtEntry {
    /// Full Zobrist key of the position this entry describes.
    pub key: u64,
    /// Best move found (may be [`Move::NONE`] if none was recorded).
    pub best_move: Move,
    /// The (possibly bounded) search score.
    pub score: i32,
    /// The search depth this result was computed at, in plies.
    pub depth: i16,
    /// Whether `score` is exact, a lower bound, or an upper bound.
    pub bound: Bound,
}

/// A fixed-size, direct-mapped transposition table.
pub struct TranspositionTable {
    /// One slot per bucket; `None` means the slot has never been written.
    /// The length is always a power of two so [`index`] can mask instead of
    /// dividing.
    ///
    /// [`index`]: TranspositionTable::index
    entries: Vec<Option<TtEntry>>,
}

impl TranspositionTable {
    /// Allocate a table using at most `size_mb` megabytes.
    ///
    /// The entry count is the largest power of two whose total size fits in the
    /// budget, computed from `size_of::<Option<TtEntry>>()`. A minimum of one
    /// entry is always allocated so the mask arithmetic stays valid even for a
    /// zero-megabyte request.
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<Option<TtEntry>>();
        let budget_bytes = size_mb.saturating_mul(1024 * 1024);

        // How many entries fit, then round *down* to a power of two.
        let fits = budget_bytes / entry_size;
        let count = if fits < 1 {
            1
        } else {
            // Largest power of two <= fits.
            1usize << (usize::BITS - 1 - fits.leading_zeros())
        };

        TranspositionTable {
            entries: vec![None; count],
        }
    }

    /// The number of slots in the table (always a power of two).
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Always `false` — the table always has at least one slot. Present so the
    /// `len`/`is_empty` pair is idiomatic.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Empty every slot, discarding all cached results.
    pub fn clear(&mut self) {
        for slot in self.entries.iter_mut() {
            *slot = None;
        }
    }

    /// The slot a key maps to. Requires a power-of-two length so the mask
    /// `len - 1` selects the low bits of the key.
    #[inline]
    fn index(&self, key: u64) -> usize {
        (key as usize) & (self.entries.len() - 1)
    }

    /// Look up a position. Returns the entry only if the slot is filled *and*
    /// its stored key matches `key`, guarding against index collisions.
    pub fn probe(&self, key: u64) -> Option<&TtEntry> {
        let slot = &self.entries[self.index(key)];
        match slot {
            Some(entry) if entry.key == key => Some(entry),
            _ => None,
        }
    }

    /// Insert or replace the result for a position.
    ///
    /// Uses a depth-preferred, always-replace-on-empty policy: the slot is
    /// overwritten when it is empty or when the incoming `depth` is at least the
    /// stored depth. This keeps deeper (more expensive) results — including a
    /// deep result for a *colliding* position — while still letting a re-search
    /// of the same position at equal-or-greater depth refresh the entry.
    pub fn store(&mut self, key: u64, best_move: Move, score: i32, depth: i16, bound: Bound) {
        let idx = self.index(key);
        let replace = match &self.entries[idx] {
            None => true,
            Some(existing) => depth >= existing.depth,
        };
        if replace {
            self.entries[idx] = Some(TtEntry {
                key,
                best_move,
                score,
                depth,
                bound,
            });
        }
    }

    /// Approximate table occupancy in per-mille (0..=1000), for the UCI
    /// `info hashfull` field. Samples the first 1000 slots (or all of them, if
    /// the table is smaller) and reports the fraction that are filled.
    pub fn hashfull_permill(&self) -> usize {
        let sample = self.entries.len().min(1000);
        if sample == 0 {
            return 0;
        }
        let filled = self.entries[..sample]
            .iter()
            .filter(|slot| slot.is_some())
            .count();
        filled * 1000 / sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Move, Square};

    #[test]
    fn new_gives_power_of_two_count() {
        for &mb in &[1usize, 16] {
            let tt = TranspositionTable::new(mb);
            let len = tt.len();
            assert!(len > 0, "expected non-empty table for {mb} MB");
            assert!(
                len.is_power_of_two(),
                "expected power-of-two entry count for {mb} MB, got {len}"
            );
        }
    }

    #[test]
    fn store_then_probe_roundtrips() {
        let mut tt = TranspositionTable::new(1);
        let key = 0xDEAD_BEEF_1234_5678;
        let mv = Move::normal(Square::E2, Square::E4);
        tt.store(key, mv, 42, 7, Bound::Exact);

        let entry = tt.probe(key).expect("stored entry should be found");
        assert_eq!(entry.key, key);
        assert_eq!(entry.score, 42);
        assert_eq!(entry.depth, 7);
        assert_eq!(entry.best_move, mv);
        assert_eq!(entry.bound, Bound::Exact);
    }

    #[test]
    fn probe_rejects_colliding_key() {
        let mut tt = TranspositionTable::new(1);
        let len = tt.len() as u64;
        // Two keys with identical low bits (same slot) but different high bits.
        let key_a = 0x0000_0000_0000_0000 | 5;
        let key_b = (len * 3) | 5; // same low bits `& (len-1)`, different high bits
        assert_eq!(
            (key_a & (len - 1)),
            (key_b & (len - 1)),
            "test keys must map to the same slot"
        );
        assert_ne!(key_a, key_b);

        tt.store(key_a, Move::NONE, 10, 4, Bound::Exact);
        // The slot is filled with key_a; probing key_b must not return it.
        assert!(tt.probe(key_b).is_none());
        assert!(tt.probe(key_a).is_some());
    }

    #[test]
    fn depth_preferred_replacement() {
        let mut tt = TranspositionTable::new(1);
        let len = tt.len() as u64;
        let key_a = 5u64;
        let key_b = (len * 7) | 5; // same slot, different key
        assert_eq!(key_a & (len - 1), key_b & (len - 1));
        assert_ne!(key_a, key_b);

        // Store a deep entry for key_a.
        let deep_move = Move::normal(Square::D2, Square::D4);
        tt.store(key_a, deep_move, 100, 5, Bound::Lower);

        // A shallower entry for a *different* key at the same slot must NOT
        // clobber the depth-5 entry.
        tt.store(key_b, Move::normal(Square::G1, Square::F3), 20, 3, Bound::Upper);
        let entry = tt.probe(key_a).expect("depth-5 entry should survive");
        assert_eq!(entry.depth, 5);
        assert_eq!(entry.score, 100);
        assert_eq!(entry.best_move, deep_move);
        // key_b was rejected, so probing it finds nothing.
        assert!(tt.probe(key_b).is_none());

        // A deeper entry (depth 6) for a different key at the same slot DOES
        // replace it (new depth >= stored depth).
        let deeper_move = Move::normal(Square::B1, Square::C3);
        tt.store(key_b, deeper_move, 55, 6, Bound::Exact);
        assert!(tt.probe(key_a).is_none(), "deeper store should evict key_a");
        let entry = tt.probe(key_b).expect("depth-6 entry should be present");
        assert_eq!(entry.depth, 6);
        assert_eq!(entry.score, 55);
        assert_eq!(entry.best_move, deeper_move);
    }

    #[test]
    fn clear_empties_table() {
        let mut tt = TranspositionTable::new(1);
        let key = 0xABCD_1234_5678_9ABC;
        tt.store(key, Move::NONE, 1, 1, Bound::Exact);
        assert!(tt.probe(key).is_some());
        tt.clear();
        assert!(tt.probe(key).is_none());
        assert_eq!(tt.hashfull_permill(), 0);
    }

    #[test]
    fn hashfull_reports_occupancy() {
        let mut tt = TranspositionTable::new(1);
        assert_eq!(tt.hashfull_permill(), 0);
        // Fill a handful of distinct slots.
        for i in 0..500u64 {
            tt.store(i, Move::NONE, 0, 1, Bound::Exact);
        }
        assert!(tt.hashfull_permill() > 0);
    }
}

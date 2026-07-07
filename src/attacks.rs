//! Attack generation for every piece type.
//!
//! Two families of pieces need different machinery:
//!
//! * **Leapers** (knight, king, pawn) always attack the same relative squares,
//!   so we precompute a `[Bitboard; 64]` lookup table once and just index it.
//!
//! * **Sliders** (bishop, rook, queen) attack along rays that stop at the first
//!   blocker, so their attacks depend on the occupied squares. We use *magic
//!   bitboards*: for each square we mask off the relevant occupancy bits, hash
//!   them with a precomputed "magic" multiplier into a small dense index, and
//!   look the answer up in one flat table. The magic numbers are found at
//!   startup by trial-and-error, and every table entry is filled in using the
//!   slow-but-obviously-correct [`slow_sliding_attacks`] reference walker — the
//!   same walker the tests use to prove the fast path agrees everywhere.

use crate::bitboard::Bitboard;
use crate::types::{Color, Direction, Square};
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Ray directions for the two slider kinds.
// ---------------------------------------------------------------------------

const ROOK_DIRS: [Direction; 4] = [
    Direction::North,
    Direction::South,
    Direction::East,
    Direction::West,
];

const BISHOP_DIRS: [Direction; 4] = [
    Direction::NorthEast,
    Direction::NorthWest,
    Direction::SouthEast,
    Direction::SouthWest,
];

// ---------------------------------------------------------------------------
// Slow reference walker — the safety net.
// ---------------------------------------------------------------------------

/// Walk each ray from `sq` one step at a time, adding squares until we run off
/// the board or hit a blocker. The blocker's own square *is* included (that is
/// the square the slider can capture on). This is deliberately simple so it is
/// obviously correct; it seeds the magic tables and validates them in tests.
fn slow_sliding_attacks(sq: Square, occupied: Bitboard, directions: &[Direction]) -> Bitboard {
    let mut attacks = Bitboard::EMPTY;
    for &dir in directions {
        let mut current = sq;
        // `offset` returns `None` once we would leave the board (or wrap a file),
        // so this loop naturally terminates at the edge.
        while let Some(next) = current.offset(dir) {
            attacks.set(next);
            if occupied.contains(next) {
                break; // stop *after* including the blocker.
            }
            current = next;
        }
    }
    attacks
}

// ---------------------------------------------------------------------------
// Leapers: knight, king, pawn.
// ---------------------------------------------------------------------------

/// Build a leaper table by applying a set of `(file_delta, rank_delta)` steps to
/// every source square, keeping only steps that stay on the 8x8 board.
fn build_leaper_table(steps: &[(i8, i8)]) -> [Bitboard; Square::NUM] {
    let mut table = [Bitboard::EMPTY; Square::NUM];
    for (i, entry) in table.iter_mut().enumerate() {
        let sq = Square(i as u8);
        let f = sq.file() as i8;
        let r = sq.rank() as i8;
        let mut bb = Bitboard::EMPTY;
        for &(df, dr) in steps {
            let nf = f + df;
            let nr = r + dr;
            if (0..8).contains(&nf) && (0..8).contains(&nr) {
                bb.set(Square::make(nf as u8, nr as u8));
            }
        }
        *entry = bb;
    }
    table
}

static KNIGHT_TABLE: LazyLock<[Bitboard; Square::NUM]> = LazyLock::new(|| {
    build_leaper_table(&[
        (1, 2),
        (2, 1),
        (2, -1),
        (1, -2),
        (-1, -2),
        (-2, -1),
        (-2, 1),
        (-1, 2),
    ])
});

static KING_TABLE: LazyLock<[Bitboard; Square::NUM]> = LazyLock::new(|| {
    build_leaper_table(&[
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ])
});

/// White and black pawn attack tables, indexed `[color][square]`.
static PAWN_TABLE: LazyLock<[[Bitboard; Square::NUM]; Color::NUM]> = LazyLock::new(|| {
    // White pawns attack one rank up; black pawns one rank down.
    let white = build_leaper_table(&[(-1, 1), (1, 1)]);
    let black = build_leaper_table(&[(-1, -1), (1, -1)]);
    [white, black]
});

/// The squares a knight on `sq` attacks (its eight L-shaped jumps that stay on
/// the board). Independent of any other pieces.
#[inline]
pub fn knight_attacks(sq: Square) -> Bitboard {
    KNIGHT_TABLE[sq.index()]
}

/// The squares a king on `sq` attacks (the up-to-eight adjacent squares).
#[inline]
pub fn king_attacks(sq: Square) -> Bitboard {
    KING_TABLE[sq.index()]
}

/// The squares a pawn of `color` on `sq` attacks — the two forward diagonals
/// only. This is *captures*, not pushes: a pawn never captures straight ahead.
#[inline]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    PAWN_TABLE[color.index()][sq.index()]
}

// ---------------------------------------------------------------------------
// Sliders: magic bitboards.
// ---------------------------------------------------------------------------

/// Per-square magic-bitboard descriptor. To look up attacks:
/// `index = (((occupied & mask) * magic) >> shift) + offset` into the shared
/// attack table.
#[derive(Clone, Copy)]
struct Magic {
    mask: Bitboard,
    magic: u64,
    shift: u32,
    offset: usize,
}

impl Magic {
    #[inline]
    fn index(&self, occupied: Bitboard) -> usize {
        let relevant = (occupied & self.mask).0;
        (relevant.wrapping_mul(self.magic) >> self.shift) as usize + self.offset
    }
}

/// The relevant-occupancy mask for a slider on `sq`: every square along its rays
/// *excluding the board edges*. Edge squares never change what lies beyond them
/// (there is nothing beyond), so leaving them out shrinks the index space — this
/// is the defining trick of magic bitboards.
fn slider_mask(sq: Square, directions: &[Direction]) -> Bitboard {
    // Files A/H and ranks 1/8, minus the piece's own square's file/rank so we
    // don't wrongly drop relevant squares on the piece's own edge lines.
    let file_a = Bitboard(crate::bitboard::FILE_A_BB);
    let file_h = Bitboard(crate::bitboard::FILE_H_BB);
    let rank_1 = Bitboard(crate::bitboard::RANK_1_BB);
    let rank_8 = Bitboard(crate::bitboard::RANK_8_BB);

    let mut edges = Bitboard::EMPTY;
    edges |= file_a & !Bitboard::file_bb(sq);
    edges |= file_h & !Bitboard::file_bb(sq);
    edges |= rank_1 & !Bitboard::rank_bb(sq);
    edges |= rank_8 & !Bitboard::rank_bb(sq);

    // The full ray reach on an empty board, minus the edge squares.
    slow_sliding_attacks(sq, Bitboard::EMPTY, directions) & !edges
}

/// Enumerate the `2^n` occupancy subsets of an `n`-bit mask. `index` selects one
/// subset by treating its bits as the on/off state of each mask bit in order.
fn occupancy_for_index(index: usize, mask: Bitboard) -> Bitboard {
    let mut result = Bitboard::EMPTY;
    let bits: Vec<Square> = mask.collect();
    for (i, &sq) in bits.iter().enumerate() {
        if index & (1 << i) != 0 {
            result.set(sq);
        }
    }
    result
}

/// A tiny deterministic PRNG (xorshift64). Fixed seed => reproducible magics.
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Magic candidates work best with few set bits, so AND three draws together.
    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

/// Everything needed to answer slider queries: the flat attack table plus the
/// per-square descriptors, built once for a given ray set.
struct SliderTable {
    magics: [Magic; Square::NUM],
    attacks: Vec<Bitboard>,
}

impl SliderTable {
    #[inline]
    fn attacks(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        let m = &self.magics[sq.index()];
        self.attacks[m.index(occupied)]
    }
}

/// Find magics and fill the attack table for one slider kind (given its rays).
///
/// For each square we: build the occupancy mask, enumerate every occupancy
/// subset with its correct attack set (from the slow walker), then search for a
/// multiplier that maps each subset to a distinct table slot — allowing
/// "constructive" collisions where two subsets that share the same attack set
/// land together. The whole thing is deterministic thanks to the fixed seed.
fn build_slider_table(directions: &[Direction]) -> SliderTable {
    let mut rng = XorShift64(0x1234_5678_9abc_def0);

    // Placeholder; overwritten for every square below.
    let placeholder = Magic {
        mask: Bitboard::EMPTY,
        magic: 0,
        shift: 0,
        offset: 0,
    };
    let mut magics = [placeholder; Square::NUM];
    let mut attacks: Vec<Bitboard> = Vec::new();

    for s in 0..Square::NUM {
        let sq = Square(s as u8);
        let mask = slider_mask(sq, directions);
        let n = mask.count();
        let count = 1usize << n;
        let shift = 64 - n;

        // Precompute (occupancy, correct attacks) for every subset.
        let mut occupancies = Vec::with_capacity(count);
        let mut references = Vec::with_capacity(count);
        for i in 0..count {
            let occ = occupancy_for_index(i, mask);
            occupancies.push(occ);
            references.push(slow_sliding_attacks(sq, occ, directions));
        }

        let offset = attacks.len();
        attacks.resize(offset + count, Bitboard::EMPTY);

        // Trial-and-error search for a working magic multiplier.
        // `used[i]` marks which epoch last wrote slot i, so we can detect
        // destructive collisions without re-zeroing the table each attempt.
        let mut used = vec![0u32; count];
        let mut epoch = 0u32;
        loop {
            let magic = rng.sparse();

            // A quick sanity filter used by every magic generator: the high bits
            // must actually be populated, or the multiply spreads too thinly.
            let hi = (mask.0.wrapping_mul(magic) >> 56).count_ones();
            if hi < 6 {
                continue;
            }

            epoch += 1;
            let mut ok = true;
            for i in 0..count {
                let idx = ((occupancies[i].0.wrapping_mul(magic)) >> shift) as usize;
                if used[idx] != epoch {
                    // First use this attempt: claim the slot.
                    used[idx] = epoch;
                    attacks[offset + idx] = references[i];
                } else if attacks[offset + idx] != references[i] {
                    // Destructive collision: two different attack sets clash.
                    ok = false;
                    break;
                }
            }

            if ok {
                magics[s] = Magic {
                    mask,
                    magic,
                    shift,
                    offset,
                };
                break;
            }
        }
    }

    SliderTable { magics, attacks }
}

static ROOK_TABLE: LazyLock<SliderTable> = LazyLock::new(|| build_slider_table(&ROOK_DIRS));
static BISHOP_TABLE: LazyLock<SliderTable> = LazyLock::new(|| build_slider_table(&BISHOP_DIRS));

/// The squares a bishop on `sq` attacks, given the `occupied` squares. Rays stop
/// at (and include) the first blocker on each diagonal.
#[inline]
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    BISHOP_TABLE.attacks(sq, occupied)
}

/// The squares a rook on `sq` attacks, given the `occupied` squares. Rays stop
/// at (and include) the first blocker along each file/rank.
#[inline]
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    ROOK_TABLE.attacks(sq, occupied)
}

/// The squares a queen on `sq` attacks — simply the union of a rook's and a
/// bishop's attacks from the same square.
#[inline]
pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    bishop_attacks(sq, occupied) | rook_attacks(sq, occupied)
}

/// Force all lookup tables to be built now, rather than lazily on first use.
/// Optional — every public function initializes on demand — but handy to pay the
/// one-time cost up front (e.g. before a timed search).
pub fn init() {
    LazyLock::force(&KNIGHT_TABLE);
    LazyLock::force(&KING_TABLE);
    LazyLock::force(&PAWN_TABLE);
    LazyLock::force(&ROOK_TABLE);
    LazyLock::force(&BISHOP_TABLE);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knight_counts() {
        assert_eq!(knight_attacks(Square::D4).count(), 8);
        assert_eq!(knight_attacks(Square::A1).count(), 2);
        assert_eq!(knight_attacks(Square::H8).count(), 2);
    }

    #[test]
    fn knight_targets_from_corner() {
        // a1 knight -> b3 and c2 only.
        let a1 = knight_attacks(Square::A1);
        assert!(a1.contains(Square::B3));
        assert!(a1.contains(Square::C2));
        assert_eq!(a1.count(), 2);
    }

    #[test]
    fn king_counts() {
        assert_eq!(king_attacks(Square::A1).count(), 3);
        assert_eq!(king_attacks(Square::E4).count(), 8);
    }

    #[test]
    fn king_targets_from_corner() {
        let a1 = king_attacks(Square::A1);
        assert!(a1.contains(Square::A2));
        assert!(a1.contains(Square::B1));
        assert!(a1.contains(Square::B2));
        assert_eq!(a1.count(), 3);
    }

    #[test]
    fn pawn_attacks_white_center() {
        // A white pawn on e4 attacks d5 and f5.
        let e4 = pawn_attacks(Color::White, Square::E4);
        assert!(e4.contains(Square::D5));
        assert!(e4.contains(Square::F5));
        assert_eq!(e4.count(), 2);
    }

    #[test]
    fn pawn_attacks_edges() {
        // White a2 pawn: only b3 (no wrap to the h-file).
        let a2 = pawn_attacks(Color::White, Square::A2);
        assert!(a2.contains(Square::B3));
        assert_eq!(a2.count(), 1);

        // Black h7 pawn: only g6.
        let h7 = pawn_attacks(Color::Black, Square::H7);
        assert!(h7.contains(Square::G6));
        assert_eq!(h7.count(), 1);
    }

    #[test]
    fn empty_board_rook() {
        assert_eq!(rook_attacks(Square::A1, Bitboard::EMPTY).count(), 14);
        // A rook always sees 14 squares on an empty board, from any square.
        assert_eq!(rook_attacks(Square::E4, Bitboard::EMPTY).count(), 14);
        assert_eq!(rook_attacks(Square::H8, Bitboard::EMPTY).count(), 14);
    }

    #[test]
    fn empty_board_bishop() {
        assert_eq!(bishop_attacks(Square::A1, Bitboard::EMPTY).count(), 7);
        assert_eq!(bishop_attacks(Square::D4, Bitboard::EMPTY).count(), 13);
        assert_eq!(bishop_attacks(Square::H1, Bitboard::EMPTY).count(), 7);
    }

    #[test]
    fn empty_board_queen() {
        // Queen = rook + bishop; on d4 that's 14 + 13 = 27 squares.
        assert_eq!(queen_attacks(Square::D4, Bitboard::EMPTY).count(), 27);
    }

    #[test]
    fn rook_blocker_up_the_file() {
        // Rook on a1 with a blocker on a4: it sees a2, a3, a4 up the file
        // (a4 is the capturable blocker) but not a5 and beyond.
        let occ = Bitboard::from_square(Square::A4);
        let att = rook_attacks(Square::A1, occ);
        assert!(att.contains(Square::A2));
        assert!(att.contains(Square::A3));
        assert!(att.contains(Square::A4));
        assert!(!att.contains(Square::A5));
        assert!(!att.contains(Square::A6));
    }

    #[test]
    fn bishop_blocker_on_diagonal() {
        // Bishop on c1 with a blocker on e3: sees d2, e3, not f4+.
        let occ = Bitboard::from_square(Square::E3);
        let att = bishop_attacks(Square::C1, occ);
        assert!(att.contains(Square::D2));
        assert!(att.contains(Square::E3));
        assert!(!att.contains(Square::F4));
    }

    /// A cheap deterministic PRNG for generating occupancy subsets in the
    /// exhaustive magic-vs-reference cross-check.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    #[test]
    fn magic_matches_reference_everywhere() {
        let mut rng = Rng(0xdead_beef_cafe_babe);
        for s in 0..Square::NUM {
            let sq = Square(s as u8);

            // Always include the two easy extremes.
            for &occ in &[Bitboard::EMPTY, Bitboard::FULL] {
                assert_eq!(
                    rook_attacks(sq, occ),
                    slow_sliding_attacks(sq, occ, &ROOK_DIRS),
                    "rook mismatch at {sq} with occ {occ:?}"
                );
                assert_eq!(
                    bishop_attacks(sq, occ),
                    slow_sliding_attacks(sq, occ, &BISHOP_DIRS),
                    "bishop mismatch at {sq} with occ {occ:?}"
                );
            }

            // Plus many pseudo-random occupancy subsets.
            for _ in 0..1000 {
                // AND three draws for a realistic sparse-ish occupancy.
                let occ = Bitboard(rng.next() & rng.next() & rng.next());
                assert_eq!(
                    rook_attacks(sq, occ),
                    slow_sliding_attacks(sq, occ, &ROOK_DIRS),
                    "rook mismatch at {sq} with occ {occ:?}"
                );
                assert_eq!(
                    bishop_attacks(sq, occ),
                    slow_sliding_attacks(sq, occ, &BISHOP_DIRS),
                    "bishop mismatch at {sq} with occ {occ:?}"
                );
            }
        }
    }

    #[test]
    fn queen_is_rook_plus_bishop() {
        let occ = Bitboard::from_square(Square::D2) | Bitboard::from_square(Square::F6);
        for s in 0..Square::NUM {
            let sq = Square(s as u8);
            assert_eq!(
                queen_attacks(sq, occ),
                rook_attacks(sq, occ) | bishop_attacks(sq, occ)
            );
        }
    }
}

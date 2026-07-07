//! Bitboards: a 64-bit integer where bit `i` (i = 0..63) is set iff square `i`
//! is "occupied" by whatever set we're describing (all white pawns, all squares
//! a rook attacks, etc.). Bit 0 = a1, bit 63 = h8.
//!
//! Rust gives us the hardware bit instructions for free through `u64` methods
//! (`count_ones` = POPCNT, `trailing_zeros` = TZCNT), so — unlike Stockfish's
//! per-compiler `#ifdef` maze — we just call them.

use crate::types::{Direction, Square};
use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};
use std::sync::LazyLock;

#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Bitboard(pub u64);

// File masks (all 8 squares of a file). FILE_A = a1,a2,..,a8.
pub const FILE_A_BB: u64 = 0x0101_0101_0101_0101;
pub const FILE_B_BB: u64 = FILE_A_BB << 1;
pub const FILE_C_BB: u64 = FILE_A_BB << 2;
pub const FILE_D_BB: u64 = FILE_A_BB << 3;
pub const FILE_E_BB: u64 = FILE_A_BB << 4;
pub const FILE_F_BB: u64 = FILE_A_BB << 5;
pub const FILE_G_BB: u64 = FILE_A_BB << 6;
pub const FILE_H_BB: u64 = FILE_A_BB << 7;

// Rank masks (all 8 squares of a rank). RANK_1 = a1..h1.
pub const RANK_1_BB: u64 = 0xFF;
pub const RANK_2_BB: u64 = RANK_1_BB << 8;
pub const RANK_3_BB: u64 = RANK_1_BB << 16;
pub const RANK_4_BB: u64 = RANK_1_BB << 24;
pub const RANK_5_BB: u64 = RANK_1_BB << 32;
pub const RANK_6_BB: u64 = RANK_1_BB << 40;
pub const RANK_7_BB: u64 = RANK_1_BB << 48;
pub const RANK_8_BB: u64 = RANK_1_BB << 56;

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);
    pub const FULL: Bitboard = Bitboard(u64::MAX);

    /// A bitboard with a single bit set for `sq`.
    #[inline]
    pub const fn from_square(sq: Square) -> Bitboard {
        Bitboard(1u64 << sq.0)
    }

    /// A mask of the whole file that `sq` sits on.
    #[inline]
    pub const fn file_bb(sq: Square) -> Bitboard {
        Bitboard(FILE_A_BB << sq.file())
    }

    /// A mask of the whole rank that `sq` sits on.
    #[inline]
    pub const fn rank_bb(sq: Square) -> Bitboard {
        Bitboard(RANK_1_BB << (8 * sq.rank()))
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    /// Is square `sq` set?
    #[inline]
    pub const fn contains(self, sq: Square) -> bool {
        self.0 & (1u64 << sq.0) != 0
    }

    /// More than one bit set? (Faster than `count() > 1`.)
    #[inline]
    pub const fn more_than_one(self) -> bool {
        self.0 & (self.0.wrapping_sub(1)) != 0
    }

    /// Number of set bits (population count).
    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// The least-significant set square (lowest index). Panics if empty.
    #[inline]
    pub const fn lsb(self) -> Square {
        debug_assert!(self.0 != 0);
        Square(self.0.trailing_zeros() as u8)
    }

    /// The most-significant set square (highest index). Panics if empty.
    #[inline]
    pub const fn msb(self) -> Square {
        debug_assert!(self.0 != 0);
        Square((63 - self.0.leading_zeros()) as u8)
    }

    /// Remove and return the least-significant set square.
    #[inline]
    pub fn pop_lsb(&mut self) -> Square {
        let sq = self.lsb();
        self.0 &= self.0 - 1; // clear the lowest set bit
        sq
    }

    /// Set the bit for `sq`.
    #[inline]
    pub fn set(&mut self, sq: Square) {
        self.0 |= 1u64 << sq.0;
    }

    /// Clear the bit for `sq`.
    #[inline]
    pub fn clear(&mut self, sq: Square) {
        self.0 &= !(1u64 << sq.0);
    }

    /// Shift the whole set one step in a direction, dropping bits that would
    /// wrap around the board edge. This is how pawn pushes/attacks are built.
    #[inline]
    pub const fn shift(self, d: Direction) -> Bitboard {
        let b = self.0;
        let not_a = !FILE_A_BB;
        let not_h = !FILE_H_BB;
        Bitboard(match d {
            Direction::North => b << 8,
            Direction::South => b >> 8,
            Direction::East => (b & not_h) << 1,
            Direction::West => (b & not_a) >> 1,
            Direction::NorthEast => (b & not_h) << 9,
            Direction::NorthWest => (b & not_a) << 7,
            Direction::SouthEast => (b & not_h) >> 7,
            Direction::SouthWest => (b & not_a) >> 9,
        })
    }
}

// ---------------------------------------------------------------------------
// Between / Line tables — the geometry legal-move generation leans on.
//
// Two squares that share a rank, file, or diagonal lie on a common ray; these
// tables answer the two questions that come up constantly when reasoning about
// checks and pins:
//
//   * `between(a, b)` — the squares *strictly* between `a` and `b` on their
//     shared line (empty if they don't share one, or are adjacent). Used to test
//     whether a move blocks a sliding check, and to find the single blocker on a
//     potential pin ray.
//
//   * `line(a, b)` — every square on the full line through `a` and `b`
//     (both endpoints included, extended to the board edges), or empty if they
//     share no line. Used to test whether a pinned piece stays on its pin ray.
//
// Both are precomputed once into `[64][64]` tables via `LazyLock`.
// ---------------------------------------------------------------------------

/// The eight ray directions, walked one step at a time to fill the tables. Kept
/// local so `bitboard` stays self-contained (no dependency on the attack tables).
const ALL_DIRS: [Direction; 8] = [
    Direction::North,
    Direction::South,
    Direction::East,
    Direction::West,
    Direction::NorthEast,
    Direction::NorthWest,
    Direction::SouthEast,
    Direction::SouthWest,
];

/// For each ordered pair `(a, b)`, the squares strictly between them on their
/// shared rank/file/diagonal — empty when they don't share a line or are adjacent.
static BETWEEN_TABLE: LazyLock<[[Bitboard; Square::NUM]; Square::NUM]> = LazyLock::new(|| {
    let mut table = [[Bitboard::EMPTY; Square::NUM]; Square::NUM];
    for a in 0..Square::NUM {
        let from = Square(a as u8);
        // Walk each direction; every square we step through is "between" `from`
        // and any square further along that same ray.
        for &dir in &ALL_DIRS {
            let mut path = Bitboard::EMPTY;
            let mut current = from;
            while let Some(next) = current.offset(dir) {
                // Squares strictly between `from` and `next` are those already on
                // `path` (i.e. stepped through before reaching `next`).
                table[a][next.index()] = path;
                path.set(next);
                current = next;
            }
        }
    }
    table
});

/// For each pair `(a, b)` that share a rank/file/diagonal, the full line through
/// them extended to both board edges (both endpoints included); empty otherwise.
static LINE_TABLE: LazyLock<[[Bitboard; Square::NUM]; Square::NUM]> = LazyLock::new(|| {
    let mut table = [[Bitboard::EMPTY; Square::NUM]; Square::NUM];
    for a in 0..Square::NUM {
        let from = Square(a as u8);
        for &dir in &ALL_DIRS {
            // The full ray from `from` in `dir`, plus `from` itself, is the same
            // line for every square lying on it (both directions of the axis).
            let mut ray = Bitboard::from_square(from);
            let mut current = from;
            while let Some(next) = current.offset(dir) {
                ray.set(next);
                current = next;
            }
            // Walk the ray again to record: every square on it shares this line
            // with `from`, along the *full axis* (this direction and its opposite).
            let opp = opposite_dir(dir);
            let mut full = ray;
            let mut current = from;
            while let Some(next) = current.offset(opp) {
                full.set(next);
                current = next;
            }
            let mut current = from;
            while let Some(next) = current.offset(dir) {
                table[a][next.index()] = full;
                current = next;
            }
        }
    }
    table
});

/// The direction pointing the opposite way along the same axis.
const fn opposite_dir(d: Direction) -> Direction {
    match d {
        Direction::North => Direction::South,
        Direction::South => Direction::North,
        Direction::East => Direction::West,
        Direction::West => Direction::East,
        Direction::NorthEast => Direction::SouthWest,
        Direction::NorthWest => Direction::SouthEast,
        Direction::SouthEast => Direction::NorthWest,
        Direction::SouthWest => Direction::NorthEast,
    }
}

/// The squares strictly between `a` and `b` on their shared rank/file/diagonal.
///
/// Empty if `a` and `b` are not aligned, are the same square, or are adjacent.
/// The endpoints are *not* included.
#[inline]
pub fn between(a: Square, b: Square) -> Bitboard {
    BETWEEN_TABLE[a.index()][b.index()]
}

/// The full line through `a` and `b`, extended to both board edges and including
/// both endpoints. Empty if `a` and `b` share no rank/file/diagonal.
#[inline]
pub fn line(a: Square, b: Square) -> Bitboard {
    LINE_TABLE[a.index()][b.index()]
}

/// Force the between/line tables to be built now instead of on first use.
pub fn init() {
    LazyLock::force(&BETWEEN_TABLE);
    LazyLock::force(&LINE_TABLE);
}

// --- Bitwise operators so bitboards read like set algebra ------------------

impl BitAnd for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitand(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 & rhs.0)
    }
}
impl BitOr for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 | rhs.0)
    }
}
impl BitXor for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitxor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 ^ rhs.0)
    }
}
impl Not for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}
impl BitAndAssign for Bitboard {
    #[inline]
    fn bitand_assign(&mut self, rhs: Bitboard) {
        self.0 &= rhs.0;
    }
}
impl BitOrAssign for Bitboard {
    #[inline]
    fn bitor_assign(&mut self, rhs: Bitboard) {
        self.0 |= rhs.0;
    }
}
impl BitXorAssign for Bitboard {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Bitboard) {
        self.0 ^= rhs.0;
    }
}
impl Shl<u32> for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn shl(self, rhs: u32) -> Bitboard {
        Bitboard(self.0 << rhs)
    }
}
impl Shr<u32> for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn shr(self, rhs: u32) -> Bitboard {
        Bitboard(self.0 >> rhs)
    }
}

/// Combine a square with a bitboard via `|` (e.g. `bb | Square::E4`).
impl BitOr<Square> for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitor(self, sq: Square) -> Bitboard {
        self | Bitboard::from_square(sq)
    }
}

/// Iterating a bitboard yields its set squares, lowest-index first.
///
/// Because `Bitboard` is `Copy`, `for sq in bb { .. }` iterates over a copy and
/// leaves the original untouched.
impl Iterator for Bitboard {
    type Item = Square;
    #[inline]
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(self.pop_lsb())
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.count() as usize;
        (n, Some(n))
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bitboard(0x{:016x})", self.0)
    }
}

impl fmt::Display for Bitboard {
    /// Pretty 8x8 grid with rank 8 on top, like a real board.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  +-----------------+")?;
        for rank in (0..8).rev() {
            write!(f, "{} | ", rank + 1)?;
            for file in 0..8 {
                let sq = Square::make(file, rank);
                write!(f, "{} ", if self.contains(sq) { 'X' } else { '.' })?;
            }
            writeln!(f, "|")?;
        }
        writeln!(f, "  +-----------------+")?;
        write!(f, "    a b c d e f g h")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_square() {
        let bb = Bitboard::from_square(Square::A1);
        assert_eq!(bb.0, 1);
        assert!(bb.contains(Square::A1));
        assert!(!bb.contains(Square::B1));
        assert_eq!(bb.count(), 1);
        assert!(!bb.more_than_one());
    }

    #[test]
    fn set_and_clear() {
        let mut bb = Bitboard::EMPTY;
        bb.set(Square::E4);
        bb.set(Square::D5);
        assert_eq!(bb.count(), 2);
        assert!(bb.more_than_one());
        bb.clear(Square::E4);
        assert_eq!(bb.count(), 1);
        assert!(bb.contains(Square::D5));
    }

    #[test]
    fn lsb_msb_pop() {
        let mut bb = Bitboard::from_square(Square::C1) | Bitboard::from_square(Square::F8);
        assert_eq!(bb.lsb(), Square::C1);
        assert_eq!(bb.msb(), Square::F8);
        assert_eq!(bb.pop_lsb(), Square::C1);
        assert_eq!(bb.lsb(), Square::F8);
    }

    #[test]
    fn iterate_squares() {
        let bb = Bitboard::from_square(Square::A1)
            | Bitboard::from_square(Square::E4)
            | Bitboard::from_square(Square::H8);
        let squares: Vec<Square> = bb.collect();
        assert_eq!(squares, vec![Square::A1, Square::E4, Square::H8]);
    }

    #[test]
    fn file_and_rank_masks() {
        assert_eq!(Bitboard::file_bb(Square::A1).0, FILE_A_BB);
        assert_eq!(Bitboard::file_bb(Square::E4).0, FILE_E_BB);
        assert_eq!(Bitboard::rank_bb(Square::A1).0, RANK_1_BB);
        assert_eq!(Bitboard::rank_bb(Square::E4).0, RANK_4_BB);
    }

    #[test]
    fn shift_no_wrap() {
        // A single pawn on h4 shifted east must vanish (no wrap to a5).
        let h4 = Bitboard::from_square(Square::H4);
        assert_eq!(h4.shift(Direction::East), Bitboard::EMPTY);
        // e4 shifted north -> e5.
        let e4 = Bitboard::from_square(Square::E4);
        assert_eq!(e4.shift(Direction::North), Bitboard::from_square(Square::E5));
        // a-file west vanishes.
        assert_eq!(
            Bitboard::from_square(Square::A5).shift(Direction::West),
            Bitboard::EMPTY
        );
    }

    #[test]
    fn full_board_counts() {
        assert_eq!(Bitboard::FULL.count(), 64);
        assert_eq!(Bitboard::EMPTY.count(), 0);
    }

    // -- between / line geometry --------------------------------------------

    #[test]
    fn between_on_rank() {
        // a1..h1: strictly between a1 and d1 are b1 and c1.
        let bb = between(Square::A1, Square::D1);
        assert!(bb.contains(Square::B1));
        assert!(bb.contains(Square::C1));
        assert!(!bb.contains(Square::A1));
        assert!(!bb.contains(Square::D1));
        assert_eq!(bb.count(), 2);
    }

    #[test]
    fn between_on_file_and_diagonal() {
        // Up the e-file: between e2 and e5 are e3, e4.
        let file = between(Square::E2, Square::E5);
        assert_eq!(file.count(), 2);
        assert!(file.contains(Square::E3) && file.contains(Square::E4));

        // Along the a1-h8 diagonal: between a1 and d4 are b2, c3.
        let diag = between(Square::A1, Square::D4);
        assert_eq!(diag.count(), 2);
        assert!(diag.contains(Square::B2) && diag.contains(Square::C3));
    }

    #[test]
    fn between_is_symmetric() {
        assert_eq!(between(Square::A1, Square::H8), between(Square::H8, Square::A1));
        assert_eq!(between(Square::C4, Square::C7), between(Square::C7, Square::C4));
    }

    #[test]
    fn between_adjacent_and_unaligned_is_empty() {
        // Adjacent squares have nothing between them.
        assert!(between(Square::A1, Square::A2).is_empty());
        assert!(between(Square::D4, Square::E5).is_empty());
        // Unaligned squares (a knight's move apart) share no line.
        assert!(between(Square::A1, Square::B3).is_empty());
        // A square with itself.
        assert!(between(Square::E4, Square::E4).is_empty());
    }

    #[test]
    fn line_extends_to_edges() {
        // The line through a1 and c1 is the whole rank 1 (8 squares).
        let rank = line(Square::A1, Square::C1);
        assert_eq!(rank.count(), 8);
        assert!(rank.contains(Square::A1) && rank.contains(Square::H1));

        // The line through c3 and e5 is the a1-h8 diagonal (8 squares).
        let diag = line(Square::C3, Square::E5);
        assert!(diag.contains(Square::A1) && diag.contains(Square::H8));
        assert_eq!(diag.count(), 8);
    }

    #[test]
    fn line_unaligned_is_empty() {
        assert!(line(Square::A1, Square::B3).is_empty());
        assert!(line(Square::E4, Square::E4).is_empty());
    }

    #[test]
    fn line_contains_both_endpoints_when_aligned() {
        let l = line(Square::B2, Square::G7);
        assert!(l.contains(Square::B2));
        assert!(l.contains(Square::G7));
        // and the full a1-h8 diagonal it lies on.
        assert!(l.contains(Square::A1) && l.contains(Square::H8));
    }
}

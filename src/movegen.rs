//! Move generation: turn a [`Position`] into the list of moves the side to move
//! may legally play.
//!
//! The strategy here is deliberately "correct first, fast later": we generate
//! every *pseudo-legal* move (a move that obeys how the piece steps and captures,
//! but might leave our own king in check) and then filter out the ones that are
//! actually illegal by making the move, testing whether our king is attacked, and
//! unmaking it. Castling is the one case we verify fully up front (the through /
//! into-check rules cannot be expressed as a simple after-the-fact king-safety
//! test), so it is never re-checked by the filter.
//!
//! Correctness is proven by `perft` (see [`crate::perft`]): the node counts must
//! match the canonical published reference values exactly.

use std::ops::Index;

use crate::attacks::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks, queen_attacks, rook_attacks,
};
use crate::bitboard::{self, Bitboard};
use crate::position::{self, Position};
use crate::types::{Color, Direction, Move, MoveType, PieceType, Square};

// ---------------------------------------------------------------------------
// MoveList — a stack-allocated, fixed-capacity move container.
// ---------------------------------------------------------------------------

/// The maximum number of pseudo-legal moves a chess position can contain is well
/// under 256 (the theoretical maximum for a legal position is 218), so a fixed
/// array never overflows and avoids any heap allocation on the perft hot path.
const MAX_MOVES: usize = 256;

/// A fixed-capacity list of moves, stored inline on the stack (no `Vec`).
pub struct MoveList {
    moves: [Move; MAX_MOVES],
    len: usize,
}

impl MoveList {
    /// An empty list.
    #[inline]
    pub fn new() -> MoveList {
        MoveList {
            moves: [Move::NONE; MAX_MOVES],
            len: 0,
        }
    }

    /// Append a move. Debug-asserts we never exceed capacity.
    #[inline]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_MOVES, "MoveList overflow");
        self.moves[self.len] = m;
        self.len += 1;
    }

    /// How many moves are stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the list is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The stored moves as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

impl Default for MoveList {
    #[inline]
    fn default() -> MoveList {
        MoveList::new()
    }
}

impl Index<usize> for MoveList {
    type Output = Move;
    #[inline]
    fn index(&self, i: usize) -> &Move {
        &self.as_slice()[i]
    }
}

/// Iterating `&MoveList` yields each `Move` by copy.
impl<'a> IntoIterator for &'a MoveList {
    type Item = Move;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, Move>>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter().copied()
    }
}

// ---------------------------------------------------------------------------
// Public entry points.
// ---------------------------------------------------------------------------

/// Generate every fully-legal move for the side to move.
///
/// Rather than making and unmaking every pseudo-legal move to test king safety,
/// we precompute the *checkers* (enemy pieces giving check) and the *pinned*
/// own pieces once, then decide each pseudo-legal move's legality directly:
///
/// * **Double check** — only the king can move (to a square the enemy doesn't
///   attack once the king is out of the way).
/// * **Single check** — legal non-king moves must land on the check-mask
///   (capture the checker or block a sliding check) and, if the moving piece is
///   pinned, stay on its pin ray. King moves must reach a safe square.
/// * **Not in check** — every pseudo-legal move is legal *except* king moves
///   (must reach a safe square), moves of pinned pieces (must stay on the pin
///   ray), and en-passant captures (verified with make/undo, since they can
///   expose a rare horizontal discovered check that pin logic alone misses).
///
/// Castling is fully validated during pseudo-legal generation, so it is passed
/// through untouched. En-passant is the only case that still uses make/undo, and
/// it is rare, so the hot path avoids make/undo entirely.
///
/// Still takes `&mut Position` (the en-passant fallback and the same signature),
/// and leaves the position byte-for-byte identical on return.
pub fn generate_legal(pos: &mut Position) -> MoveList {
    let pseudo = generate_pseudo_legal(pos);

    let us = pos.side_to_move();
    let them = !us;
    let ksq = pos.king_square(us);
    let occupied = pos.occupied();

    // Enemy pieces currently attacking our king.
    let checkers = pos.attackers_to(ksq, occupied) & pos.pieces(them);

    // Own pieces pinned to the king by an enemy slider (see `compute_pinned`).
    let pinned = compute_pinned(pos, us, them, ksq, occupied);

    let mut legal = MoveList::new();

    // Double check: only the king may move, and only to a square the enemy does
    // not attack once the king itself no longer blocks the incoming rays.
    if checkers.more_than_one() {
        let occ_without_king = occupied ^ Bitboard::from_square(ksq);
        for m in &pseudo {
            if m.from_sq() == ksq && king_destination_safe(pos, them, m.to_sq(), occ_without_king) {
                legal.push(m);
            }
        }
        return legal;
    }

    // Single check: the mask of squares a non-king move may land on to resolve
    // the check — capture the checker, or (if it slides) block between it and the
    // king. Zero-or-more checkers already handled above / below.
    let check_mask = if checkers.any() {
        let checker_sq = checkers.lsb();
        bitboard::between(ksq, checker_sq) | checkers
    } else {
        // Not in check: any target square is fine for the check-mask test.
        Bitboard::FULL
    };

    let occ_without_king = occupied ^ Bitboard::from_square(ksq);

    for m in &pseudo {
        let from = m.from_sq();
        let to = m.to_sq();

        // Castling was fully verified during pseudo-legal generation.
        if m.move_type() == MoveType::Castling {
            legal.push(m);
            continue;
        }

        // King moves: legal iff the destination is not attacked with the king
        // lifted off the board (so it can't shadow an incoming sliding ray).
        if from == ksq {
            if king_destination_safe(pos, them, to, occ_without_king) {
                legal.push(m);
            }
            continue;
        }

        // En passant is verified the slow, sure way: it can expose a rare
        // horizontal discovered check that pin logic misses, and — when the
        // captured pawn is itself the checker — its destination lies *off* the
        // check-mask, so it must skip that filter entirely. Rare, so cheap.
        if m.move_type() == MoveType::EnPassant {
            let undo = pos.make_move(m);
            let ok = !pos.is_attacked(pos.king_square(us), them);
            pos.undo_move(m, undo);
            if ok {
                legal.push(m);
            }
            continue;
        }

        // Non-king moves must resolve any check: land on the check-mask.
        if !check_mask.contains(to) {
            continue;
        }

        // A pinned piece may only move along the line through it and the king
        // (which includes capturing the pinner and retreating toward the king).
        if pinned.contains(from) && !bitboard::line(ksq, from).contains(to) {
            continue;
        }

        legal.push(m);
    }

    legal
}

/// Whether the king may step to `to`: no enemy piece attacks it once the king is
/// removed from the occupancy (`occ_without_king`), so a slider giving check
/// can't be "walked along" because the king no longer blocks its own escape ray.
#[inline]
fn king_destination_safe(
    pos: &Position,
    them: Color,
    to: Square,
    occ_without_king: Bitboard,
) -> bool {
    (pos.attackers_to(to, occ_without_king) & pos.pieces(them)).is_empty()
}

/// The set of own pieces pinned to the king by an enemy slider.
///
/// A piece is pinned when it is the *only* piece between our king and an enemy
/// slider that lines up with the king (an enemy rook/queen on the king's
/// rank/file, or bishop/queen on the king's diagonal). We find candidate snipers
/// by asking which enemy sliders would attack the king on an *empty* board, then
/// keep each ray that has exactly one blocker and that blocker is ours.
fn compute_pinned(pos: &Position, us: Color, them: Color, ksq: Square, occupied: Bitboard) -> Bitboard {
    let their_rooks_queens =
        (pos.pieces_type(PieceType::Rook) | pos.pieces_type(PieceType::Queen)) & pos.pieces(them);
    let their_bishops_queens =
        (pos.pieces_type(PieceType::Bishop) | pos.pieces_type(PieceType::Queen)) & pos.pieces(them);

    // Enemy sliders that would hit the king if nothing were in the way.
    let snipers = (rook_attacks(ksq, Bitboard::EMPTY) & their_rooks_queens)
        | (bishop_attacks(ksq, Bitboard::EMPTY) & their_bishops_queens);

    let our_pieces = pos.pieces(us);
    let mut pinned = Bitboard::EMPTY;
    for sniper in snipers {
        // Everything sitting between the king and this sniper.
        let blockers = bitboard::between(ksq, sniper) & occupied;
        // Exactly one blocker, and it is ours → pinned.
        if !blockers.more_than_one() && (blockers & our_pieces).any() {
            pinned |= blockers & our_pieces;
        }
    }
    pinned
}

/// Generate every pseudo-legal move for the side to move (no king-safety filter,
/// except that castling *is* fully validated here since it cannot be filtered
/// after the fact).
pub fn generate_pseudo_legal(pos: &Position) -> MoveList {
    let mut list = MoveList::new();
    let us = pos.side_to_move();

    generate_pawn_moves(pos, us, &mut list);
    generate_knight_moves(pos, us, &mut list);
    generate_slider_moves(pos, us, &mut list);
    generate_king_moves(pos, us, &mut list);
    generate_castling(pos, us, &mut list);

    list
}

// ---------------------------------------------------------------------------
// Pawns.
// ---------------------------------------------------------------------------

fn generate_pawn_moves(pos: &Position, us: Color, list: &mut MoveList) {
    let them = !us;
    let pawns = pos.pieces_cp(us, PieceType::Pawn);
    let empty = !pos.occupied();
    let enemies = pos.pieces(them);

    // Direction-dependent constants: white pushes north, black pushes south.
    let (forward, start_rank, promo_rank) = match us {
        Color::White => (Direction::North, 1u8, 7u8),
        Color::Black => (Direction::South, 6u8, 0u8),
    };

    for from in pawns {
        let from_rank = from.rank();

        // --- Single & double pushes (never captures) --------------------
        if let Some(one) = from.offset(forward) {
            if empty.contains(one) {
                if one.rank() == promo_rank {
                    push_promotions(list, from, one);
                } else {
                    list.push(Move::normal(from, one));
                    // Double push only from the starting rank, both squares empty.
                    if from_rank == start_rank {
                        if let Some(two) = one.offset(forward) {
                            if empty.contains(two) {
                                list.push(Move::normal(from, two));
                            }
                        }
                    }
                }
            }
        }

        // --- Diagonal captures (incl. capture-promotions) ---------------
        let targets = pawn_attacks(us, from) & enemies;
        for to in targets {
            if to.rank() == promo_rank {
                push_promotions(list, from, to);
            } else {
                list.push(Move::normal(from, to));
            }
        }

        // --- En passant --------------------------------------------------
        if let Some(ep) = pos.en_passant() {
            // We attack the ep target square iff a pawn of `us` on `from` could
            // capture onto it.
            if pawn_attacks(us, from).contains(ep) {
                list.push(Move::en_passant(from, ep));
            }
        }
    }
}

/// Emit the four promotion moves (Queen, Rook, Bishop, Knight) for a pawn going
/// from `from` to `to`.
#[inline]
fn push_promotions(list: &mut MoveList, from: Square, to: Square) {
    list.push(Move::promotion(from, to, PieceType::Queen));
    list.push(Move::promotion(from, to, PieceType::Rook));
    list.push(Move::promotion(from, to, PieceType::Bishop));
    list.push(Move::promotion(from, to, PieceType::Knight));
}

// ---------------------------------------------------------------------------
// Knights.
// ---------------------------------------------------------------------------

fn generate_knight_moves(pos: &Position, us: Color, list: &mut MoveList) {
    let not_ours = !pos.pieces(us);
    for from in pos.pieces_cp(us, PieceType::Knight) {
        for to in knight_attacks(from) & not_ours {
            list.push(Move::normal(from, to));
        }
    }
}

// ---------------------------------------------------------------------------
// Sliders: bishops, rooks, queens.
// ---------------------------------------------------------------------------

fn generate_slider_moves(pos: &Position, us: Color, list: &mut MoveList) {
    let occ = pos.occupied();
    let not_ours = !pos.pieces(us);

    for from in pos.pieces_cp(us, PieceType::Bishop) {
        for to in bishop_attacks(from, occ) & not_ours {
            list.push(Move::normal(from, to));
        }
    }
    for from in pos.pieces_cp(us, PieceType::Rook) {
        for to in rook_attacks(from, occ) & not_ours {
            list.push(Move::normal(from, to));
        }
    }
    for from in pos.pieces_cp(us, PieceType::Queen) {
        for to in queen_attacks(from, occ) & not_ours {
            list.push(Move::normal(from, to));
        }
    }
}

// ---------------------------------------------------------------------------
// King (non-castling).
// ---------------------------------------------------------------------------

fn generate_king_moves(pos: &Position, us: Color, list: &mut MoveList) {
    let from = pos.king_square(us);
    let not_ours = !pos.pieces(us);
    for to in king_attacks(from) & not_ours {
        list.push(Move::normal(from, to));
    }
}

// ---------------------------------------------------------------------------
// Castling.
// ---------------------------------------------------------------------------

fn generate_castling(pos: &Position, us: Color, list: &mut MoveList) {
    let them = !us;
    let rights = pos.castling_rights();
    let occ = pos.occupied();

    // No castling while in check.
    let king_from = pos.king_square(us);
    if pos.is_attacked(king_from, them) {
        return;
    }

    // Per-color right bits and home rank.
    let (oo, ooo, rank) = match us {
        Color::White => (position::WHITE_OO, position::WHITE_OOO, 0u8),
        Color::Black => (position::BLACK_OO, position::BLACK_OOO, 7u8),
    };

    // King-side: king e -> g, transit f,g must be empty and unattacked;
    // rook on h moves to f.
    if rights & oo != 0 {
        let f = Square::make(5, rank);
        let g = Square::make(6, rank);
        let rook = Square::make(7, rank);
        let squares_empty = !occ.contains(f) && !occ.contains(g);
        let path_safe = !pos.is_attacked(f, them) && !pos.is_attacked(g, them);
        if squares_empty && path_safe {
            list.push(Move::castling(king_from, rook));
        }
    }

    // Queen-side: king e -> c, transit d,c must be empty and unattacked; b must
    // be empty (but need not be unattacked); rook on a moves to d.
    if rights & ooo != 0 {
        let d = Square::make(3, rank);
        let c = Square::make(2, rank);
        let b = Square::make(1, rank);
        let rook = Square::make(0, rank);
        let squares_empty = !occ.contains(d) && !occ.contains(c) && !occ.contains(b);
        let path_safe = !pos.is_attacked(d, them) && !pos.is_attacked(c, them);
        if squares_empty && path_safe {
            list.push(Move::castling(king_from, rook));
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience: is a move present in a freshly generated legal list?
// ---------------------------------------------------------------------------

impl MoveList {
    /// Whether `m` is one of the stored moves (linear scan; for tests/debugging).
    pub fn contains(&self, m: Move) -> bool {
        self.as_slice().contains(&m)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_has_twenty_moves() {
        let mut pos = Position::startpos();
        let moves = generate_legal(&mut pos);
        assert_eq!(moves.len(), 20);
        // 16 pawn moves + 4 knight moves.
    }

    #[test]
    fn movelist_basics() {
        let mut list = MoveList::new();
        assert!(list.is_empty());
        let m = Move::normal(Square::E2, Square::E4);
        list.push(m);
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
        assert_eq!(list[0], m);
        assert_eq!(list.as_slice(), &[m]);
        let collected: Vec<Move> = (&list).into_iter().collect();
        assert_eq!(collected, vec![m]);
    }

    #[test]
    fn generation_leaves_position_unchanged() {
        let mut pos = Position::startpos();
        let fen_before = pos.to_fen();
        let key_before = pos.key();
        let _ = generate_legal(&mut pos);
        assert_eq!(pos.to_fen(), fen_before);
        assert_eq!(pos.key(), key_before);
    }

    #[test]
    fn king_cannot_move_into_check() {
        // White king on e1, black rook on e8 pins the e-file; Ke1 may not step to
        // e2 (still attacked) nor stay attacked... it can go to d1/f1/d2/f2 etc.
        let mut pos = Position::from_fen("4r3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let moves = generate_legal(&mut pos);
        // King must not move onto the e-file (e2), and must not be a no-op.
        assert!(!moves.contains(Move::normal(Square::E1, Square::E2)));
        assert!(moves.contains(Move::normal(Square::E1, Square::D1)));
        assert!(moves.contains(Move::normal(Square::E1, Square::F1)));
    }
}

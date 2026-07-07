//! Static Exchange Evaluation (SEE).
//!
//! SEE answers a single, purely-material question about a capture: *if I make
//! this capture and then both sides keep recapturing on the destination square
//! with their least-valuable attacker until neither wants to continue, what is
//! the net material outcome for me?* It is a cheap, static approximation of a
//! full quiescence search restricted to one square — no board is ever mutated —
//! and it is the workhorse behind good capture ordering and pruning.
//!
//! The classic **swap-off** algorithm drives the whole thing:
//!
//!  1. Book the value of the piece we capture first (`gain[0]`).
//!  2. Pretend the capturing piece now sits on the destination square.
//!  3. The side to recapture picks its **least-valuable attacker** of that
//!     square. Removing that attacker from the occupancy can *uncover* a slider
//!     behind it (an x-ray / battery), so after each capture we re-derive the
//!     slider attackers against the updated occupancy — a rook stacked behind a
//!     rook, or a queen behind a bishop, joins the fray automatically.
//!  4. We record the running material swing at each step, then fold the array
//!     back with a negamax-minimax: at every depth the side to move will only
//!     recapture if doing so does not make things worse for them.
//!
//! ### Fixed piece values
//!
//! SEE deliberately uses **flat** piece values ([`see_value`]), *not* the
//! tapered evaluation numbers. A capture sequence is a raw material trade; the
//! positional/phase blending that [`crate::eval`] does would only add noise and,
//! worse, could make the least-valuable-attacker ordering inconsistent. King is
//! given a large sentinel value so it is always the *last* resort as an attacker
//! (you may only recapture with the king when the square is otherwise safe).

use crate::attacks::{bishop_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::position::Position;
use crate::types::{Move, MoveType, PieceType};

/// The flat SEE value of a piece type, in centipawns. These are intentionally
/// simple, round numbers — SEE only compares *relative* material, so the exact
/// figures matter less than their ordering. The king's value is a large
/// sentinel: it must dominate every other piece so a king is only ever chosen as
/// the least-valuable attacker when nothing cheaper is available.
#[inline]
pub fn see_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn => 100,
        PieceType::Knight => 320,
        PieceType::Bishop => 330,
        PieceType::Rook => 500,
        PieceType::Queen => 900,
        PieceType::King => 10_000,
    }
}

/// The full net material outcome (in centipawns, from the moving side's
/// perspective) of the capture sequence initiated by `m`.
///
/// A non-capturing, non-promoting move returns `0`. Callers in the search only
/// ever pass captures and promotions, but the guard keeps the function total and
/// safe to call on any move.
pub fn see(pos: &Position, m: Move) -> i32 {
    // We reuse the threshold engine and binary-search the exact value would be
    // wasteful; instead run the full swap-off directly. This mirrors `see_ge`
    // but keeps the entire `gain[]` array so it can return the precise number.
    see_swap(pos, m)
}

/// `true` iff `see(pos, m) >= threshold`.
///
/// This is the form the search actually wants ("is this capture at least
/// break-even?", `threshold = 0`). We compute the full swap value and compare;
/// the swap-off is already cheap enough that a specialised early-exit variant
/// buys little here, and sharing one implementation keeps the two in lock-step.
#[inline]
pub fn see_ge(pos: &Position, m: Move, threshold: i32) -> bool {
    see_swap(pos, m) >= threshold
}

/// The shared swap-off core. Returns the net material of the capture chain on
/// `m.to_sq()`, from the perspective of the side that plays `m`.
fn see_swap(pos: &Position, m: Move) -> i32 {
    let to = m.to_sq();
    let from = m.from_sq();

    // The side initiating the capture (whose perspective the result is in).
    let us = pos.side_to_move();

    // --- Book the first captured piece. -------------------------------------
    //
    // `gain[0]` is the material we win outright by making `m`: the victim's
    // value, plus (for a promotion) the value the pawn gains by promoting.
    let mut occupied = pos.occupied();

    // The value of whatever piece is standing on `to` *after* our move — this is
    // what the opponent captures if they recapture. For a normal capture that is
    // our attacker; for a promotion it is the promoted piece.
    let mut on_square: i32;

    let mut gain = [0i32; 32];

    match m.move_type() {
        MoveType::EnPassant => {
            // The victim is the pawn sitting *behind* the target square (same
            // file as `to`, same rank as `from`); remove it from the occupancy.
            gain[0] = see_value(PieceType::Pawn);
            let cap_sq = crate::types::Square::make(to.file(), from.rank());
            occupied.clear(cap_sq);
            on_square = see_value(PieceType::Pawn); // our pawn now stands on `to`.
        }
        MoveType::Promotion => {
            // Value gained by promoting: we lose the pawn, gain the promo piece.
            let promo = see_value(m.promotion_type());
            let victim = match pos.piece_at(to) {
                Some(p) => see_value(p.piece_type),
                None => 0, // a bare (non-capturing) promotion.
            };
            gain[0] = victim + promo - see_value(PieceType::Pawn);
            // The promoted piece is what now stands on `to` and can be recaptured.
            on_square = promo;
        }
        _ => {
            // Normal capture (callers only pass captures here). A non-capture is
            // handled defensively as a 0-value exchange.
            let victim = match pos.piece_at(to) {
                Some(p) => see_value(p.piece_type),
                None => return 0, // not a capture: nothing to evaluate.
            };
            gain[0] = victim;
            // Our attacker now stands on `to`.
            on_square = see_value(
                pos.piece_at(from)
                    .map(|p| p.piece_type)
                    .unwrap_or(PieceType::Pawn),
            );
        }
    }

    // Remove our own attacker from `from`: it has moved onto `to`.
    occupied.clear(from);

    // --- Iterate the recaptures. --------------------------------------------
    //
    // `side` is whoever is on move to (potentially) recapture next. After our
    // move it is the opponent's turn.
    let mut side = !us;
    let mut depth = 1usize;

    // All attackers of `to` for *both* colors, recomputed against `occupied` as
    // it shrinks so batteries/x-rays are picked up. We seed it once and refresh
    // only the slider portion after each capture (leapers/pawns never uncover).
    let mut attackers = pos.attackers_to(to, occupied) & occupied;

    loop {
        // Restrict to the side that is on move to recapture.
        let side_attackers = attackers & pos.pieces(side);
        if side_attackers.is_empty() {
            break; // no recapture available: the exchange ends here.
        }

        // Pick this side's least-valuable attacker.
        let (attacker_sq, attacker_pt) = match least_valuable_attacker(pos, side_attackers) {
            Some(pair) => pair,
            None => break,
        };

        if depth >= gain.len() {
            break; // pathological chain longer than the buffer; stop safely.
        }

        // The recapturing side wins whatever is currently standing on `to`, but
        // now exposes its own attacker to be captured in turn.
        gain[depth] = on_square - gain[depth - 1];
        on_square = see_value(attacker_pt);

        // Remove the chosen attacker from the occupancy and re-derive sliders so
        // any piece it was shielding now counts as an attacker of `to`.
        occupied.clear(attacker_sq);
        attackers = refreshed_attackers(pos, to, occupied, attackers);

        // A king may only recapture if the square is not then defended by the
        // other side. If the opponent still has an attacker after the king would
        // take, the king cannot legally make that capture — the chain stops with
        // the king's capture un-played.
        if attacker_pt == PieceType::King && (attackers & pos.pieces(!side)).any() {
            break;
        }

        side = !side;
        depth += 1;
    }

    // --- Negamax-minimax the gains back down. -------------------------------
    //
    // Walking from the deepest recapture back to the root, each side takes the
    // better of "stop here" (keep the current `gain`) or "let the capture happen"
    // (`-max(-gain[d-1], gain[d])`). The result at `gain[0]` is the value each
    // side would rationally allow.
    while depth > 1 {
        depth -= 1;
        gain[depth - 1] = -std::cmp::max(-gain[depth - 1], gain[depth]);
    }

    gain[0]
}

/// Find the least-valuable attacker in `attackers` and return its square and
/// piece type. `attackers` must already be masked to a single side.
#[inline]
fn least_valuable_attacker(
    pos: &Position,
    attackers: Bitboard,
) -> Option<(crate::types::Square, PieceType)> {
    // Probe piece types cheapest-first; the first one that intersects wins.
    for pt in [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ] {
        let bb = attackers & pos.pieces_type(pt);
        if bb.any() {
            return Some((bb.lsb(), pt));
        }
    }
    None
}

/// Recompute the attacker set of `to` after removing a piece from `occupied`.
///
/// Leapers (knights, kings) and pawns can never be *uncovered* by removing a
/// blocker, so only the sliding attackers can change. We therefore keep the
/// existing set (minus pieces no longer on the board) and re-add every
/// bishop/rook/queen that now sees `to` through the thinned occupancy.
#[inline]
fn refreshed_attackers(
    pos: &Position,
    to: crate::types::Square,
    occupied: Bitboard,
    prev: Bitboard,
) -> Bitboard {
    let bishops_queens =
        pos.pieces_type(PieceType::Bishop) | pos.pieces_type(PieceType::Queen);
    let rooks_queens = pos.pieces_type(PieceType::Rook) | pos.pieces_type(PieceType::Queen);

    let diagonal = bishop_attacks(to, occupied) & bishops_queens;
    let straight = rook_attacks(to, occupied) & rooks_queens;

    // Keep the previously-found non-slider attackers (masked to what is still on
    // the board) and union in the freshly-visible sliders.
    (prev | diagonal | straight) & occupied
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Move, PieceType, Square};

    /// Parse a FEN and build the given normal capture, then run SEE.
    fn see_fen(fen: &str, from: Square, to: Square) -> i32 {
        let pos = Position::from_fen(fen).unwrap_or_else(|e| panic!("bad fen {fen}: {e}"));
        see(&pos, Move::normal(from, to))
    }

    #[test]
    fn value_ordering_is_flat_and_monotone() {
        // The flat values must strictly increase pawn < knight < bishop < rook <
        // queen < king so least-valuable-attacker ordering is well-defined.
        assert!(see_value(PieceType::Pawn) < see_value(PieceType::Knight));
        assert!(see_value(PieceType::Knight) < see_value(PieceType::Bishop));
        assert!(see_value(PieceType::Bishop) < see_value(PieceType::Rook));
        assert!(see_value(PieceType::Rook) < see_value(PieceType::Queen));
        assert!(see_value(PieceType::Queen) < see_value(PieceType::King));
    }

    #[test]
    fn undefended_piece_wins_full_value() {
        // White rook on a1 captures a totally undefended black knight on a7.
        // Nothing recaptures, so SEE = knight value = 320.
        let fen = "4k3/n7/8/8/8/8/8/R3K3 w - - 0 1";
        assert_eq!(see_fen(fen, Square::A1, Square::A7), 320);
    }

    #[test]
    fn pawn_takes_pawn_defended_by_pawn_is_zero() {
        // White pawn d4 x black pawn e5; the e5 pawn is defended by a black pawn
        // on d6 (which attacks e5). We win a pawn (+100) then lose our pawn to the
        // recapture (-100): net 0.
        //   white pawn d4, black pawns e5 and d6.
        let fen = "4k3/8/3p4/4p3/3P4/8/8/4K3 w - - 0 1";
        assert_eq!(see_fen(fen, Square::D4, Square::E5), 0);
    }

    #[test]
    fn rook_takes_pawn_defended_by_pawn_loses_the_exchange() {
        // White rook e1 x black pawn e5; the pawn is defended by a black pawn on
        // d6. We win the pawn (+100) but the recapture takes our rook (-500):
        // net 100 - 500 = -400. SEE must be negative.
        let fen = "4k3/8/3p4/4p3/8/8/8/4R1K1 w - - 0 1";
        assert_eq!(see_fen(fen, Square::E1, Square::E5), -400);
    }

    #[test]
    fn knight_takes_pawn_defended_by_pawn_loses() {
        // White knight on c4 x black pawn d5, defended by a black pawn on c6.
        // Win pawn (+100), lose knight to recapture (-320): net -220.
        let fen = "4k3/8/2p5/3p4/2N5/8/8/4K3 w - - 0 1";
        assert_eq!(see_fen(fen, Square::C4, Square::D5), -220);
    }

    #[test]
    fn pawn_takes_queen_even_if_defended_is_winning() {
        // White pawn on e4 x black queen on d5, defended by a black pawn on c6.
        // Win the queen (+900); the recapture takes our pawn (-100): net +800.
        // Winning a queen with a pawn is hugely positive even when defended.
        let fen = "4k3/8/2p5/3q4/4P3/8/8/4K3 w - - 0 1";
        assert_eq!(see_fen(fen, Square::E4, Square::D5), 800);
    }

    #[test]
    fn xray_battery_of_two_rooks_is_counted() {
        // Stacked white rooks on e1 and e2 capture a black pawn on e5, which is
        // defended once by a black rook on e8 (down the file).
        //   White: Re1, Re2, Ke1-area king g1.  Black: pawn e5, Re8, Kg8.
        // Sequence on e5: Rxe5 (+100), ...Rxe5 (-500), Rxe5 (+500).
        //   gains: [100, -500, 500] -> minimax -> +100.
        // Without the x-ray (the second white rook behind the first), the second
        // white recapture would be missing and SEE would read 100 - 500 = -400,
        // so this test proves the battery is seen.
        let fen = "4r1k1/8/8/4p3/8/8/4R3/4R1K1 w - - 0 1";
        assert_eq!(see_fen(fen, Square::E2, Square::E5), 100);
    }

    #[test]
    fn xray_battery_missing_second_rook_loses_exchange() {
        // Control for the battery test: only ONE white rook on the e-file. Now
        // Rxe5 (+100) is answered by ...Rxe5 (-500) with no follow-up, so SEE is
        // 100 - 500 = -400.
        let fen = "4r1k1/8/8/4p3/8/8/8/4R1K1 w - - 0 1";
        assert_eq!(see_fen(fen, Square::E1, Square::E5), -400);
    }

    #[test]
    fn en_passant_capture_is_evaluated() {
        // White pawn d5 takes en passant on c6 after ...c5. The captured pawn is
        // on c5. If nothing defends c6, SEE = pawn value = 100.
        let fen = "4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1";
        let pos = Position::from_fen(fen).unwrap();
        let m = Move::en_passant(Square::D5, Square::C6);
        assert_eq!(see(&pos, m), 100);
    }

    #[test]
    fn defended_equal_trade_of_knights_is_zero() {
        // White knight on e4 x black knight on d6 (a knight, value 320), defended
        // by a black pawn on c7. Win knight (+320), lose knight to pawn recapture
        // (-320): net 0. An even trade nets zero.
        let fen = "4k3/2p5/3n4/8/4N3/8/8/4K3 w - - 0 1";
        assert_eq!(see_fen(fen, Square::E4, Square::D6), 0);
    }

    #[test]
    fn non_capture_returns_zero() {
        // A quiet move that captures nothing must score 0.
        let fen = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1";
        assert_eq!(see_fen(fen, Square::A1, Square::A4), 0);
    }

    #[test]
    fn see_ge_matches_see_value() {
        // `see_ge` must agree with the sign of `see` at the given threshold.
        let fen = "4k3/8/3p4/4p3/8/8/8/4R1K1 w - - 0 1"; // rook takes defended pawn: -400
        let pos = Position::from_fen(fen).unwrap();
        let m = Move::normal(Square::E1, Square::E5);
        assert!(!see_ge(&pos, m, 0), "a losing capture must fail the >= 0 test");
        assert!(see_ge(&pos, m, -400), "it must clear its own exact value");
        assert!(!see_ge(&pos, m, -399), "one above its value must fail");
    }
}

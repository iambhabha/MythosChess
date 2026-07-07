//! Perft — the move generator's correctness proof.
//!
//! `perft(pos, depth)` counts the number of leaf nodes in the game tree of a
//! given depth, exploring only *legal* moves via make/undo. These counts are a
//! famously unforgiving test: a single mis-generated or mis-made move (a missed
//! en-passant, a bad under-promotion, castling through check, a stale castling
//! right) shifts the total, so matching the canonical published numbers to the
//! node is strong evidence the whole board / movegen / make-undo stack is right.

use crate::movegen::generate_legal;
use crate::position::Position;
use crate::types::Move;

/// Count the leaf nodes reachable from `pos` in exactly `depth` plies, moving
/// through legal moves only.
///
/// `depth == 0` is defined as a single node (the position itself). At `depth == 1`
/// we return the number of legal moves directly (a "bulk count"), which is both
/// correct and faster than descending one more ply just to count 1 per leaf.
///
/// `pos` is restored byte-for-byte on return.
pub fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = generate_legal(pos);

    // Bulk-count the last ply: each legal move is exactly one leaf.
    if depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0u64;
    for m in &moves {
        let undo = pos.make_move(m);
        nodes += perft(pos, depth - 1);
        pos.undo_move(m, undo);
    }
    nodes
}

/// Per-root-move node counts: for every legal move from `pos`, how many leaves
/// its subtree has at `depth`. Invaluable for bisecting a movegen bug against a
/// known-good reference (compare move-by-move to find the diverging subtree).
///
/// `pos` is restored byte-for-byte on return.
pub fn perft_divide(pos: &mut Position, depth: u32) -> Vec<(Move, u64)> {
    let moves = generate_legal(pos);
    let mut out = Vec::with_capacity(moves.len());

    for m in &moves {
        let undo = pos.make_move(m);
        let count = if depth <= 1 { 1 } else { perft(pos, depth - 1) };
        pos.undo_move(m, undo);
        out.push((m, count));
    }
    out
}

/// Format a `perft_divide` result the way most engines print it: one
/// `<move>: <count>` line per root move, then the total. Handy for eyeballing.
pub fn format_divide(divide: &[(Move, u64)]) -> String {
    let mut s = String::new();
    let mut total = 0u64;
    for (m, count) in divide {
        s.push_str(&format!("{m}: {count}\n"));
        total += count;
    }
    s.push_str(&format!("\nNodes searched: {total}\n"));
    s
}

// ---------------------------------------------------------------------------
// Tests — validation against the canonical published reference node counts.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    const POS3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
    const POS4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
    const POS5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1";
    const POS6: &str = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1";

    fn perft_fen(fen: &str, depth: u32) -> u64 {
        let mut pos = Position::from_fen(fen).unwrap_or_else(|e| panic!("parse {fen}: {e}"));
        perft(&mut pos, depth)
    }

    // -- FAST tests: run under `cargo test` in debug, each well under a few sec.

    #[test]
    fn perft_startpos_fast() {
        assert_eq!(perft_fen(STARTPOS, 1), 20);
        assert_eq!(perft_fen(STARTPOS, 2), 400);
        assert_eq!(perft_fen(STARTPOS, 3), 8902);
        assert_eq!(perft_fen(STARTPOS, 4), 197281);
    }

    #[test]
    fn perft_kiwipete_fast() {
        assert_eq!(perft_fen(KIWIPETE, 1), 48);
        assert_eq!(perft_fen(KIWIPETE, 2), 2039);
        assert_eq!(perft_fen(KIWIPETE, 3), 97862);
    }

    #[test]
    fn perft_pos3_fast() {
        assert_eq!(perft_fen(POS3, 1), 14);
        assert_eq!(perft_fen(POS3, 2), 191);
        assert_eq!(perft_fen(POS3, 3), 2812);
        assert_eq!(perft_fen(POS3, 4), 43238);
    }

    #[test]
    fn perft_pos4_fast() {
        assert_eq!(perft_fen(POS4, 1), 6);
        assert_eq!(perft_fen(POS4, 2), 264);
        assert_eq!(perft_fen(POS4, 3), 9467);
    }

    #[test]
    fn perft_pos5_fast() {
        assert_eq!(perft_fen(POS5, 1), 44);
        assert_eq!(perft_fen(POS5, 2), 1486);
        assert_eq!(perft_fen(POS5, 3), 62379);
    }

    #[test]
    fn perft_pos6_fast() {
        assert_eq!(perft_fen(POS6, 1), 46);
        assert_eq!(perft_fen(POS6, 2), 2079);
        assert_eq!(perft_fen(POS6, 3), 89890);
    }

    // -- DEEP tests: `#[ignore]`; run with `cargo test --release -- --ignored`.

    #[test]
    #[ignore]
    fn perft_startpos_deep() {
        assert_eq!(perft_fen(STARTPOS, 5), 4865609);
        assert_eq!(perft_fen(STARTPOS, 6), 119060324);
    }

    #[test]
    #[ignore]
    fn perft_kiwipete_deep() {
        assert_eq!(perft_fen(KIWIPETE, 4), 4085603);
        assert_eq!(perft_fen(KIWIPETE, 5), 193690690);
    }

    #[test]
    #[ignore]
    fn perft_pos3_deep() {
        assert_eq!(perft_fen(POS3, 5), 674624);
        assert_eq!(perft_fen(POS3, 6), 11030083);
    }

    #[test]
    #[ignore]
    fn perft_pos4_deep() {
        assert_eq!(perft_fen(POS4, 4), 422333);
        assert_eq!(perft_fen(POS4, 5), 15833292);
    }

    #[test]
    #[ignore]
    fn perft_pos5_deep() {
        assert_eq!(perft_fen(POS5, 4), 2103487);
    }

    // -- perft_divide restores the position and totals correctly. -----------

    #[test]
    fn divide_totals_match_perft() {
        let mut pos = Position::from_fen(KIWIPETE).unwrap();
        let fen_before = pos.to_fen();
        let divide = perft_divide(&mut pos, 3);
        assert_eq!(pos.to_fen(), fen_before, "divide must restore the position");
        let total: u64 = divide.iter().map(|(_, c)| c).sum();
        assert_eq!(total, 97862);
        assert_eq!(divide.len(), 48); // one entry per root move
    }
}

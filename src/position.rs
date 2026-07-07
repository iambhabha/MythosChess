//! The board: a full chess position, plus FEN parsing and make/undo move.
//!
//! This is the heart of the engine. Everything above it (move generation,
//! search, evaluation) reads and mutates a [`Position`], so correctness here is
//! non-negotiable — a subtle make/undo bug corrupts the whole search tree.
//!
//! We keep three redundant views of where the pieces are, all kept in sync by
//! the [`add_piece`](Position::add_piece) / [`remove_piece`](Position::remove_piece)
//! / [`move_piece`](Position::move_piece) helpers:
//!
//! * a **mailbox** `[Option<Piece>; 64]` for "what is on this square?" lookups,
//! * **occupancy bitboards per color** for "all white pieces" queries,
//! * **occupancy bitboards per piece type** for "all rooks" queries.
//!
//! The [`Zobrist`](crate::zobrist) hash is maintained *incrementally*: every
//! piece add/remove XORs its key in/out, and side/castling/en-passant changes
//! are folded in as they happen. [`compute_key`](Position::compute_key)
//! recomputes the same value from scratch and is used by the tests to prove the
//! incremental bookkeeping never drifts.

use std::fmt;

use crate::attacks::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks,
};
use crate::bitboard::Bitboard;
use crate::types::{Color, Move, MoveType, Piece, PieceType, Square};
use crate::zobrist::ZOBRIST;

// ---------------------------------------------------------------------------
// Castling-right bit constants.
//
// The four rights pack into a single `u8`: one bit per (color, side). This
// matches the layout the Zobrist castling table is indexed by.
// ---------------------------------------------------------------------------

/// White may castle king-side (short, O-O).
pub const WHITE_OO: u8 = 1;
/// White may castle queen-side (long, O-O-O).
pub const WHITE_OOO: u8 = 2;
/// Black may castle king-side (short, O-O).
pub const BLACK_OO: u8 = 4;
/// Black may castle queen-side (long, O-O-O).
pub const BLACK_OOO: u8 = 8;

// ---------------------------------------------------------------------------
// Position.
// ---------------------------------------------------------------------------

/// A complete chess position: piece placement plus all the state the rules of
/// chess need (side to move, castling rights, en-passant square, move clocks)
/// and a Zobrist hash of the whole thing.
#[derive(Clone)]
pub struct Position {
    /// Mailbox: what piece (if any) sits on each square, indexed by `Square`.
    board: [Option<Piece>; 64],
    /// Occupancy per color, indexed by `Color::index()`.
    by_color: [Bitboard; 2],
    /// Occupancy per piece type, indexed by `PieceType::index()`.
    by_type: [Bitboard; 6],
    /// Whose turn it is.
    side_to_move: Color,
    /// Castling rights as a bitmask of `WHITE_OO | WHITE_OOO | BLACK_OO | BLACK_OOO`.
    castling_rights: u8,
    /// The square a pawn could be captured *on* by en passant (the target square
    /// behind a pawn that just made a double push), or `None`.
    en_passant: Option<Square>,
    /// Plies since the last capture or pawn move (for the 50-move rule).
    halfmove_clock: u16,
    /// The move number, incremented after each Black move; starts at 1.
    fullmove_number: u16,
    /// The Zobrist hash of this position, maintained incrementally.
    key: u64,
}

/// The information needed to reverse a [`make_move`](Position::make_move):
/// the pieces/state it destroyed that cannot be recovered from the move alone.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Undo {
    /// The piece captured by the move (including the pawn removed by en passant),
    /// or `None` if the move was not a capture.
    pub captured: Option<Piece>,
    /// The castling rights *before* the move.
    pub castling_rights: u8,
    /// The en-passant square *before* the move.
    pub en_passant: Option<Square>,
    /// The halfmove clock *before* the move.
    pub halfmove_clock: u16,
    /// The Zobrist key *before* the move.
    pub key: u64,
}

impl Position {
    // -----------------------------------------------------------------------
    // Construction.
    // -----------------------------------------------------------------------

    /// An empty board with White to move and no rights — the scratch state that
    /// FEN parsing fills in. Not a legal position on its own (no kings).
    fn empty() -> Position {
        Position {
            board: [None; 64],
            by_color: [Bitboard::EMPTY; 2],
            by_type: [Bitboard::EMPTY; 6],
            side_to_move: Color::White,
            castling_rights: 0,
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            key: 0,
        }
    }

    /// The standard chess starting position.
    pub fn startpos() -> Position {
        // Unwrap is safe: this constant FEN is always valid.
        Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("start position FEN must parse")
    }

    // -----------------------------------------------------------------------
    // Piece bookkeeping — the three board views + the hash, kept in lockstep.
    // -----------------------------------------------------------------------

    /// Put `piece` on `sq` (which must be empty). Updates mailbox, both
    /// occupancy bitboards, and the Zobrist key.
    #[inline]
    fn add_piece(&mut self, sq: Square, piece: Piece) {
        debug_assert!(self.board[sq.index()].is_none());
        self.board[sq.index()] = Some(piece);
        self.by_color[piece.color.index()].set(sq);
        self.by_type[piece.piece_type.index()].set(sq);
        self.key ^= ZOBRIST.piece(piece, sq);
    }

    /// Take the piece off `sq` (which must be occupied) and return it. Updates
    /// mailbox, both occupancy bitboards, and the Zobrist key.
    #[inline]
    fn remove_piece(&mut self, sq: Square) -> Piece {
        let piece = self.board[sq.index()].expect("remove_piece on an empty square");
        self.board[sq.index()] = None;
        self.by_color[piece.color.index()].clear(sq);
        self.by_type[piece.piece_type.index()].clear(sq);
        self.key ^= ZOBRIST.piece(piece, sq);
        piece
    }

    /// Move the piece on `from` to `to` (which must be empty). A single helper so
    /// the two bitboards and the hash all stay consistent.
    #[inline]
    fn move_piece(&mut self, from: Square, to: Square) {
        let piece = self.board[from.index()].expect("move_piece from an empty square");
        debug_assert!(self.board[to.index()].is_none());
        self.board[from.index()] = None;
        self.board[to.index()] = Some(piece);
        let mask = Bitboard::from_square(from) | Bitboard::from_square(to);
        self.by_color[piece.color.index()] ^= mask;
        self.by_type[piece.piece_type.index()] ^= mask;
        self.key ^= ZOBRIST.piece(piece, from) ^ ZOBRIST.piece(piece, to);
    }

    // -----------------------------------------------------------------------
    // Accessors.
    // -----------------------------------------------------------------------

    /// Whose turn it is to move.
    #[inline]
    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    /// The piece on `sq`, or `None` if the square is empty.
    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.board[sq.index()]
    }

    /// Every occupied square, regardless of color.
    #[inline]
    pub fn occupied(&self) -> Bitboard {
        self.by_color[0] | self.by_color[1]
    }

    /// All pieces of color `c`.
    #[inline]
    pub fn pieces(&self, c: Color) -> Bitboard {
        self.by_color[c.index()]
    }

    /// All pieces of type `pt`, both colors.
    #[inline]
    pub fn pieces_type(&self, pt: PieceType) -> Bitboard {
        self.by_type[pt.index()]
    }

    /// All pieces of color `c` and type `pt` (e.g. the white rooks).
    #[inline]
    pub fn pieces_cp(&self, c: Color, pt: PieceType) -> Bitboard {
        self.by_color[c.index()] & self.by_type[pt.index()]
    }

    /// Where color `c`'s king stands. Assumes exactly one king (always true in a
    /// legal position).
    #[inline]
    pub fn king_square(&self, c: Color) -> Square {
        self.pieces_cp(c, PieceType::King).lsb()
    }

    /// The current castling-rights mask.
    #[inline]
    pub fn castling_rights(&self) -> u8 {
        self.castling_rights
    }

    /// The current en-passant target square, if any.
    #[inline]
    pub fn en_passant(&self) -> Option<Square> {
        self.en_passant
    }

    /// The Zobrist hash of this position.
    #[inline]
    pub fn key(&self) -> u64 {
        self.key
    }

    /// Plies since the last capture or pawn move (the 50-move rule counter).
    #[inline]
    pub fn halfmove_clock(&self) -> u16 {
        self.halfmove_clock
    }

    /// The full-move number (increments after each Black move).
    #[inline]
    pub fn fullmove_number(&self) -> u16 {
        self.fullmove_number
    }

    // -----------------------------------------------------------------------
    // Attack queries.
    // -----------------------------------------------------------------------

    /// A bitboard of *every* piece (both colors) that attacks `sq`, given the
    /// supplied `occupied` set (so callers can query attacks through a
    /// hypothetical occupancy, e.g. for x-ray or pinned-piece logic).
    ///
    /// Pawns need care: a square `sq` is attacked by a *white* pawn exactly when
    /// a *black* pawn placed on `sq` would attack that white pawn's square — the
    /// attack relation is symmetric, so we look up the *opposite* color's pawn
    /// attacks from `sq` and intersect with the pawns of the color we want.
    pub fn attackers_to(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        let pawns = self.by_type[PieceType::Pawn.index()];
        let knights = self.by_type[PieceType::Knight.index()];
        let kings = self.by_type[PieceType::King.index()];
        let bishops_queens =
            self.by_type[PieceType::Bishop.index()] | self.by_type[PieceType::Queen.index()];
        let rooks_queens =
            self.by_type[PieceType::Rook.index()] | self.by_type[PieceType::Queen.index()];

        // White pawns attacking `sq`: pawns that sit where a black pawn on `sq`
        // would attack; and vice-versa for black pawns.
        let white_pawn_attackers = pawn_attacks(Color::Black, sq) & self.by_color[Color::White.index()] & pawns;
        let black_pawn_attackers = pawn_attacks(Color::White, sq) & self.by_color[Color::Black.index()] & pawns;

        white_pawn_attackers
            | black_pawn_attackers
            | (knight_attacks(sq) & knights)
            | (king_attacks(sq) & kings)
            | (bishop_attacks(sq, occupied) & bishops_queens)
            | (rook_attacks(sq, occupied) & rooks_queens)
    }

    /// Is `sq` attacked by any piece of color `by`, on the current board?
    pub fn is_attacked(&self, sq: Square, by: Color) -> bool {
        (self.attackers_to(sq, self.occupied()) & self.by_color[by.index()]).any()
    }

    /// Is the side-to-move's king currently attacked (i.e. in check)?
    #[inline]
    pub fn in_check(&self) -> bool {
        let us = self.side_to_move;
        self.is_attacked(self.king_square(us), !us)
    }

    // -----------------------------------------------------------------------
    // FEN parsing.
    // -----------------------------------------------------------------------

    /// Parse a full FEN string into a [`Position`].
    ///
    /// The six FEN fields are: piece placement, side to move, castling rights,
    /// en-passant square, halfmove clock, fullmove number. We are tolerant of a
    /// truncated FEN: a missing halfmove clock defaults to 0 and a missing
    /// fullmove number to 1.
    pub fn from_fen(fen: &str) -> Result<Position, String> {
        let mut pos = Position::empty();

        let mut fields = fen.split_whitespace();

        // --- Field 1: piece placement (rank 8 first, down to rank 1). -------
        let placement = fields
            .next()
            .ok_or_else(|| "FEN is empty".to_string())?;

        let mut rank: i32 = 7; // FEN starts at rank 8 (index 7).
        let mut file: i32 = 0;
        for ch in placement.chars() {
            match ch {
                '/' => {
                    if file != 8 {
                        return Err(format!("FEN rank {} has {} files, expected 8", 8 - rank, file));
                    }
                    rank -= 1;
                    file = 0;
                    if rank < 0 {
                        return Err("FEN has more than 8 ranks".to_string());
                    }
                }
                '1'..='9' => {
                    let n = (ch as u8 - b'0') as i32;
                    file += n;
                    if file > 8 {
                        return Err(format!("FEN rank overflows past file h: {placement}"));
                    }
                }
                _ => {
                    let piece = Piece::from_char(ch)
                        .ok_or_else(|| format!("FEN has invalid piece char '{ch}'"))?;
                    if file >= 8 {
                        return Err(format!("FEN rank overflows past file h: {placement}"));
                    }
                    if rank < 0 {
                        return Err("FEN has too many ranks".to_string());
                    }
                    pos.add_piece(Square::make(file as u8, rank as u8), piece);
                    file += 1;
                }
            }
        }
        if rank != 0 || file != 8 {
            return Err(format!("FEN placement is not a full 8x8 board: {placement}"));
        }

        // --- Field 2: side to move. -----------------------------------------
        let side = fields.next().unwrap_or("w");
        pos.side_to_move = match side {
            "w" | "W" => Color::White,
            "b" | "B" => Color::Black,
            other => return Err(format!("FEN side to move must be 'w' or 'b', got '{other}'")),
        };

        // --- Field 3: castling rights. --------------------------------------
        let castling = fields.next().unwrap_or("-");
        if castling != "-" {
            for ch in castling.chars() {
                match ch {
                    'K' => pos.castling_rights |= WHITE_OO,
                    'Q' => pos.castling_rights |= WHITE_OOO,
                    'k' => pos.castling_rights |= BLACK_OO,
                    'q' => pos.castling_rights |= BLACK_OOO,
                    other => {
                        return Err(format!("FEN has invalid castling char '{other}'"));
                    }
                }
            }
        }

        // --- Field 4: en-passant target square. -----------------------------
        let ep = fields.next().unwrap_or("-");
        if ep != "-" {
            let bytes = ep.as_bytes();
            if bytes.len() != 2 || !(b'a'..=b'h').contains(&bytes[0]) || !(b'1'..=b'8').contains(&bytes[1]) {
                return Err(format!("FEN en-passant square is malformed: '{ep}'"));
            }
            let file = bytes[0] - b'a';
            let rank = bytes[1] - b'1';
            pos.en_passant = Some(Square::make(file, rank));
        }

        // --- Field 5: halfmove clock (default 0). ---------------------------
        pos.halfmove_clock = match fields.next() {
            Some(s) => s
                .parse::<u16>()
                .map_err(|_| format!("FEN halfmove clock is not a number: '{s}'"))?,
            None => 0,
        };

        // --- Field 6: fullmove number (default 1). --------------------------
        pos.fullmove_number = match fields.next() {
            Some(s) => s
                .parse::<u16>()
                .map_err(|_| format!("FEN fullmove number is not a number: '{s}'"))?,
            None => 1,
        };
        if pos.fullmove_number == 0 {
            pos.fullmove_number = 1;
        }

        // Fold the non-piece state into the (piece-only, so far) key so that the
        // incremental key matches `compute_key` right out of the gate.
        pos.key ^= ZOBRIST.castle(pos.castling_rights);
        if let Some(ep_sq) = pos.en_passant {
            pos.key ^= ZOBRIST.ep(ep_sq.file());
        }
        if pos.side_to_move == Color::Black {
            pos.key ^= ZOBRIST.side();
        }

        Ok(pos)
    }

    // -----------------------------------------------------------------------
    // FEN output.
    // -----------------------------------------------------------------------

    /// Render this position back into a FEN string. Round-trips with
    /// [`from_fen`](Position::from_fen).
    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        // --- Field 1: piece placement, rank 8 down to rank 1. ---------------
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let sq = Square::make(file, rank);
                match self.board[sq.index()] {
                    Some(piece) => {
                        if empty > 0 {
                            fen.push((b'0' + empty) as char);
                            empty = 0;
                        }
                        fen.push(piece.to_char());
                    }
                    None => empty += 1,
                }
            }
            if empty > 0 {
                fen.push((b'0' + empty) as char);
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        // --- Field 2: side to move. -----------------------------------------
        fen.push(' ');
        fen.push(match self.side_to_move {
            Color::White => 'w',
            Color::Black => 'b',
        });

        // --- Field 3: castling rights. --------------------------------------
        fen.push(' ');
        if self.castling_rights == 0 {
            fen.push('-');
        } else {
            if self.castling_rights & WHITE_OO != 0 {
                fen.push('K');
            }
            if self.castling_rights & WHITE_OOO != 0 {
                fen.push('Q');
            }
            if self.castling_rights & BLACK_OO != 0 {
                fen.push('k');
            }
            if self.castling_rights & BLACK_OOO != 0 {
                fen.push('q');
            }
        }

        // --- Field 4: en-passant square. ------------------------------------
        fen.push(' ');
        match self.en_passant {
            Some(sq) => fen.push_str(&sq.to_string()),
            None => fen.push('-'),
        }

        // --- Fields 5 & 6: clocks. ------------------------------------------
        fen.push(' ');
        fen.push_str(&self.halfmove_clock.to_string());
        fen.push(' ');
        fen.push_str(&self.fullmove_number.to_string());

        fen
    }

    // -----------------------------------------------------------------------
    // Zobrist recompute (reference; used to validate the incremental key).
    // -----------------------------------------------------------------------

    /// Recompute the Zobrist hash from scratch. Used by tests to prove the
    /// incrementally-maintained [`key`](Position::key) never drifts.
    pub fn compute_key(&self) -> u64 {
        let mut key = 0u64;
        for sq in 0..64 {
            let sq = Square(sq as u8);
            if let Some(piece) = self.board[sq.index()] {
                key ^= ZOBRIST.piece(piece, sq);
            }
        }
        key ^= ZOBRIST.castle(self.castling_rights);
        if let Some(ep_sq) = self.en_passant {
            key ^= ZOBRIST.ep(ep_sq.file());
        }
        if self.side_to_move == Color::Black {
            key ^= ZOBRIST.side();
        }
        key
    }

    // -----------------------------------------------------------------------
    // make / undo.
    // -----------------------------------------------------------------------

    /// Apply a pseudo-legal move, returning the [`Undo`] needed to reverse it.
    ///
    /// Handles all move kinds: normal moves and captures, double pawn pushes
    /// (which set the en-passant target), en-passant captures (which remove the
    /// captured pawn from *behind* the target square), castling (moving both the
    /// king and rook), and promotions. All three board views, the side to move,
    /// castling rights, en-passant square, both clocks, and the Zobrist key are
    /// updated. The Zobrist key is maintained incrementally.
    pub fn make_move(&mut self, m: Move) -> Undo {
        let us = self.side_to_move;
        let them = !us;
        let from = m.from_sq();
        let to = m.to_sq(); // NB: for castling this is the ROOK's square.
        let move_type = m.move_type();

        // Snapshot the reversible state before we touch anything.
        let undo = Undo {
            captured: None, // filled in below for capturing moves.
            castling_rights: self.castling_rights,
            en_passant: self.en_passant,
            halfmove_clock: self.halfmove_clock,
            key: self.key,
        };

        // The moving piece (read before we start mutating the board).
        let moving = self.board[from.index()].expect("make_move from an empty square");

        // Clear the old en-passant key up front; if this move creates a new one
        // we XOR the new key in later. (Zero-length no-op if there was none.)
        if let Some(ep_sq) = self.en_passant {
            self.key ^= ZOBRIST.ep(ep_sq.file());
        }
        self.en_passant = None;

        let mut captured: Option<Piece> = None;

        match move_type {
            MoveType::Castling => {
                // `to` is the rook's square. Derive the king's true destination
                // and the rook's destination from which side we're castling.
                let king_from = from;
                let rook_from = to;
                let (king_to, rook_to) = if rook_from.file() > king_from.file() {
                    // King-side: king to g-file, rook to f-file.
                    (
                        Square::make(6, king_from.rank()),
                        Square::make(5, king_from.rank()),
                    )
                } else {
                    // Queen-side: king to c-file, rook to d-file.
                    (
                        Square::make(2, king_from.rank()),
                        Square::make(3, king_from.rank()),
                    )
                };
                self.move_piece(king_from, king_to);
                self.move_piece(rook_from, rook_to);
            }

            MoveType::EnPassant => {
                // The captured pawn is not on `to`; it's on the square directly
                // "behind" the target from the mover's perspective (same file as
                // `to`, same rank as `from`).
                let cap_sq = Square::make(to.file(), from.rank());
                captured = Some(self.remove_piece(cap_sq));
                self.move_piece(from, to);
            }

            MoveType::Promotion => {
                // Capture on the destination, if any.
                if self.board[to.index()].is_some() {
                    captured = Some(self.remove_piece(to));
                }
                // Replace the pawn with the promoted piece.
                self.remove_piece(from);
                self.add_piece(to, Piece::new(us, m.promotion_type()));
            }

            MoveType::Normal => {
                // Capture on the destination, if any.
                if self.board[to.index()].is_some() {
                    captured = Some(self.remove_piece(to));
                }
                self.move_piece(from, to);

                // A double pawn push exposes an en-passant target on the square
                // it jumped over.
                if moving.piece_type == PieceType::Pawn {
                    let from_rank = from.rank() as i32;
                    let to_rank = to.rank() as i32;
                    if (to_rank - from_rank).abs() == 2 {
                        let ep_rank = (from_rank + to_rank) / 2;
                        let ep_sq = Square::make(from.file(), ep_rank as u8);
                        self.en_passant = Some(ep_sq);
                        self.key ^= ZOBRIST.ep(ep_sq.file());
                    }
                }
            }
        }

        // --- Castling rights: revoke on king/rook departure or rook capture. -
        // XOR out the old rights key, mutate, XOR the new one back in.
        self.key ^= ZOBRIST.castle(self.castling_rights);
        self.castling_rights &= castling_mask(from);
        self.castling_rights &= castling_mask(to);
        self.key ^= ZOBRIST.castle(self.castling_rights);

        // --- 50-move clock: reset on a pawn move or any capture, else +1. ----
        if moving.piece_type == PieceType::Pawn || captured.is_some() {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        // --- Fullmove number: bumps after Black completes a move. ------------
        if us == Color::Black {
            self.fullmove_number += 1;
        }

        // --- Side to move flips. ---------------------------------------------
        self.side_to_move = them;
        self.key ^= ZOBRIST.side();

        Undo {
            captured,
            ..undo
        }
    }

    /// Reverse a [`make_move`]. `m` must be the exact move that was made and
    /// `undo` the value it returned; the position is restored byte-for-byte
    /// (including the Zobrist key).
    pub fn undo_move(&mut self, m: Move, undo: Undo) {
        // The side that *made* the move is the opposite of the current one.
        let us = !self.side_to_move;
        let from = m.from_sq();
        let to = m.to_sq(); // rook's square for castling.
        let move_type = m.move_type();

        match move_type {
            MoveType::Castling => {
                let king_from = from;
                let rook_from = to;
                let (king_to, rook_to) = if rook_from.file() > king_from.file() {
                    (
                        Square::make(6, king_from.rank()),
                        Square::make(5, king_from.rank()),
                    )
                } else {
                    (
                        Square::make(2, king_from.rank()),
                        Square::make(3, king_from.rank()),
                    )
                };
                // Move the king and rook back to their origins.
                self.move_piece(king_to, king_from);
                self.move_piece(rook_to, rook_from);
            }

            MoveType::EnPassant => {
                // Put the moving pawn back, then restore the captured pawn on the
                // square it actually stood on.
                self.move_piece(to, from);
                let cap_sq = Square::make(to.file(), from.rank());
                self.add_piece(cap_sq, undo.captured.expect("en-passant undo without a captured pawn"));
            }

            MoveType::Promotion => {
                // Remove the promoted piece and restore the pawn on `from`.
                self.remove_piece(to);
                self.add_piece(from, Piece::new(us, PieceType::Pawn));
                if let Some(captured) = undo.captured {
                    self.add_piece(to, captured);
                }
            }

            MoveType::Normal => {
                self.move_piece(to, from);
                if let Some(captured) = undo.captured {
                    self.add_piece(to, captured);
                }
            }
        }

        // Restore all the scalar state directly from the snapshot — cheaper and
        // less error-prone than trying to invert each field.
        self.side_to_move = us;
        self.castling_rights = undo.castling_rights;
        self.en_passant = undo.en_passant;
        self.halfmove_clock = undo.halfmove_clock;
        if us == Color::Black {
            self.fullmove_number -= 1;
        }
        self.key = undo.key;
    }

    // -----------------------------------------------------------------------
    // Null move — "pass the turn" without moving a piece.
    // -----------------------------------------------------------------------

    /// Hand the turn to the opponent without moving any piece, returning the
    /// [`Undo`] needed to reverse it.
    ///
    /// A null move is a search device (used by null-move pruning): we ask "if I
    /// could pass, is my position still so good the opponent can't rescue it?".
    /// It is only ever applied when the side to move is **not in check** (passing
    /// while in check would leave the king captured). No piece moves, but the
    /// en-passant right is cleared (a passed turn cannot be answered by an
    /// en-passant capture of a pawn that never double-pushed), the side to move
    /// flips, the clocks advance, and the Zobrist key is kept in sync.
    pub fn make_null_move(&mut self) -> Undo {
        let us = self.side_to_move;

        // Snapshot the reversible state so `undo_null_move` restores it exactly.
        let undo = Undo {
            captured: None, // a null move never captures.
            castling_rights: self.castling_rights,
            en_passant: self.en_passant,
            halfmove_clock: self.halfmove_clock,
            key: self.key,
        };

        // Clear any en-passant target and fold its key out of the hash.
        if let Some(ep_sq) = self.en_passant {
            self.key ^= ZOBRIST.ep(ep_sq.file());
        }
        self.en_passant = None;

        // A passed turn is a reversible "move": bump the 50-move clock. (Castling
        // rights never change — no king or rook moved.)
        self.halfmove_clock += 1;

        // Fullmove number bumps after Black completes its turn, exactly as in
        // `make_move`, so the FEN round-trips.
        if us == Color::Black {
            self.fullmove_number += 1;
        }

        // Flip the side to move and toggle its key.
        self.side_to_move = !us;
        self.key ^= ZOBRIST.side();

        undo
    }

    /// Reverse a [`make_null_move`], restoring the side to move, en-passant
    /// square, clocks, and Zobrist key from the snapshot in `undo`.
    pub fn undo_null_move(&mut self, undo: Undo) {
        // The side that passed is the opposite of the current one.
        let us = !self.side_to_move;
        self.side_to_move = us;
        self.castling_rights = undo.castling_rights;
        self.en_passant = undo.en_passant;
        self.halfmove_clock = undo.halfmove_clock;
        if us == Color::Black {
            self.fullmove_number -= 1;
        }
        self.key = undo.key;
    }
}

/// The castling-rights mask that survives a piece leaving/arriving on `sq`.
///
/// A king or rook moving *from* its home square, or an enemy capturing a rook
/// *on* its home square, revokes the matching right. We express this as an AND
/// mask: `rights &= castling_mask(sq)` clears exactly the affected bit(s), and
/// is a harmless no-op for every other square.
#[inline]
fn castling_mask(sq: Square) -> u8 {
    match sq {
        Square::E1 => !(WHITE_OO | WHITE_OOO), // white king moved
        Square::H1 => !WHITE_OO,               // white king-side rook
        Square::A1 => !WHITE_OOO,              // white queen-side rook
        Square::E8 => !(BLACK_OO | BLACK_OOO), // black king moved
        Square::H8 => !BLACK_OO,               // black king-side rook
        Square::A8 => !BLACK_OOO,              // black queen-side rook
        _ => 0xFF,                             // nothing to revoke
    }
}

// ---------------------------------------------------------------------------
// Pretty printing.
// ---------------------------------------------------------------------------

impl fmt::Display for Position {
    /// A human-friendly board grid (rank 8 on top) plus the FEN on the last line.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  +-----------------+")?;
        for rank in (0..8).rev() {
            write!(f, "{} | ", rank + 1)?;
            for file in 0..8 {
                let sq = Square::make(file, rank);
                let c = match self.board[sq.index()] {
                    Some(piece) => piece.to_char(),
                    None => '.',
                };
                write!(f, "{c} ")?;
            }
            writeln!(f, "|")?;
        }
        writeln!(f, "  +-----------------+")?;
        writeln!(f, "    a b c d e f g h")?;
        write!(f, "FEN: {}", self.to_fen())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const KIWIPETE_FEN: &str =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    // -- FEN --------------------------------------------------------------

    #[test]
    fn startpos_to_fen() {
        assert_eq!(Position::startpos().to_fen(), STARTPOS_FEN);
    }

    #[test]
    fn fen_round_trip() {
        let fens = [
            STARTPOS_FEN,
            KIWIPETE_FEN,
            // An en-passant position: after 1. e4 the ep square is e3... use a
            // position with a live ep target on c6.
            "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3",
            // A promotion-ready position (white pawn on the 7th).
            "8/P7/8/8/8/8/8/k1K5 w - - 0 1",
            // Another with black to move and partial castling rights.
            "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 5 12",
        ];
        for fen in fens {
            let pos = Position::from_fen(fen).unwrap_or_else(|e| panic!("parse {fen}: {e}"));
            assert_eq!(pos.to_fen(), fen, "round-trip failed for {fen}");
        }
    }

    #[test]
    fn from_fen_defaults_clocks() {
        // Missing halfmove/fullmove default to 0 / 1.
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -").unwrap();
        assert_eq!(pos.halfmove_clock(), 0);
        assert_eq!(pos.fullmove_number(), 1);
    }

    #[test]
    fn from_fen_rejects_garbage() {
        assert!(Position::from_fen("").is_err());
        assert!(Position::from_fen("xxxx w - - 0 1").is_err());
        assert!(Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1").is_err());
    }

    // -- Zobrist key consistency ------------------------------------------

    #[test]
    fn incremental_key_matches_recompute_after_parse() {
        let fens = [
            STARTPOS_FEN,
            KIWIPETE_FEN,
            "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3",
            "8/P7/8/8/8/8/8/k1K5 w - - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 5 12",
        ];
        for fen in fens {
            let pos = Position::from_fen(fen).unwrap();
            assert_eq!(pos.key(), pos.compute_key(), "key mismatch for {fen}");
        }
    }

    // -- Attack / check queries -------------------------------------------

    #[test]
    fn startpos_not_in_check() {
        assert!(!Position::startpos().in_check());
    }

    #[test]
    fn detects_check() {
        // Black king on e8, white rook on e1 down the open e-file: black in check.
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/4R2K b - - 0 1").unwrap();
        assert!(pos.in_check());
        // And the white king (not to move) is not attacked.
        assert!(!pos.is_attacked(pos.king_square(Color::White), Color::Black));
    }

    #[test]
    fn attackers_to_sanity() {
        // White: knight b1, bishop c1-diagonal, pawn d2; probe attacks on d3/e2 etc.
        // Build a crafted spot: a white pawn on e4 attacks d5 and f5.
        let pos = Position::from_fen("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1").unwrap();
        let attackers = pos.attackers_to(Square::D5, pos.occupied());
        assert!(attackers.contains(Square::E4), "e4 pawn should attack d5");

        // A knight on d4 attacks e6; place one and check.
        let pos = Position::from_fen("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1").unwrap();
        let attackers = pos.attackers_to(Square::E6, pos.occupied());
        assert!(attackers.contains(Square::D4), "d4 knight should attack e6");

        // A rook sees along a file until blocked.
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let attackers = pos.attackers_to(Square::A8, pos.occupied());
        assert!(attackers.contains(Square::A1), "a1 rook should attack a8");
    }

    // -- make / undo restores everything ----------------------------------

    /// Assert that making `m` then undoing it restores the position exactly,
    /// including FEN, the running key, and that the running key equals the
    /// full recompute both before and after.
    fn assert_make_undo_restores(fen: &str, m: Move) {
        let mut pos = Position::from_fen(fen).unwrap_or_else(|e| panic!("parse {fen}: {e}"));
        let fen_before = pos.to_fen();
        let key_before = pos.key();
        assert_eq!(key_before, pos.compute_key(), "key wrong before move ({fen})");

        let undo = pos.make_move(m);
        // After the move the running key must still match the full recompute.
        assert_eq!(
            pos.key(),
            pos.compute_key(),
            "incremental key drifted after {m} from {fen}"
        );

        pos.undo_move(m, undo);
        assert_eq!(pos.to_fen(), fen_before, "FEN not restored after undo of {m} from {fen}");
        assert_eq!(pos.key(), key_before, "key not restored after undo of {m} from {fen}");
        assert_eq!(pos.key(), pos.compute_key(), "key wrong after undo ({fen})");
    }

    #[test]
    fn make_undo_normal_move() {
        assert_make_undo_restores(STARTPOS_FEN, Move::normal(Square::G1, Square::F3));
    }

    #[test]
    fn make_undo_double_push_sets_ep() {
        // Verify the ep target actually gets set mid-move, then undoes cleanly.
        let mut pos = Position::startpos();
        let m = Move::normal(Square::E2, Square::E4);
        let undo = pos.make_move(m);
        assert_eq!(pos.en_passant(), Some(Square::E3));
        assert_eq!(pos.key(), pos.compute_key());
        pos.undo_move(m, undo);
        assert_eq!(pos.en_passant(), None);
        assert_eq!(pos.to_fen(), STARTPOS_FEN);
    }

    #[test]
    fn make_undo_capture() {
        // Kiwipete has plenty of captures; white pawn e4 has nothing, so craft one:
        // white pawn d5 captures black pawn... use a simple crafted capture.
        // White rook a1 captures black rook a8.
        assert_make_undo_restores(
            "r6k/8/8/8/8/8/8/R6K w - - 0 1",
            Move::normal(Square::A1, Square::A8),
        );
    }

    #[test]
    fn make_undo_en_passant_capture() {
        // Black just played ...c5, white pawn on d5 takes en passant on c6.
        let fen = "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3";
        assert_make_undo_restores(fen, Move::en_passant(Square::D5, Square::C6));

        // Also verify the captured pawn is actually removed mid-move.
        let mut pos = Position::from_fen(fen).unwrap();
        let m = Move::en_passant(Square::D5, Square::C6);
        let undo = pos.make_move(m);
        assert_eq!(pos.piece_at(Square::C5), None, "ep-captured pawn must be gone");
        assert_eq!(pos.piece_at(Square::C6).unwrap().piece_type, PieceType::Pawn);
        assert_eq!(pos.key(), pos.compute_key());
        pos.undo_move(m, undo);
        assert_eq!(pos.piece_at(Square::C5).unwrap().color, Color::Black);
        assert_eq!(pos.to_fen(), fen);
    }

    #[test]
    fn make_undo_castling_both_sides() {
        // White king-side (e1 -> h1 encoding), queen-side, and both black sides.
        let fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";

        // White O-O: king e1 -> g1, rook h1 -> f1.
        let oo = Move::castling(Square::E1, Square::H1);
        assert_make_undo_restores(fen, oo);
        {
            let mut pos = Position::from_fen(fen).unwrap();
            let u = pos.make_move(oo);
            assert_eq!(pos.piece_at(Square::G1).unwrap().piece_type, PieceType::King);
            assert_eq!(pos.piece_at(Square::F1).unwrap().piece_type, PieceType::Rook);
            assert_eq!(pos.piece_at(Square::E1), None);
            assert_eq!(pos.piece_at(Square::H1), None);
            assert_eq!(pos.key(), pos.compute_key());
            pos.undo_move(oo, u);
            assert_eq!(pos.to_fen(), fen);
        }

        // White O-O-O: king e1 -> c1, rook a1 -> d1.
        let ooo = Move::castling(Square::E1, Square::A1);
        assert_make_undo_restores(fen, ooo);
        {
            let mut pos = Position::from_fen(fen).unwrap();
            let u = pos.make_move(ooo);
            assert_eq!(pos.piece_at(Square::C1).unwrap().piece_type, PieceType::King);
            assert_eq!(pos.piece_at(Square::D1).unwrap().piece_type, PieceType::Rook);
            pos.undo_move(ooo, u);
            assert_eq!(pos.to_fen(), fen);
        }

        // Black O-O and O-O-O (black to move).
        let fen_b = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1";
        let boo = Move::castling(Square::E8, Square::H8);
        assert_make_undo_restores(fen_b, boo);
        {
            let mut pos = Position::from_fen(fen_b).unwrap();
            let u = pos.make_move(boo);
            assert_eq!(pos.piece_at(Square::G8).unwrap().piece_type, PieceType::King);
            assert_eq!(pos.piece_at(Square::F8).unwrap().piece_type, PieceType::Rook);
            pos.undo_move(boo, u);
            assert_eq!(pos.to_fen(), fen_b);
        }
        let booo = Move::castling(Square::E8, Square::A8);
        assert_make_undo_restores(fen_b, booo);
    }

    #[test]
    fn make_undo_promotion() {
        // White pawn a7 -> a8 promoting to a queen (no capture).
        let fen = "k7/P7/8/8/8/8/8/K7 w - - 0 1";
        let promo = Move::promotion(Square::A7, Square::A8, PieceType::Queen);
        assert_make_undo_restores(fen, promo);
        {
            let mut pos = Position::from_fen(fen).unwrap();
            let u = pos.make_move(promo);
            let p = pos.piece_at(Square::A8).unwrap();
            assert_eq!(p.piece_type, PieceType::Queen);
            assert_eq!(p.color, Color::White);
            assert_eq!(pos.key(), pos.compute_key());
            pos.undo_move(promo, u);
            assert_eq!(pos.to_fen(), fen);
        }

        // Promotion with capture: pawn b7 takes a rook on a8 and promotes.
        let fen_cap = "r6k/1P6/8/8/8/8/8/K7 w - - 0 1";
        let promo_cap = Move::promotion(Square::B7, Square::A8, PieceType::Knight);
        assert_make_undo_restores(fen_cap, promo_cap);
    }

    #[test]
    fn make_undo_revokes_castling_on_rook_capture() {
        // White rook captures black's a8 rook -> black loses queen-side right.
        let fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
        let mut pos = Position::from_fen(fen).unwrap();
        let m = Move::normal(Square::A1, Square::A8);
        let u = pos.make_move(m);
        // Black queen-side (q) is gone; white queen-side (Q) is gone too (rook left a1).
        assert_eq!(pos.castling_rights() & BLACK_OOO, 0);
        assert_eq!(pos.castling_rights() & WHITE_OOO, 0);
        assert_eq!(pos.key(), pos.compute_key());
        pos.undo_move(m, u);
        assert_eq!(pos.to_fen(), fen);
    }

    // -- A full move sequence made then unwound ----------------------------

    #[test]
    fn sequence_make_then_full_unwind() {
        let start = STARTPOS_FEN;
        let mut pos = Position::from_fen(start).unwrap();

        let moves = [
            Move::normal(Square::E2, Square::E4), // double push, sets ep
            Move::normal(Square::C7, Square::C5), // double push, sets ep
            Move::normal(Square::G1, Square::F3), // knight develop
            Move::normal(Square::D7, Square::D6),
            Move::normal(Square::F1, Square::E2), // bishop develop
            Move::normal(Square::B8, Square::C6),
            Move::castling(Square::E1, Square::H1), // white O-O
        ];

        let mut undos = Vec::new();
        for &m in &moves {
            let u = pos.make_move(m);
            assert_eq!(pos.key(), pos.compute_key(), "key drift after {m}");
            undos.push((m, u));
        }

        // Unwind in reverse.
        while let Some((m, u)) = undos.pop() {
            pos.undo_move(m, u);
            assert_eq!(pos.key(), pos.compute_key(), "key drift after undo {m}");
        }

        assert_eq!(pos.to_fen(), start, "position not restored after full unwind");
    }

    #[test]
    fn make_undo_null_move_restores_everything() {
        // A null move must round-trip the FEN and the running key exactly, and the
        // incremental key must equal the full recompute right after the pass —
        // including from a position that has a live en-passant target.
        let fens = [
            STARTPOS_FEN,
            KIWIPETE_FEN,
            // Live en-passant target on c6 (Black to answer): the pass must clear
            // it and fold its key out cleanly.
            "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3",
            // Black to move, partial rights, a non-zero halfmove clock.
            "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 5 12",
        ];
        for fen in fens {
            let mut pos = Position::from_fen(fen).unwrap_or_else(|e| panic!("parse {fen}: {e}"));
            let fen_before = pos.to_fen();
            let key_before = pos.key();

            let undo = pos.make_null_move();
            // After passing, the ep square is gone and the key still matches the
            // full recompute.
            assert_eq!(pos.en_passant(), None, "null move must clear ep ({fen})");
            assert_eq!(
                pos.key(),
                pos.compute_key(),
                "incremental key drifted after null move from {fen}"
            );

            pos.undo_null_move(undo);
            assert_eq!(pos.to_fen(), fen_before, "FEN not restored after undo_null_move ({fen})");
            assert_eq!(pos.key(), key_before, "key not restored after undo_null_move ({fen})");
            assert_eq!(pos.key(), pos.compute_key(), "key wrong after undo_null_move ({fen})");
        }
    }

    #[test]
    fn kiwipete_various_make_undo() {
        // Exercise make/undo on a rich middlegame from a variety of move kinds.
        let moves = [
            Move::normal(Square::E5, Square::G6),   // knight
            Move::normal(Square::D5, Square::E6),   // pawn capture
            Move::castling(Square::E1, Square::H1), // O-O
            Move::castling(Square::E1, Square::A1), // O-O-O
            Move::normal(Square::F3, Square::F6),   // queen slide (capture)
        ];
        for m in moves {
            assert_make_undo_restores(KIWIPETE_FEN, m);
        }
    }
}

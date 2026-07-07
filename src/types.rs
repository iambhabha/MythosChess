//! Core value types shared across the whole engine.
//!
//! These mirror the roles of Stockfish's `types.h` (Color, PieceType, Piece,
//! Square, Move) but are written in idiomatic Rust: small `Copy` newtypes /
//! enums with `const fn` helpers, so the compiler can fold them away.

use std::fmt;
use std::ops::Not;

// ---------------------------------------------------------------------------
// Color — the side to move / the owner of a piece.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    /// Number of colors — handy for sizing arrays: `[T; Color::NUM]`.
    pub const NUM: usize = 2;

    /// Use as an array index.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The opposite color.
    #[inline]
    pub const fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

impl Not for Color {
    type Output = Color;
    #[inline]
    fn not(self) -> Color {
        self.flip()
    }
}

// ---------------------------------------------------------------------------
// PieceType — pawn..king, without a color.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceType {
    pub const NUM: usize = 6;
    pub const ALL: [PieceType; Self::NUM] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn from_index(i: usize) -> Option<PieceType> {
        match i {
            0 => Some(PieceType::Pawn),
            1 => Some(PieceType::Knight),
            2 => Some(PieceType::Bishop),
            3 => Some(PieceType::Rook),
            4 => Some(PieceType::Queen),
            5 => Some(PieceType::King),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Piece — a colored piece (e.g. white knight). We store `Option<Piece>` in the
// board mailbox, so `Piece` itself is always a real piece.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Piece {
    pub color: Color,
    pub piece_type: PieceType,
}

impl Piece {
    #[inline]
    pub const fn new(color: Color, piece_type: PieceType) -> Piece {
        Piece { color, piece_type }
    }

    /// A dense 0..12 index: white pieces 0..5, black pieces 6..11.
    /// Useful for Zobrist tables and NNUE indexing later.
    #[inline]
    pub const fn index(self) -> usize {
        self.color.index() * PieceType::NUM + self.piece_type.index()
    }

    /// The FEN character for this piece (uppercase = white, lowercase = black).
    pub const fn to_char(self) -> char {
        let c = match self.piece_type {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
        };
        match self.color {
            Color::White => c.to_ascii_uppercase(),
            Color::Black => c,
        }
    }

    /// Parse a FEN piece character (e.g. 'N' = white knight, 'q' = black queen).
    pub const fn from_char(c: char) -> Option<Piece> {
        let color = if c.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let piece_type = match c.to_ascii_lowercase() {
            'p' => PieceType::Pawn,
            'n' => PieceType::Knight,
            'b' => PieceType::Bishop,
            'r' => PieceType::Rook,
            'q' => PieceType::Queen,
            'k' => PieceType::King,
            _ => return None,
        };
        Some(Piece { color, piece_type })
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

// ---------------------------------------------------------------------------
// Square — one of the 64 board squares, 0 = a1 .. 63 = h8 (rank-major).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Square(pub u8);

#[rustfmt::skip]
impl Square {
    pub const NUM: usize = 64;

    // Named constants for all 64 squares (rank-major: a1..h1, a2..h2, ...).
    pub const A1: Square = Square(0);  pub const B1: Square = Square(1);  pub const C1: Square = Square(2);  pub const D1: Square = Square(3);
    pub const E1: Square = Square(4);  pub const F1: Square = Square(5);  pub const G1: Square = Square(6);  pub const H1: Square = Square(7);
    pub const A2: Square = Square(8);  pub const B2: Square = Square(9);  pub const C2: Square = Square(10); pub const D2: Square = Square(11);
    pub const E2: Square = Square(12); pub const F2: Square = Square(13); pub const G2: Square = Square(14); pub const H2: Square = Square(15);
    pub const A3: Square = Square(16); pub const B3: Square = Square(17); pub const C3: Square = Square(18); pub const D3: Square = Square(19);
    pub const E3: Square = Square(20); pub const F3: Square = Square(21); pub const G3: Square = Square(22); pub const H3: Square = Square(23);
    pub const A4: Square = Square(24); pub const B4: Square = Square(25); pub const C4: Square = Square(26); pub const D4: Square = Square(27);
    pub const E4: Square = Square(28); pub const F4: Square = Square(29); pub const G4: Square = Square(30); pub const H4: Square = Square(31);
    pub const A5: Square = Square(32); pub const B5: Square = Square(33); pub const C5: Square = Square(34); pub const D5: Square = Square(35);
    pub const E5: Square = Square(36); pub const F5: Square = Square(37); pub const G5: Square = Square(38); pub const H5: Square = Square(39);
    pub const A6: Square = Square(40); pub const B6: Square = Square(41); pub const C6: Square = Square(42); pub const D6: Square = Square(43);
    pub const E6: Square = Square(44); pub const F6: Square = Square(45); pub const G6: Square = Square(46); pub const H6: Square = Square(47);
    pub const A7: Square = Square(48); pub const B7: Square = Square(49); pub const C7: Square = Square(50); pub const D7: Square = Square(51);
    pub const E7: Square = Square(52); pub const F7: Square = Square(53); pub const G7: Square = Square(54); pub const H7: Square = Square(55);
    pub const A8: Square = Square(56); pub const B8: Square = Square(57); pub const C8: Square = Square(58); pub const D8: Square = Square(59);
    pub const E8: Square = Square(60); pub const F8: Square = Square(61); pub const G8: Square = Square(62); pub const H8: Square = Square(63);

    /// Build a square from file (0..7) and rank (0..7).
    #[inline]
    pub const fn make(file: u8, rank: u8) -> Square {
        Square((rank << 3) | file)
    }

    /// Build a square from a raw 0..63 index, checking the range.
    #[inline]
    pub const fn from_index(i: usize) -> Option<Square> {
        if i < 64 { Some(Square(i as u8)) } else { None }
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// File 0..7 (a..h).
    #[inline]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    /// Rank 0..7 (1..8).
    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    /// Mirror vertically (a1 <-> a8). Used to view the board from Black's side.
    #[inline]
    pub const fn flip_rank(self) -> Square {
        Square(self.0 ^ 56)
    }

    /// Mirror horizontally (a1 <-> h1).
    #[inline]
    pub const fn flip_file(self) -> Square {
        Square(self.0 ^ 7)
    }

    /// Add a direction offset, returning `None` if it would leave the board
    /// (detected by the file jumping more than 2 columns — catches wraps).
    #[inline]
    pub const fn offset(self, d: Direction) -> Option<Square> {
        let target = self.0 as i8 + d as i8;
        if target < 0 || target >= 64 {
            return None;
        }
        let target = target as u8;
        // Reject horizontal wrap: a legal single step never changes file by >2.
        let df = (target & 7) as i8 - (self.0 & 7) as i8;
        if df > 2 || df < -2 {
            None
        } else {
            Some(Square(target))
        }
    }
}

impl fmt::Display for Square {
    /// Algebraic notation, e.g. "e4".
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        write!(f, "{file}{rank}")
    }
}

// ---------------------------------------------------------------------------
// Direction — a step offset on the 0..63 board (also used to shift bitboards).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i8)]
pub enum Direction {
    North = 8,
    South = -8,
    East = 1,
    West = -1,
    NorthEast = 9,
    NorthWest = 7,
    SouthEast = -7,
    SouthWest = -9,
}

// ---------------------------------------------------------------------------
// Move — a packed 16-bit move, same layout idea as Stockfish.
//
//   bits  0..5  : destination square (0..63)
//   bits  6..11 : origin square (0..63)
//   bits 12..13 : promotion piece: 0=Knight, 1=Bishop, 2=Rook, 3=Queen
//   bits 14..15 : move type: 0=Normal, 1=Promotion, 2=EnPassant, 3=Castling
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum MoveType {
    Normal = 0,
    Promotion = 1,
    EnPassant = 2,
    Castling = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move(u16);

impl Move {
    /// The "no move" sentinel (from == to == a1).
    pub const NONE: Move = Move(0);
    /// The null move (a distinct impossible move; from == to == b1).
    pub const NULL: Move = Move(65);

    #[inline]
    const fn encode(from: Square, to: Square, promo_code: u16, ty: MoveType) -> Move {
        Move(
            (to.0 as u16)
                | ((from.0 as u16) << 6)
                | (promo_code << 12)
                | ((ty as u16) << 14),
        )
    }

    /// A normal (non-special) move.
    #[inline]
    pub const fn normal(from: Square, to: Square) -> Move {
        Move::encode(from, to, 0, MoveType::Normal)
    }

    /// A promotion move; `promo` must be Knight/Bishop/Rook/Queen.
    #[inline]
    pub const fn promotion(from: Square, to: Square, promo: PieceType) -> Move {
        // promo_code = piece_type - Knight  (Knight=1 -> 0 .. Queen=4 -> 3)
        Move::encode(from, to, (promo as u16) - 1, MoveType::Promotion)
    }

    #[inline]
    pub const fn en_passant(from: Square, to: Square) -> Move {
        Move::encode(from, to, 0, MoveType::EnPassant)
    }

    /// Castling, encoded (Stockfish-style) as "king moves to the rook's square".
    #[inline]
    pub const fn castling(from: Square, rook: Square) -> Move {
        Move::encode(from, rook, 0, MoveType::Castling)
    }

    #[inline]
    pub const fn to_sq(self) -> Square {
        Square((self.0 & 0x3f) as u8)
    }

    #[inline]
    pub const fn from_sq(self) -> Square {
        Square(((self.0 >> 6) & 0x3f) as u8)
    }

    #[inline]
    pub const fn move_type(self) -> MoveType {
        match (self.0 >> 14) & 3 {
            0 => MoveType::Normal,
            1 => MoveType::Promotion,
            2 => MoveType::EnPassant,
            _ => MoveType::Castling,
        }
    }

    /// The promotion piece type, only meaningful when `move_type == Promotion`.
    #[inline]
    pub const fn promotion_type(self) -> PieceType {
        // promo_code + Knight
        match PieceType::from_index(((self.0 >> 12) & 3) as usize + 1) {
            Some(pt) => pt,
            None => PieceType::Knight,
        }
    }

    /// The raw 16-bit encoding (used later for history-table indexing).
    #[inline]
    pub const fn raw(self) -> u16 {
        self.0
    }

    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == Move::NONE.0
    }
}

impl fmt::Display for Move {
    /// UCI long-algebraic notation, e.g. "e2e4", "e7e8q", or "e1g1" (castling).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            return write!(f, "0000");
        }
        let from = self.from_sq();
        // Internally castling is stored king -> rook; standard UCI writes it as
        // king -> its landing square (g-file for O-O, c-file for O-O-O).
        let to = if let MoveType::Castling = self.move_type() {
            let rook = self.to_sq();
            let king_to_file = if rook.file() > from.file() { 6 } else { 2 };
            Square::make(king_to_file, from.rank())
        } else {
            self.to_sq()
        };
        write!(f, "{from}{to}")?;
        if let MoveType::Promotion = self.move_type() {
            let c = Piece::new(Color::Black, self.promotion_type()).to_char();
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Move({self})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_flip() {
        assert_eq!(!Color::White, Color::Black);
        assert_eq!(Color::Black.flip(), Color::White);
    }

    #[test]
    fn square_make_and_parts() {
        let e4 = Square::make(4, 3);
        assert_eq!(e4, Square::E4);
        assert_eq!(e4.file(), 4);
        assert_eq!(e4.rank(), 3);
        assert_eq!(e4.to_string(), "e4");
    }

    #[test]
    fn square_flips() {
        assert_eq!(Square::A1.flip_rank(), Square::A8);
        assert_eq!(Square::A1.flip_file(), Square::H1);
    }

    #[test]
    fn square_offset_stays_on_board() {
        assert_eq!(Square::E4.offset(Direction::North), Some(Square::E5));
        assert_eq!(Square::H4.offset(Direction::East), None); // would wrap
        assert_eq!(Square::A1.offset(Direction::South), None); // off board
    }

    #[test]
    fn piece_char_roundtrip() {
        for c in "PNBRQKpnbrqk".chars() {
            let p = Piece::from_char(c).unwrap();
            assert_eq!(p.to_char(), c);
        }
        assert_eq!(Piece::from_char('x'), None);
    }

    #[test]
    fn move_normal_roundtrip() {
        let m = Move::normal(Square::E2, Square::E4);
        assert_eq!(m.from_sq(), Square::E2);
        assert_eq!(m.to_sq(), Square::E4);
        assert_eq!(m.move_type(), MoveType::Normal);
        assert_eq!(m.to_string(), "e2e4");
    }

    #[test]
    fn move_promotion_roundtrip() {
        let m = Move::promotion(Square::E7, Square::E8, PieceType::Queen);
        assert_eq!(m.move_type(), MoveType::Promotion);
        assert_eq!(m.promotion_type(), PieceType::Queen);
        assert_eq!(m.to_string(), "e7e8q");

        let n = Move::promotion(Square::A7, Square::A8, PieceType::Knight);
        assert_eq!(n.promotion_type(), PieceType::Knight);
        assert_eq!(n.to_string(), "a7a8n");
    }

    #[test]
    fn move_special_types() {
        assert_eq!(
            Move::en_passant(Square::E5, Square::D6).move_type(),
            MoveType::EnPassant
        );
        assert_eq!(
            Move::castling(Square::E1, Square::H1).move_type(),
            MoveType::Castling
        );
        // Castling renders in standard UCI as king -> landing square, not the
        // internal king -> rook encoding.
        assert_eq!(Move::castling(Square::E1, Square::H1).to_string(), "e1g1");
        assert_eq!(Move::castling(Square::E1, Square::A1).to_string(), "e1c1");
        assert_eq!(Move::castling(Square::E8, Square::H8).to_string(), "e8g8");
        assert_eq!(Move::castling(Square::E8, Square::A8).to_string(), "e8c8");
    }
}

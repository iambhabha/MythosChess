//! NNUE: a small, real "efficiently updatable" neural-network evaluation.
//!
//! This module owns the *feature convention* and the *network shape* for Mythos's
//! learned evaluation. The trainer (`src/bin/train.rs`) reuses the exact same
//! feature function and constants from here via `use mythos::nnue::...`, so the
//! net that is trained and the net that is served can never disagree about what a
//! feature index means.
//!
//! ## Architecture — a perspective ("HalfKP-lite") net
//!
//! * **Input**: 768 binary features = 2 (friendly / enemy) × 6 (piece type) × 64
//!   (square). Each *perspective* color builds its own feature set, with the board
//!   vertically flipped when viewed from Black so "my back rank" is always rank 1.
//! * **Layer 1** `W1` (`[HIDDEN][768]`, row-major) + `b1` (`[HIDDEN]`): the input
//!   feature vector is sparse, so an accumulator is just the sum of the `W1`
//!   columns of the active features, plus the bias.
//! * We build **two** accumulators — one from the side-to-move's perspective, one
//!   from the not-side-to-move's — and concatenate their CReLU activations into a
//!   `2*HIDDEN` "combined" vector (side-to-move half first).
//! * **Layer 2** `W2` (`[2*HIDDEN]`) + `b2` (scalar) reduces that to a single
//!   output, which is scaled to centipawns.
//!
//! CReLU (clamped ReLU) is `x.clamp(0.0, 1.0)`, matching the trainer exactly.

use std::fs;
use std::io::{self, Write};
#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

use crate::position::Position;
use crate::types::{Color, Move, MoveType, PieceType, Square};

// ---------------------------------------------------------------------------
// Architecture constants. The trainer imports these so the two stay in lockstep.
// ---------------------------------------------------------------------------

/// Hidden-layer width (per perspective). The combined layer is `2 * HIDDEN`.
pub const HIDDEN: usize = 256;
/// King buckets: the perspective's own king square is mapped (via [`king_bucket`])
/// into one of `KING_BUCKETS` coarse zones that select a feature block, so the
/// piece features are *king-relative* — the "HalfKA" idea that gives a much
/// stronger eval than plain piece-square features. Fewer, coarser buckets train
/// better on limited data (each sees more positions) and keep the net small/fast.
pub const KING_BUCKETS: usize = 16;
/// Features within one king bucket: 2 (friendly/enemy) × 6 (piece type) × 64 (square).
pub const FEATS_PER_BUCKET: usize = 768;
/// Number of input features: one 768-block per king bucket (king-bucketed HalfKA).
pub const NUM_FEATURES: usize = KING_BUCKETS * FEATS_PER_BUCKET; // 16 * 768 = 12288
/// Output scale: the raw network output is multiplied by this to get centipawns,
/// and the training target squashes `score_cp / SCALE` through a sigmoid.
pub const SCALE: f32 = 400.0;

/// Quantization scales for the **integer inference path** (post-training
/// quantization of the trained f32 net — no retrain, the `.nnue` file stays f32).
///
/// The engine evaluates with integers so the per-move accumulator update runs as
/// `i16` (16 lanes / AVX2 register vs 8 for `f32` — ~2× throughput on the hot path)
/// and the layer-2 dot product as `i16`. `QA` scales the feature transformer:
/// weights/biases and the accumulator are `round(w * QA)`, and the CReLU activation
/// clamps to `[0, QA]` (float `[0, 1]`). `QB` scales the layer-2 weights. The dot
/// product is an `i32` sum of `act * w2_q`, then descaled by `QA*QB` back to the
/// float output that feeds `* SCALE`.
///
/// `QA = 512`, `QB = 512`. The accumulator is bounded not by the pathological
/// `QA * max|w1| * 32` but by the *actual* pre-activation range: measured over 2000
/// real positions the largest `|b1 + Σ w1|` element is ≈ 7.4 (the feature weights
/// cancel heavily), so the i16 accumulator peaks at `512 * 7.4 ≈ 3.8k` — an ~8×
/// headroom to the ±32767 cap, safe even if a rare position doubles that. This finer
/// scale (4× the original 127/64) cuts the mean post-quantization eval error from
/// ~10 cp to ~2-3 cp, so the integer eval tracks the float net closely. `w2_q` peaks
/// at `512 * 1.35 ≈ 691`, and the `i32` layer-2 sum stays far inside range.
pub const QA: i32 = 512;
pub const QB: i32 = 512;

/// File-format magic: the ASCII bytes "NNUE" as a little-endian `u32`.
const MAGIC: u32 = 0x4E4E_5545;

/// Half of a king bucket's 768-feature block: within a bucket, 0..384 are
/// "friendly" pieces, 384..768 are "enemy" pieces (from the chosen perspective).
const HALF: usize = FEATS_PER_BUCKET / 2; // 384

// ---------------------------------------------------------------------------
// Feature extraction — the single source of truth for the input convention.
// ---------------------------------------------------------------------------

/// Fill `out` with the indices of every active input feature for `pos`, seen from
/// `perspective`. Clears `out` first.
///
/// For a piece of color `c` on square `sq`, seen from perspective color `P`:
///
/// ```text
/// oriented_sq = if P == White { sq.index() } else { sq.index() ^ 56 }  // vflip for Black
/// friendly    = (c == P)
/// feature_idx = (if friendly { 0 } else { 1 }) * 384
///             + piece_type.index() * 64
///             + oriented_sq
/// ```
///
/// The `^ 56` is a vertical flip (`Square::flip_rank`), so Black views the board
/// from its own side: the mapping is symmetric between the two colors.
pub fn active_features(pos: &Position, perspective: Color, out: &mut Vec<usize>) {
    out.clear();
    let king_sq = pos.king_square(perspective);
    for i in 0..64 {
        // `from_index(0..64)` is always `Some`, but we match rather than unwrap.
        let sq = match crate::types::Square::from_index(i) {
            Some(s) => s,
            None => continue,
        };
        if let Some(piece) = pos.piece_at(sq) {
            out.push(feature_index(
                perspective,
                king_sq,
                piece.color,
                piece.piece_type,
                sq,
            ));
        }
    }
}

/// The input-feature index of a piece of color `c` and type `pt` on square `sq`,
/// seen from perspective color `perspective` whose own king stands on `king_sq`.
/// This is the *single source of truth* for the feature convention — both
/// [`active_features`] (from-scratch) and the incremental [`Accumulator`] read
/// their indices from here, so the two paths can never disagree.
///
/// King-bucketed ("HalfKA"): the perspective's king square maps to one of
/// [`KING_BUCKETS`] zones ([`king_bucket`]) that selects a 768-feature block, and
/// within it the piece is placed by the usual friendly/type/square scheme.
/// Everything is vertically flipped for Black so each side views the board from
/// its own back rank.
///
/// ```text
/// oriented_sq   = if perspective == White { sq }      else { sq ^ 56 }
/// oriented_king = if perspective == White { king_sq } else { king_sq ^ 56 }
/// friendly      = (c == perspective)
/// within        = (if friendly { 0 } else { 1 }) * 384 + pt.index() * 64 + oriented_sq
/// idx           = king_bucket(oriented_king) * 768 + within
/// ```
#[inline]
pub fn feature_index(
    perspective: Color,
    king_sq: Square,
    c: Color,
    pt: PieceType,
    sq: Square,
) -> usize {
    let (oriented_sq, oriented_king) = if perspective == Color::White {
        (sq.index(), king_sq.index())
    } else {
        (sq.index() ^ 56, king_sq.index() ^ 56)
    };
    let friendly = c == perspective;
    let within = (if friendly { 0 } else { 1 }) * HALF + pt.index() * 64 + oriented_sq;
    king_bucket(oriented_king) * FEATS_PER_BUCKET + within
}

/// Map an (already perspective-oriented) king square 0..64 to its coarse bucket
/// 0..[`KING_BUCKETS`]. The board is split into a 4×4 grid of 2×2 squares, so the
/// king's rough zone (not its exact square) picks the feature block — 16 buckets
/// that each still see plenty of training data. Both the trainer (`train_nnue.py`)
/// and this engine must compute the bucket identically.
#[inline]
pub fn king_bucket(oriented_king_sq: usize) -> usize {
    let rank = oriented_king_sq / 8;
    let file = oriented_king_sq % 8;
    (rank / 2) * 4 + (file / 2)
}

// ---------------------------------------------------------------------------
// The network.
// ---------------------------------------------------------------------------

/// The trained weights of the perspective net.
///
/// * `w1`: layer-1 weights, shape `[HIDDEN][NUM_FEATURES]` stored row-major as a
///   flat `Vec<f32>` of length `HIDDEN * NUM_FEATURES`. Element `(j, f)` — the
///   weight from feature `f` into hidden neuron `j` — is at index `j * NUM_FEATURES + f`.
/// * `w1t`: the **transposed** copy of `w1`, shape `[NUM_FEATURES][HIDDEN]`, so each
///   feature's `HIDDEN`-wide column is *contiguous*: `w1t[f * HIDDEN + j] ==
///   w1[j * NUM_FEATURES + f]`. This is derived in memory from `w1` (see
///   [`build_w1t`]) and is what the accumulator add/sub touch, because a contiguous
///   column is SIMD-friendly (a strided gather across `w1` is not). It is *not*
///   part of the on-disk format — `save` only writes `w1`.
/// * `b1`: layer-1 biases, length `HIDDEN`.
/// * `w2`: layer-2 weights, length `2 * HIDDEN` (side-to-move half first).
/// * `b2`: layer-2 bias (scalar).
pub struct Net {
    pub w1: Vec<f32>,
    pub w1t: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: f32,
    // ---- Derived integer weights for the quantized inference path ----
    // Built from the f32 weights above by [`Net::build_quant`], rebuilt whenever
    // `w1`/`b1`/`w2` change. Not part of the on-disk format. The engine's
    // accumulator + evaluation read *these*; the f32 weights are kept for the
    // trainer and the float reference path ([`Net::evaluate_float`]).
    /// Quantized transposed FT weights, `round(w1t * QA)`, layout `[NUM_FEATURES][HIDDEN]`.
    w1t_q: Vec<i16>,
    /// Quantized layer-1 biases, `round(b1 * QA)`, length `HIDDEN`.
    b1_q: Vec<i16>,
    /// Quantized layer-2 weights, `round(w2 * QB)`, length `2 * HIDDEN`.
    w2_q: Vec<i16>,
}

/// Round `x` to the nearest `i16`, saturating at the `i16` bounds (a stray
/// out-of-range weight clamps rather than wrapping into a wildly wrong value).
#[inline]
fn q_i16(x: f32) -> i16 {
    x.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

/// Build the transposed feature-transformer weights from the row-major `w1`.
///
/// `w1` is `[HIDDEN][NUM_FEATURES]` (feature `f`'s column strided by `NUM_FEATURES`);
/// the result is `[NUM_FEATURES][HIDDEN]` so that feature `f`'s whole `HIDDEN`-wide
/// column lives contiguously at `out[f * HIDDEN .. f * HIDDEN + HIDDEN]`. The
/// accumulator update then adds/subtracts that contiguous slice, which vectorizes
/// cleanly. Deriving `w1t` keeps the `.nnue` file format unchanged.
fn build_w1t(w1: &[f32]) -> Vec<f32> {
    debug_assert_eq!(w1.len(), HIDDEN * NUM_FEATURES);
    let mut w1t = vec![0.0f32; NUM_FEATURES * HIDDEN];
    for j in 0..HIDDEN {
        let row = j * NUM_FEATURES;
        for f in 0..NUM_FEATURES {
            w1t[f * HIDDEN + j] = w1[row + f];
        }
    }
    w1t
}

impl Net {
    /// A correctly-sized, all-zeros net. Evaluates every position to 0.
    pub fn zeros() -> Net {
        let w1 = vec![0.0; HIDDEN * NUM_FEATURES];
        let w1t = build_w1t(&w1);
        let mut net = Net {
            w1,
            w1t,
            b1: vec![0.0; HIDDEN],
            w2: vec![0.0; 2 * HIDDEN],
            b2: 0.0,
            w1t_q: Vec::new(),
            b1_q: Vec::new(),
            w2_q: Vec::new(),
        };
        net.build_quant();
        net
    }

    /// Rebuild the transposed FT weights (`w1t`) and the quantized inference weights
    /// from the current f32 `w1`/`b1`/`w2`.
    ///
    /// `w1t` and the `*_q` buffers are derived, in-memory-only copies that the
    /// accumulator update and evaluation read. They are set automatically by
    /// [`Net::zeros`] and [`Net::from_bytes`]; call this after mutating the f32
    /// weights directly (e.g. a trainer updating them) so the derived copies stay
    /// consistent.
    pub fn rebuild_w1t(&mut self) {
        self.w1t = build_w1t(&self.w1);
        self.build_quant();
    }

    /// (Re)build the quantized integer weights (`w1t_q`, `b1_q`, `w2_q`) from the
    /// current f32 weights. Assumes `w1t` is already up to date.
    fn build_quant(&mut self) {
        self.w1t_q = self.w1t.iter().map(|&x| q_i16(x * QA as f32)).collect();
        self.b1_q = self.b1.iter().map(|&x| q_i16(x * QA as f32)).collect();
        self.w2_q = self.w2.iter().map(|&x| q_i16(x * QB as f32)).collect();
    }

    /// Load a net from a file in the binary format written by [`Net::save`].
    pub fn load(path: &str) -> io::Result<Net> {
        let bytes = fs::read(path)?;
        Net::from_bytes(&bytes).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "not a valid Mythos NNUE file (bad magic, dims, or length)",
            )
        })
    }

    /// A short human-readable description of this net's shape, e.g.
    /// `"NNUE 768->256->1"`, for a UCI `info string`.
    pub fn describe(&self) -> String {
        format!("NNUE {NUM_FEATURES}->{HIDDEN}->1")
    }

    /// Parse a net from raw file bytes. Returns `None` if the magic, dimensions,
    /// or total length do not match this architecture.
    pub fn from_bytes(bytes: &[u8]) -> Option<Net> {
        // Header: magic u32, hidden u32, num_features u32.
        let mut off = 0usize;
        let magic = read_u32(bytes, &mut off)?;
        if magic != MAGIC {
            return None;
        }
        let hidden = read_u32(bytes, &mut off)? as usize;
        let num_features = read_u32(bytes, &mut off)? as usize;
        if hidden != HIDDEN || num_features != NUM_FEATURES {
            return None;
        }

        let w1_len = HIDDEN * NUM_FEATURES;
        let b1_len = HIDDEN;
        let w2_len = 2 * HIDDEN;

        let mut w1 = vec![0.0f32; w1_len];
        for w in w1.iter_mut() {
            *w = read_f32(bytes, &mut off)?;
        }
        let mut b1 = vec![0.0f32; b1_len];
        for b in b1.iter_mut() {
            *b = read_f32(bytes, &mut off)?;
        }
        let mut w2 = vec![0.0f32; w2_len];
        for w in w2.iter_mut() {
            *w = read_f32(bytes, &mut off)?;
        }
        let b2 = read_f32(bytes, &mut off)?;

        // Reject trailing garbage: the file must be exactly the expected size.
        if off != bytes.len() {
            return None;
        }

        // Derive the transposed FT weights in memory (see `build_w1t`). The disk
        // format only ever stores `w1`.
        let w1t = build_w1t(&w1);
        let mut net = Net {
            w1,
            w1t,
            b1,
            w2,
            b2,
            w1t_q: Vec::new(),
            b1_q: Vec::new(),
            w2_q: Vec::new(),
        };
        net.build_quant();
        Some(net)
    }

    /// Serialize this net to `path` in the binary format [`Net::from_bytes`] reads.
    pub fn save(&self, path: &str) -> io::Result<()> {
        // Guard against a mis-sized net (e.g. hand-built): the writer assumes the
        // canonical dimensions.
        debug_assert_eq!(self.w1.len(), HIDDEN * NUM_FEATURES);
        debug_assert_eq!(self.b1.len(), HIDDEN);
        debug_assert_eq!(self.w2.len(), 2 * HIDDEN);

        let mut buf: Vec<u8> = Vec::with_capacity(
            12 + 4 * (self.w1.len() + self.b1.len() + self.w2.len() + 1),
        );
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&(HIDDEN as u32).to_le_bytes());
        buf.extend_from_slice(&(NUM_FEATURES as u32).to_le_bytes());
        for &w in &self.w1 {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        for &b in &self.b1 {
            buf.extend_from_slice(&b.to_le_bytes());
        }
        for &w in &self.w2 {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf.extend_from_slice(&self.b2.to_le_bytes());

        let mut f = fs::File::create(path)?;
        f.write_all(&buf)?;
        f.flush()?;
        Ok(())
    }

    /// Compute one accumulator (length `HIDDEN`) for `pos` from `perspective`:
    /// `acc[j] = b1[j] + Σ_f W1[j][f]` over the active features `f`.
    ///
    /// `scratch` is a reusable feature buffer so callers can avoid re-allocating.
    fn accumulate(&self, pos: &Position, perspective: Color, scratch: &mut Vec<usize>) -> Vec<f32> {
        active_features(pos, perspective, scratch);
        let mut acc = self.b1.clone();
        for &f in scratch.iter() {
            // W1 row-major: neuron j, feature f is at j * NUM_FEATURES + f.
            let mut base = f;
            for a in acc.iter_mut() {
                *a += self.w1[base];
                base += NUM_FEATURES;
            }
        }
        acc
    }

    /// Evaluate `pos`, returning a **side-to-move-relative** score in centipawns
    /// (positive = the side to move is better), clamped to about ±10000.
    ///
    /// This uses the same **quantized integer** path as the search's incremental
    /// [`evaluate_acc`](Net::evaluate_acc): it rebuilds the accumulator from scratch
    /// (`O(pieces)`) and applies the integer layer-2 output, so
    /// `evaluate(pos) == evaluate_acc(refresh(net, pos), stm)` holds *exactly*.
    /// For the pre-quantization float value, see [`evaluate_float`](Net::evaluate_float).
    pub fn evaluate(&self, pos: &Position) -> i32 {
        let acc = Accumulator::refresh(self, pos);
        self.evaluate_acc(&acc, pos.side_to_move())
    }

    /// Evaluate from a **maintained** [`Accumulator`] instead of recomputing it
    /// from the board. This is the incremental fast path used inside the search:
    /// the accumulator is kept up to date across make/undo, so evaluation is just
    /// the integer layer-2 output over the two already-summed hidden vectors.
    ///
    /// `stm` is the side to move for the position the accumulator describes. The
    /// result is **bit-identical** to [`Net::evaluate`] on the same position: both
    /// read the same quantized `W1` columns into the same `b1_q`-seeded sums, and
    /// integer addition is order-independent (unlike the old float path, which
    /// matched only to ~1e-6).
    pub fn evaluate_acc(&self, acc: &Accumulator, stm: Color) -> i32 {
        // Pick the side-to-move accumulator first (W2's first half), then the
        // not-side-to-move one, mirroring the combined-vector layout.
        let (acc_stm, acc_nstm) = match stm {
            Color::White => (&acc.white, &acc.black),
            Color::Black => (&acc.black, &acc.white),
        };

        // Integer dot product `Σ clamp(acc,0,QA) * w2_q`, then descale by `QA*QB`
        // back to the float output and multiply by SCALE for centipawns.
        let raw = self.output_q(acc_stm, acc_nstm);
        let o = self.b2 + raw as f32 / (QA * QB) as f32;
        let cp = (o * SCALE).round();
        cp.clamp(-10_000.0, 10_000.0) as i32
    }

    /// The raw integer layer-2 dot product `Σ_j clamp(acc[j], 0, QA) * w2_q[j]`
    /// over the combined `2*HIDDEN` vector (side-to-move half first). Returns the
    /// undescaled `i32` sum. Runtime-dispatches to an AVX2 `i16`-madd kernel when
    /// available, else the scalar reference (the two agree exactly — integer).
    #[inline]
    fn output_q(&self, acc_stm: &[i16; HIDDEN], acc_nstm: &[i16; HIDDEN]) -> i32 {
        #[cfg(target_arch = "x86_64")]
        {
            if have_avx2() {
                // SAFETY: guarded by a runtime AVX2 check. `w2_q` is `2*HIDDEN` long
                // and each accumulator half is exactly `HIDDEN`; the kernel reads
                // them with unaligned 16-wide (i16) loads.
                return unsafe { output_q_avx2(&self.w2_q, acc_stm, acc_nstm) };
            }
        }
        self.output_q_scalar(acc_stm, acc_nstm)
    }

    /// Scalar reference for [`Net::output_q`].
    #[inline]
    fn output_q_scalar(&self, acc_stm: &[i16; HIDDEN], acc_nstm: &[i16; HIDDEN]) -> i32 {
        let mut sum: i32 = 0;
        for j in 0..HIDDEN {
            sum += crelu_q(acc_stm[j]) * self.w2_q[j] as i32;
            sum += crelu_q(acc_nstm[j]) * self.w2_q[HIDDEN + j] as i32;
        }
        sum
    }

    /// The pre-quantization **float** evaluation, kept as a reference so tests can
    /// bound the quantized eval's error. Rebuilds both accumulators from scratch in
    /// `f32` and applies the float layer-2 output — this is what the net computed
    /// before integer quantization.
    pub fn evaluate_float(&self, pos: &Position) -> i32 {
        let stm = pos.side_to_move();
        let nstm = !stm;
        let mut scratch: Vec<usize> = Vec::with_capacity(32);
        let acc_stm = self.accumulate(pos, stm, &mut scratch);
        let acc_nstm = self.accumulate(pos, nstm, &mut scratch);
        let acc_stm: &[f32; HIDDEN] = (&acc_stm[..]).try_into().expect("accumulate returns HIDDEN");
        let acc_nstm: &[f32; HIDDEN] =
            (&acc_nstm[..]).try_into().expect("accumulate returns HIDDEN");
        let o = self.output_float(acc_stm, acc_nstm);
        let cp = (o * SCALE).round();
        cp.clamp(-10_000.0, 10_000.0) as i32
    }

    /// Float layer-2 output `b2 + Σ w2·CReLU(combined)` — the reference for the
    /// quantized [`output_q`](Net::output_q), used only by [`evaluate_float`].
    #[inline]
    fn output_float(&self, acc_stm: &[f32; HIDDEN], acc_nstm: &[f32; HIDDEN]) -> f32 {
        let mut o = self.b2;
        for j in 0..HIDDEN {
            o += self.w2[j] * crelu(acc_stm[j]);
            o += self.w2[HIDDEN + j] * crelu(acc_nstm[j]);
        }
        o
    }
}

// ---------------------------------------------------------------------------
// The incremental accumulator — the "efficiently updatable" part of NNUE.
// ---------------------------------------------------------------------------

/// The two layer-1 hidden vectors (one per perspective color), maintained
/// incrementally across make/undo so evaluation never re-scans the board.
///
/// `white[j] = b1[j] + Σ W1[j][f]` over the active features `f` from **White's**
/// perspective; `black` is the same from **Black's** perspective. Because every
/// input feature is binary, adding or removing a piece is just adding or
/// subtracting that feature's `W1` column into both vectors — an `O(HIDDEN)`
/// update per changed piece instead of an `O(pieces * HIDDEN)` rebuild.
#[derive(Clone)]
pub struct Accumulator {
    pub white: [i16; HIDDEN],
    pub black: [i16; HIDDEN],
    /// The king squares this accumulator is bucketed on: `white_king` for the
    /// `white` vector (White's own king), `black_king` for `black`. A move that
    /// relocates a king changes that side's whole feature block, so it is detected
    /// here and handled by a full [`refresh`](Accumulator::refresh).
    white_king: Square,
    black_king: Square,
}

impl Accumulator {
    /// Build an accumulator from scratch for `pos`: seed each perspective from
    /// `b1`, then add the `W1` column of every piece on the board. This is the
    /// reference every incremental update is checked against, and it reproduces
    /// [`Net::evaluate`]'s summation order (iterate squares 0..64, add each
    /// active feature) so `evaluate_acc(refresh(net, pos), stm)` is bit-identical
    /// to `evaluate(pos)`.
    pub fn refresh(net: &Net, pos: &Position) -> Accumulator {
        let mut acc = Accumulator {
            white: [0; HIDDEN],
            black: [0; HIDDEN],
            white_king: pos.king_square(Color::White),
            black_king: pos.king_square(Color::Black),
        };
        acc.white.copy_from_slice(&net.b1_q);
        acc.black.copy_from_slice(&net.b1_q);

        for i in 0..64 {
            let sq = match Square::from_index(i) {
                Some(s) => s,
                None => continue,
            };
            if let Some(piece) = pos.piece_at(sq) {
                acc.add_piece(net, piece.color, piece.piece_type, sq);
            }
        }
        acc
    }

    /// Whether two accumulators agree **exactly**, element for element.
    ///
    /// The accumulator is now integer (`i16`), and integer addition is
    /// order-independent, so an incrementally-maintained accumulator and a
    /// from-scratch [`refresh`] must be bit-identical (unlike the old float path,
    /// which matched only to ~1e-6). Any mismatch is a genuine feature-mapping bug
    /// in [`apply_move`](Accumulator::apply_move). The name is kept for callers.
    pub fn close_to(&self, other: &Accumulator) -> bool {
        self.white == other.white && self.black == other.black
    }

    /// Add the `W1` columns of the piece `(color, pt)` on `sq` into **both**
    /// perspective vectors (White vector uses the White-perspective feature index,
    /// Black vector the Black-perspective one).
    #[inline]
    fn add_piece(&mut self, net: &Net, color: Color, pt: PieceType, sq: Square) {
        let wf = feature_index(Color::White, self.white_king, color, pt, sq);
        let bf = feature_index(Color::Black, self.black_king, color, pt, sq);
        add_column(&mut self.white, &net.w1t_q, wf);
        add_column(&mut self.black, &net.w1t_q, bf);
    }

    /// Subtract the `W1` columns of the piece `(color, pt)` on `sq` from **both**
    /// perspective vectors — the exact inverse of [`add_piece`](Accumulator::add_piece).
    #[inline]
    fn remove_piece(&mut self, net: &Net, color: Color, pt: PieceType, sq: Square) {
        let wf = feature_index(Color::White, self.white_king, color, pt, sq);
        let bf = feature_index(Color::Black, self.black_king, color, pt, sq);
        sub_column(&mut self.white, &net.w1t_q, wf);
        sub_column(&mut self.black, &net.w1t_q, bf);
    }

    /// Produce the accumulator for the position *after* `m` is played, given the
    /// `parent` accumulator (for the position *before* the move) and `pos_before`,
    /// the position **before** the move is made. Only the features touched by the
    /// move are updated, so this is `O(HIDDEN)` per changed piece.
    ///
    /// The feature diff mirrors [`Position::make_move`] exactly:
    ///
    /// * **Normal**: remove the mover from `from`; if `to` holds an enemy piece,
    ///   remove it; add the mover on `to`.
    /// * **Promotion**: remove the pawn from `from`; remove any captured piece on
    ///   `to`; add the promoted piece on `to`.
    /// * **EnPassant**: remove our pawn from `from`, add it on `to`, and remove the
    ///   enemy pawn that stood on `(file(to), rank(from))`.
    /// * **Castling**: `to` is the *rook* square. Move the king to the g/c-file and
    ///   the rook to the f/d-file (kingside if `rook_file > king_file`), same rank.
    pub fn apply_move(
        net: &Net,
        parent: &Accumulator,
        pos_before: &Position,
        m: Move,
    ) -> Accumulator {
        let mut acc = parent.clone();

        let us = pos_before.side_to_move();
        let them = !us;
        let from = m.from_sq();
        let to = m.to_sq(); // NB: for castling this is the ROOK's square.

        // The moving piece, read from the pre-move board.
        let moving = pos_before
            .piece_at(from)
            .expect("apply_move from an empty square");

        // A king move (including castling) changes that side's king bucket, so every
        // feature in that perspective is re-indexed — a piece-by-piece diff no longer
        // applies. Kings move infrequently enough that rebuilding both accumulators
        // from the post-move board is cheap and keeps the code simple and correct.
        if moving.piece_type == PieceType::King {
            let mut after = pos_before.clone();
            after.make_move(m);
            return Accumulator::refresh(net, &after);
        }

        match m.move_type() {
            MoveType::Normal => {
                acc.remove_piece(net, us, moving.piece_type, from);
                if let Some(cap) = pos_before.piece_at(to) {
                    acc.remove_piece(net, cap.color, cap.piece_type, to);
                }
                acc.add_piece(net, us, moving.piece_type, to);
            }

            MoveType::Promotion => {
                acc.remove_piece(net, us, PieceType::Pawn, from);
                if let Some(cap) = pos_before.piece_at(to) {
                    acc.remove_piece(net, cap.color, cap.piece_type, to);
                }
                acc.add_piece(net, us, m.promotion_type(), to);
            }

            MoveType::EnPassant => {
                // The captured pawn sits on the square with `to`'s file and
                // `from`'s rank (directly "behind" the target).
                let cap_sq = Square::make(to.file(), from.rank());
                acc.remove_piece(net, us, PieceType::Pawn, from);
                acc.add_piece(net, us, PieceType::Pawn, to);
                acc.remove_piece(net, them, PieceType::Pawn, cap_sq);
            }

            MoveType::Castling => {
                // `to` is the rook's square; derive king/rook destinations exactly
                // as `Position::make_move` does.
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
                acc.remove_piece(net, us, PieceType::King, king_from);
                acc.add_piece(net, us, PieceType::King, king_to);
                acc.remove_piece(net, us, PieceType::Rook, rook_from);
                acc.add_piece(net, us, PieceType::Rook, rook_to);
            }
        }

        acc
    }
}

/// Add the transposed FT column of feature `f` into `acc` (`acc[j] += w1t[j]`).
///
/// `w1t` is the transposed weights `[NUM_FEATURES][HIDDEN]`, so feature `f`'s whole
/// `HIDDEN`-wide column is the *contiguous* slice `w1t[f*HIDDEN .. f*HIDDEN+HIDDEN]`
/// (unlike the strided column of the row-major `w1`). Runtime-dispatches to an AVX2
/// kernel when available, else the scalar loop — the two agree to float precision.
#[inline]
fn add_column(acc: &mut [i16; HIDDEN], w1t: &[i16], f: usize) {
    let col = &w1t[f * HIDDEN..f * HIDDEN + HIDDEN];
    #[cfg(target_arch = "x86_64")]
    {
        if have_avx2() {
            // SAFETY: guarded by a runtime AVX2 check; `col` and `acc` are both
            // exactly `HIDDEN` i16s, and the kernel uses unaligned loads/stores.
            unsafe {
                add_column_avx2(acc, col);
            }
            return;
        }
    }
    add_column_scalar(acc, col);
}

/// Subtract the transposed FT column of feature `f` from `acc` (`acc[j] -= w1t[j]`).
/// The exact inverse of [`add_column`]; same AVX2/scalar dispatch.
#[inline]
fn sub_column(acc: &mut [i16; HIDDEN], w1t: &[i16], f: usize) {
    let col = &w1t[f * HIDDEN..f * HIDDEN + HIDDEN];
    #[cfg(target_arch = "x86_64")]
    {
        if have_avx2() {
            // SAFETY: see `add_column` — runtime-guarded, exact `HIDDEN` lengths.
            unsafe {
                sub_column_avx2(acc, col);
            }
            return;
        }
    }
    sub_column_scalar(acc, col);
}

/// Scalar `acc[j] += col[j]` over the `HIDDEN`-wide contiguous i16 column.
#[inline]
fn add_column_scalar(acc: &mut [i16; HIDDEN], col: &[i16]) {
    for (a, &c) in acc.iter_mut().zip(col.iter()) {
        *a += c;
    }
}

/// Scalar `acc[j] -= col[j]` over the `HIDDEN`-wide contiguous i16 column.
#[inline]
fn sub_column_scalar(acc: &mut [i16; HIDDEN], col: &[i16]) {
    for (a, &c) in acc.iter_mut().zip(col.iter()) {
        *a -= c;
    }
}

/// Attempt to load the default net file, `mythos.nnue`, from the usual places:
/// first alongside the running executable (so a shipped net travels with the
/// binary), then the current working directory (handy during development).
/// Returns the first net that loads, or `None` if neither exists / is valid — in
/// which case the caller stays on the hand-crafted evaluation.
pub fn load_default() -> Option<Net> {
    const DEFAULT_NAME: &str = "mythos.nnue";

    // (a) The directory the current executable lives in.
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
    {
        let candidate = dir.join(DEFAULT_NAME);
        if let Some(path) = candidate.to_str()
            && let Ok(net) = Net::load(path)
        {
            return Some(net);
        }
    }

    // (b) The current working directory.
    if let Ok(net) = Net::load(DEFAULT_NAME) {
        return Some(net);
    }

    None
}

/// Quantized clamped ReLU: clamp an `i16` accumulator value to `[0, QA]` and widen
/// to `i32` for the layer-2 dot product. The float `[0, 1]` activation range maps to
/// the integer `[0, QA]` range.
#[inline]
pub fn crelu_q(x: i16) -> i32 {
    (x as i32).clamp(0, QA)
}

/// Clamped ReLU: the activation used between the two layers (float reference path).
#[inline]
pub fn crelu(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// AVX2 SIMD kernels (x86_64), selected at runtime.
//
// The hot NNUE math is two shapes: an `acc[j] ±= col[j]` over 256 f32 (the
// accumulator update, run per changed piece), and a CReLU-then-dot-product over
// 512 f32 (the layer-2 output). Both are 8-wide vectorizable with `__m256`.
//
// AVX2 is *not* enabled at compile time (baseline x86-64 target), so we detect it
// once at runtime and route to either a `#[target_feature(enable = "avx2")]`
// kernel or the scalar fallback. The SIMD result only *reorders* the float sums,
// so it agrees with the scalar path to ~1e-6.
// ---------------------------------------------------------------------------

/// Whether the running CPU supports both AVX2 and FMA, detected once and cached.
///
/// The output kernel uses `_mm256_fmadd_ps`, which needs the `fma` feature in
/// addition to `avx2`, so we gate on both. On non-x86_64 this is never called (the
/// callers are behind `#[cfg(target_arch = "x86_64")]`).
#[cfg(target_arch = "x86_64")]
#[inline]
fn have_avx2() -> bool {
    static AVX2_FMA: OnceLock<bool> = OnceLock::new();
    *AVX2_FMA
        .get_or_init(|| std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"))
}

/// AVX2 `acc[j] += col[j]` over `HIDDEN` (=256) i16 = 16 vector adds (16 lanes each).
///
/// # Safety
/// The caller must have verified AVX2 support (via [`have_avx2`]). `acc` is exactly
/// `HIDDEN` i16; `col` must be at least `HIDDEN` i16 (it is a `HIDDEN`-wide column
/// slice). Uses unaligned loads/stores, so no alignment requirement.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn add_column_avx2(acc: &mut [i16; HIDDEN], col: &[i16]) {
    use std::arch::x86_64::*;
    debug_assert!(col.len() >= HIDDEN);
    let a = acc.as_mut_ptr();
    let c = col.as_ptr();
    let mut j = 0;
    while j < HIDDEN {
        // SAFETY: `j` steps by 16 and stops before `HIDDEN`, so every `.add(j)` plus
        // a 16-wide (256-bit) load/store stays within the `HIDDEN`-long `acc`/`col`.
        unsafe {
            let va = _mm256_loadu_si256(a.add(j) as *const __m256i);
            let vc = _mm256_loadu_si256(c.add(j) as *const __m256i);
            _mm256_storeu_si256(a.add(j) as *mut __m256i, _mm256_add_epi16(va, vc));
        }
        j += 16;
    }
}

/// AVX2 `acc[j] -= col[j]` over `HIDDEN` (=256) i16 = 16 vector subs.
///
/// # Safety
/// Same contract as [`add_column_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn sub_column_avx2(acc: &mut [i16; HIDDEN], col: &[i16]) {
    use std::arch::x86_64::*;
    debug_assert!(col.len() >= HIDDEN);
    let a = acc.as_mut_ptr();
    let c = col.as_ptr();
    let mut j = 0;
    while j < HIDDEN {
        // SAFETY: as in `add_column_avx2` — bounded 16-wide loads/stores.
        unsafe {
            let va = _mm256_loadu_si256(a.add(j) as *const __m256i);
            let vc = _mm256_loadu_si256(c.add(j) as *const __m256i);
            _mm256_storeu_si256(a.add(j) as *mut __m256i, _mm256_sub_epi16(va, vc));
        }
        j += 16;
    }
}

/// AVX2 quantized layer-2 output: the raw `i32` dot product
/// `Σ clamp(acc, 0, QA) * w2_q` over the two `HIDDEN` halves.
///
/// CReLU is `_mm256_max_epi16(x, 0)` then `_mm256_min_epi16(x, QA)` (still `i16`);
/// the multiply-accumulate uses `_mm256_madd_epi16` (`i16 × i16 → i32`, adjacent
/// pairs summed) — no `u8` packing, so no lane-permutation bookkeeping. The stm
/// half (dotted with `w2_q[0..HIDDEN]`) and the nstm half (`w2_q[HIDDEN..]`) are
/// accumulated into the same `i32` lane-vector, then horizontally summed once at the
/// end — exactly matching the scalar `output_q_scalar`.
///
/// # Safety
/// The caller must have verified AVX2 support (via [`have_avx2`]). `w2_q` must be at
/// least `2 * HIDDEN` i16; each accumulator half is exactly `HIDDEN` i16. Uses
/// unaligned loads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn output_q_avx2(w2_q: &[i16], acc_stm: &[i16; HIDDEN], acc_nstm: &[i16; HIDDEN]) -> i32 {
    use std::arch::x86_64::*;
    debug_assert!(w2_q.len() >= 2 * HIDDEN);

    let w2p = w2_q.as_ptr();
    let stm = acc_stm.as_ptr();
    let nstm = acc_nstm.as_ptr();

    // SAFETY: `j` steps by 16 and stops before `HIDDEN`; the stm/nstm loads read
    // within the `HIDDEN`-long accumulators, and the `w2_q` loads read within its
    // `2*HIDDEN`-long buffer (`j` for the first half, `HIDDEN + j` for the second).
    let sum = unsafe {
        let zero = _mm256_setzero_si256();
        let qa = _mm256_set1_epi16(QA as i16);
        let mut sum = _mm256_setzero_si256();
        let mut j = 0;
        while j < HIDDEN {
            // stm half: clamp(acc_stm[j..], 0, QA) madd w2_q[j..].
            let xs = _mm256_loadu_si256(stm.add(j) as *const __m256i);
            let cs = _mm256_min_epi16(_mm256_max_epi16(xs, zero), qa);
            let ws = _mm256_loadu_si256(w2p.add(j) as *const __m256i);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(cs, ws));

            // nstm half: clamp(acc_nstm[j..], 0, QA) madd w2_q[HIDDEN + j..].
            let xn = _mm256_loadu_si256(nstm.add(j) as *const __m256i);
            let cn = _mm256_min_epi16(_mm256_max_epi16(xn, zero), qa);
            let wn = _mm256_loadu_si256(w2p.add(HIDDEN + j) as *const __m256i);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(cn, wn));

            j += 16;
        }
        sum
    };

    hsum256_epi32(sum)
}

/// Horizontal sum of the 8 `i32` lanes of a `__m256i`.
///
/// # Safety
/// Caller must have AVX2. Pure lane arithmetic, no memory access.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn hsum256_epi32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    // Fold the high 128 into the low 128, then reduce the 4 i32 lanes.
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(lo, hi); // 4 lanes
    let shuf = _mm_shuffle_epi32(s, 0b_10_11_00_01); // swap within pairs
    let s = _mm_add_epi32(s, shuf); // [l0+l1, ., l2+l3, .]
    let hi64 = _mm_unpackhi_epi64(s, s); // move l2+l3 into lane 0
    let s = _mm_add_epi32(s, hi64);
    _mm_cvtsi128_si32(s)
}

// ---------------------------------------------------------------------------
// Little-endian primitive readers (inverse of `to_le_bytes` used in `save`).
// ---------------------------------------------------------------------------

/// Read a little-endian `u32` at `*off`, advancing `*off`. `None` if out of range.
#[inline]
fn read_u32(bytes: &[u8], off: &mut usize) -> Option<u32> {
    let end = off.checked_add(4)?;
    if end > bytes.len() {
        return None;
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&bytes[*off..end]);
    *off = end;
    Some(u32::from_le_bytes(b))
}

/// Read a little-endian `f32` at `*off`, advancing `*off`. `None` if out of range.
#[inline]
fn read_f32(bytes: &[u8], off: &mut usize) -> Option<f32> {
    let end = off.checked_add(4)?;
    if end > bytes.len() {
        return None;
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&bytes[*off..end]);
    *off = end;
    Some(f32::from_le_bytes(b))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_net_evaluates_to_zero() {
        let net = Net::zeros();
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
        ];
        for fen in fens {
            let pos = Position::from_fen(fen).unwrap();
            assert_eq!(net.evaluate(&pos), 0, "zeros net must score 0 for {fen}");
        }
    }

    #[test]
    fn save_load_round_trips() {
        // Build a net with a few distinct, recognizable values.
        let mut net = Net::zeros();
        net.w1[0] = 0.5;
        net.w1[NUM_FEATURES + 3] = -0.25;
        net.w1[HIDDEN * NUM_FEATURES - 1] = 1.5;
        net.b1[0] = -0.75;
        net.b1[HIDDEN - 1] = 0.125;
        net.w2[0] = 2.0;
        net.w2[2 * HIDDEN - 1] = -2.0;
        net.b2 = 0.333;

        let dir = std::env::temp_dir();
        let path = dir.join("mythos_nnue_roundtrip_test.bin");
        let path_str = path.to_str().unwrap();

        net.save(path_str).unwrap();
        let loaded = Net::load(path_str).unwrap();

        assert_eq!(loaded.w1, net.w1);
        assert_eq!(loaded.b1, net.b1);
        assert_eq!(loaded.w2, net.w2);
        assert_eq!(loaded.b2, net.b2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn start_position_has_32_valid_features_per_perspective() {
        let pos = Position::startpos();
        let mut feats = Vec::new();
        for perspective in [Color::White, Color::Black] {
            active_features(&pos, perspective, &mut feats);
            assert_eq!(feats.len(), 32, "start position has 32 pieces");
            for &f in &feats {
                assert!(f < NUM_FEATURES, "feature {f} out of range");
            }
        }
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut net = Net::zeros();
        net.b2 = 1.0;
        let dir = std::env::temp_dir();
        let path = dir.join("mythos_nnue_badmagic_test.bin");
        let path_str = path.to_str().unwrap();
        net.save(path_str).unwrap();

        let mut bytes = std::fs::read(path_str).unwrap();
        bytes[0] ^= 0xFF; // corrupt the magic
        assert!(Net::from_bytes(&bytes).is_none());
        let _ = std::fs::remove_file(&path);
    }

    // -- Incremental accumulator ------------------------------------------

    /// A deterministic "random-ish" net: every weight is a small, reproducible
    /// value derived from its index by a cheap hash. This gives every hidden
    /// neuron and feature a distinct nonzero weight (so a wrong feature index or
    /// a sign error in an update cannot hide behind a zero), without needing an
    /// RNG crate or a real net file.
    fn pseudo_random_net() -> Net {
        // A splitmix-style scramble mapped into roughly [-0.5, 0.5).
        fn weight(i: usize) -> f32 {
            let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x ^= x >> 29;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 32;
            // Take 16 bits -> [0, 65535] -> [-0.5, ~0.5).
            ((x & 0xFFFF) as f32) / 65536.0 - 0.5
        }

        let mut net = Net::zeros();
        for (i, w) in net.w1.iter_mut().enumerate() {
            *w = weight(i);
        }
        for (i, b) in net.b1.iter_mut().enumerate() {
            *b = weight(i + 0x1000_0000);
        }
        for (i, w) in net.w2.iter_mut().enumerate() {
            *w = weight(i + 0x2000_0000);
        }
        net.b2 = weight(0x3000_0000);
        // `w1` was mutated directly, so refresh the derived transposed copy that the
        // accumulator update reads.
        net.rebuild_w1t();
        net
    }

    /// The two accumulators agree element-wise within a tiny float epsilon.
    fn accs_close(a: &Accumulator, b: &Accumulator) -> bool {
        a.close_to(b)
    }

    #[test]
    fn evaluate_acc_matches_from_scratch_evaluate() {
        // The core invariant: evaluating from a freshly refreshed accumulator must
        // give the *identical* centipawn score as the from-scratch `evaluate`.
        let net = pseudo_random_net();
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
            "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3",
            "8/P7/8/8/8/8/8/k1K5 w - - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 5 12",
        ];
        for fen in fens {
            let pos = Position::from_fen(fen).unwrap();
            let acc = Accumulator::refresh(&net, &pos);
            assert_eq!(
                net.evaluate_acc(&acc, pos.side_to_move()),
                net.evaluate(&pos),
                "acc eval must equal from-scratch eval for {fen}"
            );
        }
    }

    /// Apply a sequence of moves, threading the accumulator incrementally, and
    /// after each move assert the incremental accumulator equals a from-scratch
    /// refresh of the resulting position (and that the eval matches too).
    fn assert_incremental_matches(net: &Net, fen: &str, moves: &[Move]) {
        let mut pos = Position::from_fen(fen).unwrap_or_else(|e| panic!("bad fen {fen}: {e}"));
        let mut acc = Accumulator::refresh(net, &pos);
        // Sanity: the starting accumulator itself agrees with the eval.
        assert_eq!(net.evaluate_acc(&acc, pos.side_to_move()), net.evaluate(&pos));

        for &m in moves {
            // Compute the child accumulator from the pre-move position, then make
            // the move on the scratch board to advance `pos`.
            let child = Accumulator::apply_move(net, &acc, &pos, m);
            pos.make_move(m);

            let fresh = Accumulator::refresh(net, &pos);
            assert!(
                accs_close(&child, &fresh),
                "incremental accumulator drifted after {m} in {fen}"
            );
            assert_eq!(
                net.evaluate_acc(&child, pos.side_to_move()),
                net.evaluate(&pos),
                "incremental eval drifted after {m} in {fen}"
            );
            acc = child;
        }
    }

    #[test]
    fn incremental_normal_and_capture_and_double_push() {
        let net = pseudo_random_net();
        // A full opening line: double pushes (ep targets), a knight develop, a
        // bishop develop, castling, and captures along the way.
        assert_incremental_matches(
            &net,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &[
                Move::normal(Square::E2, Square::E4), // white double push
                Move::normal(Square::D7, Square::D5), // black double push
                Move::normal(Square::E4, Square::D5), // capture (pawn takes pawn)
                Move::normal(Square::G8, Square::F6), // knight develop
                Move::normal(Square::G1, Square::F3), // knight develop
                Move::normal(Square::F6, Square::D5), // knight recaptures pawn
            ],
        );
    }

    #[test]
    fn incremental_en_passant() {
        let net = pseudo_random_net();
        // White pawn on d5, black just played ...c5 (ep target on c6): d5xc6 e.p.
        assert_incremental_matches(
            &net,
            "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3",
            &[Move::en_passant(Square::D5, Square::C6)],
        );
    }

    #[test]
    fn incremental_castling_both_sides() {
        let net = pseudo_random_net();
        // White king-side, then continue and let Black castle king-side too.
        assert_incremental_matches(
            &net,
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            &[
                Move::castling(Square::E1, Square::H1), // white O-O
                Move::castling(Square::E8, Square::H8), // black O-O
            ],
        );
        // White queen-side, then Black queen-side.
        assert_incremental_matches(
            &net,
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            &[
                Move::castling(Square::E1, Square::A1), // white O-O-O
                Move::castling(Square::E8, Square::A8), // black O-O-O
            ],
        );
    }

    #[test]
    fn incremental_promotion_quiet_and_capture() {
        let net = pseudo_random_net();
        // A quiet promotion to a queen. (The enemy king sits off the promotion
        // square — a king-bucketed net requires both kings present, and no legal
        // position lets a pawn promote onto the enemy king anyway.)
        assert_incremental_matches(
            &net,
            "8/P6k/8/8/8/8/8/K7 w - - 0 1",
            &[Move::promotion(Square::A7, Square::A8, PieceType::Queen)],
        );
        // A capturing promotion: b7 takes the rook on a8, promoting to a knight.
        assert_incremental_matches(
            &net,
            "r6k/1P6/8/8/8/8/8/K7 w - - 0 1",
            &[Move::promotion(Square::B7, Square::A8, PieceType::Knight)],
        );
    }

    #[test]
    fn incremental_mixed_sequence_on_kiwipete() {
        let net = pseudo_random_net();
        // A rich middlegame with knight moves, a pawn capture, and castling mixed
        // together — the accumulator must stay exact across all of them.
        assert_incremental_matches(
            &net,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &[
                Move::normal(Square::D5, Square::E6),   // pawn captures pawn
                Move::normal(Square::E7, Square::E6),   // queen recaptures
                Move::castling(Square::E1, Square::H1), // white O-O
                Move::normal(Square::A6, Square::E2),   // bishop captures bishop
                Move::normal(Square::F3, Square::E2),   // queen recaptures bishop
            ],
        );
    }

    // -- Quantization ------------------------------------------------------

    /// FENs covering opening, middlegame, and endgame material.
    const EVAL_FENS: [&str; 4] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
        "8/P6k/8/8/8/8/8/K7 w - - 0 1",
    ];

    /// The AVX2 integer output kernel must be **bit-identical** to the scalar
    /// reference (both are integer, so there is no rounding slack — a mismatch is a
    /// real SIMD bug, e.g. a wrong clamp or a lane-permutation error).
    #[test]
    fn output_q_simd_matches_scalar() {
        let net = pseudo_random_net();
        for fen in EVAL_FENS {
            let pos = Position::from_fen(fen).unwrap();
            let acc = Accumulator::refresh(&net, &pos);
            let (stm, nstm) = match pos.side_to_move() {
                Color::White => (&acc.white, &acc.black),
                Color::Black => (&acc.black, &acc.white),
            };
            let scalar = net.output_q_scalar(stm, nstm);
            #[cfg(target_arch = "x86_64")]
            if have_avx2() {
                // SAFETY: guarded by the runtime AVX2 check; `w2_q` is `2*HIDDEN`
                // and each half is exactly `HIDDEN` i16.
                let simd = unsafe { output_q_avx2(&net.w2_q, stm, nstm) };
                assert_eq!(simd, scalar, "SIMD output_q disagrees with scalar for {fen}");
            }
        }
    }

    /// The quantized integer eval must track the pre-quantization float eval. This
    /// bounds the post-training-quantization error and would catch a structural bug
    /// (wrong scale, missing descale, sign flip — those blow up by >>1 pawn). The
    /// bound is generous because `pseudo_random_net` has large weights that push
    /// activations near the CReLU clamp, amplifying rounding well past what a real
    /// trained net (small weights) exhibits.
    #[test]
    fn quantized_eval_tracks_float() {
        let net = pseudo_random_net();
        for fen in EVAL_FENS {
            let pos = Position::from_fen(fen).unwrap();
            let q = net.evaluate(&pos);
            let f = net.evaluate_float(&pos);
            assert!(
                (q - f).abs() <= 150,
                "quantized eval {q} strays too far from float {f} for {fen}"
            );
            if f.abs() >= 400 {
                assert_eq!(q.signum(), f.signum(), "quantized eval flipped sign for {fen}");
            }
        }
    }

    /// Diagnostic (run with `--ignored --nocapture`): load the real 25M net and the
    /// first ~2000 FENs from a data shard, and report how far the quantized integer
    /// eval strays from the float reference. Tells us whether quantization is costing
    /// eval quality (mean abs error in centipawns).
    #[test]
    #[ignore]
    fn measure_quant_error_on_real_net() {
        let net = match Net::load("mythos_hka16_25m.nnue") {
            Ok(n) => n,
            Err(e) => {
                eprintln!("skip: cannot load net: {e}");
                return;
            }
        };
        let data = match std::fs::read_to_string("sf_big.txt.w0") {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip: cannot read shard: {e}");
                return;
            }
        };
        let mut n = 0usize;
        let mut sum_abs = 0i64;
        let mut max_abs = 0i32;
        let mut hist = [0usize; 6]; // <=2, <=5, <=10, <=20, <=50, >50
        let mut max_preact = 0.0f32; // largest |b1 + Σ w1| element seen (float, unclamped)
        let mut scratch: Vec<usize> = Vec::with_capacity(32);
        for line in data.lines().take(2000) {
            let fen = line.split('|').next().unwrap_or("").trim();
            let pos = match Position::from_fen(fen) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let q = net.evaluate(&pos);
            let f = net.evaluate_float(&pos);
            let d = (q - f).abs();
            n += 1;
            sum_abs += d as i64;
            max_abs = max_abs.max(d);
            let b = if d <= 2 { 0 } else if d <= 5 { 1 } else if d <= 10 { 2 }
                    else if d <= 20 { 3 } else if d <= 50 { 4 } else { 5 };
            hist[b] += 1;
            for persp in [Color::White, Color::Black] {
                for &v in net.accumulate(&pos, persp, &mut scratch).iter() {
                    max_preact = max_preact.max(v.abs());
                }
            }
        }
        let mean = if n > 0 { sum_abs as f64 / n as f64 } else { 0.0 };
        eprintln!("QUANT ERROR over {n} real positions: mean={mean:.2}cp max={max_abs}cp");
        eprintln!("  |err| buckets  <=2:{} <=5:{} <=10:{} <=20:{} <=50:{} >50:{}",
                  hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]);
        eprintln!("  max|preact|={max_preact:.3}  -> safe QA <= {:.0} (i16 cap 32767)",
                  30000.0 / max_preact);
    }
}

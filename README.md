# Shark 🦈

A UCI chess engine written in **Rust**, built by studying the
[Stockfish](https://github.com/official-stockfish/Stockfish) C++ engine and
reimplementing its ideas idiomatically.

This is a learning-driven, from-scratch engine. We are **not** copying
Stockfish line-by-line — we understand each subsystem, then write the Rust
equivalent. (Stockfish is GPL v3; if Shark is ever distributed as a derivative,
it must also be GPL v3 with source — hence the license below.)

## Roadmap

The engine is built in phases. Each phase ends with something you can *run and
test*, so we never have a giant untested pile of code.

- [x] **Phase 0 — Foundation** ✅ *complete & perft-verified*
  - [x] Core types: `Color`, `PieceType`, `Piece`, `Square`, `Direction`, `Move`
  - [x] `Bitboard` primitive (set/clear/iterate/shift, popcount, lsb/msb)
  - [x] Attack tables + **magic bitboards** (sliding pieces)
  - [x] Zobrist hashing keys
  - [x] `Position`: board state, FEN parsing, make/undo move, incremental Zobrist
  - [x] Legal move generation
  - [x] **`perft`** — matches all canonical node counts (startpos d6 = 119,060,324;
        Kiwipete d5 = 193,690,690), ~51M nodes/sec in release
- [x] **Phase 1 — A playable engine** ✅ *complete — Shark plays real chess*
  - [x] Alpha-beta (negamax) search with iterative deepening + PVS
  - [x] Quiescence search (avoids the horizon effect)
  - [x] Transposition table (Zobrist-keyed, depth-preferred)
  - [x] Move ordering: TT move, MVV-LVA, killers, history heuristic
  - [x] PeSTO tapered evaluation (material + piece-square tables)
  - [x] Draw detection (50-move, repetition) + mate/stalemate
  - [x] Time management (movetime / clock+increment / depth / nodes)
  - [x] UCI protocol loop with threaded search (`stop` works mid-think)
- [~] **Phase 3 — Strength & polish** *(in progress — search is much deeper)*
  - [x] Null-move pruning, reverse-futility (static null move)
  - [x] Late move reductions (LMR)
  - [x] Aspiration windows
  - [x] Late move pruning (LMP) + frontier futility pruning
  - [x] Delta pruning in quiescence + small contempt (anti-draw)
  - [x] `selfplay` match harness to measure Elo of every change
  - [x] Static Exchange Evaluation (SEE) for captures
  - [x] Positional evaluation — mobility, king safety, passed pawns,
        pawn structure, bishop pair, rook on open file, tempo
  - [ ] Better time management + move-ordering/history tuning
  - [ ] Faster pin-aware legal move generation (more depth)
  - [ ] Lazy SMP multithreading
  - [ ] Syzygy tablebases (`shakmaty-syzygy`)
  - [ ] Profile-guided optimization
- [ ] **Phase 4 — Neural network evaluation (NNUE)** *(the big future jump)*
  - [ ] Scalar inference first (correct, then fast)
  - [ ] Incremental accumulator + king-bucket features
  - [ ] SIMD acceleration

> **Measured progress:** in the same 3-second search, the strengthened engine
> reaches **depth 15** where the original Phase-1 baseline reaches **depth 8**
> (+7 plies) — a large practical strength gain from the search techniques above.

## Layout

```
src/
  types.rs      core value types (Color, Piece, Square, Move, ...)
  bitboard.rs   the 64-bit board-set primitive
  attacks.rs    knight/king/pawn tables + magic-bitboard sliders
  zobrist.rs    position hashing keys
  position.rs   board state, FEN, make/undo move
  movegen.rs    legal move generation
  perft.rs      move-generation correctness counter
  eval.rs       PeSTO tapered evaluation
  see.rs        static exchange evaluation (capture math)
  tt.rs         transposition table
  search.rs     alpha-beta search (the brain)
  uci.rs        UCI protocol loop
  lib.rs        module wiring + re-exports
  main.rs       the `shark` binary entry point
  bin/
    selfplay.rs  match harness: plays two engines, reports Elo
```

## Build & test

```sh
cargo build --release          # compile the optimized engine
cargo test                     # run the ~109 unit tests
cargo test --release -- --ignored   # run the deep perft tests too
cargo run --release            # start the UCI engine (talk to it or plug into a GUI)
cargo run --release -- bench   # quick perft speed benchmark
```

## Playing against Shark

Shark speaks UCI, so any UCI GUI can drive it. Point the GUI at the built
binary `target/release/shark.exe` (or `shark` on Linux/macOS). Good free GUIs:
**Cute Chess**, **Arena**, **BanksiaGUI**, **En Croissant**. You can also talk to
it by hand:

```
uci
position startpos moves e2e4 e7e5
go movetime 2000
```

## License

GPL-3.0-or-later.

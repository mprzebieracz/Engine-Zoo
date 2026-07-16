# Performance baseline

## Status

Benchmark collection is deliberately deferred on this Mac for Phase 0. No
benchmark was run and no performance number is claimed here. This file is a
runnable-later template: fill it from release builds on the same machine and
record the exact environment before using the values for refactor decisions.

## Environment

| Field | Value |
|---|---|
| Host / OS | Deferred on this Mac |
| CPU | TBD |
| GPU / CUDA | TBD / TBD |
| Rust toolchain (`rustc -Vv`) | TBD |
| libtorch version and `LIBTORCH` | TBD |
| Build flags | `cargo bench --workspace --release` (when enabled) |
| Git revision | TBD |

## Measurements

Record median and units for each deterministic benchmark. Keep positions,
seeds, simulation counts, and batch sizes fixed between before/after runs.

| Workload | Median | Units | Position / configuration |
|---|---:|---|---|
| Connect Four legal generation + step | TBD | ns/op | TBD |
| Chess legal generation + step | TBD | ns/op | TBD |
| Chess v2 encode H1/H4/H8 | TBD | ns/op | TBD |
| Move/action encode + decode | TBD | ns/op | TBD |
| PUCT with deterministic mock evaluator | TBD | simulations/s | TBD |
| Gumbel with deterministic mock evaluator | TBD | simulations/s | TBD |
| Repetition-aware Chess MCTS | TBD | simulations/s | TBD |
| Self-play | TBD | positions/s | TBD |
| CPU batcher | TBD | requests/s | TBD |
| CUDA batcher | TBD | requests/s | TBD / unavailable if no CUDA |
| CUDA transferred legal-logit count | TBD | logits/request | TBD |
| Replay insertion/sample | TBD | samples/s | TBD |

## Layout checks

Record `std::mem::size_of` results alongside the benchmark revision:

| Type | Bytes |
|---|---:|
| `ChessPosition` | TBD |
| `ChessAzState<1>` | TBD |
| `ChessAzState<4>` | TBD |
| `ChessAzState<8>` | TBD |
| `Connect4` | TBD |
| Current MCTS node | TBD |

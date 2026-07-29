# AlphaZero refactor baseline

This is a measurement record for the AlphaZero refactor, captured on
2026-07-22 on the local Apple Silicon development machine. It is a reference
point, not a performance claim or a portable benchmark result.

## Environment

| Item | Recorded value |
| --- | --- |
| Target | `aarch64-apple-darwin` (macOS 26.5.2) |
| Stable Rust | `rustc 1.96.1 (31fca3adb 2026-06-26)` |
| Nightly Rust | `rustc 1.99.0-nightly (d0babd8b6 2026-07-15)` |
| `tch` / `torch-sys` | `0.20.0` (pinned in `Cargo.lock`) |
| LibTorch | CPU `2.7.1`, local installation (path omitted) |
| Python PyTorch | Not installed (`import torch` failed) |
| GPU baseline | Not available; no CUDA benchmark was run |

Commands that exercise `tch` used:

```sh
LIBTORCH=/path/to/libtorch \
DYLD_LIBRARY_PATH="$LIBTORCH/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
cargo <command>
```

The LibTorch path is local-machine configuration, not a repository
requirement. Set the equivalent environment for `torch-sys`.

## Layout measurements

All values are bytes, measured with `std::mem::size_of` for this target and
toolchain. They are guardrails for this refactor, not a cross-platform ABI.

| Type | Size |
| --- | ---: |
| `search::mcts::Node<games::Connect4Move, ()>` | 44 |
| `search::mcts::Node<chess::ChessMove, ()>` | 44 |
| `games::ChessPosition` | 112 |
| `alphazero::representation::ChessAzState<1>` | 120 |
| `alphazero::representation::ChessAzState<4>` | 480 |
| `alphazero::representation::ChessAzState<8>` | 960 |

`ChessAzState` grows linearly because it owns fixed, in-place history frames;
this is the memory budget the later search/replay refactor must preserve or
change deliberately.

## Historical deterministic CPU MCTS harness

The recorded measurement below used the retired `connect4_mcts` harness: a
uniform legal-logit evaluator with zero value, one warmup, and 200 Connect
Four searches of 256 simulations. The harness was removed after its workload
was covered by `engine-bench`; reproduce the current equivalent with:

```sh
cargo run -p engine-bench -- search --algorithm puct --simulations 256 --human
```

Recorded run:

```text
connect4_mcts: 200 searches x 256 simulations in 33.055084ms; 1548930 simulations/s
```

This number includes only the CPU search and dummy evaluator on the recorded
machine. It does not measure neural inference, batching, self-play throughput,
GPU performance, or a cross-machine comparison.

## Verification snapshot

The prescribed Phase 0 commands are:

```sh
cargo +nightly fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test --workspace --all-features --release
```

The shared worktree began later refactor phases while this baseline was being
captured, so it did not remain a stable revision long enough for a complete
workspace test count. The completed focused search run recorded **33 passed,
0 failed, 1 ignored** (`cargo test -p search`). The release workspace run
reached **153 passed, 1 failed** before Cargo stopped at the unrelated
`chess::tests::perft_kiwipete_reference` failure (expected 48 legal moves,
observed 43). Therefore that partial number is not a complete workspace count.

The other verification limits observed in this shared, in-progress worktree:

- `cargo +nightly fmt --check` reported pre-existing formatting differences in
  files owned by concurrent refactor work; no bulk formatting was applied.
- Workspace clippy was blocked by an unrelated redundant closure in
  `games/src/connect4/mod.rs` (and, at an earlier point, unused imports in
  concurrent chess work).
- Subsequent Phase 3 edits temporarily made the search crate non-compiling
  while its modules were migrated together. That is why this document records
  completed commands and their exact limits rather than implying a green,
  atomic workspace baseline.

Before comparing a later result with this record, rerun all four commands and
the harness from a single clean revision with the same LibTorch installation.

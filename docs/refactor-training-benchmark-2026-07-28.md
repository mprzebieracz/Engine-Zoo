# Refactor training benchmark — 2026-07-28

This record was captured on the local Linux development host: 16 logical CPU
cores, NVIDIA GeForce RTX 4080 SUPER (16 GiB), CUDA 13.3, LibTorch 2.7.1,
and `tch` 0.20.0. It is a local regression record, not a portable result.

## Requested workload and comparison limit

The requested workload is one chess iteration: 200 self-play games followed
by 80 training steps of 2,048 samples, CUDA FP16 inference, 16 workers, and
an intended 800 PUCT simulations per move.

There is no exact pre-refactor H1 comparison. The pre-refactor chess model
has one fixed 19-plane state with no history setting; canonical H1 has 21
planes and requires the SE trunk plus the 73-plane policy head. Therefore:

- the historical runs below are useful legacy baselines only;
- `benchmark-chess-canonical-h1.toml` is the reproducible post-refactor H1
  experiment, not an apples-to-apples model comparison;
- direct performance comparisons must normalize by generated positions and
  actual simulations, not merely game count.

## Completed historical runs

Both used the old 10-block, 64-channel residual model and 19-plane chess
representation. The merge-base (`91904c4`) lacked the declared
`src/visualization.rs` module; an empty stub was added only in its isolated
temporary worktree so its `train` binary could build. The training/search code
was unchanged.

| Revision | Self-play | Positions | Training | Notes |
| --- | ---: | ---: | ---: | --- |
| `624bbef` (before `4836d45`) | 47.872 s | 10,767 | 5.231 s | 53.84 moves/game |
| `91904c4` (merge-base) | 43.540 s | 10,388 | 4.996 s | 51.94 moves/game |

This is a 9.0% reduction in self-play wall time and 4.5% in training time
between these two local runs. It is **not** a fixed-800 result: the historical
CLI default used playout-cap randomization (25% at 800 simulations and 75% at
100), or approximately 275 simulations/move on average. Its raw JSON records
were part of the superseded refactor experiment and are no longer retained in
the working tree; the canonical current baselines are documented in
`docs/default-chess-h4-baseline.md` and `docs/tensorrt-h4-benchmark.md`.

## Current-refactor observations

Two current runs were intentionally interrupted before a completed epoch:

- `benchmark-chess-classic-10x64.toml` ran the compatibility model at fixed
  800 simulations for more than 16 minutes without an epoch metric.
- `benchmark-chess-canonical-h1.toml` is the requested canonical H1
  configuration and starts successfully, but showed the same behavior.

Both emitted a LibTorch warning on every staging allocation:

```text
Tensor.pin_memory(): argument 'device' is deprecated
```

The active batcher used approximately one CPU core while self-play workers
waited. The persistent run state remains at `total_games_generated: 0` until
the entire epoch completes, so it offers no progress visibility. These are
real findings, but are not completion-time measurements.

## Reproduce

```sh
cargo build --release --bin train
target/release/train init \
  --experiment experiments/benchmark-chess-canonical-h1.toml \
  --run-dir benchmark-results/h1
target/release/train run --run-dir benchmark-results/h1 --iterations 1 --device cuda
```

`benchmark-chess-classic-10x64.toml` exists for a model-compatible current
comparison. To compare it to the historical CLI, configure both sides with
the same budget schedule; do not compare it to the completed legacy rows
above as-is.

## Next benchmark and profiling work

1. Add periodic coordinator metrics (completed games, moves, positions/s,
   backend evaluations/s, queue depth, batch-size histogram).
2. Fix/avoid the deprecated `pin_memory(device)` path, then rerun H1 once to
   completion before treating it as a baseline.
3. Add deterministic fixed-position benchmarks for representation encode,
   legal-action mapping, MCTS (leaf batches 1/4/8/16), and batcher
   forward-only/end-to-end inference.
4. Profile a short fixed workload with Nsight Systems plus CPU `perf record`
   to split coordinator/search, representation, queue wait, H2D, inference,
   and backward/optimizer time.
5. Run the full 200-game H1 training workload three times after those
   instrumentation fixes; report median and range with positions and actual
   simulation counts.

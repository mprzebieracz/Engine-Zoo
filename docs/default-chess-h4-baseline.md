# Default chess H4 performance baseline

These existing measurements are the pre-finalization baseline. Compare the
post-finalization CUDA run against commit
`efdd74d8c7e2701d330b066f4686df55b7e08457` as recorded in
[alphazero-closure-baseline.md](alphazero-closure-baseline.md). No replacement
measurements were taken during the non-CUDA finalization pass.

This document is the results record for
`benchmarks/configs/default-chess-h4-cuda-500.toml`. The fixture is an
immutable 500-game, 80-step full iteration matching the user-facing H4 default
apart from benchmark-only disabled progress output.

## Capture command

```bash
cargo run --release -p engine-bench -- iteration \
  --experiment benchmarks/configs/default-chess-h4-cuda-500.toml \
  --samples 2 --name default-chess-h4-cuda-500 \
  --output benchmark-results/default-chess-h4-cuda-500-baseline.json --human
```

## Results

| Field | Value |
|---|---|
| Capture date | 2026-07-29 12:20:05 UTC |
| Commit | `350bbc210942e5ab9774b0529f37dece7e02ad6d` (dirty worktree) |
| Host | AMD Ryzen 7 7800X3D, 16 logical CPUs; Linux 7.1.3-arch1-1 |
| GPU / driver | NVIDIA GeForce RTX 4080 SUPER (16 GiB) / 610.43.02 |
| CUDA-enabled LibTorch build | `tch` 0.20; CUDA available (one device) |
| Raw artifact | `benchmark-results/default-chess-h4-cuda-500-baseline.json` |
| Samples | 2, no warm-up iterations |
| Full-iteration time | mean 211.384 s; range 203.885–218.883 s |
| Positions | mean 96,419; range 94,396–98,442 |
| Positions/s | mean 456.37; range 449.75–462.99 |
| Inference states/s | mean 38,428.27; range 36,358.07–40,498.47 |
| Average inference batch | mean 76.45; range 71.21–81.69 |
| Training wall time | mean 12.944 s; range 12.856–13.033 s |
| Forward + backward time | mean 11.094 s; range 11.025–11.162 s |
| Host-to-device time | mean 0.174 s; range 0.167–0.180 s |
| Optimizer time | mean 1.537 s; range 1.525–1.549 s |

The fixture uses the default H4 chess model: SE 12x128 with 16-channel SE
layers and a 128-hidden WDL head (63 input planes). Each sample runs 500
self-play games and 80 training steps on native CUDA FP16. Self-play uses 24
threads, Root-Gumbel PUCT with a 25% full 256/16 and 75% fast 64/8 schedule,
and a 16-leaf search batch. The global inference batcher uses preferred batch
32, maximum batch 256, a 4,096-state queue, and a 2 ms maximum wait.

### Per-sample measurements

| Sample | Full iteration | Positions | Positions/s | Inference states/s | Avg. batch | Forward + backward | H2D | Optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 218.883 s | 98,442 | 449.75 | 40,498.47 | 81.69 | 11.162 s | 0.180 s | 1.549 s |
| 2 | 203.885 s | 94,396 | 462.99 | 36,358.07 | 71.21 | 11.025 s | 0.167 s | 1.525 s |

## Profile artifacts

No usable profiler artifact was produced for this baseline. A full Nsight
Systems CUDA + OS runtime trace was attempted against the exact 500-game H4
fixture, but Nsight's temporary filesystem had only approximately 3.1 GiB
available and the capture was aborted. A bounded CUDA-only retry also failed
before capture because Nsight could not write its CUDA configuration.

Accordingly, there is no profiler interpretation or profile-derived bottleneck
claim for this baseline. The benchmark measurements above remain valid and are
unchanged.

To capture the profile, first free enough space in, or redirect, Nsight's
temporary directory. Then rerun either the bounded CUDA-only capture for a
quick diagnostic or the full CUDA + OS runtime capture against
`benchmarks/configs/default-chess-h4-cuda-500.toml`, and record the resulting
artifact and command here before drawing profiling conclusions.

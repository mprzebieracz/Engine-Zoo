# TensorRT H4 benchmark

This is the measured comparison of the fixed H4 12x128 WDL checkpoint on the
RTX 4080 SUPER. The checkpoint was created with 64 H4 self-play games and 10
native training steps, then exported and compiled once. The complete artifacts
are in `benchmark-results/tensorrt-h4-comparison/`.

TensorRT is a fixed-weight inference backend. These results cover TensorRT
self-play. They do **not** make the 80 optimizer steps run in TensorRT:
training remains native LibTorch, after which the new checkpoint must be
exported and compiled for the next TensorRT self-play generation.

## Compilation

| Item | Measured value |
|---|---:|
| TensorRT compilation time | 13.520014 s |
| Compiled module | `h4-wdl.trt.ts` |
| Module size | 12,916,060 bytes (13 MB) |

## Raw inference

Each cell is the median of 100 synchronized calls. Native variants use FP16
network execution; TensorRT uses FP16 internally with the existing FP32 output
contract. GPU-memory reporting is unavailable: the public `tch` API used by
the benchmark does not expose it.

| Batch | Native FP32 host: ms / positions/s | Native FP16 host: ms / positions/s | TensorRT: ms / positions/s |
|---:|---:|---:|---:|
| 8 | 1.709 / 4,680 | 1.818 / 4,399 | 0.396 / 20,214 |
| 16 | 1.618 / 9,886 | 1.673 / 9,565 | 0.404 / 39,622 |
| 32 | 1.740 / 18,389 | 1.742 / 18,368 | 0.381 / 83,924 |
| 64 | 1.739 / 36,802 | 1.735 / 36,886 | 0.609 / 105,024 |
| 128 | 2.077 / 61,634 | 1.940 / 65,967 | 1.025 / 124,934 |
| 256 | 3.377 / 75,801 | 3.323 / 77,036 | 1.592 / 160,792 |

Against native FP16, TensorRT's measured absolute output differences across
these batch sizes were policy max 0.109375–0.140625 and mean
0.013044–0.014539; value max 0.054178–0.253062 and mean
0.019981–0.054405. This is an FP16 engine comparison, not a bitwise-equivalence
test.

## Fixed-checkpoint Root-Gumbel self-play

Both backends use the same H4 model, seed, Root-Gumbel 256/64 schedule,
24 threads, 16-leaf local batch, preferred global batch 32, maximum global
batch 256, and 2 ms batch wait. Their observed average global batches are
therefore relevant to the raw 64-sized result rather than an artificial batch
256 result. A p95 request latency is not available because `BatcherStats` has
no request-latency histogram; the artifact records queue/backend maxima instead.

| Games | Backend | Wall time | Positions/s | Backend evals/s | Avg. global batch | Games/s |
|---:|---|---:|---:|---:|---:|---:|
| 128 | Native | 66.188 s | 505.89 | 31,761.35 | 63.32 | 1.93 |
| 128 | TensorRT | 25.625 s | 1,238.28 | 77,884.67 | 63.44 | 5.00 |
| 500 | Native | 218.894 s | 567.11 | 35,719.77 | 67.08 | 2.28 |
| 500 | TensorRT | 93.137 s | 1,392.16 | 87,593.18 | 67.54 | 5.37 |

At 500 games, TensorRT cut fixed-checkpoint self-play wall time by 125.757 s
(218.894 s to 93.137 s; 2.35x faster). Including one 13.520014 s compile,
the first-generation total is 106.657 s, still saving 112.237 s. The compile
cost amortizes after about 53.8 games at this measured rate.

The separately captured native H4 500-game/80-step full iteration took
218.614840 s in
[`full-iteration-native-500-80.json`](../benchmark-results/tensorrt-h4-comparison/full-iteration-native-500-80.json).
It is a native baseline only, not an end-to-end TensorRT training result. A
full handoff benchmark must run TensorRT self-play, native training, export and
compile the resulting checkpoint, and then time that whole sequence.

## Artifacts

- [Raw inference JSON](../benchmark-results/tensorrt-h4-comparison/raw-inference.json)
- [128-game native self-play JSON](../benchmark-results/tensorrt-h4-comparison/selfplay-native-128.json)
- [128-game TensorRT self-play JSON](../benchmark-results/tensorrt-h4-comparison/selfplay-tensorrt-128.json)
- [500-game native self-play JSON](../benchmark-results/tensorrt-h4-comparison/selfplay-native-500.json)
- [500-game TensorRT self-play JSON](../benchmark-results/tensorrt-h4-comparison/selfplay-tensorrt-500.json)

# Training

Self-play produces compact replay samples: native state, sparse policy target, outcome, weights, and deterministic metadata. The representation encodes states only when the trainer samples them, avoiding permanent dense tensor storage.

The trainer seeds each replay batch from the experiment seed and global step. A bounded scoped prefetch thread overlaps CPU sampling/encoding with device work without detached threads or unbounded memory. Each sampled batch transfers to the device once; microbatches are device views rather than repeated host-to-device copies.

`TrainConfig` makes Adam parameters and learning-rate schedules explicit. Supported schedules are constant, linear warmup plus cosine decay, and piecewise values. The effective learning rate and timing breakdown are written to `metrics.jsonl` with batcher statistics.

Optimizer moments are not serializable through the pinned `tch` API. Until that changes, a resume is accurately marked weights-only even when replay persistence is introduced.

## Default chess recipe

`scripts/train_chess.py` launches `experiments/chess-puct-wdl.toml`. The recipe
uses native CUDA FP16 inference with 24 self-play workers, 16-leaf local MCTS
batches, and dynamic inference batches of 32–256 states with a 2 ms wait.
It uses Root-Gumbel PUCT: 25% of positions use a 256-playout, 16-action budget;
the remainder use 64 playouts and 8 considered root actions. Fast positions
retain value targets but have zero policy-target weight.

## TensorRT chess recipe

`experiments/chess-puct-wdl-tensorrt.toml` preserves the H4 Root-Gumbel
training recipe while using fixed-weight TensorRT inference for self-play. Its
preferred batch size of 128 and 1 ms batching wait won the 64-game TensorRT H4
batching screen; it retains the 500-game default workload. The native default
remains at 32 and 2 ms. See [TensorRT
inference](tensorrt.md) for runtime setup and the initialization command that
compiles and reloads the module per generation.

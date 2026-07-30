# Training

Self-play produces compact replay samples: native state, sparse policy target, outcome, weights, and deterministic metadata. The representation encodes states only when the trainer samples them, avoiding permanent dense tensor storage.

Set `training.max_replay_reuse_per_iteration` to pace training against fresh
self-play: an enabled cap samples no more than that many training positions per
newly generated position, subject to `train_steps` as an upper bound. It keeps
the value head from repeatedly fitting a small, stale generation while the
replay buffer retains older positions for diversity. Existing experiments leave
this unset and preserve their configured training-step count.

`training.value_loss_weight` defaults to `1.0` and scales only the value term
in the combined objective. The H4 TensorRT chess recipe uses `0.25` to reduce
value-head pressure on the shared trunk while preserving policy training.

The trainer seeds each replay batch from the experiment seed and global step. A bounded scoped prefetch thread overlaps CPU sampling/encoding with device work without detached threads or unbounded memory. Each sampled batch transfers to the device once; microbatches are device views rather than repeated host-to-device copies.

`TrainConfig` makes Adam parameters and learning-rate schedules explicit. Supported schedules are constant, linear warmup plus cosine decay, and piecewise values. The effective learning rate and timing breakdown are written to `metrics.jsonl` with batcher statistics.

Optimizer moments are not serializable through the pinned `tch` API. Until that changes, a resume is accurately marked weights-only even when replay persistence is introduced.

## Checkpoint retention

Every completed training iteration atomically updates
`checkpoints/latest.safetensors`. Completed iterations whose number is a
multiple of 25 are also retained as immutable archives under `checkpoints/`,
for example `generation-000025.safetensors`. Resuming always uses the latest
weights, while the periodic archives provide rollback and evaluation points.

## Default chess recipe

`scripts/train_chess.py` launches
`experiments/chess-puct-wdl-tensorrt.toml` into
`runs/chess-puct-wdl-tensorrt-v4` on CUDA. It uses Torch-TensorRT FP16
self-play and native CUDA training with the validated H4 defaults: 112
self-play threads, 32-leaf local MCTS batches, preferred inference batch 128,
maximum batch 256, a 1 ms wait, and compile shapes opt=128 / max=256. New runs
are seeded from `runs/chess-puct-wdl-tensorrt-v3/checkpoints/latest.safetensors`
unless `--seed-checkpoint` is overridden. The script discovers a matching
Torch-TensorRT Python runtime automatically; use `--tensor-rt-python` only when
an explicit interpreter override is needed. See
[TensorRT inference](tensorrt.md) for runtime details.

It uses Root-Gumbel PUCT: 25% of positions use a 256-playout, 16-action budget;
the remainder use 64 playouts and 8 considered root actions. Fast positions
retain value targets but have zero policy-target weight.

## Native chess fallback

`experiments/chess-puct-wdl.toml` is the explicit native fallback. It preserves
the same H4 Root-Gumbel schedule while using native CUDA FP16 inference with a
preferred batch size of 32 and 2 ms wait. Use it when TensorRT is unavailable,
or when testing native inference. The TensorRT recipe uses 112 self-play
threads, preferred batch 128, and a 1 ms batching wait from the 64-game H4
self-play screen.

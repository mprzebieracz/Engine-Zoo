# Training

Self-play produces compact replay samples: native state, sparse policy target, outcome, weights, and deterministic metadata. The representation encodes states only when the trainer samples them, avoiding permanent dense tensor storage.

Set `training.max_replay_reuse_per_iteration` to pace training against fresh
self-play: an enabled cap samples no more than that many training positions per
newly generated position, subject to `train_steps` as an upper bound. It keeps
the value head from repeatedly fitting a small, stale generation while the
replay buffer retains older positions for diversity. Existing experiments leave
this unset and preserve their configured training-step count.

`training.value_loss_weight` defaults to `1.0` and scales only the value term
in the combined objective. The current H4 TensorRT chess recipe uses `1.0`
(equal policy/value weighting). Older v2/v5 runs used `0.25` to reduce
value-head pressure on the shared trunk.

The trainer seeds each replay batch from the experiment seed and global step. A bounded scoped prefetch thread overlaps CPU sampling/encoding with device work without detached threads or unbounded memory. Each sampled batch transfers to the device once; microbatches are device views rather than repeated host-to-device copies.

Self-play workers retain their configured PUCT, Root-Gumbel PUCT, or Full Gumbel MCTS instance and its scratch capacity across moves. Search randomness is reseeded from purpose-specific game seeds; played-action selection remains a separate policy choice.

`TrainConfig` makes Adam parameters and learning-rate schedules explicit. Supported schedules are constant, linear warmup plus cosine decay, and piecewise values. The effective learning rate and timing breakdown are written to `metrics.jsonl` with batcher statistics.

Optimizer moments are not serializable through the pinned `tch` API. Until that changes, a resume is accurately marked weights-only even when replay persistence is introduced.

## Checkpoint retention

Every checkpoint is first installed at an immutable generation-addressed path,
for example `checkpoints/generation-000025.safetensors`. `state.json` records
that relative path, generation, and SHA-256; it is the authority for resume and
compiled-artifact identity. `latest.safetensors` is updated afterward as a
convenience copy. A run-level writer lock prevents concurrent trainers from
interleaving checkpoint, state, and metrics writes.

For TensorRT self-play, each compiled CModule or engine has fixed weights and a
validated sibling manifest. A checkpoint change requires a new engine artifact;
the raw builder may reuse a compatible graph/profile/environment timing cache.
Training replaces the compiled inference service between generations rather
than requesting unsupported in-place reload.

## Default chess recipe

`scripts/train` (wrapper around `scripts/train_chess.py`) launches
`experiments/chess-puct-wdl-tensorrt.toml` into
`runs/chess-puct-wdl-tensorrt-v6` on CUDA. It uses exact Torch-TensorRT FP16
self-play and native CUDA training with the validated H4 defaults: 112
self-play threads, 32-leaf local MCTS batches, preferred inference batch 128,
maximum batch 256, a 1 ms wait, and compile shapes opt=128 / max=256. New runs
are seeded from
`runs/chess-puct-wdl-tensorrt-v5/checkpoints/generation-000800.safetensors`
unless `--seed-checkpoint` is overridden. The script discovers a matching
Torch-TensorRT Python runtime automatically; use `--tensor-rt-python` only when
an explicit interpreter override is needed. See
[TensorRT inference](tensorrt.md) for runtime details.

It uses Root-Gumbel PUCT: 50% of positions use a 400-playout, 16-action budget;
the remainder use 128 playouts and 8 considered root actions with
`fast_policy_weight = 0.25`. Games are capped at 512 plies. Training uses
Adam at peak LR `3e-4` with linear warmup then cosine decay to `3e-5` over
50k steps, `value_loss_weight = 1.0`, replay capacity 2M, and
`max_replay_reuse_per_iteration = 1.0`.

## Native chess fallback

`experiments/chess-puct-wdl.toml` is the explicit native fallback. It preserves
the same H4 Root-Gumbel schedule while using native CUDA FP16 inference with a
preferred batch size of 32 and 2 ms wait. Use it when TensorRT is unavailable,
or when testing native inference. The TensorRT recipe uses 112 self-play
threads, preferred batch 128, and a 1 ms batching wait from the 64-game H4
self-play screen.

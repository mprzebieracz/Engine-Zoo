# Training

Self-play produces compact replay samples: native state, sparse policy target, outcome, weights, and deterministic metadata. The representation encodes states only when the trainer samples them, avoiding permanent dense tensor storage.

The trainer seeds each replay batch from the experiment seed and global step. A bounded scoped prefetch thread overlaps CPU sampling/encoding with device work without detached threads or unbounded memory. Each sampled batch transfers to the device once; microbatches are device views rather than repeated host-to-device copies.

`TrainConfig` makes Adam parameters and learning-rate schedules explicit. Supported schedules are constant, linear warmup plus cosine decay, and piecewise values. The effective learning rate and timing breakdown are written to `metrics.jsonl` with batcher statistics.

Optimizer moments are not serializable through the pinned `tch` API. Until that changes, a resume is accurately marked weights-only even when replay persistence is introduced.


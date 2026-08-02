# Testing strategy

The portable CI path uses CPU PyTorch and runs formatting, `cargo clippy --workspace --all-targets`, and `cargo test --workspace` with default features. It deliberately does not use `--all-features`: `raw-tensorrt` is a system feature requiring CUDA and a matching TensorRT SDK.

Python syntax and compiler-protocol unit tests run in an independent job, without CUDA or TensorRT. This keeps export/build argument, JSON protocol, and precision-policy checks portable.

The raw TensorRT job is gated by the `RAW_TENSORRT_CI` repository variable and targets a self-hosted runner labelled `cuda` and `tensorrt`. It checks and tests `alphazero` with `--features raw-tensorrt`. GPU integration and stress tests remain ignored or explicitly gated on ordinary hosts; run them only on a provisioned machine with matching CUDA, TensorRT headers, and runtime libraries.

Tests target invariants rather than timing: artifact identity, batch restoration on errors, tensor contracts, deterministic seeds, persistent variant-aware MCTS, sparse-loss equivalence, and durable checkpoint writes. Performance and strength claims require retained benchmark or arena artifacts, not unit-test thresholds.

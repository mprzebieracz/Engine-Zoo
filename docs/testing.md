# Testing strategy

Pure rules and search tests run without LibTorch. ML tests add representations, replay, batching, checkpointing, and training calculations. Tests target invariants rather than wall-clock timing: terminal handling, cache identity, evaluator validation, policy normalization, deterministic seeds/order, sparse-loss equivalence, reload barriers, and safe checkpoint writes.

CI separates pure Rust, CPU-LibTorch workspace, release/no-run benchmark compilation, frontend, and Python helper checks. The frontend is checked with `npm ci`, `npm run check`, and `npm run build`.

External strength claims require deterministic puzzle suites plus paired arenas/Fastchess, sufficient game counts, and retained artifacts. Unit tests prove mechanics; they do not prove a trained network is strong.


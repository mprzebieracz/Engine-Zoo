# Architecture

The workspace is layered so that game rules and search stay independent of ML runtimes:

```text
engine_core
  ├─ games
  └─ search
       └─ alphazero
            ├─ engine_app
            └─ checkpoint-eval
```

`GameState` uses native moves. `AlphaZeroRepresentation` owns tensor encoding and policy-action mapping. `search` owns typed PUCT, Root-Gumbel PUCT, and Full Gumbel MCTS without depending on `tch`; `alphazero` owns models, inference, replay, self-play, and training.

The application dispatches game, representation, model, backend, and search variant once. Each runtime chess engine owns its configured MCTS instance, so its arena, caches, RNG, and scratch allocations survive between moves. A request budget must match the configured PUCT or Gumbel variant.

Inference retains three explicit paths:

```text
native checkpoint       -> LibTorch backend
Torch-TensorRT artifact -> TorchScript/CModule backend
raw TensorRT artifact   -> owning C++ execution session
```

The raw session owns the TensorRT runtime, engine, context, CUDA stream, and device input allocation. It copies pinned host input and enqueues TensorRT work on that same stream before synchronizing. Rust retains output tensors and legal-policy gathering. Python builds both compiled artifact variants; C++ only loads and executes an already-built raw engine.

Compiled artifacts are fixed-weight outputs paired with a versioned manifest. The manifest records checkpoint bytes, model and I/O contracts, requested and actual precision, batch profile, compiler identity, environment, and artifact digest. Compiler cache reuse checks the complete identity; loading rejects a missing or damaged manifest and a backend/model mismatch. TensorRT timing caches are separate tactic-measurement data: they may survive weight-only checkpoint changes but never make an old engine valid.

Chess keeps an authoritative `ChessGame` with repetition bookkeeping and compact search snapshots. This avoids a heap-backed repetition map in every node while preserving draw adjudication.

See [REPOSITORY_GUIDE.md](../REPOSITORY_GUIDE.md) for the linear repository tour.

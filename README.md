# Engine Zoo

Engine Zoo is a project for implementing board-game engines. It currently
contains Chess and Connect Four, with shared rules, search, neural-network,
self-play, training, evaluation, and web-UI code.

## What is here

- `crates/games` — board states, legal moves, notation, and game rules.
- `crates/search` — reusable Monte Carlo Tree Search, including PUCT and
  Gumbel-based search.
- `crates/alphazero` — board representations, neural inference, batching,
  replay, self-play, and training.
- `crates/app` and `crates/evaluations` — command-line tools, UCI, arenas,
  puzzles, and HTTP services.
- `web` — a small Svelte interface for playing and analyzing games.
- `experiments` — checked-in experiment configurations.

## AlphaZero in this project

The training loop follows the usual AlphaZero shape:

1. Self-play uses MCTS to choose moves from the neural network's policy and
   value predictions.
2. Positions, search policies, and game outcomes are stored in replay.
3. Training updates the network from replayed positions.
4. The updated network is exported and used for the next self-play iteration.

The main performance work is around the hot paths: batched inference,
persistent search state, compact board representations, deterministic replay,
and optional TensorRT inference for fixed-weight self-play. TensorRT artifacts
and checkpoints are content-addressed and checked against their source model.

## Quick start

CPU Connect Four can be run without CUDA or TensorRT:

```bash
cargo run -p engine_app --bin train -- init \
  --experiment experiments/connect4-puct.toml \
  --run-dir runs/connect4-demo

cargo run -p engine_app --bin train -- run \
  --run-dir runs/connect4-demo --device cpu --iterations 1
```

Chess training uses LibTorch and, for the fastest self-play path, TensorRT.
See [docs/tensorrt.md](docs/tensorrt.md) and [docs/training.md](docs/training.md)
for environment setup and experiment details.

## Checks

```bash
cargo fmt --check
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

The project is licensed under GPL-3.0-or-later.

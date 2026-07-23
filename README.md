# Engine Zoo

Engine Zoo is a Rust workspace for experimenting with board-game engines: today, Chess and Connect Four backed by AlphaZero-style neural search.

It keeps four concerns separate: authoritative game rules, generic tree search, model-specific representations/inference, and application/UI code. That separation matters because a future alpha-beta engine can use `games` and `search` without inheriting PyTorch, replay, or policy-vector concepts.

## What is here

| Capability | Connect Four | Chess |
| --- | --- | --- |
| Native rules and notation | Yes | Yes (FEN/UCI/SAN at the boundary) |
| PUCT MCTS | Yes | Yes |
| Full Gumbel MCTS | Yes | Yes |
| Scalar value head | Yes | Yes |
| WDL value head | — | Yes |
| Self-play and sparse replay | Yes | Yes |
| UCI/evaluation/web analysis | Reference | Primary |

```text
games (authoritative state) → representation → batcher/network → MCTS
          ↑                                      ↓
      setup / UCI                         self-play → replay → trainer
```

`crates/core` defines only portable game/agent contracts. `crates/games` owns legal moves and terminal rules. `crates/search` is pure generic MCTS. `crates/alphazero` owns representations, neural inference, batching, replay, self-play, and training. `crates/app` and `crates/evaluations` are the executable boundaries; `web` is the Svelte frontend.

## Quick start: CPU Connect Four

The experiment is immutable after initialization. Runtime flags only select device and how long to run.

```bash
cargo run -p engine_app --bin train -- init \
  --experiment experiments/connect4-puct.toml \
  --run-dir runs/connect4-demo

cargo run -p engine_app --bin train -- run \
  --run-dir runs/connect4-demo --device cpu --iterations 1

cargo run -p engine_app --bin train -- inspect --run-dir runs/connect4-demo
```

Chess configurations are in `experiments/`: PUCT/WDL, Full Gumbel/WDL, canonical scalar ablation, and the classic 19-plane / 20,480-action scalar model. The classic model remains a supported option for existing checkpoints; it is not a deprecated runtime path.

## Chess training and LibTorch

Training/inference use `tch` and require a compatible LibTorch installation. Set `LIBTORCH` explicitly, or use a Python PyTorch installation with `LIBTORCH_USE_PYTORCH=1`. The build scripts deliberately do not guess a machine-local path.

```bash
python3 scripts/train_chess.py --device cuda
```

The script initializes `runs/chess-puct-wdl` from the checked-in experiment if needed, then invokes `train run --forever`. It does not embed a second set of training flags.

## Experiment files

An experiment TOML fixes model shape/representation, search policy, replay, training, inference batching, and random seed. `state.json` is mutable progress: iteration, generation, global step, checkpoint location, generated games, and honest resume information. Checkpoints are written as `latest.tmp.safetensors`, renamed to `latest.safetensors`, and only then recorded in state.

The canonical chess model uses the 8×8×73 policy layout and a parameterized squeeze-excitation trunk. The classic chess model uses the preserved 19-plane, 20,480-action layout with a scalar head. A SHA-256 fingerprint of the complete `ModelSpec` identifies shape-compatible model data.

## Correctness and reproducibility

- MCTS uses typed terminal outcomes and rejects invalid evaluator values/logits rather than converting them to draws.
- Cache keys include a post-reload namespace, so neural results cannot survive a successful weight reload.
- Self-play game IDs and seeds are global and worker-independent; completed games are committed to replay in game-ID order.
- Replay sampling is seeded from experiment seed plus global training step.
- Replay persists in memory for now. A resumed checkpoint is explicitly recorded as weights-only: replay count is reset and optimizer moments are not claimed restored.

Run the checks locally with:

```bash
cargo fmt --check
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cd web && npm ci && npm run check && npm run build
```

## Benchmarks

The repository has a deterministic pure-search harness in `crates/search/benches/connect4_mcts.rs`; benchmark sources are compiled in CI with `--no-run`. GPU inference/training benchmarks are intentionally not presented as measurements until they are run on suitable hardware. Methodology and the required benchmark matrix live in [docs/benchmarks.md](docs/benchmarks.md).

## Further reading

- [Repository guide](REPOSITORY_GUIDE.md): linear, file-level architecture tour.
- [Architecture](docs/architecture.md), [PUCT](docs/search/puct.md), and [Full Gumbel](docs/search/full-gumbel.md).
- [Training](docs/training.md), [Chess representation](docs/chess-representation.md), and [reproducibility](docs/reproducibility.md).
- [Testing](docs/testing.md) and [archived refactor plan](docs/archive/architecture-refactor-plan-2026-07.md).

## License

Engine Zoo is GPL-3.0-or-later. The web frontend uses Chessground, which is also GPL-3.0-or-later; see `web/README.md` for its frontend-specific notes.

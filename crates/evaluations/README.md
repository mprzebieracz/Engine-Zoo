# Checkpoint evaluation

`checkpoint-eval` plays a short, color-balanced match between one saved chess checkpoint and a strength-limited Stockfish process. It writes standard PGNs and one JSON result record per checkpoint.

Install the bundled local Stockfish release with `crates/evaluations/install_stockfish.sh`, or set `STOCKFISH_BIN` to another executable path.

Training starts `watch_checkpoints.sh` by default. The watcher evaluates each newly created numbered checkpoint whose index is divisible by `EVAL_EVERY` (default: 50). Results are written to:

```text
data/runs/chess/evaluations/ratings.jsonl
data/runs/chess/evaluations/ckpt_0050.json
data/runs/chess/evaluations/pgn/ckpt_0050/
```

Useful training settings:

```bash
STOCKFISH_BIN=/path/to/stockfish \
STOCKFISH_ELO=1400 \
EVAL_GAMES=4 BASELINE_GAMES=4 \
EVAL_SIMULATIONS=800 \
EVAL_DEVICE=cpu \
scripts/train_chess.sh
```

`EVAL_DEVICE=cpu` is the default so evaluation does not compete with GPU training. Use `cuda` only when the GPU has enough headroom.

Set `EVAL_BACKFILL=1` when starting the watcher to evaluate existing matching checkpoints too; by default it only watches checkpoints created after training starts.

Each checkpoint after the first also plays the checkpoint from one evaluation interval earlier (normally `N - 50`). The reported `rough_elo_delta` is a smoothed estimate against the selected Stockfish setting; `baseline_rough_elo_delta` is the same estimate against the prior checkpoint. Four games is a monitoring signal, not a stable absolute Elo rating.

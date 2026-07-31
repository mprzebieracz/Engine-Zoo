# Checkpoint evaluation

`checkpoint-eval` plays a short, color-balanced match between one saved chess checkpoint and a strength-limited Stockfish process. It writes standard PGNs and one JSON result record per checkpoint.

Install the bundled local Stockfish release with `python3 crates/evaluations/install_stockfish.py`, or set `STOCKFISH_BIN` to another executable path.

Checkpoint evaluation is run explicitly with the current Rust evaluation binaries. There is no background watcher.

Useful training settings:

```bash
scripts/train
```

The arena and Stockfish evaluation binaries accept their current parameters directly; use `--help` for the available options.

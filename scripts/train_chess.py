#!/usr/bin/env python3
"""Run the current chess self-play/training CLI with editable parameters."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Main workload parameters. Change these instead of editing a shell command.
DEFAULTS = {
    "run_dir": "data/runs/chess-az-v2-h4",
    "architecture": "chess-az-v2",
    "history": 4,
    "games": 500,
    "threads": 24,
    "wait_for": 128,
    "batch_timeout_ms": 5,
    "mcts_leaf_batch_size": 24,
    "v2_full_simulations": 400,
    "v2_full_root_candidates": 32,
    "v2_fast_simulations": 100,
    "v2_fast_root_candidates": 16,
    "v2_full_simulation_probability": 0.5,
    "mcts_variant": "gumbel",
    "max_moves": 512,
    "tt_entries": 1_000_000,
    "train_steps": 80,
    "train_progress_every": 10,
    "batch_size": 4096,
    "buffer": 500_000,
    "numbered_checkpoint_every": 50,
    "archive_checkpoint_minutes": 60,
    "device": "cuda",
    "inference_precision": "auto",
    "progress_every": 25,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULTS.items():
        p.add_argument(
            f"--{name.replace('_', '-')}", default=default, type=type(default)
        )
    args = p.parse_args()
    command = ["cargo", "build", "--release", "--bin", "train"]
    subprocess.run(command, cwd=ROOT, check=True)
    rust_args = ["target/release/train", "--game", "chess"]
    for name in DEFAULTS:
        value = getattr(args, name)
        rust_args += [f"--{name.replace('_', '-')}", str(value)]
    rust_args += ["--mode", "continuous", "--forever"]
    return subprocess.run(rust_args, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())

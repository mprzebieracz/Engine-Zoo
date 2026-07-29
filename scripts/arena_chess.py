#!/usr/bin/env python3
"""Run a colour-balanced checkpoint-vs-checkpoint arena."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Edit these defaults for the local machine/workload.
DEFAULTS = {
    "candidate": "data/runs/chess-az-v2-h4/best.safetensors",
    "baseline": "data/runs/chess-az-v2-h4/checkpoints/ckpt_0450.safetensors",
    "candidate_architecture": "chess-az-v2",
    "baseline_architecture": "chess-az-v2",
    "output_dir": "data/arenas/sitekarena2",
    "opening_plies": 8,
    "games": 4,
    "simulations": 400,
    "device": "cuda",
    "concurrency": 1,
    "max_moves": 512,
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fastchess", default=os.getenv("FASTCHESS_BIN", "fastchess"))
    p.add_argument("--uci", default="target/release/engine-zoo-uci")
    for name, default in DEFAULTS.items():
        p.add_argument(
            f"--{name.replace('_', '-')}",
            default=os.getenv(name.upper(), default),
            type=type(default),
        )
    p.add_argument("--openings")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="extra arguments passed to eval-arena")
    return p


def fastchess_path(value: str) -> Path:
    fastchess = shutil.which(value) or value
    fastchess_path = Path(fastchess)
    if not fastchess_path.is_absolute():
        fastchess_path = ROOT / fastchess_path

    if not fastchess_path.is_file() or not os.access(fastchess_path, os.X_OK):
        raise SystemExit(f"fastchess not found or not executable: {fastchess}")

    return fastchess_path


def build_command(args: argparse.Namespace, fastchess: Path) -> list[str]:
    command = [
        "target/release/eval-arena",
        "--fastchess", str(fastchess), "--uci", args.uci,
        "--candidate", args.candidate, "--baseline", args.baseline,
        "--candidate-architecture", args.candidate_architecture,
        "--baseline-architecture", args.baseline_architecture,
        "--opening-plies", str(args.opening_plies), "--output-dir", args.output_dir,
        "--games", str(args.games), "--simulations", str(args.simulations),
        "--device", args.device, "--concurrency", str(args.concurrency),
        "--max-moves", str(args.max_moves),
    ]

    if args.openings:
        command += ["--openings", args.openings]

    return command


def build_binaries() -> None:
    subprocess.run(
        [
            "cargo", "build", "--release", "-p", "checkpoint-eval", "--bin", "eval-arena",
            "-p", "engine_app", "--bin", "engine-zoo-uci",
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    args = parser().parse_args()
    fastchess = fastchess_path(args.fastchess)

    build_binaries()

    command = build_command(args, fastchess)
    return subprocess.run(command + args.extra, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())

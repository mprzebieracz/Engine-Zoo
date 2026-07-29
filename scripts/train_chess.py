#!/usr/bin/env python3
"""Initialize, then continuously run the checked-in chess Root-Gumbel PUCT experiment."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "experiments/chess-puct-wdl.toml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/chess-puct-wdl")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    subprocess.run(["cargo", "build", "--release", "--bin", "train"], cwd=ROOT, check=True)
    binary = ROOT / "target/release/train"
    if not (args.run_dir / "experiment.toml").exists():
        subprocess.run(
            [binary, "init", "--experiment", args.experiment, "--run-dir", args.run_dir],
            cwd=ROOT,
            check=True,
        )
    return subprocess.run(
        [binary, "run", "--run-dir", args.run_dir, "--device", args.device, "--forever"],
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the configured multi-opponent checkpoint arena."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/arena.toml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true", help="use debug binaries")
    args, extra = parser.parse_known_args()

    profile = [] if args.debug else ["--release"]
    subprocess.run(
        ["cargo", "build", *profile, "-p", "checkpoint-eval", "--bin", "eval-big-arena",
         "-p", "engine_app", "--bin", "engine-zoo-uci"],
        cwd=ROOT,
        check=True,
    )
    binary = ROOT / ("target/debug" if args.debug else "target/release") / "eval-big-arena"
    command = [str(binary), "--config", str(args.config)] + extra
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())

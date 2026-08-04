#!/usr/bin/env python3
"""Compatibility launcher for the Rust ``eval-big-arena`` command."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="run an already-built eval-big-arena binary",
    )
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    profile = "debug" if args.debug else "release"
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    binary = target_dir / profile / "eval-big-arena"
    if args.skip_build:
        if not binary.is_file():
            parser.error(f"--skip-build requested but {binary} is not available")
        command = [str(binary), *forwarded]
    else:
        command = [
            "cargo",
            "run",
            *( [] if args.debug else ["--release"] ),
            "-p",
            "checkpoint-eval",
            "--bin",
            "eval-big-arena",
            "--",
            *forwarded,
        ]
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())

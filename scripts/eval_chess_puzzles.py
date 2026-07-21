#!/usr/bin/env python3
"""Run the current Rust chess puzzle benchmark and write its reports."""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Edit these defaults for the local run.
DEFAULTS = {
    "run_dir": "data/runs/chess-az-v2-h4",
    "model": "best",
    "mode": "mcts",
    "simulations": 100,
    "wait_for_count": 1,
    "suite": "data/suites/chess_lichess_easy.jsonl",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", default=os.getenv("SERVER"))
    for name, default in DEFAULTS.items():
        p.add_argument(f"--{name.replace('_', '-')}", default=os.getenv(name.upper(), default), type=type(default))
    p.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = p.parse_args()
    out_dir = Path(os.getenv("OUT_DIR", f"{args.run_dir}/eval/puzzles"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{args.mode}_{args.simulations}_{args.stamp}"
    jsonl = Path(os.getenv("JSONL_OUT", str(out_dir / f"{stem}.jsonl")))
    html = Path(os.getenv("HTML_OUT", str(out_dir / f"{stem}.html")))
    stderr = Path(os.getenv("STDERR_LOG", str(out_dir / f"{stem}.stderr.log")))
    subprocess.run(["cargo", "build", "--release", "--bin", "eval"], cwd=ROOT, check=True)
    command = [
        "target/release/eval", "bench", "--game", "chess", "--model", args.model,
        "--mode", args.mode, "--simulations", str(args.simulations),
        "--wait-for-count", str(args.wait_for_count), "--suite", args.suite,
        "--output", str(jsonl), "--html", str(html),
    ]
    if args.server:
        command += ["--server", args.server]
    else:
        command += ["--run-dir", args.run_dir]
    with stderr.open("w") as log:
        subprocess.run(command, cwd=ROOT, stderr=log, check=True)
    print(f"wrote {jsonl}\nwrote {html}\nwrote {stderr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Initialize, then continuously run the checked-in TensorRT chess experiment."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "experiments/chess-puct-wdl-tensorrt.toml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs/chess-puct-wdl-tensorrt",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--tensor-rt-python",
        type=Path,
        help="Python executable in the matching Torch-TensorRT environment",
    )
    parser.add_argument(
        "--tensor-rt-compiler",
        type=Path,
        help="Optional TensorRT compiler script; defaults to scripts/compile_tensorrt.py",
    )
    args = parser.parse_args()

    uses_tensor_rt = args.experiment.resolve() == DEFAULT_EXPERIMENT.resolve()
    if uses_tensor_rt and args.tensor_rt_python is None:
        parser.error("the default TensorRT experiment requires --tensor-rt-python")

    subprocess.run(["cargo", "build", "--release", "--bin", "train"], cwd=ROOT, check=True)
    binary = ROOT / "target/release/train"
    if not (args.run_dir / "experiment.toml").exists():
        subprocess.run(
            [binary, "init", "--experiment", args.experiment, "--run-dir", args.run_dir],
            cwd=ROOT,
            check=True,
        )
    command = [binary, "run", "--run-dir", args.run_dir, "--device", args.device, "--forever"]
    if args.tensor_rt_python is not None:
        command.extend(["--tensor-rt-python", args.tensor_rt_python])
    if args.tensor_rt_compiler is not None:
        command.extend(["--tensor-rt-compiler", args.tensor_rt_compiler])

    return subprocess.run(
        command,
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

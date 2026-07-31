#!/usr/bin/env python3
"""Run 3 iterations of raw-TensorRT training and summarize compile/self-play timing.

Mirrors scripts/bench_tensorrt_training.py but:
- builds `train` with `--features raw-tensorrt`
- uses experiments/chess-puct-wdl-tensorrt-raw.toml (500 games, raw engine)
- passes a persistent timing cache and system TensorRT builder Python
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "experiments/chess-puct-wdl-tensorrt-raw.toml"
DEFAULT_SEED = Path(
    "/home/mati/engine-zoo/runs/chess-puct-wdl-tensorrt-v5/checkpoints/generation-000800.safetensors"
)
DEFAULT_OPT = 128
DEFAULT_MAX = 256
DEFAULT_MIN = 1

COMPILE_RE = re.compile(r"compiled TensorRT module in ([0-9.]+)s")
ITERATION_RE = re.compile(r"===== ITERATION (\d+) =====")
ITERATION_ELAPSED_RE = re.compile(r"iteration elapsed: ([0-9.]+)s")
TOTAL_RE = re.compile(r"TOTAL TRAINING TIME: ([0-9.]+)s")
ITERATION_TOTAL_RE = re.compile(
    r"iteration total: (\d+) games \| (\d+) positions \| (\d+) replay samples"
)
PROGRESS_RE = re.compile(
    r"self-play progress:\s+(\d+)/(\d+)\s+games,\s+([0-9.]+)\s+games/s,\s+([0-9.]+)\s+positions/s",
    re.IGNORECASE,
)


def _libtorch_root() -> Path | None:
    configured = os.environ.get("LIBTORCH")
    if configured:
        return Path(configured).expanduser()
    for candidate in (
        Path("/home/mati/libs/libtorch-2.11.0-cu130/libtorch"),
        Path("/home/mati/libs/libtorch"),
    ):
        if candidate.is_dir():
            return candidate
    return None


def _find_torch_python(hint: Path | None) -> Path | None:
    candidates: list[Path] = []
    if hint is not None:
        candidates.append(hint)
    if configured := os.environ.get("TRT_PYTHON"):
        candidates.append(Path(configured).expanduser())
    candidates.append(Path.home() / "venvs/engine-zoo-trt-py313/bin/python")
    candidates.append(Path(sys.executable))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().absolute()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        result = subprocess.run(
            [str(candidate), "-c", "import torch; assert torch.cuda.is_available()"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return candidate
    return None


def _runtime_environment(torch_python: Path) -> dict[str, str]:
    environment = os.environ.copy()
    trt10_libs = (
        Path.home()
        / "venvs/engine-zoo-trt-py313/lib/python3.13/site-packages/tensorrt_libs"
    )
    library_paths: list[str] = [
        str(trt10_libs),
        "/opt/cuda/targets/x86_64-linux/lib",
        "/opt/cuda/lib64",
        "/usr/lib",
    ]
    if (libtorch := _libtorch_root()) is not None:
        libtorch_lib = libtorch / "lib"
        if libtorch_lib.is_dir():
            library_paths.insert(0, str(libtorch_lib))
        environment["LIBTORCH"] = str(libtorch)
    if existing := environment.get("LD_LIBRARY_PATH"):
        library_paths.append(existing)
    environment["LD_LIBRARY_PATH"] = ":".join(library_paths)
    environment.pop("LD_PRELOAD", None)
    # Build in-process with the torch/TensorRT 10.15 venv (BuilderFlag.FP16).
    # Rust links the matching libnvinfer.so.10 via vendored 10.15 headers.
    environment.pop("TENSORRT_BUILD_PYTHON", None)
    environment["TENSORRT_PREFER_SYSTEM_BUILDER"] = "0"
    environment.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")
    environment["TRT_PYTHON"] = str(torch_python)
    return environment


def _seed_checkpoint(run_dir: Path, checkpoint: Path) -> None:
    if not checkpoint.is_file():
        raise SystemExit(f"seed checkpoint not found: {checkpoint}")
    destination = run_dir / "checkpoints" / "latest.safetensors"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination)
    (run_dir / "state.json").write_text(
        "{\n"
        '  "format_version": 2,\n'
        '  "iteration": 0,\n'
        '  "model_generation": 0,\n'
        '  "global_step": 0,\n'
        '  "total_games_generated": 0,\n'
        '  "latest_checkpoint": "checkpoints/latest.safetensors",\n'
        '  "replay_sample_count": 0,\n'
        '  "optimizer_moments_restored": false,\n'
        '  "resume_kind": "weights-only"\n'
        "}\n",
        encoding="utf-8",
    )


def _parse_report(log_text: str) -> dict:
    compiles = [float(match.group(1)) for match in COMPILE_RE.finditer(log_text)]
    iterations: list[dict] = []
    current: dict | None = None
    for line in log_text.splitlines():
        if match := ITERATION_RE.search(line):
            current = {"iteration": int(match.group(1)), "progress": []}
            iterations.append(current)
            continue
        if current is None:
            continue
        if match := PROGRESS_RE.search(line):
            current["progress"].append(
                {
                    "games_done": int(match.group(1)),
                    "games_total": int(match.group(2)),
                    "games_per_second": float(match.group(3)),
                    "positions_per_second": float(match.group(4)),
                }
            )
        if match := ITERATION_TOTAL_RE.search(line):
            current["games"] = int(match.group(1))
            current["positions"] = int(match.group(2))
            current["replay_samples"] = int(match.group(3))
        if match := ITERATION_ELAPSED_RE.search(line):
            current["elapsed_seconds"] = float(match.group(1))
            if "positions" in current and current["elapsed_seconds"] > 0:
                current["positions_per_second"] = (
                    current["positions"] / current["elapsed_seconds"]
                )
            if "games" in current and current["elapsed_seconds"] > 0:
                current["games_per_second"] = current["games"] / current["elapsed_seconds"]

    for index, compile_seconds in enumerate(compiles):
        if index == 0:
            continue
        target = iterations[index - 1] if index - 1 < len(iterations) else None
        if target is not None:
            target["post_compile_seconds"] = compile_seconds

    total = TOTAL_RE.search(log_text)
    return {
        "compile_seconds": compiles,
        "initial_compile_seconds": compiles[0] if compiles else None,
        "post_iteration_compile_seconds": compiles[1:],
        "iterations": iterations,
        "total_training_seconds": float(total.group(1)) if total else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs/chess-puct-wdl-tensorrt-raw-bench-3iter",
    )
    parser.add_argument("--seed-checkpoint", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--tensor-rt-python", type=Path)
    parser.add_argument("--tensor-rt-min-batch-size", type=int, default=DEFAULT_MIN)
    parser.add_argument("--tensor-rt-opt-batch-size", type=int, default=DEFAULT_OPT)
    parser.add_argument("--tensor-rt-max-batch-size", type=int, default=DEFAULT_MAX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmarks/results/raw-tensorrt-train-3iter",
    )
    parser.add_argument("--fresh-run-dir", action="store_true")
    args = parser.parse_args()

    torch_python = _find_torch_python(args.tensor_rt_python)
    if torch_python is None:
        parser.error("could not find a CUDA torch Python environment")
    environment = _runtime_environment(torch_python)

    if args.fresh_run_dir and args.run_dir.exists():
        shutil.rmtree(args.run_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.log"
    report_path = args.output_dir / "report.json"
    timing_cache = args.run_dir / "tensorrt-timing.cache"

    print("building train with --features raw-tensorrt ...")
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "engine_app",
            "--bin",
            "train",
            "--features",
            "raw-tensorrt",
        ],
        cwd=ROOT,
        check=True,
        env=environment,
    )
    binary = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "release" / "train"
    if not binary.is_file():
        # cargo may use a shared target dir outside the worktree
        candidates = list(Path("/tmp").glob("cursor-sandbox-cache/*/cargo-target/release/train"))
        candidates.append(ROOT / "target/release/train")
        for candidate in candidates:
            if candidate.is_file():
                binary = candidate
                break

    if not (args.run_dir / "experiment.toml").is_file():
        subprocess.run(
            [
                str(binary),
                "init",
                "--experiment",
                str(args.experiment),
                "--run-dir",
                str(args.run_dir),
            ],
            cwd=ROOT,
            check=True,
            env=environment,
        )
        _seed_checkpoint(args.run_dir, args.seed_checkpoint)

    command = [
        str(binary),
        "run",
        "--run-dir",
        str(args.run_dir),
        "--device",
        args.device,
        "--iterations",
        str(args.iterations),
        "--tensor-rt-python",
        str(torch_python),
        "--tensor-rt-compiler",
        str(ROOT / "scripts/compile_tensorrt_raw.py"),
        "--tensor-rt-min-batch-size",
        str(args.tensor_rt_min_batch_size),
        "--tensor-rt-opt-batch-size",
        str(args.tensor_rt_opt_batch_size),
        "--tensor-rt-max-batch-size",
        str(args.tensor_rt_max_batch_size),
        "--tensor-rt-timing-cache",
        str(timing_cache),
    ]

    print(
        "raw TensorRT 3-iter screen: "
        f"experiment={args.experiment} "
        f"compile min={args.tensor_rt_min_batch_size} "
        f"opt={args.tensor_rt_opt_batch_size} "
        f"max={args.tensor_rt_max_batch_size} "
        f"timing_cache={timing_cache}"
    )
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as handle:
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="")
            lines.append(line)
    code = process.wait()
    wall = time.perf_counter() - started
    report = {
        "backend": "raw-tensorrt",
        "command": command,
        "returncode": code,
        "wall_seconds": wall,
        "timing_cache": str(timing_cache),
        "compile_profile": {
            "min": args.tensor_rt_min_batch_size,
            "opt": args.tensor_rt_opt_batch_size,
            "max": args.tensor_rt_max_batch_size,
            "precision": "fp16",
        },
        "experiment": str(args.experiment),
        "run_dir": str(args.run_dir),
        "baseline_reference": {
            "positions_per_second": 1300,
            "games_per_second": 10,
            "compile_seconds": [12, 13],
            "note": "user-reported Torch-TensorRT production baseline after ~500 games",
        },
        "metrics": _parse_report("".join(lines)),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {log_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

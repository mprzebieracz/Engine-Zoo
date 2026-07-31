#!/usr/bin/env python3
"""Run a short TensorRT training screen and summarize compile/self-play timing.

Default TensorRT params match experiments/chess-puct-wdl-tensorrt.toml
(opt=128/max=256, fp16, 112 threads, leaf batch 32). The checked-in
bench recipe only reduces game count so three iterations stay measurable.

Builder timing-cache is unsupported by the TorchScript frontend that
produces Rust-loadable modules, so this harness does not pass
--tensor-rt-timing-cache.
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
DEFAULT_EXPERIMENT = ROOT / "experiments/chess-puct-wdl-tensorrt-bench-3iter.toml"
DEFAULT_SEED = (
    ROOT
    / "runs/chess-puct-wdl-tensorrt-v5/checkpoints/generation-000800.safetensors"
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


def _libtorch_root() -> Path | None:
    configured = os.environ.get("LIBTORCH")
    if configured:
        return Path(configured).expanduser()
    config = ROOT / ".cargo/config.toml"
    if not config.is_file():
        return None
    match = re.search(
        r"^LIBTORCH\s*=\s*[\"\']([^\"\']+)",
        config.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return Path(match.group(1)).expanduser() if match else None


def _can_import_tensor_rt(python: Path) -> bool:
    result = subprocess.run(
        [str(python), "-c", "import torch, torch_tensorrt"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _tensor_rt_site_lib(python: Path) -> Path:
    query = (
        "from pathlib import Path; import importlib.util; "
        "print(Path(next(iter(importlib.util.find_spec('torch_tensorrt')"
        ".submodule_search_locations))) / 'lib')"
    )
    result = subprocess.run(
        [str(python), "-c", query],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def _find_tensor_rt_python(hint: Path | None) -> Path | None:
    candidates: list[Path] = []
    if hint is not None:
        candidates.append(hint)
    if configured := os.environ.get("TRT_PYTHON"):
        candidates.append(Path(configured).expanduser())
    candidates.append(Path.home() / "venvs/engine-zoo-trt-py313/bin/python")
    candidates.append(Path("/tmp/engine-zoo-arena/venv/bin/python"))
    for environment in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        if prefix := os.environ.get(environment):
            candidates.append(Path(prefix) / "bin/python")
    candidates.append(Path(sys.executable))
    for relative in (
        ".venv/engine-zoo-trt/bin/python",
        ".venv/engine-zoo-trt-py313/bin/python",
    ):
        candidates.append(ROOT / relative)
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().absolute()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if _can_import_tensor_rt(candidate):
            return candidate
    return None


def _runtime_environment(python: Path) -> dict[str, str]:
    trt_lib = _tensor_rt_site_lib(python)
    runtime_library = trt_lib / "libtorchtrt.so"
    if not runtime_library.is_file():
        raise SystemExit(f"missing Torch-TensorRT runtime library: {runtime_library}")

    environment = os.environ.copy()
    library_paths = [str(trt_lib)]
    preload: list[str] = []
    if (libtorch := _libtorch_root()) is not None:
        libtorch_lib = libtorch / "lib"
        libtorch_so = libtorch_lib / "libtorch.so"
        if libtorch_lib.is_dir():
            library_paths.insert(0, str(libtorch_lib))
        if libtorch_so.is_file():
            preload.append(str(libtorch_so))
    nvidia_lib = trt_lib.parent.parent / "tensorrt_libs"
    if nvidia_lib.is_dir():
        library_paths.append(str(nvidia_lib))
    if existing := environment.get("LD_LIBRARY_PATH"):
        library_paths.append(existing)
    environment["LD_LIBRARY_PATH"] = ":".join(library_paths)
    preload.append(str(runtime_library))
    if existing_preload := environment.get("LD_PRELOAD"):
        preload.append(existing_preload)
    environment["LD_PRELOAD"] = ":".join(preload)
    environment.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")
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
            current = {"iteration": int(match.group(1))}
            iterations.append(current)
            continue
        if current is None:
            continue
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

    # First compile is before iteration 1; later compiles follow each iteration.
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
        default=ROOT / "runs/chess-puct-wdl-tensorrt-bench-3iter",
    )
    parser.add_argument("--seed-checkpoint", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--tensor-rt-python", type=Path)
    parser.add_argument("--tensor-rt-compiler", type=Path)
    parser.add_argument("--tensor-rt-min-batch-size", type=int, default=DEFAULT_MIN)
    parser.add_argument("--tensor-rt-opt-batch-size", type=int, default=DEFAULT_OPT)
    parser.add_argument("--tensor-rt-max-batch-size", type=int, default=DEFAULT_MAX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmarks/results/tensorrt-train-3iter",
    )
    parser.add_argument(
        "--fresh-run-dir",
        action="store_true",
        help="delete the run directory before initializing",
    )
    args = parser.parse_args()

    if not (
        0
        < args.tensor_rt_min_batch_size
        <= args.tensor_rt_opt_batch_size
        <= args.tensor_rt_max_batch_size
    ):
        parser.error("TensorRT batch sizes must satisfy 0 < min <= opt <= max")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    python = _find_tensor_rt_python(args.tensor_rt_python)
    if python is None:
        parser.error("could not find a Torch-TensorRT Python environment")
    environment = _runtime_environment(python)

    if args.fresh_run_dir and args.run_dir.exists():
        shutil.rmtree(args.run_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.log"
    report_path = args.output_dir / "report.json"

    subprocess.run(
        ["cargo", "build", "--release", "-p", "engine_app", "--bin", "train"],
        cwd=ROOT,
        check=True,
        env=environment,
    )
    binary = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "release" / "train"
    if not binary.is_file():
        binary = ROOT / "target/release/train"

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
        str(python),
        "--tensor-rt-min-batch-size",
        str(args.tensor_rt_min_batch_size),
        "--tensor-rt-opt-batch-size",
        str(args.tensor_rt_opt_batch_size),
        "--tensor-rt-max-batch-size",
        str(args.tensor_rt_max_batch_size),
    ]
    if args.tensor_rt_compiler is not None:
        command.extend(["--tensor-rt-compiler", str(args.tensor_rt_compiler)])

    print(
        "TensorRT 3-iter screen: "
        f"experiment={args.experiment} "
        f"compile min={args.tensor_rt_min_batch_size} "
        f"opt={args.tensor_rt_opt_batch_size} "
        f"max={args.tensor_rt_max_batch_size} "
        f"timing_cache=unsupported"
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
        "command": command,
        "returncode": code,
        "wall_seconds": wall,
        "timing_cache": "unsupported-by-torchscript-frontend",
        "compile_profile": {
            "min": args.tensor_rt_min_batch_size,
            "opt": args.tensor_rt_opt_batch_size,
            "max": args.tensor_rt_max_batch_size,
            "precision": "fp16",
        },
        "experiment": str(args.experiment),
        "run_dir": str(args.run_dir),
        "metrics": _parse_report("".join(lines)),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {log_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

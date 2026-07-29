#!/usr/bin/env python3
"""Initialize, then continuously run the checked-in TensorRT chess experiment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "experiments/chess-puct-wdl-tensorrt.toml"


def _can_import_tensor_rt(python: Path) -> bool:
    """Return whether *python* has the Torch-TensorRT compiler installed."""

    result = subprocess.run(
        [str(python), "-c", "import torch, torch_tensorrt"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    return result.returncode == 0


def _tensor_rt_runtime_environment(python: Path) -> dict[str, str]:
    """Prepare the dynamic linker environment for TensorRT TorchScript loads."""

    query = (
        "from pathlib import Path; import importlib.util; "
        "print(Path(next(iter(importlib.util.find_spec('torch_tensorrt').submodule_search_locations))) / 'lib')"
    )
    result = subprocess.run(
        [str(python), "-c", query],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    trt_lib = Path(result.stdout.strip())
    runtime_library = trt_lib / "libtorchtrt.so"
    if not runtime_library.is_file():
        return os.environ.copy()

    environment = os.environ.copy()
    library_paths = [str(trt_lib)]

    # Rust/tch may use a local LibTorch distribution that is not installed in
    # the system linker cache.  Keep this portable by honoring the standard
    # LIBTORCH override instead of embedding a machine-specific path.
    libtorch = environment.get("LIBTORCH")
    if not libtorch:
        config = ROOT / ".cargo/config.toml"
        if config.is_file():
            contents = config.read_text(encoding="utf-8")
            match = re.search(r'^LIBTORCH\s*=\s*[\"\']([^\"\']+)', contents, re.MULTILINE)
            if match:
                libtorch = match.group(1)
    if libtorch:
        libtorch_lib = Path(libtorch).expanduser() / "lib"
        if libtorch_lib.is_dir():
            library_paths.append(str(libtorch_lib))

    nvidia_lib = trt_lib.parent.parent / "tensorrt_libs"
    if nvidia_lib.is_dir():
        library_paths.append(str(nvidia_lib))

    existing_paths = environment.get("LD_LIBRARY_PATH")
    if existing_paths:
        library_paths.append(existing_paths)
    environment["LD_LIBRARY_PATH"] = ":".join(library_paths)

    # Keep noisy Python deprecation warnings from optional TensorRT plugins
    # out of the training log without hiding real subprocess failures.
    environment.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")

    existing_preload = environment.get("LD_PRELOAD")
    environment["LD_PRELOAD"] = ":".join(
        [str(runtime_library), existing_preload] if existing_preload else [str(runtime_library)]
    )
    return environment


def _find_tensor_rt_python() -> Path | None:
    """Find a local or PATH Python environment that can compile TensorRT models."""

    candidates: list[Path] = []

    configured = os.environ.get("TRT_PYTHON")
    if configured:
        candidates.append(Path(configured).expanduser())

    for environment in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        prefix = os.environ.get(environment)
        if prefix:
            candidates.append(Path(prefix) / "bin/python")

    candidates.append(Path(sys.executable))

    for parent in (Path.home() / "venvs", Path.home() / ".venvs"):
        if parent.is_dir():
            candidates.extend(parent.glob("*/bin/python"))

    for relative in (
        ".venv/engine-zoo-trt/bin/python",
        ".venv/engine-zoo-trt-py313/bin/python",
        "venv/engine-zoo-trt/bin/python",
        "venv/engine-zoo-trt-py313/bin/python",
    ):
        candidates.append(ROOT / relative)

    for executable in ("python3", "python"):
        resolved = shutil.which(executable)
        if resolved:
            candidates.append(Path(resolved))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().absolute()
        if candidate in seen or not candidate.is_file():
            continue

        seen.add(candidate)
        if _can_import_tensor_rt(candidate):
            return candidate

    return None


def _is_nonfatal_runtime_noise(line: str) -> bool:
    """Return whether *line* is a known non-actionable runtime warning."""

    if "Memory.cpp" in line and ("pin_memory" in line or "is_pinned" in line):
        return True

    return any(
        marker in line
        for marker in (
            "CUDA 13 is not currently supported for TRT-LLM plugins",
            "[TRT] [W] Functionality provided through tensorrt.plugin module is experimental",
            "Unable to import quantization op.",
            "Unable to import quantize op.",
            "quantized models",
            "modelopt library",
            "WARNING: [Torch-TensorRT] - Mean converter disregards dtype",
        )
    )


def _run_training(command: list[Path | str], environment: dict[str, str] | None) -> int:
    """Run training while hiding known nonfatal backend noise."""

    process = subprocess.Popen(
        [str(argument) for argument in command],
        cwd=ROOT,
        env=environment,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stderr is not None
    for line in process.stderr:
        if _is_nonfatal_runtime_noise(line):
            continue
        print(line, end="", file=sys.stderr)
    return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs/chess-puct-wdl-tensorrt-v2",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
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
        args.tensor_rt_python = _find_tensor_rt_python()
        if args.tensor_rt_python is None:
            parser.error(
                "could not find a Python environment with torch and torch_tensorrt; "
                "set TRT_PYTHON or pass --tensor-rt-python"
            )

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

    runtime_environment = (
        _tensor_rt_runtime_environment(args.tensor_rt_python)
        if uses_tensor_rt and args.tensor_rt_python is not None
        else None
    )
    return _run_training(command, runtime_environment)


if __name__ == "__main__":
    raise SystemExit(main())

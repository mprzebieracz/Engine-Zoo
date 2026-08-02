#!/usr/bin/env python3
"""Initialize, then continuously run the checked-in TensorRT chess experiment.

Defaults match the validated H4 Torch-TensorRT self-play screen:
112 threads, leaf batch 32, preferred batch 128, 1 ms wait, and TensorRT
compile shapes opt=128 / max=256. New runs are seeded from the v5
generation-800 archive unless --seed-checkpoint is overridden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "experiments/chess-puct-wdl-tensorrt.toml"
DEFAULT_RUN_DIR = ROOT / "runs/chess-puct-wdl-tensorrt-v6"
DEFAULT_SEED_CHECKPOINT = None
DEFAULT_OPT_BATCH = 128
DEFAULT_MAX_BATCH = 256
DEFAULT_MIN_BATCH = 1


def _libtorch_root() -> Path | None:
    """Resolve the LibTorch root from $LIBTORCH or `.cargo/config.toml`."""

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


def _can_import_tensor_rt(python: Path, *, raw: bool) -> bool:
    """Return whether *python* has the selected TensorRT compiler installed."""

    imports = "import torch, tensorrt" if raw else "import torch, torch_tensorrt"

    result = subprocess.run(
        [str(python), "-c", imports],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _tensor_rt_site_lib(python: Path) -> Path:
    """Return the Torch-TensorRT C++ library directory for *python*."""

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


def _raw_tensor_rt_runtime_environment(python: Path) -> dict[str, str]:
    """Linker environment for an experiment using the raw TensorRT backend.

    Needs libnvinfer from the TensorRT 10.x wheel and LibTorch, but must not
    preload libtorchtrt.so (that is only for Torch-TensorRT TorchScript).
    """

    environment = os.environ.copy()
    library_paths: list[str] = []

    libtorch = _libtorch_root()
    if libtorch is not None:
        libtorch_lib = libtorch / "lib"
        if libtorch_lib.is_dir():
            library_paths.append(str(libtorch_lib))

    # Prefer the pip TensorRT 10.x libs that match the vendored headers / FP16
    # builder used by scripts/compile_tensorrt_raw.py.
    query = (
        "from pathlib import Path; import tensorrt, importlib.util; "
        "spec = importlib.util.find_spec('tensorrt_libs') or "
        "importlib.util.find_spec('tensorrt'); "
        "root = Path(next(iter(spec.submodule_search_locations))); "
        "libs = root if (root / 'libnvinfer.so.10').exists() else root.parent / 'tensorrt_libs'; "
        "print(libs)"
    )
    probed = subprocess.run(
        [str(python), "-c", query],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    nvidia_lib = Path(probed.stdout.strip()) if probed.returncode == 0 else Path()
    if nvidia_lib.is_dir():
        library_paths.append(str(nvidia_lib))
    if existing := environment.get("LD_LIBRARY_PATH"):
        library_paths.append(existing)
    environment["LD_LIBRARY_PATH"] = ":".join(library_paths)
    environment.pop("LD_PRELOAD", None)
    environment["TENSORRT_PREFER_SYSTEM_BUILDER"] = "0"
    environment.pop("TENSORRT_BUILD_PYTHON", None)
    environment.setdefault(
        "PYTHONWARNINGS",
        "ignore::DeprecationWarning,ignore::UserWarning",
    )
    return environment


def _tensor_rt_runtime_environment(python: Path, *, raw: bool) -> dict[str, str]:
    if raw:
        return _raw_tensor_rt_runtime_environment(python)
    return _torch_tensor_rt_runtime_environment(python)


def _torch_tensor_rt_runtime_environment(python: Path) -> dict[str, str]:
    """Prepare the linker environment for Torch-TensorRT module loads."""

    trt_lib = _tensor_rt_site_lib(python)
    runtime_library = trt_lib / "libtorchtrt.so"
    if not runtime_library.is_file():
        raise SystemExit(f"missing Torch-TensorRT runtime library: {runtime_library}")

    environment = os.environ.copy()
    library_paths: list[str] = [str(trt_lib)]
    preload: list[str] = []

    libtorch = _libtorch_root()
    if libtorch is not None:
        libtorch_lib = libtorch / "lib"
        libtorch_so = libtorch_lib / "libtorch.so"
        if libtorch_lib.is_dir():
            library_paths.insert(0, str(libtorch_lib))
        if libtorch_so.is_file():
            preload.append(str(libtorch_so))

    nvidia_lib = trt_lib.parent.parent / "tensorrt_libs"
    if nvidia_lib.is_dir():
        library_paths.append(str(nvidia_lib))

    existing = environment.get("LD_LIBRARY_PATH")
    if existing:
        library_paths.append(existing)
    environment["LD_LIBRARY_PATH"] = ":".join(library_paths)

    preload.append(str(runtime_library))
    existing_preload = environment.get("LD_PRELOAD")
    if existing_preload:
        preload.append(existing_preload)
    environment["LD_PRELOAD"] = ":".join(preload)
    environment.setdefault(
        "PYTHONWARNINGS",
        "ignore::DeprecationWarning,ignore::UserWarning",
    )
    return environment


def _find_tensor_rt_python(*, raw: bool) -> Path | None:
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
        if _can_import_tensor_rt(candidate, raw=raw):
            return candidate

    return None


def _is_nonfatal_runtime_noise(line: str) -> bool:
    """Return whether *line* is a known non-actionable runtime warning."""

    if "Memory.cpp" in line and ("pin_memory" in line or "is_pinned" in line):
        return True

    if line.startswith("engine build time:"):
        # Redundant with train's "compiled TensorRT module in …s".
        return True

    return any(
        marker in line
        for marker in (
            "CUDA 13 is not currently supported for TRT-LLM plugins",
            "Functionality provided through tensorrt.plugin module is experimental",
            "Unable to import quantization op.",
            "Unable to import quantize op.",
            "quantized models",
            "modelopt library",
            "WARNING: [Torch-TensorRT] - Mean converter disregards dtype",
            "no signature found for builtin",
            "skipping _decide_input_format",
            "legacy TorchScript-based ONNX export",
            "You are using the legacy TorchScript-based ONNX export",
        )
    )


def _run_training(command: list[Path | str], environment: dict[str, str]) -> int:
    """Run training while hiding known nonfatal backend noise."""

    process = subprocess.Popen(
        [str(argument) for argument in command],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if _is_nonfatal_runtime_noise(line):
            continue
        print(line, end="")
    return process.wait()


def _seed_checkpoint(run_dir: Path, checkpoint: Path) -> None:
    """Install *checkpoint* as the run's latest weights before the first compile."""

    if not checkpoint.is_file():
        raise SystemExit(f"seed checkpoint not found: {checkpoint}")

    checkpoints = run_dir / "checkpoints"
    generation = checkpoints / "generation-000000.safetensors"
    latest = checkpoints / "latest.safetensors"
    checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, generation)
    shutil.copy2(generation, latest)
    digest = hashlib.sha256(generation.read_bytes()).hexdigest()

    state_path = run_dir / "state.json"
    state = {
        "format_version": 3,
        "iteration": 0,
        "model_generation": 0,
        "global_step": 0,
        "total_games_generated": 0,
        "checkpoint_identity": {
            "generation": 0,
            "relative_path": "checkpoints/generation-000000.safetensors",
            "sha256": digest,
        },
        "replay_sample_count": 0,
        "optimizer_moments_restored": False,
        "resume_kind": "weights-only",
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"seeded {generation} from {checkpoint}")


def _experiment_engine(path: Path) -> str:
    import tomllib

    with path.open("rb") as source:
        return tomllib.load(source).get("inference", {}).get("engine", "native")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--seed-checkpoint",
        type=Path,
        default=DEFAULT_SEED_CHECKPOINT,
        help="Checkpoint copied into a newly initialized run (required; historical checkpoints are not bundled)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tensor-rt-export-python",
        type=Path,
        help="Optional separate Python used for raw ONNX export",
    )
    parser.add_argument(
        "--tensor-rt-python",
        type=Path,
        help="Python executable in the matching Torch-TensorRT environment",
    )
    parser.add_argument(
        "--tensor-rt-compiler",
        type=Path,
        help="Optional compiler script; defaults to the experiment backend's compiler",
    )
    parser.add_argument(
        "--tensor-rt-min-batch-size",
        type=int,
        default=DEFAULT_MIN_BATCH,
        help=f"TensorRT dynamic min batch (default: {DEFAULT_MIN_BATCH})",
    )
    parser.add_argument(
        "--tensor-rt-opt-batch-size",
        type=int,
        default=DEFAULT_OPT_BATCH,
        help=f"TensorRT dynamic opt batch (default: {DEFAULT_OPT_BATCH})",
    )
    parser.add_argument(
        "--tensor-rt-max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH,
        help=f"TensorRT dynamic max batch (default: {DEFAULT_MAX_BATCH})",
    )
    parser.add_argument(
        "--tensor-rt-timing-cache",
        type=Path,
        help="Explicit raw TensorRT timing-cache path",
    )
    parser.add_argument("--disable-tensor-rt-timing-cache", action="store_true")
    parser.add_argument("--keep-tensor-rt-intermediates", action="store_true")
    args = parser.parse_args()

    if args.cache:
        parser.error(
            "--cache was removed; select engine = 'tensor-rt-raw' in the experiment "
            "and use --disable-tensor-rt-timing-cache for cold builds"
        )

    experiment_source = args.run_dir / "experiment.toml"
    if not experiment_source.is_file():
        experiment_source = args.experiment
    raw_backend = _experiment_engine(experiment_source) == "tensor-rt-raw"

    if args.tensor_rt_python is None:
        args.tensor_rt_python = _find_tensor_rt_python(raw=raw_backend)
        if args.tensor_rt_python is None:
            parser.error(
                "could not find a Python environment with torch and torch_tensorrt; "
                "set TRT_PYTHON or pass --tensor-rt-python"
            )

    if not 0 < args.tensor_rt_min_batch_size <= args.tensor_rt_opt_batch_size <= args.tensor_rt_max_batch_size:
        parser.error("TensorRT batch sizes must satisfy 0 < min <= opt <= max")

    environment = _tensor_rt_runtime_environment(args.tensor_rt_python, raw=raw_backend)
    build_cmd = ["cargo", "build", "--release", "-p", "engine_app", "--bin", "train"]
    if raw_backend:
        build_cmd.extend(["--features", "raw-tensorrt"])
    subprocess.run(
        build_cmd,
        cwd=ROOT,
        check=True,
        env=environment,
    )
    binary = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "release" / "train"
    if not binary.is_file():
        binary = ROOT / "target/release/train"

    experiment_path = args.run_dir / "experiment.toml"
    if not experiment_path.exists():
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
        if args.seed_checkpoint is None:
            parser.error("--seed-checkpoint is required; no historical checkpoint is bundled")
        _seed_checkpoint(args.run_dir, args.seed_checkpoint)

    backend = "tensor-rt-raw" if raw_backend else "tensor-rt-torch-script"
    print(
        "TensorRT defaults: "
        f"experiment={args.experiment} "
        f"run_dir={args.run_dir} "
        f"backend={backend}; "
        f"compile min={args.tensor_rt_min_batch_size} "
        f"opt={args.tensor_rt_opt_batch_size} "
        f"max={args.tensor_rt_max_batch_size}"
    )

    command: list[Path | str] = [
        binary,
        "run",
        "--run-dir",
        args.run_dir,
        "--device",
        args.device,
        "--forever",
        "--tensor-rt-python",
        args.tensor_rt_python,
        "--tensor-rt-min-batch-size",
        str(args.tensor_rt_min_batch_size),
        "--tensor-rt-opt-batch-size",
        str(args.tensor_rt_opt_batch_size),
        "--tensor-rt-max-batch-size",
        str(args.tensor_rt_max_batch_size),
    ]
    if args.tensor_rt_export_python is not None:
        command.extend(["--tensor-rt-export-python", args.tensor_rt_export_python])
    if args.tensor_rt_compiler is not None:
        command.extend(["--tensor-rt-compiler", args.tensor_rt_compiler])
    if args.tensor_rt_timing_cache is not None:
        command.extend(["--tensor-rt-timing-cache", args.tensor_rt_timing_cache])
    if args.disable_tensor_rt_timing_cache:
        command.append("--disable-tensor-rt-timing-cache")
    if args.keep_tensor_rt_intermediates:
        command.append("--keep-tensor-rt-intermediates")

    return _run_training(command, environment)


if __name__ == "__main__":
    raise SystemExit(main())

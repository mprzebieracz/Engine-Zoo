#!/usr/bin/env python3
"""Create machine-local CUDA/LibTorch/TensorRT configuration for engine-zoo.

This script intentionally runs without Rust or LibTorch.  It writes absolute
paths only to the ignored ``engine-zoo.local.toml`` and merges the discovered
build environment into ``.cargo/config.toml`` without disturbing other Cargo
configuration.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


MANAGED_CARGO_ENV = (
    "LIBTORCH",
    "LIBTORCH_BYPASS_VERSION_CHECK",
    "CUDA_HOME",
    "TENSORRT_ROOT",
    "TENSORRT_PYTHON",
)


def first_directory(paths: Iterable[Path]) -> Path | None:
    return next((path.resolve() for path in paths if path.is_dir()), None)


def first_executable(paths: Iterable[Path]) -> Path | None:
    return next((path.resolve() for path in paths if path.is_file() and os.access(path, os.X_OK)), None)


def environment_path(name: str, *, executable: bool = False) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value).expanduser()
    if (path.is_file() and executable) or (path.is_dir() and not executable):
        # A venv's bin/python is normally a symlink to its base interpreter.
        # Resolving it drops the venv prefix and, with it, its site-packages.
        return path.absolute() if executable else path.resolve()
    return None


def python_imports(python: Path, module: str) -> bool:
    completed = subprocess.run(
        [str(python), "-c", f"import {module}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def detect_python(variable: str, module: str, candidates: Iterable[Path]) -> Path | None:
    configured = environment_path(variable, executable=True)
    if configured:
        return configured
    for python in candidates:
        if python.is_file() and os.access(python, os.X_OK) and python_imports(python, module):
            return python.absolute()
    return None


def detect(root: Path) -> dict[str, Path | None]:
    python_candidates = (
        root / ".venv" / "cuda-torch" / "bin" / "python",
        root / ".venv" / "engine-zoo-trt" / "bin" / "python",
        root / ".venv" / "engine-zoo-trt-py313" / "bin" / "python",
        Path(sys.executable),
    )
    cuda_torch_python = detect_python("TORCH_PYTHON", "torch", python_candidates)
    tensorrt_candidates = (
        root / ".venv" / "tensorrt" / "bin" / "python",
        root / ".venv" / "engine-zoo-trt" / "bin" / "python",
        root / ".venv" / "engine-zoo-trt-py313" / "bin" / "python",
        Path(sys.executable),
    )
    tensorrt_python = detect_python("TENSORRT_PYTHON", "tensorrt", tensorrt_candidates)
    return {
        "cuda_torch_libtorch": environment_path("LIBTORCH")
        or first_directory((Path("/opt/libtorch"), Path("/usr/local/libtorch"), root / "third_party" / "libtorch")),
        "cuda_root": environment_path("CUDA_HOME")
        or environment_path("CUDA_PATH")
        or first_directory((Path("/usr/local/cuda"), Path("/opt/cuda"))),
        "tensorrt_root": environment_path("TENSORRT_ROOT")
        or first_directory((Path("/opt/tensorrt"), Path("/usr/local/tensorrt"), root / "third_party" / "tensorrt")),
        "cuda_torch_python": cuda_torch_python,
        "tensorrt_python": tensorrt_python,
    }


def explicit_path(value: str, *, option: str, executable: bool = False) -> Path:
    """Resolve a user-supplied path and reject the wrong kind of filesystem entry."""
    path = Path(value).expanduser()
    valid = path.is_file() and os.access(path, os.X_OK) if executable else path.is_dir()
    expected = "an executable file" if executable else "a directory"
    if not valid:
        raise argparse.ArgumentTypeError(f"{option} must be {expected}: {value}")
    return path.absolute() if executable else path.resolve()


def apply_overrides(
    found: dict[str, Path | None], args: argparse.Namespace
) -> dict[str, Path | None]:
    overrides = {
        "cuda_torch_libtorch": args.cuda_torch_libtorch,
        "cuda_root": args.cuda_root,
        "tensorrt_root": args.tensorrt_root,
        "cuda_torch_python": args.cuda_torch_python,
        "tensorrt_python": args.tensorrt_python,
    }
    resolved = dict(found)
    for name, value in overrides.items():
        if value is not None:
            resolved[name] = value
    return resolved


def toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def local_config(root: Path, found: dict[str, Path | None]) -> str:
    toolchain = ["[toolchain]"]
    for name in (
        "cuda_torch_libtorch",
        "cuda_root",
        "tensorrt_root",
        "cuda_torch_python",
        "tensorrt_python",
    ):
        if path := found[name]:
            toolchain.append(f"{name} = {toml_string(str(path))}")
    if len(toolchain) == 1:
        toolchain.append("# No CUDA/Torch/TensorRT installation was detected.")

    return "\n".join(
        (
            "# Generated by scripts/init_cuda_torch_tensorrt.py. Do not commit this file.",
            "[paths]",
            f"training_runs = {toml_string(str((root / 'runs').resolve()))}",
            f"permanent_models = {toml_string(str((root / 'models').resolve()))}",
            f"cuda_tensorrt_artifact_cache = {toml_string(str((root / 'artifacts' / 'cuda-tensorrt').resolve()))}",
            "",
            *toolchain,
            "",
            "[runtime]",
            'default_backend = "auto"',
            'device = "auto"',
            "fp16 = true",
            "cuda_tensorrt_batch_min = 1",
            "cuda_tensorrt_batch_optimal = 64",
            "cuda_tensorrt_batch_max = 256",
            "",
        )
    )


def replace_cargo_env(config: str, values: dict[str, str]) -> str:
    """Merge managed keys into Cargo's [env] table, retaining every other line."""
    lines = config.splitlines(keepends=True)
    env_start = next((index for index, line in enumerate(lines) if line.strip() == "[env]"), None)
    new_entries = [f'{key} = {{ value = {toml_string(value)}, force = false }}\n' for key, value in values.items()]
    if env_start is None:
        separator = "" if not config or config.endswith("\n") else "\n"
        return config + separator + "\n[env]\n" + "".join(new_entries)

    env_end = next(
        (index for index in range(env_start + 1, len(lines)) if re.match(r"^\s*\[[^[]", lines[index])),
        len(lines),
    )
    key_pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
    seen: set[str] = set()
    merged: list[str] = []
    for line in lines[env_start + 1 : env_end]:
        match = key_pattern.match(line)
        if match and match.group(1) in values:
            key = match.group(1)
            if key not in seen:
                merged.append(f'{key} = {{ value = {toml_string(values[key])}, force = false }}\n')
                seen.add(key)
            continue
        merged.append(line)
    merged.extend(entry for key, entry in zip(values, new_entries) if key not in seen)
    return "".join(lines[: env_start + 1] + merged + lines[env_end:])


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def merge_cargo_config(root: Path, found: dict[str, Path | None], dry_run: bool) -> None:
    env: dict[str, str] = {"LIBTORCH_BYPASS_VERSION_CHECK": "1"}
    mappings = {
        "cuda_torch_libtorch": "LIBTORCH",
        "cuda_root": "CUDA_HOME",
        "tensorrt_root": "TENSORRT_ROOT",
        "tensorrt_python": "TENSORRT_PYTHON",
    }
    for config_key, cargo_key in mappings.items():
        if path := found[config_key]:
            env[cargo_key] = str(path)
    cargo = root / ".cargo" / "config.toml"
    original = cargo.read_text(encoding="utf-8") if cargo.exists() else ""
    merged = replace_cargo_env(original, env)
    if dry_run:
        print(f"would merge CUDA/Torch/TensorRT environment into {cargo}")
        return
    cargo.parent.mkdir(parents=True, exist_ok=True)
    if cargo.exists():
        backup = cargo.with_name("config.toml.before-cuda-torch-tensorrt-init.bak")
        shutil.copy2(cargo, backup)
        print(f"backed up {cargo} to {backup}")
    atomic_write(cargo, merged)
    print(f"updated {cargo}")


def read_local_config(path: Path) -> dict[str, Path]:
    section = ""
    values: dict[str, Path] = {}
    assignment = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"\s*$')
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        match = assignment.match(line)
        if match and section in {"paths", "toolchain"}:
            values[f"{section}.{match.group(1)}"] = Path(match.group(2))
    return values


def doctor(root: Path) -> int:
    failures: list[str] = []
    config = root / "engine-zoo.local.toml"
    if not config.is_file():
        print(f"CUDA/Torch/TensorRT local configuration is missing: {config}", file=sys.stderr)
        return 1
    values = read_local_config(config)
    for name in (
        "toolchain.cuda_torch_libtorch",
        "toolchain.cuda_root",
        "toolchain.tensorrt_root",
        "toolchain.cuda_torch_python",
        "toolchain.tensorrt_python",
    ):
        path = values.get(name)
        status = str(path) if path else "not configured"
        print(f"{name}: {status}")
        if path is None or not path.is_absolute() or not path.exists():
            failures.append(name)
    libtorch = values.get("toolchain.cuda_torch_libtorch")
    libtorch_ready = libtorch is not None and (libtorch / "lib").is_dir()
    print(f"LibTorch libraries: {'yes' if libtorch_ready else 'no'}")
    if not libtorch_ready:
        failures.append("LibTorch libraries")
    cuda = values.get("toolchain.cuda_root")
    cuda_ready = cuda is not None and (cuda / "bin" / "nvcc").is_file()
    print(f"CUDA compiler: {'yes' if cuda_ready else 'no'}")
    if not cuda_ready:
        failures.append("CUDA compiler")
    tensorrt = values.get("toolchain.tensorrt_root")
    tensorrt_ready = tensorrt is not None and (tensorrt / "include").is_dir() and any(
        (tensorrt / library).is_dir() for library in ("lib", "lib64")
    )
    print(f"TensorRT headers and libraries: {'yes' if tensorrt_ready else 'no'}")
    if not tensorrt_ready:
        failures.append("TensorRT headers and libraries")
    for name, module in (("toolchain.cuda_torch_python", "torch"), ("toolchain.tensorrt_python", "tensorrt")):
        python = values.get(name)
        ready = python is not None and python.is_file() and os.access(python, os.X_OK) and python_imports(python, module)
        print(f"{module} Python environment: {'yes' if ready else 'no'}")
        if not ready:
            failures.append(f"{module} Python environment")
    for name in ("paths.permanent_models", "paths.cuda_tensorrt_artifact_cache"):
        directory = values.get(name)
        writable = directory is not None and directory.is_absolute() and os.access(directory, os.W_OK)
        print(f"writable {directory}: {'yes' if writable else 'no'}")
        if not writable:
            failures.append(name)
    for script in ("compile_tensorrt.py", "compile_tensorrt_raw.py"):
        path = root / "scripts" / script
        available = path.is_file()
        print(f"compiler script {path}: {'yes' if available else 'no'}")
        if not available:
            failures.append(script)
    nvidia_smi = first_executable((Path("/usr/bin/nvidia-smi"), Path("/bin/nvidia-smi")))
    if nvidia_smi:
        gpu = subprocess.run([str(nvidia_smi), "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False)
        print(f"GPU: {gpu.stdout.strip() if gpu.returncode == 0 else 'query failed'}")
    else:
        print("GPU: nvidia-smi not found")
        failures.append("GPU")
    if failures:
        print("CUDA/Torch/TensorRT doctor failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("CUDA/Torch/TensorRT doctor passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dry-run", action="store_true", help="show intended changes without writing files")
    parser.add_argument("--doctor", action="store_true", help="validate CUDA/Torch/TensorRT paths and GPU access")
    parser.add_argument(
        "--cuda-torch-libtorch",
        type=lambda value: explicit_path(value, option="--cuda-torch-libtorch"),
        help="LibTorch root directory; overrides LIBTORCH and auto-detection",
    )
    parser.add_argument(
        "--cuda-root",
        type=lambda value: explicit_path(value, option="--cuda-root"),
        help="CUDA root directory; overrides CUDA_HOME/CUDA_PATH and auto-detection",
    )
    parser.add_argument(
        "--tensorrt-root",
        type=lambda value: explicit_path(value, option="--tensorrt-root"),
        help="TensorRT root directory; overrides TENSORRT_ROOT and auto-detection",
    )
    parser.add_argument(
        "--cuda-torch-python",
        type=lambda value: explicit_path(value, option="--cuda-torch-python", executable=True),
        help="Python executable with torch; overrides TORCH_PYTHON and auto-detection",
    )
    parser.add_argument(
        "--tensorrt-python",
        type=lambda value: explicit_path(value, option="--tensorrt-python", executable=True),
        help="Python executable with tensorrt; overrides TENSORRT_PYTHON and auto-detection",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    found = apply_overrides(detect(root), args)
    if args.doctor:
        return doctor(root)
    config = root / "engine-zoo.local.toml"
    if args.dry_run:
        print(f"would write {config}")
    else:
        for directory in (root / "models", root / "artifacts" / "cuda-tensorrt"):
            directory.mkdir(parents=True, exist_ok=True)
        atomic_write(config, local_config(root, found))
        print(f"wrote {config}")
    merge_cargo_config(root, found, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

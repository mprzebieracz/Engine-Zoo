#!/usr/bin/env python3
"""Run the configured multi-opponent checkpoint arena.

Models declared with ``backend = "tensor-rt"`` are compiled into a per-model
Torch-TensorRT module in ``settings.trt_cache_dir`` before the arena launches.
The cache is keyed on the safetensors fingerprint plus the ModelSpec's channel
count so the same compiled artifact is reused across arena runs. Models with
``backend = "native"`` load through tch/LibTorch directly and require no extra
setup.

The wrapper writes a resolved copy of the arena config to a temporary file so
that the Rust binary sees ``tensor_rt_module`` paths filled in for every
tensor-rt model without mutating the checked-in configuration.
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
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/arena.toml"
DEFAULT_TRT_CACHE = ROOT / "data/evaluations/trt-cache"
DEFAULT_TRT_MIN = 1
DEFAULT_TRT_OPT = 32
DEFAULT_TRT_MAX = 256


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

    configured = os.environ.get("TRT_PYTHON")
    if configured:
        candidates.append(Path(configured).expanduser())

    candidates.append(Path.home() / "venvs/engine-zoo-trt-py313/bin/python")
    candidates.append(Path("/tmp/engine-zoo-arena/venv/bin/python"))
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


def _tensor_rt_runtime_environment(python: Path) -> dict[str, str]:
    """Prepare the linker environment needed to load a TensorRT module."""

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
    environment.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")
    return environment


def _resolve_path(value: str | os.PathLike[str], root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def _load_config(config_path: Path) -> tuple[dict, Path]:
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.resolve().parent
    return raw, root


def _infer_run_dir(checkpoint: Path) -> Path:
    parent = checkpoint.parent
    if parent.name == "checkpoints":
        return parent.parent
    return parent


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "model"


def _fingerprint(checkpoint: Path) -> str:
    stat = checkpoint.stat()
    return f"{stat.st_size}-{stat.st_mtime_ns}"


def _tensor_rt_runtime_identity(python: Path) -> dict[str, object]:
    """Return the runtime properties that make TensorRT engines incompatible."""

    query = (
        "import json, torch, torch_tensorrt; "
        "print(json.dumps({"
        "'torch': torch.__version__, "
        "'torch_tensorrt': torch_tensorrt.__version__, "
        "'cuda': torch.version.cuda, "
        "'device_capability': torch.cuda.get_device_capability()"
        "}, sort_keys=True))"
    )
    result = subprocess.run(
        [str(python), "-c", query],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _tensor_rt_compile_profile(
    experiment: Path,
    compiler_script: Path,
    runtime: dict[str, object],
    override_batch_shapes: tuple[int, int, int] | None,
) -> dict[str, object]:
    """Derive an engine profile from the checkpoint's immutable experiment."""

    with experiment.open("rb") as handle:
        inference = tomllib.load(handle).get("inference", {})
    default_min, default_opt, default_max = 1, 32, 256
    inferred_shapes = (
        default_min,
        int(inference.get("preferred_batch_size", default_opt)),
        int(inference.get("max_batch_size", default_max)),
    )
    if not 0 < inferred_shapes[0] <= inferred_shapes[1] <= inferred_shapes[2]:
        raise SystemExit(
            f"invalid inference batch profile in {experiment}: {inferred_shapes}"
        )
    if override_batch_shapes is not None:
        if override_batch_shapes != inferred_shapes:
            print(
                "arena: WARNING using explicit TensorRT batch override "
                f"{override_batch_shapes} instead of {inferred_shapes} from "
                f"{experiment}; this can reduce inference throughput"
            )
        batch_shapes = override_batch_shapes
    else:
        batch_shapes = inferred_shapes

    precision = "fp16" if inference.get("precision") == "fp16" else "fp32-fp16"
    with compiler_script.open("rb") as handle:
        compiler_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()

    return {
        "batch_shapes": batch_shapes,
        "precision": precision,
        "runtime": runtime,
        "compiler_sha256": compiler_sha256,
    }


def _profile_key(profile: dict[str, object]) -> str:
    encoded = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _channels(model_cli: Path, experiment: Path, env: dict[str, str]) -> int:
    result = subprocess.run(
        [str(model_cli), "channels", "--experiment", str(experiment)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def _ensure_experiment(model_cli: Path, run_dir: Path, env: dict[str, str]) -> Path:
    """Return ``experiment.toml``, migrating legacy run configs when needed."""

    experiment = run_dir / "experiment.toml"
    if experiment.is_file():
        return experiment

    result = subprocess.run(
        [str(model_cli), "ensure-experiment", "--run-dir", str(run_dir)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = Path(result.stdout.strip())
    if not resolved.is_file():
        raise SystemExit(f"failed to materialize experiment.toml under {run_dir}")
    print(f"arena: migrated experiment config to {resolved}")
    return resolved


def _prepare_tensor_rt_module(
    *,
    name: str,
    checkpoint: Path,
    experiment: Path,
    channels: int,
    cache_dir: Path,
    trt_python: Path,
    compiler_script: Path,
    model_cli: Path,
    profile: dict[str, object],
    device: str,
    env: dict[str, str],
    force_recompile: bool,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(checkpoint)
    batch_min, batch_opt, batch_max = profile["batch_shapes"]
    precision = profile["precision"]
    profile_key = _profile_key(profile)
    module_path = cache_dir / (
        f"{_sanitize_name(name)}-{fingerprint}-c{channels}-p{profile_key}.trt.ts"
    )
    if module_path.is_file() and not force_recompile:
        print(f"arena: reusing cached TensorRT module {module_path}")
        return module_path

    export_path = cache_dir / (module_path.name + ".torchscript")
    print(f"arena: exporting {checkpoint.name} to {export_path}")
    subprocess.run(
        [
            str(model_cli),
            "export-torch-script",
            "--experiment",
            str(experiment),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(export_path),
            "--device",
            "cuda" if device.startswith("cuda") else device,
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

    print(f"arena: compiling TensorRT module {module_path}")
    temporary = cache_dir / (module_path.name + ".tmp")
    try:
        # The Python compiler must load its own Torch-TensorRT wheel, not the
        # linker overrides the parent uses to open compiled modules from Rust.
        compile_env = env.copy()
        compile_env.pop("LD_PRELOAD", None)
        compile_env.pop("LD_LIBRARY_PATH", None)

        subprocess.run(
            [
                str(trt_python),
                str(compiler_script),
                "--input",
                str(export_path),
                "--output",
                str(temporary),
                "--channels",
                str(channels),
                "--min-batch-size",
                str(batch_min),
                "--opt-batch-size",
                str(batch_opt),
                "--max-batch-size",
                str(batch_max),
                "--precision",
                str(precision),
            ],
            cwd=ROOT,
            env=compile_env,
            check=True,
        )
        temporary.replace(module_path)
    finally:
        if temporary.exists():
            temporary.unlink()
        if export_path.exists():
            export_path.unlink()
    return module_path


def _dump_toml(data: dict) -> str:
    """Serialize the resolved config back into TOML.

    Kept intentionally small: we know the shape of the arena file and only
    write scalars/tables/inline arrays of tables, so a tiny writer avoids
    adding a runtime dependency on ``tomli-w``.
    """

    lines: list[str] = []

    def _emit_scalar(value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, float):
            return repr(value)
        if isinstance(value, (str, os.PathLike)):
            escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        raise TypeError(f"unsupported TOML scalar: {value!r}")

    def _emit_table(name: str, table: dict) -> None:
        lines.append(f"[{name}]")
        for key, value in table.items():
            lines.append(f"{key} = {_emit_scalar(value)}")
        lines.append("")

    def _emit_array_of_tables(name: str, tables: list[dict]) -> None:
        for table in tables:
            lines.append(f"[[{name}]]")
            for key, value in table.items():
                lines.append(f"{key} = {_emit_scalar(value)}")
            lines.append("")

    if "candidate" in data:
        _emit_table("candidate", data["candidate"])
    if "opponents" in data:
        _emit_array_of_tables("opponents", data["opponents"])
    if "settings" in data:
        _emit_table("settings", data["settings"])
    return "\n".join(lines) + "\n"


def _absolutize_paths(model: dict, root: Path, path_keys: tuple[str, ...]) -> None:
    for key in path_keys:
        if key in model and model[key]:
            model[key] = str(_resolve_path(model[key], root).resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true", help="use debug binaries")
    parser.add_argument(
        "--tensor-rt-python",
        type=Path,
        help="Python executable in the matching Torch-TensorRT environment",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="do not rebuild the eval-big-arena / UCI / model binaries",
    )
    parser.add_argument(
        "--force-tensor-rt-recompile",
        action="store_true",
        help="rebuild TensorRT modules even when a matching cached artifact exists",
    )
    args, extra = parser.parse_known_args()
    # Callers often write `arena_big.py -- --games 8`. argparse leaves the
    # bare `--` in the unknown-arg list; clap then treats later flags as
    # positionals and rejects them.
    if extra and extra[0] == "--":
        extra = extra[1:]

    config_path = args.config.resolve()
    if not config_path.is_file():
        parser.error(f"arena config not found: {config_path}")

    raw, config_root = _load_config(config_path)
    settings = raw.setdefault("settings", {})
    profile = "debug" if args.debug else "release"

    def _default_binary(name: str) -> Path:
        target_dir = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
        return target_dir / profile / name

    def _resolve_setting(key: str, default: Path) -> Path:
        value = settings.get(key)
        return _resolve_path(value, config_root) if value else default

    model_cli = _resolve_setting("model_cli", _default_binary("engine-zoo-model")).resolve()
    uci = _resolve_setting("uci", _default_binary("engine-zoo-uci")).resolve()
    fastchess = _resolve_setting(
        "fastchess", ROOT / "crates/evaluations/bin/fastchess/fastchess"
    ).resolve()
    cache_dir = _resolve_setting("trt_cache_dir", DEFAULT_TRT_CACHE).resolve()
    compiler_script = _resolve_setting(
        "tensor_rt_compiler", ROOT / "scripts/compile_tensorrt.py"
    ).resolve()
    # Always write absolute binary paths into the resolved config. Relative
    # defaults like `target/release/engine-zoo-uci` get resolved against
    # Fastchess' working directory, which is not the repo root.
    settings["model_cli"] = str(model_cli)
    settings["uci"] = str(uci)
    settings["fastchess"] = str(fastchess)
    settings["trt_cache_dir"] = str(cache_dir)
    settings["tensor_rt_compiler"] = str(compiler_script)
    override_batch_shapes: tuple[int, int, int] | None = None
    if settings.get("trt_override_batch_sizes", False):
        override_batch_shapes = (
            int(settings.get("trt_min_batch_size", DEFAULT_TRT_MIN)),
            int(settings.get("trt_opt_batch_size", DEFAULT_TRT_OPT)),
            int(settings.get("trt_max_batch_size", DEFAULT_TRT_MAX)),
        )
        if not (
            0
            < override_batch_shapes[0]
            <= override_batch_shapes[1]
            <= override_batch_shapes[2]
        ):
            parser.error("TensorRT batch sizes must satisfy 0 < min <= opt <= max")

    models: list[dict] = []
    if "candidate" in raw:
        models.append(raw["candidate"])
    for opponent in raw.get("opponents", []):
        models.append(opponent)

    needs_tensor_rt = any(
        model.get("backend") == "tensor-rt" and not model.get("tensor_rt_module")
        for model in models
    )

    trt_python: Path | None = None
    trt_env: dict[str, str] = os.environ.copy()

    if needs_tensor_rt:
        trt_python = _find_tensor_rt_python(args.tensor_rt_python)
        if trt_python is None:
            parser.error(
                "could not find a Python environment with torch and torch_tensorrt; "
                "set TRT_PYTHON or pass --tensor-rt-python"
            )
        trt_env = _tensor_rt_runtime_environment(trt_python)
        trt_runtime = _tensor_rt_runtime_identity(trt_python)
    else:
        trt_runtime = {}

    if not args.skip_build:
        subprocess.run(
            [
                "cargo",
                "build",
                *(["--release"] if profile == "release" else []),
                "-p",
                "checkpoint-eval",
                "--bin",
                "eval-big-arena",
                "-p",
                "engine_app",
                "--bin",
                "engine-zoo-uci",
                "-p",
                "engine_app",
                "--bin",
                "engine-zoo-model",
            ],
            cwd=ROOT,
            check=True,
        )

    device = str(settings.get("device", "cuda"))
    for model in models:
        _absolutize_paths(model, config_root, ("path", "tensor_rt_module"))
        if model.get("backend") != "tensor-rt":
            continue
        if model.get("tensor_rt_module") and Path(model["tensor_rt_module"]).is_file():
            continue

        checkpoint = Path(model["path"])
        if not checkpoint.is_file():
            parser.error(f"checkpoint not found for {model.get('name')}: {checkpoint}")
        run_dir = _infer_run_dir(checkpoint)
        experiment = _ensure_experiment(model_cli, run_dir, trt_env)

        channels = _channels(model_cli, experiment, trt_env)
        profile = _tensor_rt_compile_profile(
            experiment,
            compiler_script,
            trt_runtime,
            override_batch_shapes,
        )
        module_path = _prepare_tensor_rt_module(
            name=str(model.get("name", "model")),
            checkpoint=checkpoint,
            experiment=experiment,
            channels=channels,
            cache_dir=cache_dir,
            trt_python=trt_python,  # type: ignore[arg-type]
            compiler_script=compiler_script,
            model_cli=model_cli,
            profile=profile,
            device=str(model.get("device", device)),
            env=trt_env,
            force_recompile=args.force_tensor_rt_recompile,
        )
        model["tensor_rt_module"] = str(module_path)

    # Absolutize a few settings paths so the resolved config no longer depends
    # on its original directory.
    settings_paths = (
        "fastchess",
        "uci",
        "output_dir",
        "opening_file",
        "trt_cache_dir",
        "tensor_rt_compiler",
        "model_cli",
    )
    for key in settings_paths:
        value = settings.get(key)
        if not value:
            continue
        # Bare executable names (no directory component) should keep resolving
        # through PATH.
        candidate = Path(value)
        if candidate.parent == Path("") or candidate.parent == Path("."):
            continue
        settings[key] = str(_resolve_path(candidate, config_root).resolve())

    with tempfile.NamedTemporaryFile(
        prefix="arena-resolved-", suffix=".toml", mode="w", delete=False
    ) as handle:
        handle.write(_dump_toml(raw))
        resolved_config = Path(handle.name)

    try:
        eval_binary = _default_binary("eval-big-arena")
        command = [str(eval_binary), "--config", str(resolved_config), *extra]
        result = subprocess.run(command, cwd=ROOT, env=trt_env)
        if result.returncode != 0:
            return result.returncode
    finally:
        try:
            resolved_config.unlink()
        except OSError:
            pass

    output_dir_setting = settings.get("output_dir")
    output_dir = (
        _resolve_path(output_dir_setting, config_root).resolve()
        if output_dir_setting
        else ROOT / "data/evaluations/arena"
    )
    print(f"arena report: {output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

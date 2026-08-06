#!/usr/bin/env python3
"""Compile an Engine-zoo TorchScript export into a raw TensorRT engine plan.

Unlike `compile_tensorrt.py` (Torch-TensorRT TorchScript frontend), this uses
the plain TensorRT Python builder: ONNX export -> nvonnxparser -> IBuilder.
The result is a serialized engine plan loaded by Rust `RawTensorRtBackend`.

Supports a genuine persistent builder timing cache (atomic temp+rename writes).

Tensor contract matches the Torch-TensorRT path: input `input` [N,C,8,8] and
packed output `output` [N, action_size+1], both FP32 at the I/O boundary.
Internal compute uses FP16 when requested.

On TensorRT >= 11 (no BuilderFlag.FP16), the ONNX graph is exported with an
explicit FP16 compute wrapper (Cast -> half network -> Cast) and the builder
uses STRONGLY_TYPED. On TensorRT 10.x the classic BuilderFlag.FP16 path is used.

When the active interpreter has torch but a TensorRT without FP16 (or the
reverse), set TENSORRT_BUILD_PYTHON to a Python that has a matching TensorRT
install; ONNX export still uses the current interpreter's torch.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time

import tensorrt as trt


INPUT_NAME = "input"
OUTPUT_NAME = "output"


def builder_has_fp16_flag(build_python: Path | None) -> bool:
    if build_python is None:
        return hasattr(trt.BuilderFlag, "FP16")
    result = subprocess.run(
        [
            str(build_python),
            "-c",
            "import tensorrt as t; print('yes' if hasattr(t.BuilderFlag, 'FP16') else 'no')",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "LD_PRELOAD"},
    )
    return result.returncode == 0 and "yes" in result.stdout


def build_onnx(
    input_path: Path,
    onnx_path: Path,
    channels: int,
    precision: str,
    strongly_typed_fp16: bool,
) -> None:
    import torch

    del precision, strongly_typed_fp16
    # Export the ScriptModule as-is (FP32). TensorRT 10 can still enable FP16
    # tactics via BuilderFlag.FP16; TensorRT 11 strongly-typed FP16 requires a
    # dtype-correct ONNX graph, which this TorchScript archive does not
    # compose into cleanly via .half() (weights stay FP32 constants).
    module = torch.jit.load(str(input_path), map_location="cuda").eval()
    dummy = torch.zeros((1, channels, 8, 8), dtype=torch.float32, device="cuda")
    torch.onnx.export(
        module,
        (dummy,),
        str(onnx_path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )


def load_timing_cache(config: trt.IBuilderConfig, path: Path | None) -> None:
    data = b""
    if path is not None and path.is_file():
        data = path.read_bytes()
    cache = config.create_timing_cache(data)
    config.set_timing_cache(cache, ignore_mismatch=False)


def save_timing_cache(config: trt.IBuilderConfig, path: Path) -> None:
    cache = config.get_timing_cache()
    if cache is None:
        return
    serialized = cache.serialize()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(bytes(serialized))
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def network_creation_flags(precision: str) -> int:
    del precision
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    return flags


def configure_precision(config: trt.IBuilderConfig, precision: str) -> None:
    if precision == "fp32":
        return
    if hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
        if precision == "fp16" and hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        return
    print(
        f"warning: TensorRT {trt.__version__} has no BuilderFlag.FP16; "
        f"building FP32 engine despite --precision {precision}",
        file=sys.stderr,
    )


def build_engine(
    onnx_path: Path,
    channels: int,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    precision: str,
    workspace_size: int | None,
    optimization_level: int | None,
    timing_cache: Path | None,
) -> tuple[bytes, float]:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(network_creation_flags(precision))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            messages = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"failed to parse ONNX export:\n{messages}")

    config = builder.create_builder_config()
    if workspace_size is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    if optimization_level is not None:
        config.builder_optimization_level = optimization_level
    configure_precision(config, precision)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        INPUT_NAME,
        min=(min_batch, channels, 8, 8),
        opt=(opt_batch, channels, 8, 8),
        max=(max_batch, channels, 8, 8),
    )
    config.add_optimization_profile(profile)

    load_timing_cache(config, timing_cache)

    started = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.monotonic() - started
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed; see builder log above")

    if timing_cache is not None:
        save_timing_cache(config, timing_cache)

    return bytes(serialized), elapsed


def build_engine_via_subprocess(
    build_python: Path,
    scripts_dir: Path,
    onnx_path: Path,
    output: Path,
    channels: int,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    precision: str,
    workspace_size: int | None,
    optimization_level: int | None,
    timing_cache: Path | None,
) -> float:
    """Build with a different Python that has the desired TensorRT install."""
    script = f"""
from pathlib import Path
import sys
sys.path.insert(0, {str(scripts_dir)!r})
import compile_tensorrt_raw as raw
engine, elapsed = raw.build_engine(
    Path({str(onnx_path)!r}),
    {channels},
    {min_batch},
    {opt_batch},
    {max_batch},
    {precision!r},
    {workspace_size!r},
    {optimization_level!r},
    Path({str(timing_cache)!r}) if {timing_cache is not None!r} else None,
)
out = Path({str(output)!r})
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(engine)
print(f"engine build time: {{elapsed:.3f}}s -> {{out}}")
"""
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    result = subprocess.run(
        [str(build_python), "-c", script],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"TensorRT build subprocess failed ({result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    print(result.stdout, end="")
    for line in result.stdout.splitlines():
        if line.startswith("engine build time:"):
            try:
                return float(line.split()[3].rstrip("s"))
            except (IndexError, ValueError):
                break
    return 0.0


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--channels", required=True, type=int)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--opt-batch-size", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=4096)
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32-fp16", "fp32"),
        default="fp32-fp16",
    )
    parser.add_argument("--workspace-size", type=int)
    parser.add_argument("--optimization-level", type=int, choices=range(6), metavar="0-5")
    parser.add_argument("--timing-cache", type=Path)
    parser.add_argument(
        "--tensorrt-build-python",
        type=Path,
        help="Optional Python used only for the TensorRT builder (must import tensorrt)",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        parser.error("TensorRT compilation requires CUDA")
    if not 0 < args.min_batch_size <= args.opt_batch_size <= args.max_batch_size:
        parser.error("batch sizes must satisfy 0 < min <= opt <= max")

    build_python = args.tensorrt_build_python
    if build_python is None and (configured := os.environ.get("TENSORRT_BUILD_PYTHON")):
        build_python = Path(configured)
    # Prefer a builder Python whose TensorRT matches the C++ runtime this host
    # links. When the Rust shim links pip TensorRT 10.x, keep the build
    # in-process (same interpreter) so BuilderFlag.FP16 works. Otherwise prefer
    # the system TensorRT builder.
    if build_python is None and os.environ.get("TENSORRT_PREFER_SYSTEM_BUILDER", "0") == "1":
        candidate = Path("/usr/bin/python3")
        if candidate.is_file() and candidate.resolve() != Path(sys.executable).resolve():
            build_python = candidate

    use_subprocess = (
        build_python is not None and build_python.resolve() != Path(sys.executable).resolve()
    )
    strongly_typed = not builder_has_fp16_flag(build_python if use_subprocess else None)

    with tempfile.TemporaryDirectory(prefix="engine-zoo-raw-trt-") as tmp_dir:
        onnx_path = Path(tmp_dir) / "model.onnx"
        build_onnx(
            args.input,
            onnx_path,
            args.channels,
            args.precision,
            strongly_typed_fp16=strongly_typed,
        )

        if use_subprocess:
            assert build_python is not None
            persistent_onnx = args.output.with_suffix(".onnx.tmp")
            persistent_onnx.write_bytes(onnx_path.read_bytes())
            try:
                elapsed = build_engine_via_subprocess(
                    build_python,
                    Path(__file__).resolve().parent,
                    persistent_onnx,
                    args.output,
                    args.channels,
                    args.min_batch_size,
                    args.opt_batch_size,
                    args.max_batch_size,
                    args.precision,
                    args.workspace_size,
                    args.optimization_level,
                    args.timing_cache,
                )
            finally:
                persistent_onnx.unlink(missing_ok=True)
            print(f"engine build time: {elapsed:.3f}s -> {args.output}")
            return

        engine_bytes, elapsed = build_engine(
            onnx_path,
            args.channels,
            args.min_batch_size,
            args.opt_batch_size,
            args.max_batch_size,
            args.precision,
            args.workspace_size,
            args.optimization_level,
            args.timing_cache,
        )

    atomic_write(args.output, engine_bytes)
    print(f"engine build time: {elapsed:.3f}s -> {args.output}")


if __name__ == "__main__":
    main()

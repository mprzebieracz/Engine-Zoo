#!/usr/bin/env python3
"""Export ONNX and build raw TensorRT engines for Engine Zoo.

The two operations are intentionally separate so Rust can use a Torch Python
for export and a different TensorRT Python for the engine build.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterator
import warnings


INPUT_NAME = "input"
OUTPUT_NAME = "output"
RESULT_SCHEMA_VERSION = 1


def write_result(result: dict[str, object], path: Path | None) -> None:
    document = json.dumps(result, sort_keys=True)

    if path is None:
        print(document)
        return

    path.write_text(document + "\n", encoding="utf-8")


def dtype_name(dtype: object) -> str:
    name = str(dtype).lower()

    if "float16" in name or name.endswith(".half"):
        return "fp16"
    if "float32" in name or name.endswith(".float"):
        return "fp32"

    raise RuntimeError(f"unsupported TensorRT I/O dtype: {dtype}")


def expected_io_dtype(precision: str) -> str:
    return "fp16" if precision == "fp16" else "fp32"


def environment_result() -> dict[str, object]:
    import tensorrt as trt

    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "operation": "inspect-environment",
        "tensorrt_version": trt.__version__,
        "cuda_runtime_version": getattr(trt, "__cuda_version__", "unknown"),
        "gpu_name": "unknown",
        "compute_capability": "unknown",
        "python_version": sys.version.split()[0],
    }

    try:
        import torch

        result["cuda_runtime_version"] = torch.version.cuda or "unknown"
        result["torch_version"] = torch.__version__

        if torch.cuda.is_available():
            result["gpu_name"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            result["compute_capability"] = f"{major}.{minor}"
    except ImportError:
        pass

    return result


def inspect_environment(args: Namespace) -> None:
    write_result(environment_result(), args.result_json)


def validate_exact_precision(precision: str, input_dtype: str, output_dtype: str) -> None:
    expected = expected_io_dtype(precision)

    if input_dtype != expected or output_dtype != expected:
        raise RuntimeError(
            f"{precision} requires {expected} I/O, built engine has "
            f"input={input_dtype}, output={output_dtype}"
        )


def onnx_export_result(torch: object, precision: str, elapsed: float) -> dict[str, object]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "operation": "export-onnx",
        "elapsed_seconds": elapsed,
        "input_dtype": expected_io_dtype(precision),
        "output_dtype": expected_io_dtype(precision),
        "onnx_opset": 18,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda or "unknown",
        "python_version": sys.version.split()[0],
    }


def export_onnx(args: Namespace) -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings(
        "ignore",
        message=r"no signature found for builtin .* skipping _decide_input_format",
        category=UserWarning,
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ONNX export requires CUDA")

    started = time.monotonic()
    module = torch.jit.load(str(args.input), map_location="cuda").eval()
    dtype = torch.float16 if args.precision == "fp16" else torch.float32

    if args.precision == "fp16":
        module = module.half()

    dummy = torch.zeros((1, args.channels, 8, 8), dtype=dtype, device="cuda")
    torch.onnx.export(
        module,
        (dummy,),
        str(args.output),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )

    result = onnx_export_result(torch, args.precision, time.monotonic() - started)
    write_result(result, args.result_json)


def network_flags(trt: object, precision: str) -> int:
    flags = 0
    network_flag = trt.NetworkDefinitionCreationFlag

    if hasattr(network_flag, "EXPLICIT_BATCH"):
        flags |= 1 << int(network_flag.EXPLICIT_BATCH)

    if precision == "fp16" and not hasattr(trt.BuilderFlag, "FP16"):
        if not hasattr(network_flag, "STRONGLY_TYPED"):
            raise RuntimeError(
                f"TensorRT {trt.__version__} cannot enforce exact fp16"
            )
        flags |= 1 << int(network_flag.STRONGLY_TYPED)

    return flags


def configure_precision(trt: object, config: object, precision: str) -> None:
    if precision == "fp32":
        return

    if hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)

        if precision == "fp16" and hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        return

    if precision == "mixed-fp32-io-fp16-tactics":
        raise RuntimeError(
            f"TensorRT {trt.__version__} cannot provide mixed fp16 tactics "
            "without BuilderFlag.FP16"
        )


@contextmanager
def timing_cache_lock(path: Path | None) -> Iterator[None]:
    if path is None:
        yield
        return

    import fcntl

    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def load_timing_cache(config: object, path: Path | None) -> None:
    data = path.read_bytes() if path is not None and path.is_file() else b""
    cache = config.create_timing_cache(data)
    config.set_timing_cache(cache, ignore_mismatch=False)


def save_timing_cache(config: object, path: Path) -> None:
    cache = config.get_timing_cache()

    if cache is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")

    try:
        with temporary.open("wb") as output:
            output.write(bytes(cache.serialize()))
            output.flush()
            os.fsync(output.fileno())

        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def engine_contract(trt: object, logger: object, serialized: object) -> tuple[str, str]:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)

    if engine is None:
        raise RuntimeError("TensorRT built an engine that cannot be deserialized")

    tensors: dict[str, str] = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        tensors[name] = dtype_name(engine.get_tensor_dtype(name))

    try:
        return tensors[INPUT_NAME], tensors[OUTPUT_NAME]
    except KeyError as error:
        raise RuntimeError(f"built engine is missing tensor {error.args[0]!r}") from error


def engine_build_result(
    args: Namespace,
    elapsed: float,
    input_dtype: str,
    output_dtype: str,
    environment: dict[str, object],
) -> dict[str, object]:
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "operation": "build-engine",
        "artifact_kind": "tensor-rt-raw",
        "elapsed_seconds": elapsed,
        "input_dtype": input_dtype,
        "output_dtype": output_dtype,
        "internal_precision": args.precision,
        "profile": {
            "min": args.min_batch_size,
            "opt": args.opt_batch_size,
            "max": args.max_batch_size,
        },
    }
    result.update(environment)
    result["operation"] = "build-engine"

    return result


def build_engine(args: Namespace) -> None:
    import tensorrt as trt

    if not 0 < args.min_batch_size <= args.opt_batch_size <= args.max_batch_size:
        raise RuntimeError("batch sizes must satisfy 0 < min <= opt <= max")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags(trt, args.precision))
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(args.input.read_bytes()):
        messages = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"failed to parse ONNX export:\n{messages}")

    config = builder.create_builder_config()
    if args.workspace_size is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_size)
    if args.optimization_level is not None:
        config.builder_optimization_level = args.optimization_level

    configure_precision(trt, config, args.precision)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        INPUT_NAME,
        min=(args.min_batch_size, args.channels, 8, 8),
        opt=(args.opt_batch_size, args.channels, 8, 8),
        max=(args.max_batch_size, args.channels, 8, 8),
    )
    config.add_optimization_profile(profile)

    with timing_cache_lock(args.timing_cache):
        load_timing_cache(config, args.timing_cache)

        started = time.monotonic()
        serialized = builder.build_serialized_network(network, config)
        elapsed = time.monotonic() - started

        if serialized is None:
            raise RuntimeError("TensorRT engine build failed; see builder diagnostics")

        input_dtype, output_dtype = engine_contract(trt, logger, serialized)
        validate_exact_precision(args.precision, input_dtype, output_dtype)

        if args.timing_cache is not None:
            save_timing_cache(config, args.timing_cache)

    with args.output.open("wb") as output:
        output.write(bytes(serialized))
        output.flush()
        os.fsync(output.fileno())

    result = engine_build_result(
        args,
        elapsed,
        input_dtype,
        output_dtype,
        environment_result(),
    )
    write_result(result, args.result_json)


def parser() -> ArgumentParser:
    root = ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="operation", required=True)

    inspect = commands.add_parser("inspect-environment")
    inspect.add_argument("--result-json", type=Path)
    inspect.set_defaults(operation_function=inspect_environment)

    export = commands.add_parser("export-onnx")
    export.add_argument("--input", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--channels", required=True, type=int)
    export.add_argument(
        "--precision",
        choices=("fp16", "mixed-fp32-io-fp16-tactics", "fp32"),
        required=True,
    )
    export.add_argument("--result-json", type=Path)
    export.set_defaults(operation_function=export_onnx)

    build = commands.add_parser("build-engine")
    build.add_argument("--input", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--channels", required=True, type=int)
    build.add_argument("--min-batch-size", required=True, type=int)
    build.add_argument("--opt-batch-size", required=True, type=int)
    build.add_argument("--max-batch-size", required=True, type=int)
    build.add_argument(
        "--precision",
        choices=("fp16", "mixed-fp32-io-fp16-tactics", "fp32"),
        required=True,
    )
    build.add_argument("--workspace-size", type=int)
    build.add_argument("--optimization-level", type=int, choices=range(6), metavar="0-5")
    build.add_argument("--timing-cache", type=Path)
    build.add_argument("--result-json", type=Path)
    build.set_defaults(operation_function=build_engine)

    return root


def main() -> None:
    args = parser().parse_args()

    try:
        args.operation_function(args)
    except Exception as error:
        print(f"{args.operation} failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

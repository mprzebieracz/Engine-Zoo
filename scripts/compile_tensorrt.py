#!/usr/bin/env python3
"""Compile an Engine Zoo TorchScript export into a TensorRT CModule."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import time
import warnings


RESULT_SCHEMA_VERSION = 1


def write_result(result: dict[str, object], path: Path | None) -> None:
    document = json.dumps(result, sort_keys=True)

    if path is None:
        print(document)
    else:
        path.write_text(document + "\n", encoding="utf-8")


def dtype_name(dtype: object) -> str:
    name = str(dtype).lower()

    if "float16" in name:
        return "fp16"
    if "float32" in name:
        return "fp32"

    raise RuntimeError(f"unsupported Torch-TensorRT I/O dtype: {dtype}")


def environment_result(torch: object, torch_tensorrt: object) -> dict[str, object]:
    import tensorrt

    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "operation": "inspect-environment",
        "tensorrt_version": tensorrt.__version__,
        "cuda_runtime_version": torch.version.cuda or "unknown",
        "gpu_name": "unknown",
        "compute_capability": "unknown",
        "torch_version": torch.__version__,
        "torch_tensorrt_version": torch_tensorrt.__version__,
        "python_version": sys.version.split()[0],
    }

    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        result["compute_capability"] = f"{major}.{minor}"

    return result


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "inspect-environment":
        inspect = ArgumentParser(description="Inspect the TensorRT compiler environment")
        inspect.add_argument("inspect-environment")
        inspect.add_argument("--result-json", type=Path)
        args = inspect.parse_args()

        import torch
        import torch_tensorrt

        write_result(environment_result(torch, torch_tensorrt), args.result_json)
        return

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--channels", required=True, type=int)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--opt-batch-size", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=4096)
    parser.add_argument(
        "--precision",
        choices=("fp16", "mixed-fp32-io-fp16-tactics", "fp32"),
        default="mixed-fp32-io-fp16-tactics",
    )
    parser.add_argument("--workspace-size", type=int)
    parser.add_argument("--optimization-level", type=int, choices=range(6), metavar="0-5")
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    import torch
    import torch_tensorrt

    if not torch.cuda.is_available():
        parser.error("TensorRT compilation requires CUDA")
    if not 0 < args.min_batch_size <= args.opt_batch_size <= args.max_batch_size:
        parser.error("batch sizes must satisfy 0 < min <= opt <= max")

    exact_fp16 = args.precision == "fp16"
    io_dtype = torch.float16 if exact_fp16 else torch.float32
    enabled_precisions = {
        "fp16": {torch.float16},
        "mixed-fp32-io-fp16-tactics": {torch.float32, torch.float16},
        "fp32": {torch.float32},
    }[args.precision]

    module = torch.jit.load(str(args.input), map_location="cuda").eval()
    if exact_fp16:
        module = module.half()

    compile_kwargs: dict[str, object] = {
        "ir": "torchscript",
        "inputs": [
            torch_tensorrt.Input(
                min_shape=[args.min_batch_size, args.channels, 8, 8],
                opt_shape=[args.opt_batch_size, args.channels, 8, 8],
                max_shape=[args.max_batch_size, args.channels, 8, 8],
                dtype=io_dtype,
            )
        ],
        "enabled_precisions": enabled_precisions,
    }

    if args.workspace_size is not None:
        compile_kwargs["workspace_size"] = args.workspace_size
    if args.optimization_level is not None:
        compile_kwargs["optimization_level"] = args.optimization_level

    started = time.monotonic()
    engine = torch_tensorrt.compile(module, **compile_kwargs)
    elapsed = time.monotonic() - started

    sample = torch.zeros(
        [args.min_batch_size, args.channels, 8, 8],
        dtype=io_dtype,
        device="cuda",
    )
    output = engine(sample)
    input_dtype = dtype_name(sample.dtype)
    output_dtype = dtype_name(output.dtype)
    expected_dtype = "fp16" if exact_fp16 else "fp32"

    if input_dtype != expected_dtype or output_dtype != expected_dtype:
        raise RuntimeError(
            f"{args.precision} requires {expected_dtype} I/O, compiled module has "
            f"input={input_dtype}, output={output_dtype}"
        )

    engine.save(str(args.output))

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "operation": "compile",
        "artifact_kind": "tensor-rt-torch-script",
        "elapsed_seconds": elapsed,
        "input_dtype": input_dtype,
        "output_dtype": output_dtype,
        "internal_precision": args.precision,
        "tensorrt_version": "unknown",
        "cuda_runtime_version": torch.version.cuda or "unknown",
        "torch_version": torch.__version__,
        "torch_tensorrt_version": torch_tensorrt.__version__,
        "profile": {
            "min": args.min_batch_size,
            "opt": args.opt_batch_size,
            "max": args.max_batch_size,
        },
        "python_version": sys.version.split()[0],
    }
    result.update(environment_result(torch, torch_tensorrt))
    result["operation"] = "compile"
    write_result(result, args.result_json)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"compile failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

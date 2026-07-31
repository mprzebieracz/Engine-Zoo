#!/usr/bin/env python3
"""Compile an Engine-zoo TorchScript export into an opt-in TensorRT module."""

from argparse import ArgumentParser
from pathlib import Path
import warnings

# Torch-TensorRT currently emits deprecation warnings from optional plugin
# integrations during import. They are harmless for this compiler; genuine
# exceptions and compiler diagnostics remain visible on stderr.
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch_tensorrt


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, help="TorchScript file from train export-torch-script")
    parser.add_argument("--output", required=True, help="TensorRT TorchScript output path")
    parser.add_argument("--channels", required=True, type=int)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--opt-batch-size", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=4096)
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32-fp16"),
        default="fp32-fp16",
        help="TensorRT enabled precisions (fp16 is recommended for self-play)",
    )
    parser.add_argument(
        "--workspace-size",
        type=int,
        help="TensorRT workspace limit in bytes (default: TensorRT automatic limit)",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=range(6),
        metavar="0-5",
        help="TensorRT builder optimization level",
    )
    parser.add_argument(
        "--timing-cache",
        type=Path,
        help=(
            "Request a persistent TensorRT builder timing cache. "
            "The TorchScript frontend must support this artifact ABI."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("TensorRT compilation requires CUDA")
    if not 0 < args.min_batch_size <= args.opt_batch_size <= args.max_batch_size:
        parser.error("batch sizes must satisfy 0 < min <= opt <= max")
    if args.timing_cache is not None:
        parser.error(
            "--timing-cache is unsupported by the installed Torch-TensorRT "
            "TorchScript frontend; refusing to switch compiler frontends because "
            "Rust inference requires a loadable TorchScript CModule"
        )

    module = torch.jit.load(args.input, map_location="cuda").eval()
    compile_kwargs: dict[str, object] = dict(
        ir="torchscript",
        inputs=[
            torch_tensorrt.Input(
                min_shape=[args.min_batch_size, args.channels, 8, 8],
                opt_shape=[args.opt_batch_size, args.channels, 8, 8],
                max_shape=[args.max_batch_size, args.channels, 8, 8],
                dtype=torch.float32,
            )
        ],
        enabled_precisions=(
            {torch.float16}
            if args.precision == "fp16"
            else {torch.float32, torch.float16}
        ),
    )

    if args.workspace_size is not None:
        compile_kwargs["workspace_size"] = args.workspace_size

    if args.optimization_level is not None:
        compile_kwargs["optimization_level"] = args.optimization_level

    engine = torch_tensorrt.compile(module, **compile_kwargs)
    engine.save(args.output)


if __name__ == "__main__":
    main()

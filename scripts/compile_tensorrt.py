#!/usr/bin/env python3
"""Compile an Engine-zoo TorchScript export into an opt-in TensorRT module."""

from argparse import ArgumentParser

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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("TensorRT compilation requires CUDA")
    if not 0 < args.min_batch_size <= args.opt_batch_size <= args.max_batch_size:
        parser.error("batch sizes must satisfy 0 < min <= opt <= max")

    module = torch.jit.load(args.input, map_location="cuda").eval()
    engine = torch_tensorrt.compile(
        module,
        ir="torchscript",
        inputs=[
            torch_tensorrt.Input(
                min_shape=[args.min_batch_size, args.channels, 8, 8],
                opt_shape=[args.opt_batch_size, args.channels, 8, 8],
                max_shape=[args.max_batch_size, args.channels, 8, 8],
                dtype=torch.float32,
            )
        ],
        enabled_precisions={torch.float32, torch.float16},
    )
    engine.save(args.output)


if __name__ == "__main__":
    main()

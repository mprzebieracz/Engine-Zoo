# TensorRT inference

TensorRT is an opt-in inference backend for CUDA self-play. Training, checkpoints, and the default `native` backend remain ordinary `tch`/LibTorch code. Keeping TensorRT at this boundary avoids making the project or its CPU workflows depend on an NVIDIA-specific runtime.

The backend consumes a TorchScript module whose `forward` returns one tensor: policy logits followed by one scalar-value column. The repository exports that module from a normal checkpoint; a separate Python step uses Torch-TensorRT to compile it. This is deliberately a one-generation workflow: TensorRT modules contain the weights they were compiled from, so they cannot accept a subsequent safetensors reload.

## Requirements

Use the same CUDA-compatible PyTorch/LibTorch version for Rust and Python, then install a matching `torch-tensorrt` Python package. The TensorRT shared library must be loaded before the Rust process starts; its exact path varies by installation.

## Export and compile

Initialize a normal run and obtain `latest.safetensors`, then export it on CUDA:

```bash
cargo run -p engine_app --bin train -- export-torch-script \
  --experiment experiments/chess-puct-wdl.toml \
  --checkpoint runs/chess-puct-wdl/checkpoints/latest.safetensors \
  --output runs/chess-puct-wdl/model.ts \
  --device cuda

python scripts/compile_tensorrt.py \
  --input runs/chess-puct-wdl/model.ts \
  --output runs/chess-puct-wdl/model.trt.ts \
  --channels 63 \
  --min-batch-size 1 --opt-batch-size 1024 --max-batch-size 4096
```

`--channels` must equal the model's encoded-state channel count. For the standard history-four canonical chess model it is 63; use the model specification for other representations.

## Enable it

Copy the experiment TOML used by the run and change its inference section:

```toml
[inference]
engine = "tensor-rt-torch-script"
tensor_rt_module = "model.trt.ts" # relative to the run directory
preferred_batch_size = 1024
max_batch_size = 4096
max_queue = 16384
max_wait = { milliseconds = 2 }
```

Then launch exactly one generation, preloading Torch-TensorRT's LibTorch integration:

```bash
LD_LIBRARY_PATH=/path/to/torch/lib:/path/to/tensorrt_libs \
LD_PRELOAD=/path/to/libtorchtrt.so \
  cargo run -p engine_app --bin train -- run \
  --run-dir runs/chess-puct-wdl --device cuda --iterations 1
```

For the next generation, export the new checkpoint, compile a new module, update `tensor_rt_module`, and start another one-generation invocation. TensorRT inference intentionally ignores the regular `precision` setting: the compiled module accepts FP32 tensors and selects FP16 kernels internally when legal.

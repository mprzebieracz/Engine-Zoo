# Raw TensorRT inference backend

The `tensor-rt-raw` variant builds a serialized TensorRT engine from ONNX and executes it in process. It is behind the `raw-tensorrt` Cargo feature because compilation and linking require CUDA and TensorRT headers and libraries. Default CPU builds do not enable it.

## Responsibility and data flow

`scripts/compile_tensorrt_raw.py` provides separate `export-onnx` and `build-engine` operations. They may use different Python interpreters. Rust supplies temporary paths and validates structured result JSON; normal compilation removes intermediates unless `--keep-tensor-rt-intermediates` is set.

At runtime one opaque C++ session owns the TensorRT runtime, engine, execution context, CUDA device, CUDA stream, discovered tensor contract, and grow-only device input allocation. A run accepts pinned host input and a Rust-owned device output buffer:

```text
pinned host input
  -> cudaMemcpyAsync on the session stream
  -> enqueueV3 on the same stream
  -> synchronize the session stream
  -> Rust gathers legal logits on CUDA
```

This ordering prevents TensorRT from reading input before H2D completes. Rust does not expose TensorRT enum ordinals, bind tensor names, manage a separate context, or sequence raw enqueue/synchronization calls. The session is movable to the single batcher executor thread but is not concurrent.

The constructor validates one input and one packed output, dtypes, `[channels, height, width]`, `action_size + 1` output columns, and min/opt/max batch profile. Every exported C function is exception-contained and returns project-owned status codes plus diagnostic text.

## SDK discovery and build

Use a coherent TensorRT SDK and CUDA toolkit. TensorRT discovery order is:

1. `TENSORRT_ROOT` with `include/` and `lib/` or `lib64/`;
2. both `TENSORRT_INCLUDE_DIR` and `TENSORRT_LIB_DIR`;
3. `TENSORRT_PYTHON`, if its package includes both headers and libraries.

CUDA uses either `CUDA_ROOT` or both `CUDA_INCLUDE_DIR` and `CUDA_LIB_DIR`. Linux also checks `/opt/cuda` and `/usr/local/cuda`. The build does not embed a developer-specific RPATH, so configure the runtime loader path for the selected SDK. Keep TensorRT headers, Python builder, and loaded `libnvinfer` on compatible versions.

```bash
TENSORRT_ROOT=/opt/tensorrt \
CUDA_ROOT=/opt/cuda \
LIBTORCH=/opt/libtorch \
cargo build -p engine_app --bin train --features raw-tensorrt --release
```

Alternatively:

```bash
TENSORRT_INCLUDE_DIR=/opt/tensorrt/include \
TENSORRT_LIB_DIR=/opt/tensorrt/lib \
CUDA_INCLUDE_DIR=/opt/cuda/include \
CUDA_LIB_DIR=/opt/cuda/lib64 \
cargo check -p alphazero --features raw-tensorrt
```

## Compile and train

Select the backend in the experiment, not with a cache flag:

```toml
[inference]
engine = "tensor-rt-raw"
compiled_artifact = "model.raw.engine"
precision = "fp16"
preferred_batch_size = 128
max_batch_size = 256
```

Then run the normal training command with a TensorRT builder Python. Raw builds use automatic timing-cache reuse unless disabled:

```bash
cargo run -p engine_app --bin train --features raw-tensorrt -- run \
  --run-dir runs/chess-puct-wdl-tensorrt-raw \
  --device cuda \
  --tensor-rt-python "$TRT_PYTHON" \
  --tensor-rt-export-python "$TORCH_PYTHON" \
  --tensor-rt-opt-batch-size 128 \
  --tensor-rt-max-batch-size 256
```

For direct compiler diagnosis, use explicit subcommands rather than generated `python -c` source:

```bash
"$TORCH_PYTHON" scripts/compile_tensorrt_raw.py export-onnx \
  --input model.ts --output model.onnx --channels 63 \
  --precision fp16 --result-json export.json

"$TRT_PYTHON" scripts/compile_tensorrt_raw.py build-engine \
  --input model.onnx --output model.engine \
  --min-batch-size 1 --opt-batch-size 128 --max-batch-size 256 \
  --precision fp16 --timing-cache tactics.cache \
  --result-json build.json
```

Exact FP16 fails if the graph, TensorRT version, or produced engine cannot satisfy FP16 I/O. Mixed mode has FP32 I/O and may use FP16 tactics. The manifest records both the request and actual contract.

Raw engines have frozen weights and reject backend reload. Training compiles a checkpoint-specific replacement artifact and replaces the inference service between generations. Timing-cache reuse only accelerates tactic selection and never validates an engine from another checkpoint.

GPU integration and same-stream stress tests must run on the dedicated CUDA/TensorRT CI runner. Ordinary hosts run default-feature tests only.

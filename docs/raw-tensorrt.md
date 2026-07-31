# Raw TensorRT inference backend (experimental variant)

This is an alternative to the Torch-TensorRT TorchScript path in
[`docs/tensorrt.md`](tensorrt.md). It bypasses Torch-TensorRT entirely: an
ONNX export is parsed and built directly with the TensorRT Python builder
API, producing a raw serialized engine plan (not a TorchScript module). A
small C shim (`crates/alphazero/native/raw_trt_runtime.cpp`) loads and runs
that plan in-process from Rust through the TensorRT C++ runtime, so
self-play still gets in-process, no-subprocess inference.

It exists to unlock a genuine persistent **builder timing cache**, which the
Torch-TensorRT path cannot support today (see the "Builder timing-cache
variant" section of `docs/tensorrt.md`): tactic timings measured while
compiling one checkpoint carry over to the next, since architecture/profile
stay fixed across a training run and only weights change. The goal is fewer
seconds spent per-generation compiling, without regressing inference
throughput relative to the existing TensorRT TorchScript path.

Implemented as a variant behind a Cargo feature (`raw-tensorrt`, off by
default) so it does not affect existing builds or require TensorRT headers
to be present unless explicitly opted into. It implements the same
`InferenceBackend` trait as `TchInferenceBackend` (see
`crates/alphazero/src/batcher/mod.rs`), so the batching/dynamic-batch
machinery, self-play, and training code are unchanged.

## Training entry point

```bash
# Same TensorRT experiment recipe; --cache selects raw engine + timing cache.
cargo run -p engine_app --bin train -- run \
  --run-dir runs/chess-puct-wdl-tensorrt \
  --cache \
  --tensor-rt-python "$TRT_PYTHON" \
  --tensor-rt-opt-batch-size 128 \
  --tensor-rt-max-batch-size 256
```

`--cache` is a runtime override (does not rewrite `experiment.toml`). Defaults:
compiler `scripts/compile_tensorrt_raw.py`, artifact `model.raw.engine`, timing
cache `<run-dir>/tensorrt-timing.cache`. The `raw-tensorrt` feature is on by
default for `engine_app`.

## Status

Requires TensorRT **10.x** headers + libs (FP16 via `BuilderFlag.FP16`).
Defaults: vendored headers in `third_party/tensorrt-10.15.1/include` and the
pip `libnvinfer.so.10` from the Torch-TensorRT venv. System TensorRT 11.x
dropped `BuilderFlag.FP16` and is not used for this path.

## Tensor contract

Matches the existing Torch-TensorRT path exactly: one input tensor `input`
of shape `[batch, channels, 8, 8]`, one packed output tensor `output` of
shape `[batch, action_size + 1]` (policy logits followed by one scalar-value
column), both FP32. Both backends can be built from the exact same
TorchScript export (`export_torchscript` / `engine-zoo-model
export-torch-script`), so they are directly comparable.

## Building

```bash
LIBTORCH=/path/to/libtorch \
TENSORRT_INCLUDE_DIR=/usr/include \
CUDA_ROOT=/opt/cuda \
cargo build -p engine-bench --features raw-tensorrt --release
```

`TENSORRT_INCLUDE_DIR`/`TENSORRT_LIB_DIR` and `CUDA_ROOT` (or
`CUDA_INCLUDE_DIR`/`CUDA_LIB_DIR`) are overridable in
`crates/alphazero/build.rs`; defaults assume a system TensorRT package under
`/usr/include`/`/usr/lib` and a CUDA toolkit under `/opt/cuda` or
`/usr/local/cuda`.

**Version note:** this repo's tested Torch-TensorRT stack is pinned to
TensorRT `10.15.1.29` (`docs/tensorrt.md`). This raw path was built and
link-checked against whatever TensorRT happens to be installed
system-wide (`11.1.0` in this environment) because pip-installed TensorRT
wheels don't ship C++ headers and no `10.x` SDK with headers was available
here. The two paths are fully decoupled (no `LD_PRELOAD`/`LD_LIBRARY_PATH`
matching needed between them), so this is fine as an independent
comparison, but for like-for-like results against the pinned Torch-TensorRT
stack, point `TENSORRT_INCLUDE_DIR`/`TENSORRT_LIB_DIR` at a `10.15.1.29` SDK
if you have one, and use `scripts/compile_tensorrt_raw.py` against a
matching Python `tensorrt` install.

**Precision:** TensorRT 11 removed the legacy weak-typing
`BuilderFlag.FP16` in favor of "strongly typed" networks driven by the
ONNX graph's own tensor dtypes. `scripts/compile_tensorrt_raw.py` detects
this and falls back to FP32 with a warning rather than silently building an
engine whose FP16 numerics were never checked against the native reference.
On a TensorRT 10.x build (`BuilderFlag.FP16` present), `--precision fp16`
and `fp32-fp16` behave like `compile_tensorrt.py` today.

## Compiling an engine

```bash
LIBTORCH=/path/to/libtorch \
cargo run -p engine_app --bin engine-zoo-model -- export-torch-script \
  --experiment experiments/chess-puct-wdl.toml \
  --checkpoint runs/chess-puct-wdl/checkpoints/latest.safetensors \
  --output runs/chess-puct-wdl/model.ts \
  --device cuda

TRT_PYTHON=.venv/engine-zoo-trt/bin/python  # or a python with `tensorrt` installed
"$TRT_PYTHON" scripts/compile_tensorrt_raw.py \
  --input runs/chess-puct-wdl/model.ts \
  --output runs/chess-puct-wdl/model.raw.engine \
  --channels 63 \
  --min-batch-size 1 --opt-batch-size 128 --max-batch-size 256 \
  --precision fp16 \
  --timing-cache runs/chess-puct-wdl/tensorrt-timing.cache
```

Rebuild after changing the checkpoint (same architecture, new weights): the
`--timing-cache` file is reused and updated atomically (temp file + rename,
so a killed process cannot leave a truncated cache) rather than requiring a
cold rebuild each generation the way the Torch-TensorRT path currently does.

## Benchmarking against the existing backends

`engine-bench raw-inference` already compares native FP16 and Torch-TensorRT
FP16; with this feature it adds a third:

```bash
LIBTORCH=/path/to/libtorch \
cargo run -p engine-bench --features raw-tensorrt --release -- raw-inference \
  --experiment experiments/chess-puct-wdl.toml \
  --checkpoint runs/chess-puct-wdl/checkpoints/latest.safetensors \
  --tensor-rt-module runs/chess-puct-wdl/model.trt.ts \
  --raw-tensor-rt-engine runs/chess-puct-wdl/model.raw.engine \
  --batch-sizes 8,16,32,64,128,256 \
  --human
```

This reports `positions_per_second`/`backend_latency_ms` for all backends at
each batch size, plus numerical-difference metadata for the Torch-TensorRT
path. For compile-time comparisons, time `compile_tensorrt.py` vs
`compile_tensorrt_raw.py` cold (no cache / fresh checkpoint) and warm (cache
present, next generation's weights) — `scripts/bench_tensorrt_training.py`
is a useful template for scripting this across iterations.

## Known limitations / follow-ups

- **Synchronous by design.** `RawTensorRtBackend` fully synchronizes the
  device around each `enqueueV3` (matching today's non-overlapped batcher
  execution), rather than using a dedicated CUDA stream with async
  host-to-device/device-to-host transfers. That's the natural next
  optimization once this is validated for correctness and throughput parity
  — but validate parity first before adding overlap.
- **Fixed weights.** Like the Torch-TensorRT path, an engine's weights are
  frozen at compile time; `reload_weights` is rejected. Compile a new
  engine for the next checkpoint.
- **No CUDA graphs.** Deliberately not attempted first: the AlphaZero
  reference implementation this was inspired by found CUDA graph mode
  increased total runtime for their workload and disabled it by default.
- Two independent copies of `libcudart` end up loaded in the same process
  (one from the CUDA toolkit this shim links against, one bundled with
  LibTorch) — the same situation Torch-TensorRT's own `libtorchtrt.so`
  already creates in this codebase. Established as safe in practice, but
  worth knowing about if you see unexpected CUDA errors during testing.

# TensorRT inference

TensorRT is a fixed-weight CUDA self-play option. Native LibTorch still performs training and writes safetensors checkpoints. After each training generation, the runner compiles a new artifact and replaces the inference service before the next self-play generation; compiled backends do not support in-place weight reload.

Two compiled variants remain intentionally separate:

- `tensor-rt-torch-script`: `scripts/compile_tensorrt.py` builds a Torch-TensorRT CModule loaded through `tch::CModule`.
- `tensor-rt-raw`: `scripts/compile_tensorrt_raw.py` exports ONNX and builds a TensorRT engine plan loaded by the raw runtime session.

Python remains the build-time compiler for both. Rust owns checkpoint selection, compilation orchestration, temporary paths, manifest validation, and backend loading. The raw C++ code only loads and executes a completed engine.

## Runtime stack

The established host stack is PyTorch/LibTorch 2.11.0 with CUDA 13.0, Torch-TensorRT 2.11.0+cu130, and TensorRT 10.15.1.29. Treat this as a tested combination, not a promise that arbitrary mixed versions work. Torch-TensorRT, LibTorch, CUDA, and the registered `libtorchtrt` custom classes must be ABI-compatible.

For the TorchScript path, create and inspect a matching compiler environment:

```bash
TRT_VENV=.venv/engine-zoo-trt
python3.13 -m venv "$TRT_VENV"
TRT_PYTHON="$TRT_VENV/bin/python"

"$TRT_PYTHON" -m pip install --upgrade pip
"$TRT_PYTHON" -m pip install \
  'torch==2.11.0+cu130' \
  'torch-tensorrt==2.11.0+cu130' \
  'tensorrt-cu13==10.15.1.29' \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.nvidia.com

"$TRT_PYTHON" -c 'import torch, torch_tensorrt, tensorrt; print(torch.__version__, torch.version.cuda, torch_tensorrt.__version__, tensorrt.__version__)'
```

At runtime, put the matching shared libraries on the loader path. The TorchScript backend also needs the matching `libtorchtrt` custom-class archive loaded after LibTorch.

## Experiment configuration

Backend selection belongs to immutable experiment configuration:

```toml
[inference]
engine = "tensor-rt-torch-script" # or "tensor-rt-raw"
compiled_artifact = "model.trt.ts" # or "model.raw.engine"
precision = "fp16"
preferred_batch_size = 128
max_batch_size = 256
max_queued_states = 4096

[inference.max_wait]
milliseconds = 1
```

`tensor_rt_module` remains a read alias for old experiment files; new files should use `compiled_artifact`. Do not use the removed `--cache` backend switch. A raw experiment automatically selects the raw compiler and automatic timing-cache policy.

Initialize and run normally:

```bash
cargo run -p engine_app --bin train -- init \
  --experiment experiments/chess-puct-wdl-tensorrt.toml \
  --run-dir runs/chess-puct-wdl-tensorrt

cargo run -p engine_app --bin train -- run \
  --run-dir runs/chess-puct-wdl-tensorrt \
  --device cuda --iterations 2 \
  --tensor-rt-python "$TRT_PYTHON" \
  --tensor-rt-min-batch-size 1 \
  --tensor-rt-opt-batch-size 128 \
  --tensor-rt-max-batch-size 256
```

For raw TensorRT, `--tensor-rt-export-python` may select a Torch-capable ONNX exporter separately from the TensorRT builder interpreter. `--keep-tensor-rt-intermediates` preserves the otherwise temporary TorchScript, ONNX, and result files for diagnosis.

## Precision

Build precision is explicit:

- `fp32` requires FP32 I/O and FP32 build policy.
- `fp16` requires exact FP16 I/O; unsupported exact FP16 fails closed.
- `mixed-fp32-io-fp16-tactics` keeps FP32 I/O while allowing FP16 tactics.

The compiler inspects actual artifact I/O and returns structured JSON. Rust rejects a mismatch and records requested precision and actual input/output dtypes in the artifact manifest. There is no silent FP16-to-FP32 fallback.

## Artifact lifecycle

An engine or CModule contains checkpoint weights. The training/compiler path validates its sibling `manifest.json` against checkpoint SHA-256, model fingerprint, tensor/value contract, requested build settings, actual I/O dtypes, batch profile, compiler identity, target environment, and artifact SHA-256 before cache reuse. The loader independently rejects a missing manifest, artifact digest mismatch, wrong backend, or wrong model contract.

Compilation is serialized by an artifact lock. It writes unique same-directory temporaries, validates compiler JSON, installs the complete artifact, then writes the manifest last. A missing, malformed, stale, or digest-mismatched manifest means recompile required; file existence, size, and modification time are never sufficient.

TensorRT timing caches have a separate lifecycle. A raw build uses an automatic graph/profile/environment-specific cache by default, which can be reused after checkpoint weights change. The engine artifact cannot. Use `--disable-tensor-rt-timing-cache` for a deliberate cold build or `--tensor-rt-timing-cache PATH` for an explicit cache path; neither option changes the backend.

See [raw-tensorrt.md](raw-tensorrt.md) for SDK discovery and raw-session details.

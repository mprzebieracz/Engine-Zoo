# TensorRT inference

TensorRT is the default CUDA self-play backend for chess. Training and
checkpoints remain normal `tch`/LibTorch code. A compiled module contains fixed
weights, so the supported workflow is TensorRT self-play followed by native
training; export and compile the new checkpoint before the next TensorRT
self-play generation. The native experiment remains an explicit fallback.

The module's `forward` returns a packed tensor: policy logits followed by one
scalar-value column.

## Pinned, matching runtime

The working stack on this host is:

- LibTorch/PyTorch `2.11.0`, CUDA `13.0`
- Torch-TensorRT `2.11.0+cu130`
- TensorRT `10.15.1.29`
- Python `3.13` compiler environment

Set `LIBTORCH` to the root of your matching LibTorch installation. The Python
compiler and C++ runtime integration must match that LibTorch release and CUDA
build *exactly*.
In particular, do not mix a system TensorRT, an older Python environment, or a
`libtorchtrt.so` built for another PyTorch release with this runtime.

Create the compiler environment with PyPI as the primary index, plus the CUDA
wheel and NVIDIA package indexes:

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
```

`tensorrt-cu13` installs the matching TensorRT bindings and shared libraries.
Verify the actual installation before compiling:

```bash
TRT_PYTHON=.venv/engine-zoo-trt/bin/python

"$TRT_PYTHON" - <<'PY'
import torch
import torch_tensorrt  # Required: registers Torch-TensorRT's Python classes.

print(torch.__version__, torch.version.cuda)
print(torch_tensorrt.__version__)
PY
```

This import smoke test succeeds on the configured host and reports
`2.11.0+cu130 13.0` and `2.11.0+cu130`. The compilation script already imports
`torch_tensorrt`; do not remove or defer that import, because it registers the
TorchScript classes needed to create the compiled module.

## Export and compile

Initialize a normal run and obtain `latest.safetensors`, then export it on CUDA:

```bash
LIBTORCH=/path/to/libtorch \
cargo run -p engine_app --bin engine-zoo-model -- export-torch-script \
  --experiment experiments/chess-puct-wdl.toml \
  --checkpoint runs/chess-puct-wdl/checkpoints/latest.safetensors \
  --output runs/chess-puct-wdl/model.ts \
  --device cuda

TRT_PYTHON=.venv/engine-zoo-trt/bin/python

"$TRT_PYTHON" scripts/compile_tensorrt.py \
  --input runs/chess-puct-wdl/model.ts \
  --output runs/chess-puct-wdl/model.trt.ts \
  --channels 63 \
  --min-batch-size 1 --opt-batch-size 1024 --max-batch-size 4096
```

`--channels` must equal the model's encoded-state channel count. For the
standard history-four chess model it is 63; use the model specification for
other representations.

## Run iterative TensorRT self-play and native training

The `train` CLI now owns the safe per-generation handoff. It always compiles
the current checkpoint before opening the TensorRT inference service. For a
newly initialized run it first creates the same deterministic initial
checkpoint that native training would create. For each iteration, it:

1. produces self-play with the fixed TensorRT module;
2. trains the native model and writes the next safetensors checkpoint;
3. exports and compiles that new checkpoint atomically; and
4. reloads the new TensorRT module before the next self-play generation.

The experiment remains immutable. `tensor_rt_module` is a generated artifact
under the run directory, and is only replaced after compilation succeeds.

The default H4 TensorRT recipe is
`experiments/chess-puct-wdl-tensorrt.toml`. The 64-game self-play screen selected
112 self-play threads, a 32-leaf local search batch, preferred batch 128, and a
1 ms wait (maximum batch 256 and queued states 4096), with TensorRT compile
shapes optimal 128 and maximum 256. That validated median was about 2,115
positions/s. Initialize the run with that immutable recipe; `train run` then
automatically exports, compiles, and reloads the TensorRT artifact before and
between generations:

```bash
LIBTORCH=/path/to/libtorch
TRT_PYTHON=.venv/engine-zoo-trt/bin/python
TRT_SITE="$("$TRT_PYTHON" -c 'import site; print(site.getsitepackages()[0])')"

LD_LIBRARY_PATH="$LIBTORCH/lib:$TRT_SITE/torch_tensorrt/lib:$TRT_SITE/tensorrt_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LD_PRELOAD="$LIBTORCH/lib/libtorch.so:$TRT_SITE/torch_tensorrt/lib/libtorchtrt.so" \
  cargo run -p engine_app --bin train -- init \
  --experiment experiments/chess-puct-wdl-tensorrt.toml \
  --run-dir runs/chess-puct-wdl-tensorrt

LD_LIBRARY_PATH="$LIBTORCH/lib:$TRT_SITE/torch_tensorrt/lib:$TRT_SITE/tensorrt_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LD_PRELOAD="$LIBTORCH/lib/libtorch.so:$TRT_SITE/torch_tensorrt/lib/libtorchtrt.so" \
  cargo run -p engine_app --bin train -- run \
  --run-dir runs/chess-puct-wdl-tensorrt --device cuda --iterations 2 \
  --tensor-rt-python "$TRT_PYTHON" \
  --tensor-rt-min-batch-size 1 \
  --tensor-rt-opt-batch-size 128 \
  --tensor-rt-max-batch-size 256
```

`--tensor-rt-compiler` can point to a different compiler script; by default it
uses this checkout's `scripts/compile_tensorrt.py`. Compilation failure stops
the run before a stale module can be used for another generation. The latest
valid module remains on disk for diagnosis or recovery.

## Builder timing-cache variant

Use `train run --cache` to select the raw TensorRT backend with a persistent
builder timing cache (see [`raw-tensorrt.md`](raw-tensorrt.md)). This is a
**runtime** override: it does not rewrite `experiment.toml`. A
`tensor-rt-torch-script` experiment is compiled via
`scripts/compile_tensorrt_raw.py` into `model.raw.engine`, with timings stored
in `<run-dir>/tensorrt-timing.cache` unless `--tensor-rt-timing-cache` is set.

```bash
LD_LIBRARY_PATH="$LIBTORCH/lib:$TRT_SITE/tensorrt_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  cargo run -p engine_app --bin train --features raw-tensorrt -- run \
  --run-dir runs/chess-puct-wdl-tensorrt --device cuda --iterations 3 \
  --cache \
  --tensor-rt-python "$TRT_PYTHON" \
  --tensor-rt-min-batch-size 1 \
  --tensor-rt-opt-batch-size 128 \
  --tensor-rt-max-batch-size 256
```

Do **not** preload `libtorchtrt.so` for `--cache` (raw TensorRT does not use
it). Ensure `libnvinfer.so.10` from the TensorRT 10.x wheel is on
`LD_LIBRARY_PATH`.

The older `train run --tensor-rt-timing-cache PATH` flag alone still targets
the Torch-TensorRT TorchScript compiler, which rejects the cache before
building. Prefer `--cache` for the working timing-cache path.

## Manual enablement

Copy the experiment TOML used by the run and set:

```toml
[inference]
engine = "tensor-rt-torch-script"
tensor_rt_module = "model.trt.ts" # relative to the run directory
preferred_batch_size = 128
max_batch_size = 256
max_queued_states = 4096
max_wait = { milliseconds = 1 }
```

These are the tuned default TensorRT batching settings, paired with 112
self-play threads and a 32-leaf search batch in the experiment recipe. The
compiler's dynamic batch shapes should cover the same range: minimum 1,
optimal 128, maximum 256.

The C++ custom-class archive in `libtorchtrt.so` must match LibTorch exactly.
Before Rust loads the compiled TorchScript module, preload LibTorch first and
then that matching archive. The pip layout above has these three runtime
directories:

```bash
LIBTORCH=/path/to/libtorch
TRT_PYTHON=.venv/engine-zoo-trt/bin/python
TRT_SITE="$("$TRT_PYTHON" -c 'import site; print(site.getsitepackages()[0])')"

LD_LIBRARY_PATH="$LIBTORCH/lib:$TRT_SITE/torch_tensorrt/lib:$TRT_SITE/tensorrt_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LD_PRELOAD="$LIBTORCH/lib/libtorch.so:$TRT_SITE/torch_tensorrt/lib/libtorchtrt.so" \
  cargo run -p engine_app --bin train -- run \
  --run-dir runs/chess-puct-wdl --device cuda --iterations 1
```

The `LD_PRELOAD` order is intentional: `libtorch.so` comes first, followed by
the exact matching C++ Torch-TensorRT archive. Loading an unmatched archive can
produce missing custom-class symbols, ABI errors, or a module that cannot be
deserialized.

TensorRT selects FP16 kernels internally when legal; the compiled module still
accepts FP32 input tensors. It cannot receive new safetensors weights in place.
After TensorRT self-play, use native inference for the optimizer step, then
export and compile the resulting checkpoint for the next generation.

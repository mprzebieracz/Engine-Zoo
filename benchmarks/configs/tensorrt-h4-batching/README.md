# H4 TensorRT batching sweep

The runner dynamically writes nine benchmark-only configurations from the
default H4 Root-Gumbel recipe: 24 workers, leaf batch 16, playout-cap
randomization 256/64, inference maximum batch 256, and queue 4096. It varies
only preferred global inference batch (`32`, `64`, or `128`) and maximum wait
(`1`, `2`, or `4` ms).

Build the benchmark binary, then run the complete resumable sweep with a
TensorRT module that was compiled from the supplied checkpoint:

```bash
export LIBTORCH=/path/to/libtorch
export TRT_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"

cargo build --release -p engine-bench

python3 scripts/sweep_tensorrt_h4_batching.py \
  --checkpoint runs/chess-puct-wdl/checkpoints/latest.safetensors \
  --tensor-rt-module runs/chess-puct-wdl/model.trt.ts
```

Pass `--libtorch` and `--trt-site` instead if you do not want to set
environment variables. `TRT_SITE` must be the site-packages directory of the
Python environment that contains both `torch_tensorrt` and `tensorrt_libs`.

Each case runs exactly 64 fixed-checkpoint games and writes one JSON report to
`benchmark-results/tensorrt-h4-batching-sweep/results/`. The runner establishes
the matching LibTorch/Torch-TensorRT runtime environment, resumes valid prior
reports, and writes `aggregate.json` and `aggregate.csv`, ordered by positions
per second. Use `--dry-run` to inspect cases or `--limit N` for a partial run.

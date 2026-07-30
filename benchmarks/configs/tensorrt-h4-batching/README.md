# H4 TensorRT throughput sweep

The runner dynamically writes benchmark-only configurations from the default
H4 Root-Gumbel recipe. It screens self-play thread count, MCTS leaf batch,
preferred inference batch, and maximum inference wait. Defaults are 16/24/32
threads, leaf batches 16/32, preferred batches 128/256, and waits 1/2 ms.
Use the command-line lists to narrow or expand any dimension.

Build the benchmark binary, then run the complete resumable sweep with a
TensorRT module that was compiled from the supplied checkpoint:

```bash
export LIBTORCH=/path/to/libtorch
export TRT_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"

cargo build --release -p engine-bench

python3 scripts/sweep_tensorrt_h4_batching.py \
  --checkpoint runs/chess-puct-wdl-tensorrt-v2/checkpoints/latest.safetensors \
  --tensor-rt-module runs/chess-puct-wdl-tensorrt-v2/model.trt.ts \
  --games 64
```

For a smaller focused screen, for example, use
`--threads 24,32 --leaf-batches 16,32 --preferred-batches 128,256
--wait-ms 1,2`. The production experiment is never modified.

Pass `--libtorch` and `--trt-site` instead if you do not want to set
environment variables. `TRT_SITE` must be the site-packages directory of the
Python environment that contains both `torch_tensorrt` and `tensorrt_libs`.

Each case runs exactly 64 fixed-checkpoint games and writes one JSON report to
`benchmark-results/tensorrt-h4-comparison/batching-sweep/results/`. The runner establishes
the matching LibTorch/Torch-TensorRT runtime environment, resumes valid prior
reports, and writes `aggregate.json` and `aggregate.csv`, ordered by positions
per second. Use `--dry-run` to inspect cases or `--limit N` for a partial run.

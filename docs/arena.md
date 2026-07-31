# Multi-model chess arena

Edit `configs/arena.toml` to name a candidate and a fixed opponent roster.
Each entry lists a checkpoint path, a human `architecture` label, an optional
per-model MCTS `simulations` budget, and an inference `backend`. The ModelSpec
actually loaded comes from the checkpoint's run directory (`experiment.toml`,
or a migrated legacy `config.json`).

```bash
python3 scripts/arena_big.py
```

The wrapper builds `eval-big-arena`, `engine-zoo-uci`, and `engine-zoo-model`,
compiles missing TensorRT modules into `settings.trt_cache_dir`, writes a
resolved config with those paths filled in, and runs the Rust binary. Each
opponent gets a directory under `settings.output_dir` with cleaned PGNs,
Fastchess diagnostics, `spec.json`, and `report.json`. The root also receives
`arena.json` and a human-readable `REPORT.md` with candidate metadata,
aggregate score, and per-opponent W/D/L, score fraction, and smoothed Elo.

## Inference backends

```toml
[candidate]
backend = "tensor-rt"   # Torch-TensorRT TorchScript module, GPU-only.

[[opponents]]
backend = "native"      # Default. Loads the safetensors checkpoint via tch.
```

The UCI engine forces native inference whenever no `TensorRtModule` UCI option
is set, even when the run's `experiment.toml` requests
`engine = "tensor-rt-torch-script"`. That way a TensorRT training checkpoint
can still be served natively in the arena without its precompiled module.

## TensorRT cache

For each `backend = "tensor-rt"` model without a preset `tensor_rt_module`
path, `scripts/arena_big.py` compiles a per-checkpoint module into
`settings.trt_cache_dir` (default `data/evaluations/trt-cache`). The cache key
includes the checkpoint fingerprint, model channels, compile min/opt/max
shapes, precision, compiler contents, and Torch/TensorRT/GPU identity. This
prevents an engine optimized for one batch profile or GPU from being reused for
another.

By default the compile shapes and precision are derived from the checkpoint's
immutable `experiment.toml`, matching the runtime batcher. Set
`trt_override_batch_sizes = true` only for a deliberate benchmark; the wrapper
warns when that profile differs from the experiment. Use
`--force-tensor-rt-recompile` to rebuild a matching artifact for diagnosis.

Compilation uses `scripts/compile_tensorrt.py` inside a matching
Torch-TensorRT Python environment. Discovery order:

1. `--tensor-rt-python`
2. `$TRT_PYTHON`
3. `~/venvs/engine-zoo-trt-py313/bin/python`
4. `/tmp/engine-zoo-arena/venv/bin/python` (miracle)
5. `$VIRTUAL_ENV` / `$CONDA_PREFIX`
6. repo-local `.venv/engine-zoo-trt*/bin/python`
7. `python3` / `python` on `$PATH`

Legacy classic runs that only have `config.json` are migrated automatically
via `engine-zoo-model ensure-experiment` before compilation.

## Overriding the roster

```bash
python3 scripts/arena_big.py --config configs/arena.toml -- \
  --candidate-name v6-latest \
  --candidate-path runs/chess-puct-wdl-tensorrt-v6/checkpoints/latest.safetensors \
  --candidate-architecture chess-se-h4 \
  --candidate-simulations 800 \
  --games 8 \
  --output-dir data/evaluations/arena-smoke
```

Arguments after `--` go to `eval-big-arena`. `settings.games` is the total
number of games per candidate/opponent pair and must be even (colors balanced).

## Example (v6 vs v5-gen800 + v2)

The default `configs/arena.toml` pits `v6-latest` against `v5-gen800` and
`v2-latest`, all on the native (LibTorch) backend:

```bash
python3 scripts/arena_big.py --config configs/arena.toml -- --games 8
```

Set `backend = "tensor-rt"` on a model when you want a one-time compile into
`settings.trt_cache_dir`. TensorRT engines are GPU-architecture specific, so
compile on the machine that will run the arena.

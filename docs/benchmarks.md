# Benchmark methodology

Timing is evidence, not a unit-test assertion. Do not claim a performance improvement without recording machine, compiler, device/driver, configuration, median, spread, and workload.

Use the reproducible benchmark CLI rather than ad hoc benchmark binaries. Run `cargo run -p engine-bench -- search --algorithm puct --human` for the deterministic Connect Four search workload, or `cargo run -p engine-bench -- suite --human` for the complete matrix. Run benchmarks only on an intentional benchmark machine. Required future matrices cover PUCT leaf batches, Full Gumbel budgets, cold/hot caches, representative chess positions, arena allocations, and depth.

The benchmark CLI defaults to `--device cuda`; it deliberately errors when
`tch::Cuda::is_available()` is false rather than silently measuring CPU. Use
`--device cpu` for an intentional CPU run, or `--device auto` only when a
portable fallback is wanted. Every neural workload report records its selected
device and precision. For example:

```bash
cargo run --release -p engine-bench -- inference --samples 20 --human
cargo run --release -p engine-bench -- suite --device cuda --precision fp32 --output results.json
```

## Full-iteration CUDA baselines

The `iteration` workload measures the production `TrainingRun::step` lifecycle
from outside the production crates: self-play, training, checkpoint write, and
inference reload. CUDA is the default and the timer synchronizes at the
iteration boundary. Every measured sample starts from a fresh temporary run,
so its replay contents, optimizer state, model weights, and seed are
reproducible.

The checked-in 200-game baseline is the standard comparison point: canonical
chess H1, 12x128 SE model, FP16 inference, 16 self-play workers, 800 PUCT
simulations per move, 2,048 training batch, 256 micro-batch, and 80 training
steps. The optional 500-game version changes only the game count.

```bash
# Capture a named before/after artifact. The parent directory is created.
cargo run --release -p engine-bench -- iteration \
  --name before-cache-change \
  --output benchmark-results/before-cache-change.json \
  --human

cargo run --release -p engine-bench -- iteration \
  --name after-cache-change \
  --output benchmark-results/after-cache-change.json \
  --human

# Use the longer workload or deliberately override a parameter.
cargo run --release -p engine-bench -- iteration \
  --experiment benchmarks/configs/full-iteration-cuda-500.toml \
  --name cuda-500-baseline \
  --output benchmark-results/cuda-500-baseline.json

cargo run --release -p engine-bench -- iteration \
  --games 200 --train-steps 80 --batch-size 2048 --micro-batch-size 256 \
  --seed 1 --device cuda --precision fp16 \
  --name explicit-cuda-200 \
  --output benchmark-results/explicit-cuda-200.json
```

The report records the effective experiment TOML, seed, machine/build metadata,
elapsed wall time, positions/s, configured PUCT simulations/s, training phase
metrics, and batcher statistics. `TrainMetrics` provides training subphase
durations. Without benchmark timers in production, the remaining phase is
truthfully labelled as the combined self-play/checkpoint/reload/persistence
time; do not treat it as an isolated checkpoint or reload measurement. Use
`engine-profile` with Nsight Systems to inspect that timeline.

Representation/replay measurements should cover H1/H4/H8 construction/encoding/action round trips and replay batch sizes 256/1024/4096. GPU measurements should separately report forward-only, legal-logit gather, end-to-end inference, transfer, backward, optimizer, latency percentiles, and samples/s.

## Profiling prerequisites

`crates/benchmarks` (`engine-bench`) owns benchmark workloads;
`crates/profiling` (`engine-profile`) owns profiling-only workloads and launch
commands. Production crates remain instrumentation-free: do not add profiler
hooks, tracing allocations, or profiler-only branches to search, inference,
self-play, or training code. The profiling Cargo profile keeps optimized code
while retaining symbols (`debug = 1`, `strip = false`). See
[profiling.md](profiling.md) for the compact command reference.

### LibTorch runtime

All inference, batching, and training workloads require a LibTorch runtime
compatible with the repository's `tch` dependency and the selected device.
Use one of these explicit setups:

```bash
# Official LibTorch distribution.
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Or use the active Python PyTorch installation.
export LIBTORCH_USE_PYTORCH=1
```

Do not mix a CPU-only LibTorch installation with CUDA benchmark commands. For
CUDA workloads, use a CUDA-enabled LibTorch/PyTorch build supported by the
installed NVIDIA driver. Rebuild after changing the selected LibTorch runtime;
the build scripts intentionally do not infer machine-local installation paths.
Record the selected runtime/version in the benchmark environment report and
with the raw profiler artifacts.

`nvidia-smi` proves that the driver can see a GPU, but it does not prove that
the LibTorch dynamically loaded by `tch` has CUDA support. If the default
benchmark reports `tch::Cuda::is_available() is false`, inspect the loaded
LibTorch/PyTorch distribution and its CUDA/driver compatibility before using
`--device cpu` or `--device auto`.

### CPU profiling on Linux

CPU profiles require Linux, a `perf` binary compatible with the running
kernel, and permission to collect the requested events. On Debian/Ubuntu the
usual installation is:

```bash
sudo apt install linux-tools-common "linux-tools-$(uname -r)"
perf --version
perf stat -e cycles,instructions,cache-misses -- true
```

If the final command is denied, the machine's `kernel.perf_event_paranoid`
policy or container/VM policy must be adjusted by its administrator; do not
silently substitute wall-clock timing for hardware counters. Build the named
profiling target so Rust symbols are available:

```bash
cargo build --profile profiling -p engine-profile
perf record -g --call-graph dwarf -- \
  target/profiling/engine-profile run inference --device cpu --warmup 20 --iterations 1000
perf report
```

Install distribution debug-symbol packages as well when attribution inside
LibTorch, CUDA stubs, or system libraries matters. Record kernel version,
`perf --version`, CPU governor/frequency policy, and process affinity alongside
the benchmark JSON.

### NVIDIA GPU profiling

GPU profiles require an NVIDIA GPU, a driver that supports the CUDA runtime
used by the chosen CUDA-enabled LibTorch/PyTorch build, and compatible Nsight
tools. Verify the complete toolchain before measuring:

```bash
nvidia-smi
nsys --version
ncu --version
```

Install Nsight Systems (`nsys`) for timeline/CPU-CUDA correlation and Nsight
Compute (`ncu`) for kernel metrics. Both tools must support the installed
driver/GPU; GPU performance counters can also be restricted by administrator
policy, especially on shared machines. Use a short fixed workload for `ncu`
because detailed metric collection changes kernel timing:

```bash
cargo build --profile profiling -p engine-profile
nsys profile --trace=cuda,osrt -- \
  target/profiling/engine-profile run inference --warmup 20 --iterations 1000
ncu --target-processes all --set full -- \
  target/profiling/engine-profile run inference --warmup 20 --iterations 10
```

Do not compare an `ncu`-instrumented duration with ordinary benchmark timing.
Record GPU model, driver, CUDA runtime reported by the chosen LibTorch/PyTorch
installation, Nsight versions, power-management mode, and whether the machine
was otherwise idle. Keep raw `.nsys-rep`, `ncu` exports, and large profiler
artifacts out of Git; commit only concise summaries and interpretation.

## Historical CPU refactor benchmark and profiling record — 2026-07-29

These results are a local CPU measurement record for the completed refactor.
They are not portable performance claims and predate the CUDA-default neural
batcher and self-play fixtures above; do not compare their non-search rows to
new benchmark output.

### Regression comparison

The legacy ad-hoc PUCT benchmark (200 searches × 256 simulations) measured
**3,395,435 simulations/s**. The final reproducible `engine-bench` PUCT-256
workload, measured over seven samples, had a median of **3,384,005
simulations/s**: **-0.34%**, within the no-regression tolerance. The harnesses
are not identical, so this comparison is a regression guard rather than an
improvement claim.

### Final CPU `engine-bench` suite medians

| Workload | Median |
| --- | ---: |
| PUCT, 128 simulations | 0.036 ms |
| Root Gumbel PUCT, 128 simulations | 0.041 ms |
| Full Gumbel, 128 simulations | 0.063 ms |
| CPU inference | 0.220 ms |
| Dynamic batcher | 0.008 ms |
| Replay | 0.010 ms |
| Self-play | 0.239 ms |
| CPU training | 2.961 ms |

### CPU profile

The representative CPU PUCT run completed five 5,000-search PUCT-256 passes
in **0.6718 s**. Hardware counters recorded **9.50B instructions** and
**2.83B cycles**, with a **4.18% cache-miss rate**.

The hottest sampled functions were:

| Function | Samples |
| --- | ---: |
| `puct_child` | 17.44% |
| `Mcts::search` | 13.67% |
| `evaluate_positions` | 9.58% |

Nsight was unavailable for this campaign, so no GPU profile is recorded.

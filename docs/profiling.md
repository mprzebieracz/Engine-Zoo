# Profiling

`engine-profile` is the only profiling workload crate. Production crates have
no dependency edge back into it: AlphaZero, search, games, and application
crates contain neither profiler hooks nor profiler-only timing branches.

The executable defaults to CUDA and refuses to fall back silently. Before a
capture, verify that the LibTorch selected by the current environment exposes
CUDA and that every installed tool is on `PATH`:

```sh
cargo run -p engine-profile -- requirements
cargo build --profile profiling -p engine-profile
```

`requirements` reports the CUDA result from `tch`, plus `perf`,
`cargo-flamegraph`, `nsys`, and `ncu`. `engine-profile command <tool> ...`
performs the CUDA check and fails if the selected profiler is unavailable, so
it is the preflight command to use in automation.

## Workload contract

Each run creates a `workload.json` manifest in `--artifact-dir` before it
starts. The manifest records workload settings and device choice; external
tools write their raw artifacts in the same directory. Use a fresh directory
per capture. In particular, `iteration` creates a durable `training-run/`
there and deliberately refuses to reuse it, preventing an accidental resume
from turning a baseline into a different workload.

`inference` is a deterministic Connect Four 4x64 residual-network forward
pass with a 256-position CUDA batch. `training` runs the normal replay,
transfer, forward/backward, and optimizer path against an 8,192-sample fixed
replay. Its default is 80 training steps of 4,096 samples, with 256-sample
microbatches. `self-play` uses the production MCTS worker, dynamic batcher,
and 4x64 Connect Four network; it defaults to 200 games.

`iteration` is the high-value regression baseline. It runs the production
`TrainingRun` lifecycle—self-play, replay commit, 80 training steps,
checkpoint write, and inference reload—using
`experiments/benchmark-chess-canonical-h1.toml`. The profile crate overrides
only its explicit `--games` (default 200) and `--training-steps` (default 80)
arguments. The checked-in fixture retains its representative 16 self-play
threads, 800 PUCT simulations, 12x128 H1 chess network, FP16 inference, and
2,048-sample training batches.

This complete iteration is intentionally long. Use it for before/after
optimization baselines, not for every edit. Component workloads are for
isolating a suspected regression. `--warmup` and `--iterations` are counted
as whole workload passes; use them sparingly for the full iteration.

```sh
# A complete CUDA baseline. Pick a new output directory on every capture.
cargo run --profile profiling -p engine-profile -- \
  run iteration --artifact-dir artifacts/profiles/chess-h1-baseline

# Focused component captures; preserve default CUDA.
cargo run --profile profiling -p engine-profile -- \
  run inference --warmup 20 --iterations 1000 \
  --artifact-dir artifacts/profiles/inference
cargo run --profile profiling -p engine-profile -- \
  run training --artifact-dir artifacts/profiles/training
cargo run --profile profiling -p engine-profile -- \
  run self-play --games 200 --artifact-dir artifacts/profiles/connect4-self-play
```

Use `--device cpu` only for an intentional CPU capture. It is not a fallback
or a substitute for a CUDA baseline.

## Capturing CPU and GPU profiles

Generate commands instead of copying flags by hand. They create the artifact
directory and point outputs at it. Build the profiling profile first so native
symbols are retained.

```sh
# CPU samples and hardware-counter-compatible call stacks.
cargo run -p engine-profile -- command perf inference \
  --device cpu --warmup 20 --iterations 1000 \
  --artifact-dir artifacts/profiles/perf-inference

# SVG flamegraph; this wraps perf and rebuilds the profiling target.
cargo run -p engine-profile -- command flamegraph training \
  --device cpu --artifact-dir artifacts/profiles/flamegraph-training

# CPU/CUDA API/GPU timeline for the full baseline.
cargo run -p engine-profile -- command nsys iteration \
  --artifact-dir artifacts/profiles/nsys-chess-h1

# Kernel metrics after the Systems trace identifies a representative kernel.
cargo run -p engine-profile -- command ncu inference \
  --warmup 20 --iterations 100 \
  --artifact-dir artifacts/profiles/ncu-inference
```

The generated `perf` command writes `perf-<workload>.data`; inspect it with
`perf report`. The flamegraph command writes `flamegraph-<workload>.svg`.
Nsight Systems writes `nsys-<workload>.nsys-rep`, while Nsight Compute writes
`ncu-<workload>.ncu-rep`. Do not compare runtime under Nsight Compute with
ordinary benchmark timing: detailed counter collection changes execution.

For Nsight Compute, start with the generated one-launch capture to validate
permissions and identify a kernel. Then use the GUI or add a deliberately
chosen `--kernel-name` filter and matching launch skip/count based on the
Nsight Systems trace. Do not guess a logical warmup count as a CUDA kernel
skip: one model pass launches many kernels.

## System prerequisites

CUDA captures need a driver, a CUDA-enabled LibTorch/PyTorch runtime matched
to this repository's `tch` dependency, and an Nsight version compatible with
the installed driver/GPU. `nvidia-smi` confirms driver visibility, but only
the `tch` CUDA result in `engine-profile requirements` confirms the runtime
that this Rust binary dynamically loaded.

Nsight Compute additionally needs NVIDIA GPU performance-counter permission.
Nsight Systems timeline capture does not prove that permission. If an `ncu`
capture reports `ERR_NVGPUCTRPERM`, an administrator must enable access under
the NVIDIA driver's performance-counter security policy before kernel metrics
can be collected. On this host the Systems smoke capture succeeds, while NCU
currently reports that permission error. Keep using Systems for timeline work;
do not treat an empty NCU capture as a kernel profile.

```sh
# Point the build and execution environment at the same CUDA LibTorch.
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvidia-smi
nsys --version
ncu --version
perf --version
cargo flamegraph --help >/dev/null
```

Linux CPU sampling additionally requires permission for `perf` events. If
`perf record` is denied, have the system administrator adjust the relevant
kernel policy; do not replace counters with wall-clock guesses. Keep the GPU
clock/power state, CPU governor, driver, LibTorch, compiler, and profiler
versions alongside the raw artifacts. Commit concise derived summaries only;
do not commit `.nsys-rep`, `.ncu-rep`, `perf.data`, SVG flamegraphs, generated
checkpoints, or profiling run directories.

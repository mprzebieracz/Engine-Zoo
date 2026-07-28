# Profiling

`engine-profile` is a profiling-only crate. It contains deterministic Connect Four CUDA
workloads, but no timing harnesses or instrumentation in the production crates.

CUDA is the default device:

```sh
cargo run --profile profiling -p engine-profile -- run inference
cargo run --profile profiling -p engine-profile -- run training --iterations 100
```

Use `--device cpu` only when collecting a CPU-only profile. `engine-profile requirements`
reports CUDA and profiler availability, while `engine-profile command <tool> <workload>`
checks the selected profiler and prints a command for it.

```sh
cargo run -p engine-profile -- requirements
cargo run -p engine-profile -- command perf inference
cargo run -p engine-profile -- command flamegraph training
cargo run -p engine-profile -- command nsys inference
cargo run -p engine-profile -- command ncu training
```

## System prerequisites

For CUDA workloads, install a current NVIDIA driver and a CUDA-enabled LibTorch build that
matches the `tch` setup used by the repository. Verify the driver with `nvidia-smi` and CUDA
visibility with `engine-profile requirements`.

For CPU profiling on Linux, install the `linux-tools` package that matches the running kernel
to obtain `perf`. If `perf record` is denied, lower `kernel.perf_event_paranoid` according to
your distribution's security policy. `cargo install flamegraph` supplies the optional
`cargo-flamegraph` command.

For GPU profiling, install both NVIDIA Nsight Systems (`nsys`) and NVIDIA Nsight Compute
(`ncu`) from the CUDA toolkit or NVIDIA's profiler packages, and add their `bin` directories to
`PATH`. Nsight Systems captures CPU/GPU scheduling and CUDA API timelines; Nsight Compute is
for a single representative kernel after the timeline identifies it.

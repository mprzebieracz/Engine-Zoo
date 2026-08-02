# AlphaZero closure benchmark comparison point

The pre-finalization comparison commit is:

```text
efdd74d8c7e2701d330b066f4686df55b7e08457
```

Existing checked-in measurements remain the baseline. No benchmarks were run during this finalization pass because this computer has no CUDA support; no performance improvement is claimed.

On the provisioned CUDA benchmark host, compare the finalized code with the existing measurements for:

- native, Torch-TensorRT, and raw TensorRT latency and throughput across batch sizes;
- end-to-end H4 Root-Gumbel self-play games and positions per second;
- PUCT nodes per second and Root/Full-Gumbel time and allocations;
- training steps and samples per second, including H2D, forward/backward, and optimizer phases;
- raw TensorRT cold build versus automatic timing-cache reuse;
- repeated-move latency and allocations for persistent MCTS.

Record the tested commit, model/checkpoint, search and batching configuration, GPU, driver, CUDA, TensorRT, LibTorch, compiler identity, multiple-run distribution, and any regression above 2–3%.

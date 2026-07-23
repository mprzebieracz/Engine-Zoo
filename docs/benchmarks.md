# Benchmark methodology

Timing is evidence, not a unit-test assertion. Do not claim a performance improvement without recording machine, compiler, device/driver, configuration, median, spread, and workload.

The pure deterministic Connect Four search harness is `crates/search/benches/connect4_mcts.rs`. Compile it with `cargo bench -p search --bench connect4_mcts --no-run`; run it only on an intentional benchmark machine. Required future matrices cover PUCT leaf batches, Full Gumbel budgets, cold/hot caches, representative chess positions, arena allocations, and depth.

Representation/replay measurements should cover H1/H4/H8 construction/encoding/action round trips and replay batch sizes 256/1024/4096. GPU measurements should separately report forward-only, legal-logit gather, end-to-end inference, transfer, backward, optimizer, latency percentiles, and samples/s. No GPU benchmark was run as part of this change.


# Engine Zoo Post-Refactor Audit and Improvement Specification

**Repository reviewed:** `main(3).zip`  
**Audit date:** 2026-07-23  
**Scope:** current AlphaZero, search, game, training, evaluation, application, and web architecture.  
**Explicit non-goal:** do not implement alpha-beta or NNUE in this change set.

This document is intended to be given directly to Codex. It separates required correctness work from measured-performance work and portfolio polish. Codex should implement the phases in order and must not retain obsolete parallel paths after replacements are complete.

---

## 1. Executive assessment

The repository is substantially better than the previous revision. It now has a coherent architecture rather than an AlphaZero implementation embedded in a general engine project.

### Strong parts that should be preserved

- `engine_core`, `games`, `search`, `alphazero`, `app`, and `evaluations` have clear high-level responsibilities.
- MCTS is independent of `tch`.
- Native game moves are distinct from neural policy action indices.
- PUCT and Full Gumbel are separate search implementations.
- Full Gumbel now uses:
  - root Gumbel sampling;
  - sequential halving;
  - completed-Q transformation with the reference-style `value_scale = 0.1`;
  - improved-policy-deficit selection below the root.
- PUCT supports:
  - dynamic exploration constants;
  - FPU reduction and absolute FPU;
  - unscored virtual visits;
  - optional virtual loss;
  - root Dirichlet noise;
  - local leaf batching.
- Value perspective is represented by `PositionValue`, and backup sign flipping is centralized and tested.
- Search nodes are sparse, contiguous, arena-indexed, and currently 44 bytes for both Connect Four and chess moves on the recorded target.
- The evaluation cache is sharded and shares immutable entries through `Arc`.
- Chess has an authoritative game type, branch-local fixed-size AlphaZero states, repetition-aware search rules, exact draw handling, perft tests, and property tests.
- Scalar and WDL value heads share one training path.
- Replay stores compact game states and sparse policies, with full/fast search metadata and independent policy/value weights.
- Dynamic batching is separated from the `tch` inference implementation and already handles coalescing, splitting, bounded queues, shutdown, errors, and model reloads.
- Experiments have immutable configuration, mutable run state, atomic JSON updates, and model-format migration support.
- The frontend and evaluation infrastructure are sufficient foundations for a portfolio-quality demonstration.

### Current overall status

| Area | Assessment |
|---|---|
| Core architecture | Strong |
| Search representation | Strong |
| Chess rule correctness | Strong |
| Gumbel/PUCT separation | Good, but needs stronger golden tests |
| Training hot path | Functional but currently inefficient on CUDA |
| Model/replay lifecycle | Contains critical generation and persistence defects |
| Experiment configuration | Too restricted and partly misleading |
| Test breadth | Good for games, insufficient for search/training lifecycle |
| Portfolio presentation | Underdeveloped relative to the implementation |

The repository does not need another wholesale architecture rewrite. It needs several focused lifecycle fixes, a training-loop optimization pass, stronger algorithm tests, and removal of stale project artifacts.

---

# 2. Verification limitations

The review environment did not contain `rustc` or `cargo`, so the Rust workspace could not be compiled or executed. Python helper scripts passed syntax compilation. The frontend dependencies could not be installed in the review environment, so `npm run check` and `npm run build` were not executed.

Codex must begin by running, on a machine with the configured LibTorch installation:

```bash
cargo +nightly fmt --check
cargo +stable clippy --workspace --all-targets --all-features -- -D warnings
cargo +stable test --workspace --all-features
cargo +stable test --workspace --all-features --release
cargo +stable bench -p search --bench connect4_mcts
cd web
npm ci
npm run check
npm run build
```

Record the exact initial failures before changing code.

---

# 3. Required Phase 0: correctness defects before further training

These changes are mandatory before another long training run.

## 3.1 Preserve the `.safetensors` extension on temporary checkpoint files

### Problem

`RunDir::write_latest` currently writes to:

```rust
let temporary = latest.with_extension("safetensors.tmp");
```

For `latest.safetensors`, this produces a path whose final extension is `.tmp`. `tch::nn::VarStore` chooses its save/load format from the filename extension. The model can therefore be written using the non-safetensors format and then renamed to `.safetensors`, after which loading interprets it as safetensors.

The existing test only writes arbitrary bytes and therefore does not test a real `VarStore` round trip.

### Required implementation

In `crates/alphazero/src/experiment/run_dir.rs`:

1. Add one helper that creates a sibling temporary path while preserving `.safetensors` as the final extension.

Example result:

```text
latest.safetensors
latest.tmp.safetensors
```

Do not produce:

```text
latest.safetensors.tmp
```

2. Use the same helper for every model checkpoint that is written atomically.
3. Remove a stale temporary file before writing.
4. Rename the completed temporary file to the final path.
5. Keep JSON temporary files separate; their extension does not select a model serialization format.

Suggested helper:

```rust
fn temporary_model_path(final_path: &Path) -> anyhow::Result<PathBuf> {
    let stem = final_path
        .file_stem()
        .and_then(OsStr::to_str)
        .context("checkpoint path must have a UTF-8 file stem")?;
    Ok(final_path.with_file_name(format!("{stem}.tmp.safetensors")))
}
```

A non-UTF-8-safe implementation using `OsString` is preferable.

### Required tests

Add a CPU-only real model round-trip test:

1. Create a small `VarStore` and deterministic network.
2. Set or record known tensor values.
3. Call `RunDir::write_latest` with `vs.save`.
4. Assert the final path exists.
5. Assert no temporary file remains.
6. Create a second equivalent network and call `VarStore::load` on the final path.
7. Compare every named tensor exactly or within a strict tolerance.

This test must fail under the current `.tmp`-extension implementation.

---

## 3.2 Make evaluation-cache entries model-generation-aware

### Problem

`ChessSelfPlayWorkerFactory` owns one shared `EvalTable` for its entire lifetime. After every training iteration:

1. the training `VarStore` is saved;
2. the batcher reloads the updated weights;
3. the evaluation cache remains populated;
4. `RepresentedEvaluator::evaluation_key` still depends only on the encoded state.

A position evaluated by generation \(n\) can therefore be returned as a cache hit while self-play is using generation \(n+1\). This silently mixes old and new network evaluations and can corrupt every later generation.

Clearing the cache directly from the batcher would create inappropriate coupling between batching and MCTS. The cache key should include the evaluator namespace.

### Required architecture

Replace the raw `u64` evaluation key with a structured key:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EvaluationKey {
    pub state: u64,
    pub namespace: u64,
}
```

In `search::PolicyValueEvaluator`:

```rust
fn evaluation_key(&self, state: &G) -> Option<EvaluationKey>;
```

In `alphazero::EncodedEvaluator` add:

```rust
fn cache_namespace(&self) -> u64 {
    0
}
```

In the batcher:

- add `AtomicU64 evaluation_generation` to shared state;
- initialize it to zero when the batcher is created;
- increment it only after a successful weight reload;
- expose the current generation through `BatcherClient::cache_namespace`;
- use Acquire/Release or stronger ordering around successful reload publication.

In `RepresentedEvaluator`:

```rust
fn evaluation_key(&self, state: &G) -> Option<EvaluationKey> {
    Some(EvaluationKey {
        state: self.representation.encoded_state_key(state),
        namespace: self.encoded.cache_namespace(),
    })
}
```

Update `EvalTable` to store and compare the complete `EvaluationKey`.

Do not merely XOR the two values into one `u64`; preserving both fields avoids a deliberately introduced collision class and makes tests clearer.

### Required tests

1. The same state evaluated twice before a reload produces one backend evaluation and one cache hit.
2. The same state evaluated after a successful reload misses the old cache entry.
3. A failed reload does not publish a new namespace.
4. A new process/batcher can begin at namespace zero because its cache is empty.
5. Different encoded states in the same namespace remain distinct.
6. Cache statistics and search diagnostics report the correct hit/miss/backend counts.

---

## 3.3 Make self-play generations and seeds globally unique

### Problems

`SelfPlayConfig` currently contains `model_generation`, but the config is immutable and cloned into the worker factory once. `RunState.model_generation` changes after training, while replay samples continue to record the stale configuration value, normally zero.

`SelfPlayCoordinator::run` also assigns game IDs from `0..num_games` on every iteration. The experiment seed and game IDs therefore repeat across generations.

Additionally, `GameRequest::seed_for` includes `worker_id`. Which worker claims a game ID depends on scheduling, so runs with the same experiment seed and thread count are not strictly reproducible.

### Required architecture

Remove `model_generation` from `SelfPlayConfig`.

Add a per-call execution value:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelfPlayEpoch {
    pub model_generation: u64,
    pub first_game_id: u64,
}
```

Change:

```rust
SelfPlayCoordinator::run(factory, replay)
```

to:

```rust
SelfPlayCoordinator::run(factory, replay, epoch)
```

Extend `GameRequest`:

```rust
pub struct GameRequest {
    pub game_id: u64,          // global, never reused within a run
    pub model_generation: u64, // network that generated the position
    pub worker_id: usize,      // diagnostics only
    pub experiment_seed: u64,
}
```

Seed derivation must not depend on `worker_id`:

```rust
pub fn seed_for(self, purpose: u64) -> u64 {
    splitmix64(
        self.experiment_seed
            ^ splitmix64(self.game_id)
            ^ splitmix64(self.model_generation)
            ^ purpose,
    )
}
```

Add `total_games_generated: u64` to `RunState`. Increment the state format version and implement migration from the previous state format.

At each iteration:

```rust
let epoch = SelfPlayEpoch {
    model_generation: state.model_generation,
    first_game_id: state.total_games_generated,
};
let stats = coordinator.run(factory, replay, epoch)?;
state.total_games_generated += stats.games as u64;
```

Replay metadata must use `request.model_generation`, not immutable self-play configuration.

### Deterministic commit order

For exact reproducibility independent of thread scheduling, workers should not commit directly to the replay ring.

Use this flow:

```text
workers
  -> send (global_game_id, CompletedGame)
coordinator
  -> buffers out-of-order completions in BTreeMap
  -> commits complete trajectories in increasing game_id order
```

This makes:

- replay insertion order deterministic;
- progress output deterministic;
- ring-buffer eviction deterministic;
- one-thread and many-thread runs comparable.

Game-level channel overhead is negligible compared with playing the game.

### Required tests

1. Generation 0 and generation 1 never reuse a game ID.
2. Generation 0 and generation 1 produce different purpose seeds.
3. The same `(experiment seed, generation, game ID, purpose)` always produces the same seed.
4. Changing only `worker_id` does not change a game seed.
5. One-thread and four-thread scripted self-play produce identical exported replay order and metadata.
6. Sample metadata records the generation that actually generated it.
7. State migration from the previous format succeeds and documents how `total_games_generated` is initialized.

---

## 3.4 Correct checkpoint/replay resume semantics

### Problem

`RunState.replay_sample_count` is loaded from disk, but `ReplayBuffer` is always created empty after process restart. The state can therefore claim that replay samples were restored when they were not.

`optimizer_moments_restored` is always false, which is honest, but the resume behavior is not clearly surfaced to the user.

### Required immediate fix

Until replay persistence is implemented:

1. On startup, if a checkpoint is resumed but replay is not restored:
   - set the in-memory and next-persisted `replay_sample_count` to zero;
   - print a prominent warning;
   - log a structured `resume` record to `metrics.jsonl`.
2. Add:

```rust
pub enum ResumeKind {
    Fresh,
    WeightsOnly,
    Full,
}
```

to run-state/status reporting.
3. Do not imply a full training resume when only weights were restored.
4. Record that optimizer moments and replay were reset.

Replay persistence is specified in Phase 3.

---

## 3.5 Reject zero-game training iterations

`SelfPlayConfig::validate` checks threads and maximum moves but not `num_games`.

With `num_games = 0`, training can perform no update, increment the model generation, save the same weights, and reload them.

Add:

```rust
ensure!(self.num_games > 0, "self-play num_games must be positive");
```

Add a validation test.

---

# 4. Required Phase 1: search/evaluator correctness and diagnostics

## 4.1 Return typed errors for terminal roots

### Problem

Both PUCT and Gumbel:

- assert only `!game.is_terminal()`;
- call `SearchRules::enter_state` for the root;
- discard its `RuleResult`;
- perform network inference even if the external search rule declares the root terminal.

This violates the `SearchRules` contract and keeps a public panic at a recoverable API boundary.

### Required implementation

Extend `SearchError`:

```rust
pub enum SearchError {
    TerminalRoot { value: PositionValue },
    NoLegalMoves,
    Evaluation(EvaluationError),
}
```

At root initialization:

1. reset the path;
2. call `enter_state`;
3. combine the rule result with `game.terminal_value`;
4. if terminal, return `SearchError::TerminalRoot` before evaluator invocation;
5. if nonterminal but no legal moves are produced, return `NoLegalMoves`.

Add `Mcts::try_new`:

```rust
pub fn try_new(
    evaluator: E,
    config: SearchConfig,
    rules: R,
) -> Result<Self, SearchConfigError>;
```

The existing `new` may remain as a convenience wrapper only if its panic is clearly documented. Application and training code must use `try_new`.

### Required tests

- a root terminal through native game state invokes no evaluator;
- a root terminal only through `SearchRules` invokes no evaluator;
- PUCT and Gumbel behave identically at terminal roots;
- application callers display a normal error rather than panicking.

---

## 4.2 Stop silently converting non-finite network output into draws

### Problem

`PositionValue::new_clamped` maps:

- `NaN` to draw;
- positive infinity to win;
- negative infinity to loss.

`build_policy` and Gumbel softmax also fall back to uniform probabilities after invalid numeric input.

The evaluator API can already return `EvaluationError`, so invalid network output should not be hidden. A NaN during training or inference is a failed run, not a neutral chess evaluation.

### Required implementation

Extend `EvaluationError`:

```rust
pub enum EvaluationError {
    Message(String),
    ResultCardinality { expected: usize, actual: usize },
    LogitCardinality { row: usize, expected: usize, actual: usize },
    NonFiniteLogit { row: usize, index: usize },
    InvalidValue { row: usize, value: f32 },
}
```

At the evaluator boundary validate:

- one result per requested state;
- one logit per legal move;
- every logit is finite;
- every scalar value is finite and in `[-1, 1]`.

Construct model values through `PositionValue::new`.

Restrict clamping to operations where a finite rounding error is expected. Replace `new_clamped` with one of:

```rust
PositionValue::new(value)?
PositionValue::from_finite_clamped(value) // rejects NaN/inf first
```

The second helper may be used for arithmetic averages and transformed values, but it must reject non-finite inputs.

Once logits are validated, ordinary softmax should not need a uniform fallback. Keep explicit handling only for a logically empty distribution, which should already be a typed search error.

### Required tests

- NaN value is an evaluation error;
- infinite value is an evaluation error;
- NaN/infinite logits are evaluation errors;
- wrong per-row logit counts are typed errors;
- an evaluator error clears every outstanding visit reservation;
- no invalid value is written into a node or replay sample.

---

## 4.3 Make search diagnostics reflect actual backend work

### Problem

`SearchDiagnostics.network_evaluations` is initialized to one for the root and then incremented by `unique_games.len()`. Cache hits are counted as network evaluations, and `cache_hits` is never populated.

`SelfPlayStats.tt_hits`, `tt_misses`, and `tt_inserts` are also never populated and use “TT” terminology for a neural evaluation cache.

### Required implementation

Make `evaluate_positions` return internal evaluation statistics:

```rust
struct EvaluationBatchStats {
    requested_states: usize,
    duplicate_requests: usize,
    cache_hits: usize,
    backend_evaluations: usize,
}
```

Return evaluations and stats together.

Update root and leaf diagnostics from these exact values.

Rename:

```rust
network_evaluations -> backend_evaluations
tt_hits             -> evaluation_cache_hits
tt_misses           -> evaluation_cache_misses
tt_inserts          -> evaluation_cache_inserts
```

Aggregate into `SelfPlayStats`:

- completed simulations;
- backend evaluations;
- cache hits;
- duplicate leaves;
- maximum depth;
- full/fast searches;
- resignations.

Do not derive per-game insertion counts by subtracting a globally shared concurrent cache counter. Search-local hits/backend evaluations are deterministic and sufficient. Lifetime cache insertion totals can remain available from `EvalTable::stats`.

### Required tests

- root cache hit reports zero backend evaluations;
- duplicated leaves within one local PUCT batch count once;
- a cache miss followed by a hit produces exact counters;
- Gumbel reports its sequential backend use correctly;
- self-play aggregation equals the sum of individual search diagnostics.

---

## 4.4 Strengthen PUCT tests

The current PUCT integration tests cover a one-ply terminal win and configuration validation, but the selection implementation is complex enough to require direct golden tests.

Add deterministic tests for:

1. dynamic \(c_{\text{puct}}\);
2. absolute FPU at root and non-root;
3. FPU reduction using visited prior mass;
4. root-specific FPU reduction;
5. unscored virtual visits:
   - affect selection visits;
   - do not alter completed Q;
6. virtual loss:
   - uses the configured value;
   - is completely removed after backup/cancellation;
7. root Dirichlet noise:
   - deterministic under fixed seed;
   - only applied in exploratory mode;
   - never applied below the root;
8. a two-ply tactical/defensive game that detects incorrect sign orientation;
9. local leaf batches of 1 and \(N\):
   - produce legal normalized policies;
   - do not duplicate backend work for identical leaves;
10. cache hits and evaluator failures leave no in-flight visits.

Avoid asserting exact equality between sequential and batched policies; asynchronous reservation changes are expected. Assert invariants and tactical ordering.

---

## 4.5 Add Full Gumbel golden fixtures

The current code has useful unit tests for mixed value, value rescaling, and one-action schedules, but it does not establish end-to-end equivalence with a reference implementation.

Create a checked-in fixture file generated once from DeepMind Mctx or an independent reference script. The Rust tests must not depend on Python or JAX at test time.

Fixtures should cover:

- considered-action truncation;
- visit schedules for awkward counts such as 3, 5, 7, 13, and 16;
- completed-Q values with visited and unvisited actions;
- mixed-value interpolation;
- value rescaling;
- transformed Q;
- interior improved-policy probabilities;
- interior deficit selection over several visits;
- root survivor sequence;
- final selected root action;
- final improved-policy target.

Use tolerances, not stringified exact floats.

Also add a test that Full Gumbel always uses one in-flight visit and never accidentally calls the PUCT selector.

---

## 4.6 Clean minor search implementation debt

Perform these small cleanups while touching search:

- remove the duplicate `batch.unique_games.clear()` call;
- replace the linear `find_leaf_result` only if benchmarks show meaningful cost; local leaf batches are normally small;
- make cache shard and slot selection use independent hash portions or a secondary mix;
- document whether cache atomic counters are enabled in production; if contention appears in profiling, make detailed cache metrics optional;
- preserve the current 44-byte node unless a benchmark demonstrates a better layout;
- do not implement DAG search or LC3 in this phase.

---

# 5. Required Phase 2: training hot-path optimization

This phase should be implemented before AMP, TensorRT, or a larger network. The current code performs avoidable synchronization and transfer work on every microbatch.

## 5.1 Remove CUDA synchronization from every microbatch

### Problem

Inside each microbatch the trainer calls:

```rust
numerator.double_value(&[])
```

for policy and value metrics.

On CUDA, reading a scalar onto the host synchronizes queued device work. With a batch of 4096 and microbatch size 256, this can cause up to 32 host/device synchronization points per optimizer step.

### Required implementation

Keep backward calls per microbatch, but accumulate detached metric tensors on the device:

```rust
let policy_metric = Tensor::zeros([], (Kind::Float, device));
let value_metric = Tensor::zeros([], (Kind::Float, device));

for microbatch in ... {
    let numerator = ...;
    policy_metric += numerator.detach();
    loss.backward();
}

let policy_numerator = policy_metric.double_value(&[]);
let value_numerator = value_metric.double_value(&[]);
```

Use one host read per metric per optimizer step, or only read on configured logging intervals.

Do not retain all microbatch computation graphs.

### Required test

On CPU, compare the reported losses against the current reference formulas for:

- one microbatch;
- several microbatches;
- zero policy-weight samples;
- scalar value head;
- WDL value head.

---

## 5.2 Transfer each sampled batch to the device once

### Problem

For every microbatch, the trainer separately transfers:

- states;
- outcomes;
- policy weights;
- value weights;
- sparse policy actions;
- sparse policy probabilities;
- sparse policy row indices.

The full sampled input is small relative to a 16 GB GPU because activations remain microbatched. For chess history 4, 4096 FP32 input states are roughly 66 MB.

### Required implementation

Add:

```rust
struct DeviceReplayBatch {
    states: Tensor,
    actions: Tensor,
    probabilities: Tensor,
    rows: Tensor,
    outcomes: Tensor,
    policy_weights: Tensor,
    value_weights: Tensor,
    offsets: Vec<i64>,
}
```

At the start of an optimizer step:

1. transfer each tensor once to the target device;
2. retain policy offsets on the CPU;
3. use `narrow` views for each microbatch;
4. subtract `start` from the selected sparse row indices;
5. do not call `.to_device` inside the microbatch loop.

Keep the old per-microbatch path only as a temporary benchmark comparison, then delete it after equivalence is established.

### Required tests

- device-batch slicing selects exactly the same sparse entries as the CPU reference;
- policy rows are rebased correctly;
- microbatch and unmicrobatched losses agree;
- gradients and one optimizer update agree within tolerance on CPU.

---

## 5.3 Make replay sampling deterministic

### Problem

`ReplayBuffer::sample` calls:

```rust
index::sample(&mut rand::rng(), ...)
```

This ignores the experiment seed and cannot be reproduced after a checkpoint.

### Required implementation

Change sampling to accept an RNG:

```rust
pub fn sample<R, Rep>(
    &self,
    batch_size: usize,
    representation: &Rep,
    rng: &mut R,
) -> Option<ReplayBatch>
where
    R: rand::Rng + ?Sized;
```

Derive each optimizer-step seed from:

- experiment seed;
- global optimizer step;
- a dedicated replay-sampling purpose constant.

The same run state and replay contents must produce the same next batch.

Do not serialize an opaque thread RNG if a counter-derived seed is sufficient.

### Required tests

- equal seed and replay contents produce equal sample indices/tensors;
- different global steps produce different samples;
- resume at the same global step reproduces the next batch.

---

## 5.4 Add bounded replay prefetch

Self-play and training are currently sequential, which is appropriate on one GPU. During the training phase, however, CPU replay sampling and representation encoding can overlap the previous GPU optimizer step.

Add a scoped producer thread with a bounded channel of one or two `ReplayBatch` values:

```text
sampler/encoder thread -> bounded channel -> GPU trainer
```

Requirements:

- deterministic step-derived seeds;
- no detached background thread;
- all errors propagated;
- clean shutdown when training stops;
- bounded memory;
- configurable depth, default 1;
- CPU-only tests with a scripted slow sampler proving ordering and shutdown.

Do not overlap self-play inference and gradient training in this phase.

---

## 5.5 Reuse batcher worker scratch allocations

### Problem

Every inference pass currently creates:

```rust
let mut items = Vec::new();
let mut combined = CombinedEncodedBatch::new();
```

The individual request batches are reused, but the worker-level coalescing vectors are not.

### Required implementation

Let `run_worker` own reusable:

```rust
let mut work_items = Vec<WorkItem>::new();
let mut combined = CombinedEncodedBatch::new();
```

Each pass must:

- drain `work_items`;
- clear `combined` while retaining capacity;
- reserve based on the rows selected for the pass;
- pass borrowed scratch to the backend and finish logic;
- avoid `mem::take` that destroys retained capacity.

Add pointer/capacity tests with the mock backend to prove buffers are reused after warm-up.

---

## 5.6 Fix model-reload ordering in the batcher

### Problem

Evaluation tasks and reload requests are held in separate queues. `next_work` prioritizes reloads, so a reload can overtake evaluations submitted earlier. The training loop normally reloads after self-play has joined, but the public “reload is a barrier” contract is not generally correct.

### Required architecture

Use one ordered command queue:

```rust
enum Command {
    Evaluate(Task),
    Reload(Reload),
}
```

The worker may coalesce adjacent `Evaluate` commands up to the batch limit, but it must not cross a `Reload`.

A split task must remain before the following reload until all of its rows finish.

Semantics:

```text
evaluate A
evaluate B
reload
evaluate C
```

must guarantee:

- A and B use old weights;
- reload completes;
- C uses new weights.

### Required tests

Use a backend whose outputs include its current generation:

1. submit old-generation requests;
2. enqueue reload;
3. submit new-generation requests;
4. assert exact generation assignment;
5. repeat with a split oversized request before the reload;
6. assert failed reload unblocks all waiters and does not publish a cache namespace.

---

## 5.7 Add useful throughput metrics

Expand `BatcherStats` and training metrics with:

### Batcher

- total states;
- total inference batches;
- mean batch size;
- partial batch count;
- split request count;
- queue wait total/max;
- coalescing wait total;
- backend execution total/max;
- reload count;
- current cache namespace.

Rename lifetime maxima so `BatcherStats::since` does not pretend they are interval deltas.

### Self-play

- games/s;
- positions/s;
- completed simulations/s;
- backend evaluations/s;
- mean inference batch;
- cache hit rate;
- duplicate-leaf rate;
- mean/max search depth;
- full/fast search count.

### Training

- replay sampling/encoding time;
- host-to-device transfer time;
- forward/backward time;
- optimizer step time;
- samples/s;
- policy/value loss;
- replay reuse ratio;
- current learning rate.

Logging should remain structured JSONL. Console output should be a concise view of the same counters.

---

# 6. Required Phase 3: experiment and model configuration cleanup

## 6.1 Make experiment files the primary configuration interface

### Problems

The training CLI constructs an experiment through a restricted set of flags:

- canonical chess is always WDL;
- canonical chess always uses a 12x128 SE trunk;
- `--blocks` is silently ignored for canonical chess;
- `--channels` changes the WDL hidden layer but not the hard-coded 128 trunk;
- the default search remains PUCT;
- Full Gumbel is not selectable from the current CLI;
- inference precision and batcher parameters are hard-coded;
- the checked-in `scripts/train_chess.py` passes many flags the current CLI does not accept.

This is both an experiment correctness problem and a portfolio usability problem.

### Required CLI shape

Use a two-stage interface:

```bash
engine-zoo-train init \
  --experiment examples/experiments/chess-gumbel.toml \
  --run-dir data/runs/chess-gumbel

engine-zoo-train run \
  --run-dir data/runs/chess-gumbel \
  --iterations 10 \
  --device cuda
```

Or retain the `train` binary name with subcommands:

```text
train init
train run
train inspect
```

The immutable experiment file must define all algorithmic and model behavior. Runtime-only flags may override:

- device;
- number of iterations;
- forever;
- log verbosity;
- profiling mode.

Do not allow ordinary CLI flags to silently disagree with an existing immutable experiment.

Print the fully resolved experiment at startup and store it in the run directory.

### Example configuration coverage

Check in:

```text
examples/experiments/connect4-puct-small.toml
examples/experiments/chess-puct-wdl.toml
examples/experiments/chess-gumbel-wdl.toml
examples/experiments/chess-canonical-scalar-ablation.toml
examples/experiments/chess-classic-compat.toml
```

---

## 6.2 Parameterize modern chess network construction

Replace:

```rust
ModelSpec::chess_se(history, value_head)
```

with either a full constructor:

```rust
ModelSpec::chess_se(
    history,
    SeTrunkSpec {
        blocks,
        channels,
        se_hidden,
    },
    ChessPolicyHeadSpec { ... },
    value_head,
)
```

or named presets that resolve to a complete `ModelSpec`.

Do not make the model a free-form invalid combination. Preserve validation rules:

- classic chess compatibility remains frozen;
- canonical chess uses its canonical representation and action layout;
- every model spec resolves to one exact architecture;
- model identity includes all dimensions that affect checkpoint compatibility.

Support at least:

- canonical scalar value head;
- canonical WDL value head;
- configurable blocks/channels/SE width;
- existing classic scalar compatibility.

Add model fingerprinting:

```rust
pub struct ModelFingerprint([u8; 32]);
```

It should be derived from canonical serialized `ModelSpec` and stored with checkpoints/replay metadata.

---

## 6.3 Separate search budget from search execution configuration

`SearchBudget::Puct` currently includes `leaf_batch_size`, but `set_budget` rejects any value different from the fixed MCTS configuration.

A per-move budget should describe the amount of search, not the execution scheduler.

Change to:

```rust
pub enum SearchBudget {
    Puct {
        simulations: usize,
    },
    Gumbel {
        simulations: usize,
        max_considered_actions: usize,
    },
}
```

Keep `leaf_batch_size`, FPU, virtual visits, and PUCT constants in `PuctConfig`.

This removes a misleading field and simplifies playout-cap randomization.

---

## 6.4 Add explicit inference/runtime configuration

Add to `ExperimentConfig` or a separate immutable run configuration:

```rust
pub struct InferenceConfig {
    pub precision: InferencePrecision,
    pub preferred_batch_size: usize,
    pub max_batch_size: usize,
    pub max_wait: DurationConfig,
    pub max_queued_states: usize,
}
```

The current `Batcher::new_with_model` always selects FP32 even though the backend supports FP16.

Do not select TensorRT now. Keep `InferenceBackend` as the extension boundary.

---

## 6.5 Make optimizer and learning-rate behavior explicit

Current training always uses Adam with a fixed learning rate.

Add serializable configuration:

```rust
pub enum OptimizerSpec {
    Adam { beta1: f64, beta2: f64, epsilon: f64 },
    // Add SGD/AdamW only if supported cleanly by the pinned tch version.
}

pub enum LearningRateSchedule {
    Constant,
    LinearWarmupThenCosine {
        warmup_steps: u64,
        total_steps: u64,
        minimum_lr: f64,
    },
    Piecewise {
        boundaries: Vec<(u64, f64)>,
    },
}
```

Do not change the default optimizer solely on theoretical grounds. Make alternatives measurable.

Record the effective learning rate in every metrics entry.

AMP remains optional after the required training hot-path fixes. If added:

- keep FP32 as the reference mode;
- use `tch::autocast`;
- implement a tested Rust gradient scaler;
- save scaler state;
- reject non-finite gradients;
- compare loss curves and throughput against FP32.

---

# 7. Recommended Phase 4: replay and training-state persistence

This is strongly recommended before multi-day training, but it may follow the correctness and hot-path work.

## 7.1 Persist self-play generations, not only an opaque ring dump

The project is explicitly interested in observing how the engine learns. Persist each completed self-play generation in a versioned format:

```text
run/
  selfplay/
    generation-000000.bin.zst
    generation-000001.bin.zst
```

Each chunk should contain:

- format version;
- model fingerprint;
- model generation;
- game ID range;
- representation fingerprint/version;
- search configuration fingerprint;
- samples/trajectories;
- checksums.

Benefits:

- replay reconstruction after restart;
- offline training experiments;
- inspection of play style over time;
- policy/value debugging;
- training-dashboard data;
- architecture ablations on compatible data.

Use a stable binary codec and compression. Do not serialize raw `tch::Tensor` objects.

Add a representation-specific stable replay codec if game states cannot derive portable serialization directly.

## 7.2 Reconstruct the replay window on resume

On startup:

1. verify model/representation compatibility;
2. load recent self-play chunks newest-to-oldest until capacity is filled;
3. restore deterministic oldest-to-newest ring order;
4. set `replay_sample_count` from actual loaded entries;
5. report `ResumeKind::Full` only if replay and required training state were restored.

## 7.3 Optimizer persistence

If `tch` still does not expose sufficient optimizer serialization, choose explicitly between:

- implementing the required Adam state in a small project-owned optimizer wrapper; or
- documenting and logging that resumes are weights+replay with reset optimizer moments.

Do not claim exact continuation without optimizer moments.

A custom optimizer is welcome only after the ordinary trainer is benchmarked and stable.

---

# 8. Portfolio and repository cleanup

These changes are not algorithmic, but the current repository undersells its implementation and contains stale artifacts.

## 8.1 Remove repository debris

Delete from the repository root:

- `main.zip`;
- the generated Fastchess `config.json` containing local run paths and match statistics.

Move the 72 KB implementation plan:

```text
architecture-refactor-plan.md
```

to either:

```text
docs/archive/architecture-refactor-plan-2026-07.md
```

or replace it with concise architecture decision records under:

```text
docs/adr/
```

The main branch should present the resulting architecture, not lead with old migration debt.

## 8.2 Fix or remove stale helper scripts

`scripts/train_chess.py` currently passes flags that the current Rust `train` binary does not accept, including old architecture, mode, batching, Gumbel, and checkpoint flags.

After the experiment-file CLI is implemented:

- replace the script with a thin wrapper around one checked-in example experiment; or
- remove it and document the direct Rust command.

Review every Python script against current `--help` output in CI.

`crates/evaluations/install_stockfish.py` must validate extracted archive paths and reject absolute paths, `..`, and unsafe links before extraction.

## 8.3 Add an actual project README

The current README is too short for the implementation.

The new README should include:

1. one-sentence project statement;
2. screenshots or a short GIF of the web UI;
3. feature matrix:
   - Chess/Connect Four;
   - PUCT/Full Gumbel;
   - scalar/WDL;
   - self-play/training;
   - UCI;
   - evaluation suites;
   - web analysis;
4. architecture diagram;
5. explanation of why Rust is used;
6. quick CPU Connect Four demo requiring no CUDA;
7. chess training setup and LibTorch requirements;
8. experiment configuration example;
9. correctness/test strategy;
10. benchmark methodology and a clearly dated table;
11. repository layout;
12. roadmap;
13. license and third-party notices.

Correct the current `data/suites/` path if suites actually live under `crates/evaluations/suites`.

## 8.4 Add architecture and research documentation

Create:

```text
docs/architecture.md
docs/search/puct.md
docs/search/full-gumbel.md
docs/training.md
docs/chess-representation.md
docs/reproducibility.md
docs/benchmarks.md
docs/testing.md
```

The search documents should define:

- value perspective;
- backup equations;
- FPU;
- virtual visits/loss;
- root noise;
- completed-Q;
- Gumbel sequential halving;
- policy targets;
- known parallelism tradeoffs.

These documents should link to the exact Rust modules.

## 8.5 Add licensing and metadata

There is currently no root license.

Choose and add an explicit license. Because the frontend uses Chessground, document its GPL-3.0-or-later obligations and decide whether:

- the whole repository is GPL-compatible; or
- Rust crates and the web frontend have separate clearly documented licenses.

Add workspace metadata:

```toml
[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "..."
license = "..."
repository = "..."
authors = ["..."]
```

Update crate descriptions. In particular, `checkpoint-eval` currently describes itself as “Background Stockfish evaluation” while its README says evaluation is explicit and has no watcher.

Consider renaming it to `engine-evaluations` for consistency, but do not perform broad crate renaming unless it improves the public interface materially.

## 8.6 Improve CI

Split CI into focused jobs:

### Lightweight Rust

No LibTorch:

```bash
cargo test -p engine_core -p games -p search
cargo clippy -p engine_core -p games -p search --all-targets -- -D warnings
```

### ML workspace

Install the pinned compatible CPU PyTorch/LibTorch and run:

```bash
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Use explicit toolchains (`cargo +stable`, `cargo +nightly`) rather than relying on which setup action last changed the default.

### Release correctness

```bash
cargo test -p games -p search --release
cargo bench -p search --bench connect4_mcts --no-run
```

### Frontend

```bash
npm ci
npm run check
npm run build
```

### Helpers and hygiene

```bash
python -m compileall scripts crates/evaluations/install_stockfish.py
cargo audit
cargo deny check
```

Add coverage for the pure Rust crates with `cargo llvm-cov` if CI duration is acceptable.

## 8.7 Centralize LibTorch build support

Three near-identical `build.rs` files currently implement local LibTorch rpath behavior and assume `$HOME/libs/libtorch` when `LIBTORCH` is absent.

Consolidate the logic into one checked-in build-support module or build dependency, and:

- use `LIBTORCH` only when explicitly set;
- support `LIBTORCH_USE_PYTORCH`;
- avoid silently inventing a machine-specific path;
- document Linux/macOS behavior;
- keep GNU linker flags Linux-only.

Do not introduce a new runtime service merely for this.

## 8.8 Add a training-progress view

This is optional but high-value for the stated project goals and portfolio.

Expose read-only run endpoints:

```text
GET /api/runs
GET /api/runs/:run/state
GET /api/runs/:run/metrics
GET /api/runs/:run/selfplay/:generation/summary
```

Add a frontend route showing:

- model generation;
- games and positions generated;
- policy/value loss;
- self-play throughput;
- inference batch size;
- cache hit rate;
- W/D/L and game length over generations;
- selected example games;
- policy/value evolution on fixed probe positions.

Use the existing JSONL metrics and persisted generation chunks. Do not couple the trainer to the web server.

---

# 9. Benchmark suite required before speculative optimization

Keep the existing deterministic Connect Four search harness and add the following.

## 9.1 Pure search benchmarks

With scripted/dummy evaluators:

- PUCT at leaf batch 1, 2, 4, 8, 16;
- Full Gumbel at 32, 64, 128 simulations;
- cache disabled, cold cache, hot cache;
- Connect Four;
- representative chess positions;
- node allocations and peak arena length;
- simulations/s;
- duplicate-leaf rate;
- maximum depth.

No timing assertions in unit tests.

## 9.2 Representation and replay benchmarks

- chess H1/H4/H8 state construction;
- state encoding;
- policy action round trips;
- replay insertion;
- replay sampling/encoding at batch 256, 1024, 4096;
- replay prefetch throughput.

## 9.3 Inference benchmark binary

Add a CLI that emits JSON for:

- device;
- precision;
- model fingerprint;
- batch sizes 1, 2, 4, 8, 16, 32, 64, 128, 256;
- forward-only time;
- end-to-end evaluator time;
- legal-logit gather time;
- positions/s;
- latency percentiles.

Use actual legal-action distributions from stored self-play data in addition to synthetic batches.

## 9.4 Training benchmark binary

Measure:

- sampling/encoding;
- transfer;
- forward;
- backward;
- optimizer;
- complete samples/s;
- old versus optimized transfer path during migration;
- FP32 versus AMP only after AMP exists.

## 9.5 End-to-end learning ablations

When GPU access returns, compare configurations at equal wall time:

- PUCT 64/128/256;
- PUCT leaf batch 1/4/8/16;
- Full Gumbel 32/64/128;
- scalar versus WDL;
- network sizes;
- cache on/off;
- full-only versus playout-cap randomized training.

Record both:

- strength after equal GPU-hours;
- systems throughput.

Do not choose an algorithm solely from evaluations/s.

---

# 10. Tests required for the final AlphaZero foundation

The following matrix is the minimum acceptance suite.

## Game correctness

- Connect Four random games match an independent oracle.
- Chess start and standard perft positions.
- Castling, en passant, promotion, checkmate, stalemate.
- Exact 50-move threshold and reset conditions.
- Exact threefold identity:
  - side to move;
  - castling rights;
  - en-passant availability;
  - irreversible-history reset.
- Insufficient material policy.
- FEN clocks and history.
- Random legal playout invariants.

## Representation correctness

- every legal move maps to a unique in-range action;
- action-to-move round trip for all legal moves;
- perspective/color symmetry fixtures;
- history planes;
- repetition plane;
- halfmove/fullmove/global planes;
- encoded-state cache-key equality exactly follows encoded input equality;
- classic compatibility layout remains frozen.

## Search correctness

- typed terminal root;
- one-ply forced outcome;
- two-ply defensive sign test;
- deeper odd/even backup;
- terminal leaves bypass evaluator;
- failed evaluation cancels reservations;
- cache-generation namespace;
- cache and duplicate diagnostics;
- PUCT FPU/virtual-visit/root-noise fixtures;
- Full Gumbel reference fixtures;
- legal normalized root policy;
- deterministic fixed-seed behavior.

## Batcher correctness

- coalescing;
- timeout flush;
- maximum split;
- exact result reassembly;
- reusable allocations;
- bounded queue/backpressure;
- strict reload ordering;
- successful generation publication;
- failed reload behavior;
- backend error fan-out;
- shutdown/drop unblocks clients;
- concurrent stress test.

## Replay/training correctness

- ring order and eviction;
- deterministic sampling;
- sparse duplicate merge/normalization;
- full/fast weights;
- outcome perspective over skipped/fast plies;
- scalar and WDL exact losses;
- zero policy weight;
- microbatch equivalence;
- device-batch slicing;
- one optimizer update equivalence;
- checkpoint real-format round trip;
- resume-state truthfulness;
- replay persistence compatibility when implemented.

## Application/portfolio correctness

- every documented command appears in `--help`;
- example experiments validate;
- helper scripts invoke valid arguments;
- frontend type check/build;
- UCI smoke test;
- API session smoke test;
- evaluation command construction and path-safety tests.

---

# 11. Deletions and migrations

Codex must remove replaced debt rather than stack new layers over it.

Delete or migrate:

- `SelfPlayConfig.model_generation`;
- per-iteration game IDs restarting at zero;
- worker-dependent game randomness;
- raw `u64` cache keys without evaluator namespace;
- `.safetensors.tmp` final-extension handling;
- misleading `tt_*` statistics;
- `SearchBudget::Puct.leaf_batch_size`;
- per-microbatch host scalar reads;
- per-microbatch repeated device transfers;
- stale `scripts/train_chess.py` argument mapping;
- root `main.zip`;
- generated root Fastchess `config.json`;
- duplicated batcher `.name("batcher")` call;
- duplicate `batch.unique_games.clear()` call;
- obsolete README commands and paths.

Do not leave compatibility aliases unless a checked-in historical run or checkpoint requires them and a test protects that requirement.

---

# 12. Explicit non-goals

Do not implement in this specification:

- alpha-beta;
- NNUE;
- DAG/graph MCTS;
- full LC3 repository/event architecture;
- distributed self-play;
- simultaneous training and inference on the same GPU;
- TensorRT;
- transformer networks;
- custom chess move generation;
- a universal dynamically dispatched engine/search abstraction.

The existing interfaces should leave room for these without prematurely implementing them.

---

# 13. Implementation order and merge gates

## Phase A — training blockers

1. safetensors temporary path;
2. evaluation cache namespace;
3. self-play generation/global game IDs/seeds;
4. zero-game validation;
5. truthful resume state.

**Gate:**

```bash
cargo test -p alphazero -p search -p engine_app
```

plus a real CPU checkpoint round trip.

## Phase B — search correctness

1. typed terminal roots;
2. strict numeric validation;
3. exact diagnostics;
4. PUCT fixtures;
5. Full Gumbel golden fixtures.

**Gate:**

```bash
cargo test -p search --release
cargo test -p games --release
```

No search test may rely on a real neural network.

## Phase C — training performance

1. device-side metric accumulation;
2. one transfer per sampled batch;
3. deterministic replay RNG;
4. bounded prefetch;
5. reusable batcher scratch;
6. ordered reload queue;
7. metrics.

**Gate:**

- CPU loss/gradient equivalence tests;
- batcher stress tests;
- benchmark report before/after;
- no regression greater than 5% in the existing pure-search baseline unless explained.

## Phase D — experiment interface

1. experiment-file-first CLI;
2. parameterized model specs;
3. inference config;
4. budget cleanup;
5. optimizer/scheduler config;
6. example experiments;
7. update/remove scripts.

**Gate:**

Every example validates and can create a run directory without CUDA.

## Phase E — persistence and portfolio

1. self-play generation chunks;
2. replay restore;
3. README/docs/license;
4. CI split;
5. repository cleanup;
6. optional training dashboard.

**Gate:**

A new developer can:

1. clone the repository;
2. run Connect Four on CPU;
3. run all lightweight tests;
4. understand PUCT versus Gumbel from the docs;
5. open the frontend;
6. inspect a sample run without local absolute paths.

---

# 14. Definition of done

The AlphaZero portion is considered a stable foundation when:

- checkpoints round-trip in their declared format;
- cached evaluations can never survive a network-generation change;
- every replay sample has correct global game ID and model generation;
- fixed seeds are independent of thread scheduling;
- no non-finite model output is silently converted into a plausible search value;
- PUCT and Full Gumbel have direct deterministic reference tests;
- chess draw rules and value perspectives remain covered;
- training performs no host scalar synchronization inside the microbatch loop;
- each sampled batch is transferred to the GPU once;
- replay sampling is deterministic;
- experiment files can express scalar/WDL and PUCT/Gumbel models without source edits;
- checked-in helper scripts match the actual CLI;
- the repository contains no generated archives or machine-specific arena state;
- CI checks Rust, frontend, helpers, and release-mode pure search;
- README and architecture documentation accurately present the implemented system.

After these gates, the codebase is clean enough to pause AlphaZero infrastructure work and proceed independently with the UI or a new alpha-beta crate.

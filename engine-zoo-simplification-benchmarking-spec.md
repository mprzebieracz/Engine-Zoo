# Engine Zoo Architecture Simplification, Benchmarking, and Profiling Specification

**Repository reviewed:** `main(5).zip`  
**Purpose:** simplify the current AlphaZero architecture without losing supported features or hot-path performance, then establish a reproducible benchmark and profiling program on the available GPU machine.  
**Primary implementation consumer:** Codex.  
**Scope:** `engine_core`, `games`, MCTS, AlphaZero model/inference/self-play/training, application assembly, benchmark tooling, and documentation.  
**Explicit non-goals:** alpha-beta, NNUE, UI redesign, DAG search, LC3, distributed self-play, and new neural architectures beyond the currently supported model families.

---

# 1. Executive decision

The existing crate boundaries are fundamentally sound and should remain:

```text
engine_core
    ↓
games       search
    \       /
     alphazero
         ↓
   app / evaluations / web
```

Do **not** merge the workspace into one crate. The current separation gives useful dependency control:

- game rules do not depend on neural code;
- MCTS does not depend on `tch`;
- AlphaZero owns neural and training concerns;
- binaries and UI sit at the edge.

The main complexity is inside and above those boundaries:

1. the model configuration exposes more combinations than the implementation really supports;
2. runtime assembly is duplicated across training, UCI, HTTP sessions, analysis, and CLI agents;
3. chess model/history dispatch appears in several places;
4. generic and chess-specific self-play duplicate most of the same game loop;
5. search budget is stored both in the algorithm configuration and the per-move budget schedule;
6. the current search API supports only PUCT and Full Gumbel, while the desired explicit root-Gumbel/interior-PUCT variant is absent;
7. the `alphazero` crate has a very large flat re-export surface;
8. the main `train` binary owns model loading, inference construction, typed runtime dispatch, replay, self-play, training, checkpointing, and metric logging;
9. benchmarking is currently two small ad hoc binaries rather than one reproducible system;
10. profiling is not yet organized around named workloads and repeatable commands.

The target is therefore:

> Keep the layers, reduce the number of concepts visible at each layer, centralize runtime assembly, and preserve static dispatch inside search, representation, self-play, and training loops.

The final top-level flows should be readable in one screen.

## Training

```rust
fn run(args: RunArgs) -> Result<()> {
    let mut run = TrainingRun::open(&args.run_dir, args.device)?;

    while args.forever || run.completed_iterations() < args.iterations {
        let report = run.step()?;
        report.print();
    }

    Ok(())
}
```

## UCI

```rust
fn main() -> Result<()> {
    let settings = UciSettings::parse();
    let mut engine = ChessAlphaZeroEngine::open(settings)?;
    run_uci_loop(&mut engine)
}
```

## HTTP model use

```rust
let model = state.models.get(&request.model)?;
let result = model.analyze(&session.game, request.search)?;
```

The complicated generic implementation may remain underneath these surfaces.

---

# 2. Current state assessment

## 2.1 Strong architecture that must be preserved

- `GameState` remains a minimal native-rules trait.
- Native moves remain distinct from neural `Action` indices.
- `AlphaZeroRepresentation` remains the representation boundary.
- `search` remains free of `tch`.
- MCTS stores only legal children in a contiguous arena.
- Search values remain typed and side-to-move-oriented.
- Chess keeps authoritative `ChessGame` separate from compact search snapshots.
- Repetition and the fifty-move rule remain explicit.
- Scalar and WDL heads remain supported.
- The classic chess model remains loadable for compatibility.
- The canonical 8×8×73 chess model remains parameterized.
- The dynamic inference batcher remains backend-independent.
- Replay remains compact and sparse.
- Experiment files remain immutable; run state remains mutable.
- Self-play remains deterministic with global game IDs and worker-independent seeds.
- Hot search and self-play code remains statically dispatched.
- TensorRT remains optional and outside the core correctness path.

## 2.2 Concrete current complexity and debt

### Flat AlphaZero API

`crates/alphazero/src/lib.rs` re-exports most internal types from:

- model configuration;
- inference;
- replay;
- self-play;
- training;
- experiment management;
- MCTS;
- rules;
- analysis.

A caller cannot easily tell which types form the intended stable API and which are implementation details.

### Training binary owns the whole application

`crates/app/src/bin/train.rs` currently contains:

- CLI parsing;
- experiment initialization;
- inspection;
- TorchScript export;
- device selection;
- run opening;
- checkpoint loading;
- resume handling;
- model construction;
- inference backend construction;
- TensorRT restrictions;
- game dispatch;
- representation/history dispatch;
- replay construction;
- self-play factory construction;
- optimizer construction;
- the generation loop;
- checkpoint save/reload;
- metric logging.

This makes the primary program flow difficult to understand even though each individual piece is reasonable.

### Duplicate runtime model assembly

Model loading and MCTS assembly are repeated in:

- `train.rs`;
- UCI;
- `players.rs`;
- proxy/session handlers;
- analysis endpoints;
- evaluation code.

Several places also construct their own PUCT defaults and batcher configuration.

### Duplicate self-play loops

`selfplay/generic.rs` and `selfplay/chess.rs` share the same structure:

1. seed RNGs;
2. initialize a game;
3. choose full/fast search budget;
4. run MCTS;
5. choose a move;
6. record sparse training data;
7. optionally resign;
8. apply the move;
9. assign alternating outcomes.

Chess differs mainly in:

- authoritative `ChessGame`;
- conversion to a search state;
- repetition context;
- optional evaluation cache;
- resignation.

These differences can be expressed by a compile-time domain adapter rather than a second loop.

### Classic chess self-play uses the generic path

The classic chess model currently uses `ChessPosition` with `NoExtraRules` in the generic worker. That means its self-play path does not use the authoritative `ChessGame` repetition context in the same manner as canonical chess. The compatibility network architecture should not imply weaker chess rules.

### Search budget duplication

The configured PUCT search contains `common.simulations`, while `SearchBudgetSchedule` also contains the simulations used for each move. Gumbel likewise stores simulations and considered actions both in the search configuration and budget.

Self-play mutates the search object before every move through `set_budget`.

The algorithm parameters and the per-search work budget are separate concepts and should be represented separately.

### Search variants are not the requested complete set

The current code supports:

- PUCT;
- strict Full Gumbel.

The requested project should support three explicitly named algorithms:

- PUCT;
- root Gumbel sequential halving with PUCT below the root;
- Full Gumbel with improved-policy-deficit selection below the root.

The hybrid must not be named simply `Gumbel`.

### Stale statistics terminology

`SelfPlayStats` still contains:

```rust
tt_hits
tt_misses
tt_inserts
```

despite the project now describing a neural evaluation cache and search-local diagnostics. These fields are not the clear public measurements the training loop needs.

### Unused retained source

`crates/evaluations/src/legacy.rs` and `legacy/support.rs` are retained but not part of the current exported evaluation path. Dead retained implementations make the codebase harder to understand and should be archived in Git history rather than kept in the active source tree.

### TensorRT tooling is mixed into the training binary

TorchScript export is a model-tool operation, not part of the self-play/training control flow. Its presence in `train.rs` makes the normal workflow look more complicated than it is.

---

# 3. Design rules for the refactor

Codex must follow these rules throughout the change.

## 3.1 Prefer closed, valid-by-construction types

If the project supports three concrete model families, represent exactly those three model families. Do not expose seven independently configurable enums and rely on validation to reject unsupported combinations.

If the project supports three search algorithms, represent exactly those three search algorithms.

## 3.2 Runtime dispatch only at coarse boundaries

Runtime configuration may select:

- game;
- model family;
- chess history length;
- search algorithm;
- inference backend.

Perform those matches:

- at run startup;
- when loading a model;
- once per search call when choosing the algorithm implementation.

Do not introduce trait objects or enum matches inside:

- node selection loops;
- board encoding loops;
- neural forward operations;
- sparse loss loops;
- per-edge policy scoring.

## 3.3 One primary abstraction per boundary

Use:

- `GameState` for native search state;
- `AlphaZeroRepresentation` for tensor/action mapping;
- `SearchRules` for path-dependent search rules;
- `InferenceBackend` for one contiguous backend batch;
- `SelfPlayDomain` for adapting an authoritative game to a search state;
- `TrainingRun` for one initialized experiment lifecycle.

Do not create overlapping “provider”, “manager”, “service”, “context”, and “factory” traits for the same responsibility.

## 3.4 A trait needs a real boundary

Add a trait only when at least one of these is true:

- there are already two implementations;
- tests require a scripted implementation;
- the trait prevents an unwanted dependency;
- the trait preserves static generic reuse across games.

Do not create traits solely to avoid a match over a small closed enum.

## 3.5 Functions should express one operation

Extract:

- formulas;
- validation;
- state transitions;
- repeated setup;
- result construction;
- serialization/migration;
- logging/report conversion.

Do not extract trivial one-line getters merely to reduce function length.

As a guideline, a normal function should usually fit in roughly 20–60 lines. Larger functions are acceptable when they are a clear linear algorithm. A function requiring several unrelated comments such as “load”, “train”, “save”, and “reload” should be split.

## 3.6 Use visual separation consistently

Within a function, separate distinct phases with one blank line:

```rust
let state = domain.search_state(&game);
let context = domain.search_context(&game);
let budget = schedule.choose(&mut budget_rng);

search.set_or_receive_no_mutable_budget_here();

let result = search.search(&state, context, request)?;
let selected = select_move(&result, move_policy, &mut move_rng);

trajectory.push(make_sample(...));

domain.play(&mut game, selected);
```

Keep closely related declarations together. Do not put blank lines between every statement.

## 3.7 Comments explain invariants and choices

Good:

```rust
// The root schedule uses selection visits, including reservations, so one
// local batch cannot schedule the same candidate beyond the current round.
```

Bad:

```rust
// Increment the visit count.
visits += 1;
```

## 3.8 No performance claim without a measurement

A refactor must preserve or improve readability first and must not assume that:

- fewer allocations;
- fewer enums;
- a smaller node;
- a larger local batch;
- a new backend;
- an asynchronous design

is faster without benchmark evidence.

---

# 4. Target crate and module structure

Keep the workspace crates, but simplify their internal public surfaces.

```text
crates/
  core/
  games/
  search/
  alphazero/
  app/
  evaluations/
  benchmarks/        # new; profiling and benchmarking only
```

## 4.1 `search`

Target:

```text
search/src/
  lib.rs
  config.rs
  evaluator.rs
  rules.rs
  value.rs

  mcts/
    mod.rs
    tree.rs
    workspace.rs
    evaluation.rs
    traversal.rs

    puct.rs
    root_gumbel_puct.rs
    full_gumbel.rs
```

Public API from the crate root:

```rust
pub use config::{
    FullGumbelConfig,
    PuctConfig,
    RootGumbelPuctConfig,
    SearchAlgorithm,
    SearchBudget,
    SearchRequest,
};
pub use evaluator::{Evaluation, EvaluationError, EvaluationKey, PolicyValueEvaluator};
pub use mcts::{EvalTable, Mcts, SearchDiagnostics, SearchError, SearchResult};
pub use rules::{NoExtraRules, RuleResult, SearchRules};
pub use value::{PositionValue, InvalidValue};
```

Do not publicly expose tree nodes, workspace buffers, root candidates, pending leaves, or internal strategy structs.

## 4.2 `alphazero`

Target:

```text
alphazero/src/
  lib.rs

  model/
    mod.rs
    spec.rs
    network.rs
    basic_residual.rs
    chess_se.rs

  representation/
    mod.rs
    action.rs
    connect4.rs
    chess_classic.rs
    chess_canonical.rs
    chess_state.rs

  inference/
    mod.rs
    config.rs
    service.rs
    backend.rs
    tch_backend.rs

  self_play/
    mod.rs
    config.rs
    domain.rs
    worker.rs
    coordinator.rs

  replay/
    mod.rs
    buffer.rs
    batch.rs
    sample.rs

  training/
    mod.rs
    config.rs
    trainer.rs
    run.rs
    metrics.rs

  experiment/
    ...

  runtime/
    mod.rs
    loaded_model.rs
    chess.rs

  analysis/
    mod.rs
```

The exact file split may be adjusted, but modules should correspond to concepts visible in the architecture. Avoid `generic.rs`, `core.rs`, or `utils.rs` when a more specific name exists.

## 4.3 `app`

Binaries should parse arguments and invoke application functions. They should not assemble generic AlphaZero components directly.

Target binary size guidelines:

- `train.rs`: at most about 120 lines;
- `engine-zoo-uci.rs`: at most about 120 lines;
- `engine-zoo.rs`: already minimal;
- `eval.rs`: route to evaluation commands, not implement them.

## 4.4 `benchmarks`

Add a separate workspace crate:

```text
crates/benchmarks/
  Cargo.toml
  src/
    main.rs
    cli.rs
    environment.rs
    report.rs
    harness.rs

    search.rs
    representation.rs
    inference.rs
    batcher.rs
    self_play.rs
    training.rs
    end_to_end.rs
```

One public binary:

```bash
cargo run -p engine-bench --release -- <subcommand>
```

This keeps benchmark code, synthetic fixtures, environment detection, output schemas, and profiling workloads out of production crates.

---

# 5. Simplify the model specification

## 5.1 Problem with the current structure

The current public model description contains:

- `GameSpec`;
- `RepresentationSpec`;
- `NetworkSpec`;
- `ResidualNetworkConfig`;
- `ResidualTrunkConfig`;
- `PolicyHeadConfig`;
- `ValueHeadConfig`;
- `ModelSpec`.

The implementation supports a small number of valid combinations and rejects the rest.

This is flexible in theory but makes experiment files and runtime dispatch harder to read.

## 5.2 Required replacement

Use a closed model-family enum.

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "architecture", rename_all = "kebab-case")]
pub enum ModelSpec {
    Connect4Residual {
        blocks: usize,
        channels: i64,
        value_head: ValueHeadSpec,
    },

    ChessClassic {
        blocks: usize,
        channels: i64,
    },

    ChessSe {
        history: ChessHistory,
        blocks: usize,
        channels: i64,
        se_hidden: i64,
        value_head: ValueHeadSpec,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ValueHeadSpec {
    Scalar { hidden: i64 },
    Wdl { hidden: i64 },
}
```

If Connect Four WDL is not supported in the implementation, either:

- restrict its `value_head` to scalar by using no field; or
- implement the already-general WDL training/output path for Connect Four.

Do not expose unsupported options.

## 5.3 Derived information

`ModelSpec` must provide:

```rust
impl ModelSpec {
    pub fn game(&self) -> GameKind;
    pub fn state_shape(&self) -> [i64; 3];
    pub fn action_size(&self) -> usize;
    pub fn value_head(&self) -> ValueHeadSpec;
    pub fn fingerprint(&self) -> ModelFingerprint;
    pub fn chess_history(&self) -> Option<ChessHistory>;
    pub fn is_chess_classic(&self) -> bool;
}
```

Representation and policy-head shape are derived from the model family. There is no public invalid state such as:

- chess canonical plus dense 20,480-action policy;
- classic chess plus WDL;
- Connect Four plus 73 policy planes.

## 5.4 Network implementation

Internally:

```rust
pub struct Model {
    inner: ModelImpl,
}

enum ModelImpl {
    BasicResidual(BasicResidualNet),
    ChessSe(ChessSeNet),
}
```

or keep the current `Network` name if preferred.

The public methods are:

```rust
impl Model {
    pub fn new(path: &nn::Path, spec: &ModelSpec) -> Result<Self>;
    pub fn forward(&self, states: &Tensor, mode: ForwardMode) -> ModelOutput;
}
```

Prefer an explicit `ForwardMode` enum or keep `train: bool` if it remains idiomatic with `tch`. Do not create separate model types for scalar and WDL; the output enum already handles both.

## 5.5 Experiment migration

Increment the experiment format to version 3.

Implement a deterministic migration from the current version-2 nested model structure to the new closed enum.

Add tests for all checked-in experiments and representative old JSON/TOML files.

After migration:

- update all example TOMLs;
- update model fingerprints intentionally;
- document that old checkpoints remain loadable when their resolved tensor structure is unchanged;
- test real tensor-name compatibility for the classic and canonical models.

## 5.6 Example simplified TOML

```toml
format_version = 3
seed = 1

[model]
architecture = "chess-se"
history = "four"
blocks = 12
channels = 128
se_hidden = 16
value_head = { wdl = { hidden = 128 } }
```

The exact serde representation may be made cleaner, but it should be visibly shorter than the current nested trunk/policy/value specification.

---

# 6. Refactor search around algorithm configuration plus per-call budget

## 6.1 Public algorithms

Add:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SearchAlgorithm {
    Puct,
    RootGumbelPuct,
    FullGumbel,
}
```

The configuration enum should be explicit:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "algorithm", rename_all = "kebab-case")]
pub enum SearchConfig {
    Puct(PuctConfig),
    RootGumbelPuct(RootGumbelPuctConfig),
    FullGumbel(FullGumbelConfig),
}
```

Rename the current `GumbelConfig` to `FullGumbelConfig`.

## 6.2 Separate stable algorithm parameters from work budget

The MCTS object owns algorithm behavior:

```rust
pub struct PuctConfig {
    pub leaf_batch_size: usize,
    pub selection: PuctSelectionConfig,
    pub fpu: FpuConfig,
    pub in_flight: InFlightConfig,
    pub root_noise: Option<DirichletConfig>,
}

pub struct RootGumbelPuctConfig {
    pub leaf_batch_size: usize,
    pub puct: PuctTreeConfig,
    pub root: GumbelRootConfig,
}

pub struct FullGumbelConfig {
    pub root: GumbelRootConfig,
}

pub struct GumbelRootConfig {
    pub gumbel_scale: f32,
    pub completed_q: CompletedQConfig,
}
```

The exact grouping of PUCT fields may differ, but simulations and `max_considered_actions` must not live in these stable algorithm configurations.

One search call supplies its budget:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
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

Both `RootGumbelPuct` and `FullGumbel` consume the `Gumbel` budget.

## 6.3 Search request

```rust
#[derive(Clone, Copy, Debug)]
pub struct SearchRequest {
    pub mode: PolicyMode,
    pub budget: SearchBudget,
}
```

New API:

```rust
let result = mcts.search(&state, context, SearchRequest {
    mode: PolicyMode::Explore,
    budget,
})?;
```

Delete:

- `Mcts::set_simulations`;
- `Mcts::set_gumbel_config`;
- the self-play `set_budget` function;
- duplicated default simulations stored inside search configuration;
- configuration cloning solely to determine the current algorithm.

Add:

```rust
pub fn algorithm(&self) -> SearchAlgorithm;
```

## 6.4 Internal dispatch

`MctsKind` becomes:

```rust
enum MctsKind<G, E, R> {
    Puct(MctsCore<G, E, R, Puct>),
    RootGumbelPuct(MctsCore<G, E, R, RootGumbelPuct>),
    FullGumbel(MctsCore<G, E, R, FullGumbel>),
}
```

The match occurs once at the start of `Mcts::search`. Inner loops remain monomorphized.

## 6.5 Common root initialization

PUCT and Full Gumbel currently duplicate:

- tree clearing;
- root-node creation;
- path reset;
- root rule entry;
- terminal checks;
- root inference;
- root policy building;
- root expansion;
- initial diagnostics.

Extract a private operation with a concrete result:

```rust
struct PreparedRoot {
    network_value: PositionValue,
    diagnostics: SearchDiagnostics,
}

impl<...> MctsCore<...> {
    fn prepare_root(
        &mut self,
        game: &G,
        context: R::Context<'_>,
        root_policy: RootPolicyPreparation,
    ) -> Result<PreparedRoot, SearchError>;
}
```

Do not abstract the whole search loop. PUCT, Root-Gumbel-PUCT, and Full Gumbel should each retain a readable top-level algorithm.

## 6.6 PUCT

Preserve:

- dynamic PUCT constant;
- FPU reduction/absolute FPU;
- root-specific FPU;
- unscored virtual visits;
- optional virtual loss;
- Dirichlet noise;
- local leaf batching.

Extract formulas into named functions:

```rust
fn exploration_constant(...);
fn first_play_urgency(...);
fn completed_child_q(...);
fn virtualized_q(...);
fn puct_score(...);
```

The search loop should read approximately:

```rust
let root = self.prepare_root(...)?;
let mut batch = LeafBatch::with_capacity(config.leaf_batch_size);

while diagnostics.completed_simulations < budget.simulations {
    self.collect_puct_batch(...);
    self.evaluate_and_backup(&mut batch, &mut diagnostics)?;
}

Ok(self.puct_result(root, diagnostics))
```

## 6.7 Root Gumbel + interior PUCT

Implement this as an honestly documented hybrid.

Behavior:

1. sample root Gumbel scores;
2. restrict to at most `max_considered_actions`;
3. allocate root visits through sequential-halving rounds;
4. below the selected root child, use normal PUCT;
5. permit local leaf batches;
6. use PUCT in-flight reservations within and below the root;
7. eliminate candidates only after the current round’s required completed visits are available;
8. return the final root proposal and a defined policy target.

### Batching rule

The root scheduler must distinguish:

- completed visits;
- in-flight/reserved visits;
- target visits for the current sequential-halving round.

Within one local batch, select an active root candidate whose **selection visits** remain below its current round target. This prevents one batch from overscheduling one candidate using stale completed counts.

After evaluating the batch:

- convert reservations into completed visits;
- check whether every active candidate reached its round target;
- eliminate half using Gumbel score plus transformed completed Q;
- continue.

### Interior selection

Use the same PUCT scoring helpers as ordinary PUCT. Do not duplicate the PUCT formula.

### Policy target

Because this is a hybrid rather than the canonical Full Gumbel algorithm, document one exact target and keep it stable.

Recommended target:

```text
softmax(root network logits + transformed completed Q)
```

over every legal root action, with unvisited values completed from the root value.

Also expose visit policy in diagnostics if useful, but do not silently switch targets between experiments.

Add tests proving:

- root scheduling matches sequential-halving fixtures;
- interior selection calls PUCT;
- leaf batches larger than one are supported;
- reservations cannot cross a round target;
- elimination waits for completed results;
- the proposed action is deterministic under fixed RNG/evaluator;
- returned policy is normalized and legal.

## 6.8 Full Gumbel

Preserve the strict behavior:

- root Gumbel sequential halving;
- Mctx-style completed Q;
- improved-policy-deficit selection below the root;
- exactly one outstanding visit;
- no PUCT selection below the root.

The config must not expose `leaf_batch_size`.

The code should explicitly set its workspace batch capacity to one and test that no larger batch can occur.

## 6.9 Search diagnostics

Replace stale/ambiguous statistics with:

```rust
pub struct SearchDiagnostics {
    pub completed_simulations: usize,
    pub backend_evaluations: usize,
    pub evaluation_cache_hits: usize,
    pub duplicate_leaves: usize,
    pub nodes_created: usize,
    pub max_depth: usize,
}
```

Add algorithm-specific optional fields only if they have stable meaning:

```rust
pub struct GumbelDiagnostics {
    pub considered_actions: usize,
    pub halving_rounds: usize,
}
```

Do not put timers in per-search diagnostics. Time at the benchmark or application boundary.

---

# 7. Unify self-play with a compile-time domain adapter

## 7.1 Domain trait

Add:

```rust
pub trait SelfPlayDomain: Send + Sync + 'static {
    type Move: Copy + Eq + std::fmt::Debug + Send + Sync + 'static;

    type Game: Send + 'static;
    type State: GameState<Move = Self::Move> + Clone + Send + Sync + 'static;
    type Representation: AlphaZeroRepresentation<Self::State> + Default;
    type Rules: SearchRules<Self::State> + Default;

    fn initial_game() -> Self::Game;

    fn search_state(game: &Self::Game) -> Self::State;

    fn search_context<'a>(
        game: &'a Self::Game,
    ) -> <Self::Rules as SearchRules<Self::State>>::Context<'a>;

    fn is_terminal(game: &Self::Game) -> bool;

    fn terminal_value(game: &Self::Game) -> Option<TerminalValue>;

    fn play(game: &mut Self::Game, mv: Self::Move);
}
```

Adjust lifetime bounds as required by Rust, but preserve this conceptual surface.

## 7.2 Domain implementations

```rust
pub struct Connect4Domain;

pub struct ChessClassicDomain;

pub struct ChessCanonicalDomain<const HISTORY: usize>;
```

### Connect Four

- `Game = Connect4`;
- `State = Connect4`;
- `Representation = Connect4AzRepresentation`;
- `Rules = NoExtraRules`.

### Classic chess

- `Game = ChessGame`;
- `State = ChessPosition`;
- `Representation = ChessClassicRepresentation`;
- `Rules = ChessRepetitionRules`.

This fixes repetition-aware self-play without changing the classic tensor representation.

### Canonical chess

- `Game = ChessGame`;
- `State = ChessAzState<HISTORY>`;
- `Representation = ChessAzRepresentation<HISTORY>`;
- `Rules = ChessRepetitionRules`.

## 7.3 One factory and one worker

Replace:

- `GenericSelfPlayWorkerFactory`;
- `ChessSelfPlayWorkerFactory`;
- `GenericSelfPlayWorker`;
- `ChessSelfPlayWorker`;

with:

```rust
pub struct SelfPlayWorkerFactory<D: SelfPlayDomain> {
    inference: InferenceClient,
    config: SelfPlayConfig,
    cache: Option<Arc<EvalTable<D::Move>>>,
    marker: PhantomData<fn() -> D>,
}

pub struct SelfPlayWorker<D: SelfPlayDomain> {
    representation: D::Representation,
    mcts: Mcts<...>,
    config: SelfPlayConfig,
}
```

The evaluation cache should be available to every domain when configured, not only canonical chess.

## 7.4 One readable game loop

The `play_game` method should be organized into named helpers:

```rust
fn play_game(&mut self, request: GameRequest) -> Result<CompletedGame<D::State>> {
    let mut rng = GameRngs::new(request);
    let mut game = D::initial_game();
    let mut trajectory = self.new_trajectory();
    let mut stats = SelfPlayStats::for_one_game();
    let resignation = ResignationState::new(&self.config, &mut rng.resignation);

    while !D::is_terminal(&game) && trajectory.len() < self.config.max_moves {
        let turn = self.play_turn(&game, &mut rng, &mut stats)?;
        trajectory.push(turn.sample);

        if resignation.should_resign(turn.root_value, trajectory.len()) {
            stats.resignations += 1;
            assign_outcomes(&mut trajectory, Outcome::Loss);
            return Ok(CompletedGame { stats, trajectory });
        }

        D::play(&mut game, turn.selected_move);
    }

    assign_outcomes(
        &mut trajectory,
        last_mover_outcome(D::terminal_value(&game), D::is_terminal(&game)),
    );

    stats.moves = trajectory.len();
    Ok(CompletedGame { stats, trajectory })
}
```

The exact helper breakdown may differ. Keep the main loop linear and visible.

## 7.5 General move-selection policy

Rename the Gumbel-specific setting:

```rust
GumbelMoveSelection
```

to a general self-play setting:

```rust
pub enum MoveSelection {
    SearchProposal,
    PolicyTemperature,
}
```

- `SearchProposal` uses `SearchResult::selected_move`;
- `PolicyTemperature` samples the returned policy using the configured schedule.

Validate sensible combinations but do not hard-code the name of one algorithm into the self-play layer.

## 7.6 Search budgets

`SearchBudgetSchedule` remains responsible for:

- fixed budget;
- playout-cap randomization;
- fast policy weight;
- full probability.

It validates the budget family against `SearchAlgorithm`.

It no longer mutates the MCTS object. The selected budget is placed directly into `SearchRequest`.

## 7.7 Self-play stats

Replace stale fields with aggregate search work:

```rust
#[derive(Clone, Debug, Default, Serialize)]
pub struct SelfPlayStats {
    pub games: usize,
    pub moves: usize,
    pub full_searches: usize,
    pub fast_searches: usize,
    pub resignations: usize,

    pub completed_simulations: u64,
    pub backend_evaluations: u64,
    pub evaluation_cache_hits: u64,
    pub duplicate_leaves: u64,
    pub nodes_created: u64,
    pub maximum_search_depth: usize,
}
```

Every played turn adds its `SearchDiagnostics`.

---

# 8. Centralize inference construction

## 8.1 Rename the public concept

The current `Batcher` is more than a queue: it owns the backend thread, model reload ordering, generation namespace, metrics, and clients.

Expose it as:

```rust
pub struct InferenceService { ... }

#[derive(Clone)]
pub struct InferenceClient { ... }
```

The internal module and worker may still use “batcher” terminology.

If avoiding a rename is considered more valuable than the clarity gain, keep `Batcher`, but then hide its backend-specific constructors behind one loader.

## 8.2 One loader

```rust
impl InferenceService {
    pub fn load(
        model: &ModelSpec,
        source: InferenceSource<'_>,
        device: Device,
        config: &InferenceConfig,
    ) -> Result<Self>;
}
```

`InferenceSource` may be:

```rust
pub enum InferenceSource<'a> {
    Checkpoint(&'a Path),
    TensorRtModule(&'a Path),
}
```

Normal callers should not choose between:

- `new_with_model`;
- `new_with_model_precision`;
- `new_with_tensor_rt_torchscript`.

Backend constructors remain available in an advanced submodule for tests and benchmarks.

## 8.3 Public module surface

```rust
pub mod inference {
    pub use config::{InferenceConfig, InferenceEngine, InferencePrecision};
    pub use service::{InferenceClient, InferenceService, InferenceStats};

    pub mod backend {
        pub use ...; // intended for tests, custom backends, and engine-bench
    }
}
```

`CombinedEncodedBatch` should become `BackendBatch` and live under `inference::backend`, not the crate root.

## 8.4 Operational metrics versus profiling

Keep low-frequency operational counters:

- submitted states;
- inference batches;
- average batch size;
- partial batches;
- queue wait;
- backend execution;
- reloads.

These are useful during normal training and are updated once per request or inference batch, not once per tree edge.

Do not add:

- timestamps per MCTS node;
- timers per edge;
- heap allocations for events;
- string-based tracing in selection loops.

## 8.5 Named worker threads

Use named threads:

```text
inference-batcher
selfplay-00
selfplay-01
...
replay-prefetch
```

This improves external profiler output and debugging with negligible runtime cost.

Use scoped thread builders where available.

---

# 9. Introduce a typed `TrainingRun` workflow

## 9.1 Purpose

The binary should not manually pass ten objects through `run_iterations`.

Create:

```rust
pub struct TrainingRun {
    inner: TrainingRunKind,
}

enum TrainingRunKind {
    Connect4(TypedTrainingRun<Connect4Domain>),
    ChessClassic(TypedTrainingRun<ChessClassicDomain>),
    ChessH1(TypedTrainingRun<ChessCanonicalDomain<1>>),
    ChessH4(TypedTrainingRun<ChessCanonicalDomain<4>>),
    ChessH8(TypedTrainingRun<ChessCanonicalDomain<8>>),
}
```

This enum dispatches once per iteration. `TypedTrainingRun<D>` is monomorphized.

## 9.2 Owned lifecycle

```rust
struct TypedTrainingRun<D: SelfPlayDomain> {
    run_dir: RunDir,
    experiment: ExperimentConfig,
    state: RunState,
    device: Device,

    var_store: nn::VarStore,
    model: Model,
    optimizer: nn::Optimizer,

    inference: InferenceService,
    replay: ReplayBuffer<D::State>,
    self_play: SelfPlayCoordinator,
    workers: SelfPlayWorkerFactory<D>,

    marker: PhantomData<fn() -> D>,
}
```

The exact ownership may be adjusted to satisfy borrowing and backend lifecycle constraints.

## 9.3 Public API

```rust
impl TrainingRun {
    pub fn open(run_dir: &Path, device: Device) -> Result<Self>;

    pub fn step(&mut self) -> Result<IterationReport>;

    pub fn run(&mut self, limit: RunLimit) -> Result<()>;

    pub fn state(&self) -> &RunState;
}
```

`step` should read clearly:

```rust
fn step(&mut self) -> Result<IterationReport> {
    let self_play = self.generate_self_play()?;
    let training = self.train_network()?;

    self.advance_state(&self_play, training.as_ref());
    self.save_checkpoint()?;
    self.reload_inference()?;

    let report = self.build_report(self_play, training);
    self.persist_report(&report)?;

    Ok(report)
}
```

TensorRT’s non-reloadable behavior must be handled explicitly:

- either `TrainingRun` only accepts reloadable native inference;
- or a separate one-generation `SelfPlayRun` accepts TensorRT.

The preferred simplification is:

> Normal self-play/training uses reloadable native inference. TensorRT is benchmark and self-play-generation tooling until automatic per-generation compilation exists.

Preserve the existing one-generation TensorRT functionality through a separate command if required, but do not let it complicate the normal training lifecycle.

## 9.4 Trainer object

Instead of free functions with many arguments:

```rust
pub struct Trainer {
    config: TrainConfig,
    optimizer: nn::Optimizer,
}
```

```rust
impl Trainer {
    pub fn new(vs: &nn::VarStore, config: TrainConfig) -> Result<Self>;

    pub fn train<S, R>(
        &mut self,
        model: &Model,
        replay: &ReplayBuffer<S>,
        representation: &R,
        device: Device,
        seed: TrainingSeed,
    ) -> Option<TrainMetrics>;
}
```

The generic method remains monomorphized. Configuration and optimizer lifecycle are grouped.

## 9.5 Move TorchScript export out of `train`

Add a small tool binary:

```text
engine-zoo-model
```

with:

```text
engine-zoo-model export-torchscript
engine-zoo-model inspect
```

or put export under `engine-bench backend` if it is only used for TensorRT experiments.

The `train` command should contain only:

- `init`;
- `run`;
- `inspect`.

---

# 10. Centralize loaded AlphaZero engine assembly

## 10.1 Generic engine

```rust
pub struct AlphaZeroEngine<D: SelfPlayDomain> {
    inference: InferenceService,
    search: Mcts<...>,
    representation: D::Representation,
}
```

For application play, a narrower `EngineDomain` trait may be used if reusing `SelfPlayDomain` would expose unnecessary self-play concepts. Prefer reuse if the associated types align naturally.

## 10.2 Chess runtime enum

```rust
pub enum ChessAlphaZeroEngine {
    Classic(AlphaZeroEngine<ChessClassicDomain>),
    H1(AlphaZeroEngine<ChessCanonicalDomain<1>>),
    H4(AlphaZeroEngine<ChessCanonicalDomain<4>>),
    H8(AlphaZeroEngine<ChessCanonicalDomain<8>>),
}
```

Public operations:

```rust
impl ChessAlphaZeroEngine {
    pub fn open(request: LoadChessEngine) -> Result<Self>;

    pub fn select_move(
        &mut self,
        game: &ChessGame,
        search: SearchRequest,
    ) -> Result<ChessMove>;

    pub fn analyze(
        &mut self,
        game: &ChessGame,
        request: AnalyzeRequest,
    ) -> Result<Analysis>;
}
```

This removes history/model dispatch from UCI, proxy, sessions, and evaluation callers.

## 10.3 Shared model registry for the server

The current HTTP session path constructs and loads an inference service during engine turns. Loading weights and starting a backend thread should not be repeated for each move.

Add:

```rust
pub struct ModelRegistry {
    models: DashMap<ModelId, Arc<LoadedModel>>,
}
```

A standard `RwLock<HashMap<...>>` is also sufficient; do not add `dashmap` without need.

`LoadedModel` owns:

- model spec;
- checkpoint identity;
- inference service;
- immutable metadata.

Each request creates or checks out a short-lived searcher using a cloned `InferenceClient`.

If multiple concurrent requests need one mutable MCTS per session, store search state with the session. Do not duplicate model weights per session.

## 10.4 One search-default builder

Application code should not contain several private `puct_search` functions.

Add explicit named constructors:

```rust
impl PuctConfig {
    pub fn analysis_default(leaf_batch_size: usize) -> Self;
}

impl SearchRequest {
    pub fn deterministic_puct(simulations: usize) -> Self;
}
```

Experiment/self-play defaults remain in experiment files. Application defaults should be named for their use rather than duplicated literals.

---

# 11. Clean the public API

## 11.1 AlphaZero root exports

After the refactor, `alphazero::lib.rs` should expose modules and a small convenience surface.

Example:

```rust
pub mod analysis;
pub mod experiment;
pub mod inference;
pub mod model;
pub mod replay;
pub mod representation;
pub mod runtime;
pub mod self_play;
pub mod training;

pub use model::{Model, ModelSpec};
pub use runtime::{ChessAlphaZeroEngine, TrainingRun};
```

Do not re-export every MCTS configuration through `alphazero`; callers needing search types should import them from `search`.

## 11.2 Naming

Use these terms consistently:

- `model`: tensor architecture and weights;
- `representation`: game state/action encoding;
- `inference`: forward-only neural execution;
- `search`: MCTS improvement;
- `self-play`: game generation;
- `replay`: stored training samples;
- `training`: optimizer updates;
- `experiment`: immutable configuration plus run state;
- `runtime`: assembled model/search objects for a program.

Avoid mixing:

- model/network;
- TT/evaluation cache;
- position/state/game;
- search config/search budget;
- batch/local leaf batch/inference batch/training microbatch.

Document these terms in `docs/architecture.md`.

---

# 12. Code formatting and readability policy

Add `docs/code-style.md` with project-specific rules.

## Required rules

1. `rustfmt` is authoritative.
2. Separate logical phases with a blank line.
3. Group related declarations.
4. Prefer early returns for invalid/terminal cases.
5. Name formulas.
6. Keep mutation local.
7. Avoid booleans whose meaning is unclear at call sites; use enums.
8. Avoid `#[allow(clippy::too_many_arguments)]` in normal workflow code. Use state-owning structs.
9. Avoid broad `utils` modules.
10. Avoid public fields unless they are configuration/data records.
11. Avoid `expect` at user/configuration boundaries.
12. `expect` is acceptable for locally established internal invariants and must state the invariant.
13. No hidden fallback from invalid numeric values to uniform policy/draw.
14. No performance-sensitive heap allocation added without benchmark coverage.
15. No dynamic dispatch in per-node search code.

## Review examples

Codex should actively rewrite blocks such as:

```rust
let a = ...;
let b = ...;
let c = ...;
f();
g();
let d = ...;
let e = ...;
```

into visually grouped phases when `d/e` belong to a later operation.

Do not mechanically add blank lines where all declarations form one coherent construction.

---

# 13. Benchmarking architecture

## 13.1 Benchmark before refactoring

Before changing architecture:

1. commit or tag the current state;
2. add the benchmark harness with minimal production changes;
3. run the baseline suite on the GPU machine;
4. save machine metadata and raw results;
5. then perform the simplification refactor.

Suggested tag:

```text
pre-simplification-benchmark-baseline
```

This prevents the project from discovering after the refactor that no comparable baseline exists.

## 13.2 One benchmark CLI

Add:

```bash
engine-bench environment
engine-bench search
engine-bench representation
engine-bench inference
engine-bench batcher
engine-bench self-play
engine-bench training
engine-bench end-to-end
engine-bench suite
```

All commands accept:

```text
--config <toml>
--warmup <n or duration>
--samples <n>
--output <json>
--human
```

`--human` prints a concise table. JSON is always suitable for machine comparison.

## 13.3 Report schema

```rust
pub struct BenchmarkReport {
    pub schema_version: u32;
    pub benchmark: String;
    pub started_at: String;
    pub git: GitMetadata;
    pub build: BuildMetadata;
    pub host: HostMetadata;
    pub workload: serde_json::Value;
    pub samples: Vec<BenchmarkSample>;
    pub summary: BenchmarkSummary;
}
```

Record:

- commit;
- dirty state;
- Cargo profile;
- Rust version;
- target triple;
- `tch` version;
- LibTorch/PyTorch version if available;
- CUDA driver/runtime;
- GPU name and memory;
- CPU model/core count;
- OS/kernel;
- model fingerprint;
- experiment/search config;
- warmup;
- repetitions;
- median;
- mean;
- minimum;
- maximum;
- standard deviation or median absolute deviation.

Do not require GPU metadata for CPU-only benchmarks.

## 13.4 Custom Cargo profile

Add:

```toml
[profile.profiling]
inherits = "release"
debug = 1
strip = false
lto = "thin"
```

Use:

```bash
cargo build --profile profiling -p engine-bench
```

for external profilers.

Keep ordinary production release builds unchanged.

## 13.5 Benchmark fixtures

Add deterministic fixtures:

```text
benchmarks/
  configs/
  fixtures/
    chess_positions.jsonl
    inference_positions.bin
    replay_connect4.bin
    replay_chess_h4.bin
  results/
    README.md
```

Do not commit huge replay buffers or profiler traces.

Commit:

- small representative fixture data;
- JSON summaries;
- Markdown interpretations.

Ignore:

- `.nsys-rep`;
- raw Nsight exports;
- perf data;
- flamegraph intermediates;
- large generated checkpoints.

---

# 14. Benchmark workloads

## 14.1 Pure MCTS CPU benchmark

Use scripted or uniform evaluators so no GPU is involved.

Games:

- Connect Four initial/midgame/tactical;
- chess opening/middlegame/endgame positions;
- repetition-sensitive chess state.

Algorithms:

### PUCT

- simulations: 64, 128, 256, 800;
- leaf batch: 1, 2, 4, 8, 16, 32;
- cache: off/cold/hot;
- unscored virtual visits;
- selected representative FPU settings.

### Root Gumbel + PUCT

- simulations: 32, 64, 128, 256;
- considered actions: 8, 16, 32;
- leaf batch: 1, 2, 4, 8, 16;
- cache: off/cold/hot.

### Full Gumbel

- simulations: 32, 64, 128, 256;
- considered actions: 8, 16, 32;
- effective batch must remain one.

Measure:

- searches/s;
- simulations/s;
- nodes created/s;
- nodes per simulation;
- backend requests, even though scripted;
- duplicate leaves;
- maximum depth;
- arena peak capacity;
- cache hit rate.

The benchmark must reuse the `Mcts` object across searches as production does.

## 14.2 Representation benchmark

Measure:

- Connect Four state encoding;
- classic chess encoding;
- canonical H1/H4/H8 encoding;
- `ChessAzState::from_game`;
- legal move to action;
- action to move;
- complete legal-action mapping for one position;
- encoded-state cache key;
- replay sample conversion to one batch.

Report:

- ns/state;
- states/s;
- allocations if measurable;
- bytes written.

## 14.3 Raw model inference benchmark

Model families:

- Connect Four residual;
- classic chess 10×64 scalar;
- canonical chess 12×128 scalar;
- canonical chess 12×128 WDL.

Precisions:

- FP32;
- FP16 on CUDA.

Batch sizes:

```text
1, 2, 4, 8, 16, 32, 64, 128, 256
```

Continue to 512/1024 only if memory and latency remain sensible.

Measure two paths:

1. forward-only model tensor input to raw outputs;
2. complete `InferenceBackend` path:
   - host staging;
   - H2D;
   - model;
   - legal-logit gather;
   - D2H;
   - Rust evaluations.

Report:

- median latency;
- positions/s;
- policy/value output shape;
- maximum allocated GPU memory if available;
- numerical comparison FP32 versus FP16.

Use real legal-action distributions from representative positions, not only a fixed count.

## 14.4 Dynamic batcher benchmark

Sweep:

- client threads: 1, 2, 4, 8, 16, 24, 32;
- request size: 1, 2, 4, 8, 16, 32;
- preferred batch: 8, 16, 32, 64, 128;
- max wait: 0, 100 µs, 500 µs, 2 ms;
- max batch: 64, 128, 256.

Do not execute the full Cartesian product by default. Define:

- a quick suite;
- an extended sweep.

Measure:

- states/s;
- request latency p50/p95/p99;
- average inference batch;
- partial-batch fraction;
- queue wait;
- backend utilization time;
- backpressure frequency.

## 14.5 Replay benchmark

For each representation:

- insert trajectories;
- ring overwrite;
- deterministic sample;
- encode sample into batch;
- batch sizes 256, 1024, 2048, 4096;
- prefetch depths 1 and 2.

Measure:

- positions/s;
- wall time;
- peak host memory;
- sparse policy entries/position.

## 14.6 Training benchmark

Use a prefilled deterministic replay buffer large enough to avoid sampling a tiny repeated set.

Canonical primary model:

```text
Chess SE
history 4
12 blocks
128 channels
WDL
```

Sweep:

- batch size: 1024, 2048, 4096;
- microbatch: 64, 128, 256, 512, 1024 as memory permits;
- prefetch: 1, 2;
- FP32;
- AMP only after a correct implementation exists.

Run:

- 10 warmup optimizer steps;
- at least 50 measured steps.

Measure:

- replay sample/encode time;
- H2D time;
- forward/backward time;
- optimizer time;
- complete samples/s;
- GPU memory;
- loss values;
- synchronization count only if profiler-derived.

## 14.7 Self-play benchmark

Use a fixed checkpoint and fixed experiment seed.

Primary model:

```text
Chess SE H4 12×128 WDL
```

Thread counts on the Ryzen 7 7800X3D:

```text
4, 8, 12, 16, 24
```

Do not assume hardware-thread count is optimal.

### PUCT

- simulations 64, 128, 256;
- leaf batch 1, 2, 4, 8, 16;
- evaluation cache off and representative capacity.

### Root Gumbel + PUCT

- simulations 64, 128;
- considered actions 8, 16;
- leaf batch 1, 2, 4, 8.

### Full Gumbel

- simulations 32, 64, 128;
- considered actions 8, 16.

For each configuration:

- warm up inference;
- run enough games for at least several minutes or a stable sample;
- use identical maximum-move and resignation policy;
- save full configuration.

Measure:

- games/s;
- positions/s;
- simulations/s;
- backend evaluations/s;
- average inference batch;
- request latency;
- cache hit rate;
- duplicate-leaf rate;
- moves/game;
- CPU utilization;
- GPU utilization externally;
- mean/max search depth.

## 14.8 End-to-end generation benchmark

After selecting promising parameters, run:

```text
self-play generation
→ replay insertion
→ configured train steps
→ checkpoint save
→ inference reload
```

Measure total wall time and phase breakdown.

At least compare:

- best PUCT throughput candidate;
- best Root-Gumbel-PUCT candidate;
- Full Gumbel candidate.

This benchmark answers whether a search that is slower per evaluation may still produce more useful games per hour because it uses fewer evaluations.

## 14.9 Short learning experiments

System benchmarks do not establish learning efficiency.

After the throughput matrix:

### Connect Four validation

Run several complete short trainings with different search variants. Connect Four should show whether the whole pipeline learns and allows faster repeated comparisons.

### Chess preliminary ablation

Choose at most three configurations:

- best PUCT;
- best Root-Gumbel-PUCT;
- best Full Gumbel.

Train for equal:

- GPU-hours;
- generated positions;
- neural evaluations.

Evaluate all checkpoints with one fixed deterministic evaluation search and paired openings.

Record:

- strength;
- policy/value loss;
- positions generated;
- evaluations used;
- wall time.

Do not choose the final self-play algorithm from throughput alone.

---

# 15. Profiling without contaminating production code

## 15.1 First use external profilers

The refactor should create named functions and named threads so external profilers are understandable without fine-grained instrumentation.

CPU:

- `perf` or `samply` for flamegraphs;
- `perf stat` for cycles, instructions, cache misses, branch misses;
- a heap profiler for replay/search allocation studies when needed.

GPU:

- Nsight Systems for CPU/GPU timeline, CUDA synchronization, transfer, and kernel gaps;
- Nsight Compute only for kernels identified as material by the system timeline.

Profile these dedicated workloads:

```bash
engine-bench profile search
engine-bench profile self-play
engine-bench profile training
```

The “profile” subcommand should run a long stable loop and print almost nothing.

## 15.2 No per-node production timers

Do not insert `Instant::now` into:

- `select_puct_child`;
- Gumbel interior selection;
- tree descent;
- backup;
- move-to-action mapping.

Sampling-profiler results are sufficient at this granularity.

## 15.3 Optional coarse profiling feature

Only if external timelines cannot identify major phases, add:

```toml
[features]
profiling = []
```

Under that feature, add a tiny internal macro:

```rust
profile_scope!("inference.forward");
```

Scopes may exist only at coarse boundaries:

- self-play generation;
- one MCTS search;
- inference queue wait;
- inference backend;
- model forward;
- legal-policy gather;
- replay sampling;
- one optimizer step;
- checkpoint save/reload.

The default build must compile scopes to no operations and must not allocate strings.

If NVTX is used, keep the dependency optional and isolated in the benchmark/profiling crate or a small internal module.

## 15.4 Operational metrics remain in production

Keep:

- `SearchDiagnostics`;
- `SelfPlayStats`;
- `InferenceStats`;
- `TrainMetrics`.

These are algorithm/workflow outputs, not deep profiler instrumentation.

Make detailed metric collection configurable only if measurement proves it has material overhead.

---

# 16. Documentation outputs

After benchmarks, add:

```text
docs/benchmarks/
  methodology.md
  hardware-rtx4080-7800x3d.md
  search.md
  inference.md
  self-play.md
  training.md
  end-to-end.md

docs/profiling/
  cpu-search.md
  gpu-self-play.md
  gpu-training.md
```

Each result document must state:

- date;
- commit;
- dirty state;
- hardware;
- software versions;
- exact command;
- configuration;
- warmup;
- repetitions;
- raw JSON path;
- interpretation;
- next action.

Do not write “X is faster” without the tested workload and uncertainty.

Add one README performance table only after the full documented result exists.

---

# 17. Refactor implementation phases

## Phase 0 — capture baseline

1. Add `engine-bench` environment and current-workload support with minimal changes.
2. Run:
   - current PUCT;
   - current Full Gumbel;
   - inference;
   - batcher;
   - training;
   - end-to-end.
3. Save JSON and Markdown.
4. Tag the commit.

**Gate:** baseline results are reproducible from checked-in commands/configs.

## Phase 1 — model and public API simplification

1. Introduce closed `ModelSpec`.
2. Add v2→v3 experiment migration.
3. Move model internals under `alphazero::model`.
4. Reduce root re-exports.
5. Update example experiments.
6. Add checkpoint compatibility tests.

**Gate:** all existing model/checkpoint tests pass; current checkpoints load.

## Phase 2 — search API and three algorithms

1. Separate algorithm config from per-search budget.
2. Add `SearchRequest`.
3. Rename Full Gumbel config.
4. Implement Root-Gumbel-PUCT.
5. Extract root preparation and shared PUCT formulas.
6. Remove mutable per-move budget setters.
7. Update diagnostics.

**Gate:** PUCT and Full Gumbel fixtures still pass; new hybrid fixtures pass.

## Phase 3 — unify self-play

1. Add `SelfPlayDomain`.
2. Implement three domains.
3. Replace both worker loops with one worker.
4. Make classic chess repetition-aware.
5. Generalize evaluation cache.
6. Generalize move selection.
7. Aggregate diagnostics.

**Gate:** deterministic scripted self-play fixtures pass for every domain.

## Phase 4 — inference and runtime assembly

1. Introduce `InferenceService`/loader.
2. Move advanced backend types into submodule.
3. Add named threads.
4. Add `ChessAlphaZeroEngine`.
5. Add server model registry.
6. Remove repeated model loading and PUCT builders.

**Gate:** UCI, HTTP, analysis, and CLI use the same runtime loader.

## Phase 5 — training workflow

1. Add `TrainingRun` and `TypedTrainingRun<D>`.
2. Add `Trainer`.
3. Move iteration lifecycle out of binary.
4. Move TorchScript export to model tooling.
5. Make `train.rs` a thin CLI.

**Gate:** one CPU Connect Four generation runs through the new flow.

## Phase 6 — cleanup

1. Remove unused legacy evaluation source.
2. Remove old factories/workers/setters/config variants.
3. Remove stale root audit specification.
4. Regenerate repository guide.
5. Update architecture and code-style docs.
6. Run dead-code and unused-dependency checks.

**Gate:** no parallel old/new implementation remains.

## Phase 7 — final benchmark and profiling campaign

1. Run the complete benchmark matrix.
2. Select focused extended sweeps based on quick-suite results.
3. Profile representative self-play and training.
4. Implement only measured optimizations.
5. Repeat benchmarks after each optimization.
6. Document results.

---

# 18. Test requirements

## 18.1 Model

- every `ModelSpec` validates by construction;
- state/action/value shape;
- scalar/WDL output;
- v2 experiment migration;
- fingerprints stable for canonical serialization;
- old checkpoint tensor names load;
- invalid dimensions rejected.

## 18.2 Search API

- budget family mismatch is a typed error;
- PUCT accepts only PUCT budget;
- both Gumbel algorithms accept Gumbel budget;
- Full Gumbel batch is one;
- algorithm introspection returns the exact variant;
- no search configuration mutation between moves.

## 18.3 Root Gumbel + PUCT

- root Gumbel fixture;
- round targets;
- in-flight scheduling;
- no premature elimination;
- interior PUCT fixture;
- local batch 1/N invariants;
- deterministic proposal;
- policy target normalization.

## 18.4 Self-play domains

- Connect Four produces the same deterministic trajectory as the old generic worker under scripted evaluation;
- canonical chess produces the same deterministic trajectory;
- classic chess now handles repetition and fifty-move draws through authoritative game/context;
- worker count does not affect trajectories/replay order;
- resignation and outcome orientation;
- full/fast policy weights;
- diagnostics aggregation.

## 18.5 Runtime

- every model family loads through one loader;
- UCI and HTTP choose the same move for identical model/game/search request;
- model registry loads one inference service per model identity;
- concurrent clients share batching;
- model reload namespaces cache entries correctly.

## 18.6 Training workflow

- `TrainingRun::step` order:
  - self-play;
  - training;
  - state advance;
  - save;
  - reload;
  - metrics;
- failed save does not publish state;
- failed reload does not publish cache namespace;
- CPU Connect Four integration;
- deterministic replay sampling;
- weights-only resume remains honest.

## 18.7 Benchmarks

- report JSON schema round-trip;
- environment detection tolerates missing CUDA;
- benchmark harness warmup is excluded;
- summary calculations;
- fixed workload reproducibility;
- benchmark commands never mutate a real training run unless explicitly requested.

---

# 19. Performance acceptance rules

A readability refactor is accepted when:

- all correctness tests pass;
- no hot path gains dynamic dispatch;
- no new per-node allocation is introduced;
- no benchmark regression exceeds 5% without an explanation;
- a regression caused by corrected semantics is documented rather than hidden.

For search and inference, use medians from repeated runs.

Do not reject a simplification because of a one-off noisy result.

For production selection, optimize:

```text
learning progress per wall-clock GPU hour
```

not merely:

- simulations/s;
- evaluations/s;
- games/s.

---

# 20. Required deletions

After replacements are complete, delete:

- old `GenericSelfPlayWorker*`;
- old `ChessSelfPlayWorker*`;
- `set_budget`;
- duplicated simulations inside algorithm config;
- old `SearchConfig::Gumbel` naming;
- stale `tt_*` self-play stats;
- repeated private `puct_search` constructors;
- repeated direct batcher/model loading in app paths;
- TorchScript export from `train.rs`;
- unused `evaluations/legacy.rs` and support;
- root `engine-zoo-post-refactor-audit-and-improvement-spec.md`;
- obsolete docs describing old public types;
- old benchmark binaries after `engine-bench` covers their workloads.

Git history is the archive. Do not keep dead source “for reference”.

---

# 21. Explicit non-goals

Do not implement as part of this specification:

- alpha-beta;
- NNUE;
- UI redesign;
- replay persistence redesign;
- optimizer-state serialization;
- AMP unless separately validated;
- transformer networks;
- DAG MCTS;
- LC3;
- per-tree asynchronous event pipelines;
- distributed actors;
- multi-GPU training;
- custom CUDA kernels;
- automatic TensorRT recompilation.

The architecture may leave clean extension points for them.

---

# 22. Definition of done

The refactor is complete when all of the following are true:

1. A reader can understand each binary’s main flow in one screen.
2. The normal training binary delegates to `TrainingRun`.
3. The model spec exposes only valid supported model families.
4. The project supports:
   - PUCT;
   - Root-Gumbel-PUCT;
   - Full Gumbel.
5. Per-search work budget is passed to `search`, not stored/mutated in MCTS.
6. One self-play worker implementation serves Connect Four, classic chess, and canonical chess.
7. Classic chess uses authoritative repetition-aware rules.
8. Runtime model/history dispatch is centralized.
9. UCI, HTTP, analysis, and CLI do not duplicate model assembly.
10. Model weights are not reloaded for every HTTP engine move.
11. `alphazero::lib.rs` has a small, intentional public surface.
12. No unused legacy implementation remains.
13. The benchmark CLI can reproduce CPU, GPU, self-play, training, and end-to-end results.
14. Baseline and post-refactor results are documented.
15. CPU and GPU profiles identify the major costs without per-node production instrumentation.
16. The README links to real benchmark/profiling reports.
17. All tests, clippy, frontend checks, and benchmark compilation pass.

At that point the AlphaZero codebase should be both:

- simple enough to explain and extend;
- measured enough that the next optimization is driven by evidence rather than architectural speculation.

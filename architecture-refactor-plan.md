# Engine Zoo Architecture Refactor Plan

**Repository reviewed:** uploaded `main.zip` (Rust workspace with `core`, `games`, `algorithms`, `app`, `evaluations`, and the Svelte frontend)  
**Primary objective:** finish the AlphaZero implementation, remove architectural coupling and obsolete compatibility debt, preserve the existing performance work, and leave a clean foundation for adding alpha-beta without forcing alpha-beta through neural-network abstractions.  
**Execution model:** implement one phase at a time. Every phase must compile, pass tests, and satisfy its stated performance gate before the next phase begins.

> The repository could not be compiled in the review environment because `rustc` and `cargo` are not installed. The plan is based on a static review of the complete uploaded source tree. The first phase therefore requires establishing compiler-verified behavioral and performance baselines on the development machine before changing architecture.

---

## 1. Final architectural decisions

These decisions are fixed for this refactor. Codex should not invent a different plugin architecture or broaden the scope while implementing a phase.

1. **Keep both Chess and Connect Four.**
   - Chess is the main showcase and performance target.
   - Connect Four is a small, useful validation domain for generic game/search abstractions.
   - Do not add Go during this refactor.

2. **Keep AlphaZero, PUCT, Gumbel MCTS, intra-search leaf batching, cross-thread network batching, sparse replay policies, evaluation caching, and the current chess v2 model.**

3. **Do not require every engine to support every game.**
   - Supported combinations are explicit at the app boundary.
   - Initial target matrix:

   | Game         |              AlphaZero |              Alpha-beta |
   | ------------ | ---------------------: | ----------------------: |
   | Chess        |           yes, primary |       near-term primary |
   | Connect Four | yes, generic reference | optional test/reference |
   | Go 9x9       |   not in this refactor |                      no |

4. **Use static dispatch in search and inference hot paths.**
   - Runtime configuration is matched once in `app`, UCI, training, or evaluation code.
   - Do not replace hot-path generics with `dyn Game`, `dyn Move`, or a universal boxed engine.

5. **Native game moves and AlphaZero policy actions are distinct types.**
   - Game rules, MCTS tree edges, agents, alpha-beta, UCI, and sessions use native moves.
   - Network inference, replay targets, and checkpoint policy layouts use `Action`, a newtype representing a policy-vector index.

6. **`GameState` contains only game-tree semantics.**
   - It does not contain tensor shape, policy size, state encoding, parsing, formatting, display, model identity, or repetition tables external to the state.

7. **AlphaZero representation is explicit and keeps the repository’s naming convention.**
   - It owns `STATE_SHAPE`, `ACTION_SIZE`, `encode_state`, move/action conversion, and the encoded-state cache key.

8. **Chess authoritative state and search snapshots are distinct.**
   - One authoritative `ChessGame` owns full repetition counts and the real bounded history.
   - Compact `Copy` snapshots are constructed for search.
   - Remove the duplicated authoritative `ChessAzGame<HISTORY>` type.

9. **Pure search is separated from `tch`.**
   - MCTS and the alpha-beta kernel live in a lightweight `search` crate.
   - Neural networks, batching, replay, training, checkpoints, and AlphaZero representations live in an `alphazero` crate.

10. **Old chess-v1 compatibility must not shape the default architecture.**
    - The reusable scalar-value residual network used by Connect Four is retained and renamed; it is not “legacy.”
    - Old chess-v1 representation/checkpoint support is removed from the default build.
    - If specific old chess checkpoints must remain usable, isolate that support behind a `legacy-chess` Cargo feature and a `compat/v1` module. Do not retain legacy branches in ordinary Chess v2 paths.

---

## 2. Current architecture: concrete problems found

### 2.1 `Game` mixes unrelated responsibilities

Current file: `crates/core/src/game.rs`

The current trait combines:

- game rules and state transitions;
- a fixed neural policy space (`ACTION_SIZE`);
- neural input shape (`STATE_SHAPE`);
- state tensor encoding;
- human/protocol parsing and formatting;
- app metadata (`NAME`);
- construction (`Default`);
- rendering (`Display`).

This forces AlphaZero concepts into alpha-beta and makes one game state synonymous with one model representation.

### 2.2 `Action = u32` means both move and policy index

Current file: `crates/core/src/game.rs`

This is visible in Chess:

- `encode_move` uses the old `64 * 64 * 5` layout;
- `encode_az_move` uses the `8 * 8 * 73` layout;
- both are returned as the same `Action` type;
- `ChessPosition::step_without_repetition` accepts a policy action and decodes it before applying a chess move.

Consequences:

- game rules depend on a model encoding;
- MCTS stores policy indices instead of native moves;
- alpha-beta would have to return an AlphaZero action;
- the same chess move has different numeric meanings under different model formats.

### 2.3 `RepetitionGame` combines rules and model-cache identity

Current file: `crates/core/src/rules.rs`

The trait includes:

- repetition hash;
- halfmove clock;
- mutation of an encoded repetition feature;
- mutation to a repetition draw;
- neural evaluation cache key.

These are not one capability. `evaluation_cache_key` depends on the model representation, not on chess repetition rules.

### 2.4 Common MCTS nodes contain chess-only fields

Current file: `crates/algorithms/src/alphazero/mcts/mod.rs`

Every node currently contains:

```text
hash
repetitions_before_current
repetition_cached
```

Connect Four pays for this metadata even though it has no repetition rule.

### 2.5 Evaluators are not typed against their game/representation

Current file: `crates/algorithms/src/alphazero/evaluator.rs`

`Evaluator` accepts an encoded buffer and numeric legal actions. The type system does not prevent pairing a model with the wrong game or action encoding. Compatibility is checked indirectly by tensor dimensions or runtime indexing.

### 2.6 Chess authoritative state is duplicated

Current files:

- `crates/games/src/chess/game.rs`
- `crates/games/src/chess/az_game.rs`

`ChessGame` and `ChessAzGame<HISTORY>` both own repetition bookkeeping and real-game progression. UCI, proxy sessions, self-play, and puzzles branch between them and reconstruct v2 history separately.

### 2.7 `algorithms` forces all search code to depend on `tch`

Current file: `crates/algorithms/Cargo.toml`

A future alpha-beta implementation would inherit neural-network dependencies, build scripts, checkpoint code, replay code, and training code.

### 2.8 Generic and Chess-specific self-play are duplicated

Current files:

- `crates/algorithms/src/alphazero/selfplay/mod.rs`
- `crates/app/src/chess_selfplay.rs`

Worker orchestration, trajectory handling, progress reporting, search-budget selection, resignation, and reward assignment are partially duplicated. `SelfPlayConfig` also contains `chess_v2_*` fields inside an otherwise generic configuration.

### 2.9 Run configuration has two competing sources of truth

Current files:

- `crates/algorithms/src/alphazero/checkpoint.rs`
- `crates/algorithms/src/alphazero/network.rs`

`RunConfig` stores `game`, `net`, and `architecture`. For Chess v2, `net` is present but ignored when constructing the v2 network. `Legacy` also means both the reusable Connect Four network and old chess-v1 compatibility.

### 2.10 App and evaluation code contain repeated runtime history enums

Current files include:

- `crates/app/src/proxy.rs`
- `crates/app/src/uci.rs`
- `crates/evaluations/src/puzzle.rs`

Each has history-specific dispatch and duplicated wrappers around `ChessAzGame<1|4|8>`. Runtime matching on history is acceptable, but the duplicated authoritative game types are not.

---

## 3. Performance invariants

The refactor must preserve these existing optimizations. They are acceptance requirements, not optional implementation details.

1. **Contiguous MCTS children in one arena.**
   - Keep `first_child .. first_child + num_children`.
   - Keep `u32` node indices unless a benchmark justifies changing them.
   - Clear and reuse arena capacity between searches.

2. **Compact search snapshots.**
   - `ChessPosition`, `ChessHistoryState<1|4|8>`, and `Connect4` should remain `Copy` if their fields permit it.
   - MCTS may require `G: Clone`; cloning a `Copy` state must compile to a plain copy.
   - Do not insert `Arc`, `Rc`, `Box`, `HashMap`, or heap-allocated history into per-simulation chess states.

3. **Intra-search leaf batching and virtual loss.**
   - Preserve `leaf_batch_size`, duplicate-leaf coalescing, virtual loss, and one backup per collected simulation.

4. **Cross-thread GPU batching.**
   - Preserve the `Batcher` worker architecture, grow-only staging buffers, pinned CPU memory, request coalescing, and allocation handoff from client to worker and back.

5. **Gather legal policy logits on device.**
   - Preserve the current CUDA path that maps legal moves to policy actions, gathers only those logits, and transfers only gathered logits to the host.
   - The move/action refactor must not cause a full policy-vector D2H transfer.

6. **Sparse replay policies.**
   - Store only nonzero policy targets.
   - Do not create dense `[batch, ACTION_SIZE]` targets in the replay buffer or training pipeline.

7. **Evaluation cache correctness.**
   - Cache identity must include all features used by `encode_state`.
   - Chess history, repetition planes, halfmove/fullmove planes, side to move, castling, and en passant must not be accidentally omitted.

8. **No dynamic dispatch in MCTS selection/expansion/backup.**
   - Generic `G`, evaluator, representation adapter, and search-rule types are monomorphized.

9. **No per-simulation allocations introduced.**
   - Path buffers, legal-move buffers, pending leaves, evaluator batches, and policy buffers remain reusable.

10. **No performance claim without a benchmark.**
    - If a cleaner layout changes node size, state size, or allocation behavior, benchmark before accepting it.

---

## 4. Target crate dependency graph

```text
engine_core
    ├── games
    └── search

engine_core + games + search
    └── alphazero

engine_core + games + search + alphazero
    ├── engine_app
    └── engine_evaluations
```

Rules:

- `engine_core`: no `anyhow`, `serde`, `chess`, `tch`, or app dependencies.
- `games`: depends on `engine_core`, `chess`, and serialization/error crates needed for setup loading; does not depend on AlphaZero.
- `search`: depends on `engine_core`, `rand`, and `rand_distr`; `games` is allowed only as a dev-dependency for tests/benchmarks.
- `alphazero`: depends on `engine_core`, `games`, `search`, `tch`, serialization, and training dependencies.
- `app` and `evaluations`: perform explicit runtime dispatch over concrete game/model combinations.

---

## 5. Target repository layout

```text
crates/
├── core/
│   └── src/
│       ├── lib.rs
│       ├── game.rs
│       ├── agent.rs
│       └── notation.rs
│
├── games/
│   └── src/
│       ├── lib.rs
│       ├── setup.rs
│       ├── connect4/
│       │   ├── mod.rs
│       │   ├── state.rs
│       │   ├── mv.rs
│       │   ├── notation.rs
│       │   └── tests.rs
│       └── chess/
│           ├── mod.rs
│           ├── position.rs
│           ├── game.rs
│           ├── history.rs
│           ├── repetition.rs
│           ├── notation.rs
│           ├── setup.rs
│           ├── zobrist.rs
│           └── tests.rs
│
├── search/
│   └── src/
│       ├── lib.rs
│       ├── evaluator.rs
│       ├── rules.rs
│       ├── mcts/
│       │   ├── mod.rs
│       │   ├── core.rs
│       │   ├── tree.rs
│       │   ├── traversal.rs
│       │   ├── evaluation.rs
│       │   ├── cache.rs
│       │   ├── puct.rs
│       │   ├── gumbel.rs
│       │   └── tests.rs
│       └── alphabeta/
│           ├── mod.rs
│           ├── score.rs
│           ├── transposition.rs
│           └── tests.rs
│
├── alphazero/
│   └── src/
│       ├── lib.rs
│       ├── representation/
│       │   ├── mod.rs
│       │   ├── connect4.rs
│       │   ├── chess_v2.rs
│       │   └── compat_v1.rs        # only with legacy-chess feature
│       ├── evaluator.rs
│       ├── chess_rules.rs
│       ├── batcher/
│       ├── network/
│       ├── replay/
│       ├── selfplay/
│       ├── trainer/
│       ├── checkpoint/
│       └── analysis/
│
├── app/
└── evaluations/
```

Do not perform this physical crate split first. Stabilize the logical APIs inside the current workspace, then move files with mostly mechanical import changes.

---

## 6. Final core contracts

### 6.1 `GameState`

Target file: `crates/core/src/game.rs`

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminalValue {
    Win,
    Draw,
    Loss,
}

impl TerminalValue {
    pub const fn as_f32(self) -> f32 {
        match self {
            Self::Win => 1.0,
            Self::Draw => 0.0,
            Self::Loss => -1.0,
        }
    }
}

/// A state that can be explored as a two-player, alternating-turn,
/// zero-sum, perfect-information game tree.
pub trait GameState: Send + 'static {
    type Move: Copy + Eq + std::fmt::Debug + Send + 'static;

    fn initial() -> Self
    where
        Self: Sized;

    fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_;

    /// Applies a legal native game move and changes the side to move.
    fn play(&mut self, mv: Self::Move);

    /// Terminal value from the current side-to-move perspective.
    fn terminal_value(&self) -> Option<TerminalValue>;

    fn is_terminal(&self) -> bool {
        self.terminal_value().is_some()
    }
}
```

Important constraints:

- Do not put `Clone`, `Copy`, `Default`, or `Display` on the trait.
- MCTS adds `G: Clone` where it copies states.
- Alpha-beta may add `G: Copy` or use a future make/unmake capability.
- `initial()` replaces the semantic use of `Default` in generic algorithm code. Concrete types may still implement `Default`.

### 6.2 `Agent`

Target file: `crates/core/src/agent.rs`

```rust
pub trait Agent<G: GameState> {
    fn select_move(&mut self, state: &G, mode: PolicyMode) -> G::Move;

    fn select_deterministic_move(&mut self, state: &G) -> G::Move {
        self.select_move(state, PolicyMode::Deterministic)
    }
}
```

Agents return native moves. No general agent returns an AlphaZero `Action`.

### 6.3 Notation

Target file: `crates/core/src/notation.rs`

```rust
pub trait GameNotation<G: GameState> {
    fn parse_move(&self, state: &G, text: &str) -> Option<G::Move>;
    fn format_move(&self, state: &G, mv: G::Move) -> String;
}
```

Use zero-sized concrete notation types:

- `ChessUciNotation`
- `Connect4Notation`

SAN remains a Chess-specific function because it is not a generic game notation requirement.

---

## 7. Native `Move` versus AlphaZero `Action`

### 7.1 Type definitions

`Action` moves out of `engine_core` and into AlphaZero representation code.

Target file: `crates/alphazero/src/representation/mod.rs`

```rust
/// Index into the fixed policy vector of one AlphaZero representation.
/// This is not a native game move.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Action(u32);

impl Action {
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}
```

Native move choices:

- Chess: initially use `chess::ChessMove` directly as `GameState::Move`. Do not add a wrapper merely for architectural purity. Add a packed project-owned move type only if alpha-beta profiling later justifies it.
- Connect Four: add `Connect4Move(u8)` with checked construction and `column()`.

### 7.2 Ownership table

| Layer                        |                    Uses native `Move` |   Uses AlphaZero `Action` |
| ---------------------------- | ------------------------------------: | ------------------------: |
| game rules / `GameState`     |                                   yes |                        no |
| MCTS nodes and search result |                                   yes |                        no |
| alpha-beta                   |                                   yes |                        no |
| `Agent`                      |                                   yes |                        no |
| UCI, CLI, session game loop  |                                   yes |                        no |
| AlphaZero representation     |              yes, input to conversion | yes, output of conversion |
| encoded evaluator batch      |                                    no |                       yes |
| neural policy tensor         |                                    no |                       yes |
| replay policy target         |                                    no |                       yes |
| model-analysis DTO           | move text plus optional policy action |     yes, explicitly named |

### 7.3 Conversion points

Only `AlphaZeroRepresentation` converts between them:

1. MCTS generates native legal moves.
2. The represented evaluator calls `move_to_action` for each legal move.
3. The low-level batcher gathers those policy logits.
4. MCTS stores and returns native moves.
5. Self-play converts `SearchResult<Move>` back to sparse `(Action, probability)` replay targets.
6. `action_to_move` is used for round-trip tests, debugging, model inspection, and compatibility loading. It is not the normal MCTS expansion path.

---

## 8. AlphaZero representation contract

Target file: `crates/alphazero/src/representation/mod.rs`

```rust
pub trait AlphaZeroRepresentation<G: GameState>:
    Clone + Send + Sync + 'static
{
    const STATE_SHAPE: [usize; 3];
    const ACTION_SIZE: usize;

    fn encode_state(&self, state: &G, output: &mut [f32]);

    fn move_to_action(&self, state: &G, mv: G::Move) -> Action;

    fn action_to_move(&self, state: &G, action: Action) -> Option<G::Move>;

    /// Identifies every field that can change `encode_state` or legal-action
    /// policy lookup for this representation.
    fn encoded_state_key(&self, state: &G) -> u64;

    fn state_size() -> usize {
        Self::STATE_SHAPE.into_iter().product()
    }
}
```

Concrete types:

```rust
pub struct Connect4AzRepresentation;
pub struct ChessAzRepresentation<const HISTORY: usize>;

#[cfg(feature = "legacy-chess")]
pub struct ChessV1Representation;
```

Responsibilities moved from current game implementations:

- `Connect4::STATE_SHAPE`, `ACTION_SIZE`, and `encode_state` move to `Connect4AzRepresentation`.
- `ChessLegacyState::STATE_SHAPE`, `ACTION_SIZE`, and `encode_state` move to optional `ChessV1Representation`.
- `ChessAzState<H>::STATE_SHAPE`, `ACTION_SIZE`, `encode_state`, `encode_az_move`, `decode_az_move`, and `evaluation_cache_key` move to `ChessAzRepresentation<H>`.

`STATE_SHAPE` uses `[usize; 3]`. Convert to `i64` only at the `tch` boundary.

---

## 9. Typed evaluator layers

The current low-level batching optimization is retained, but it is wrapped in a typed game-facing evaluator.

### 9.1 Search-facing evaluator

Target file: `crates/search/src/evaluator.rs`

```rust
#[derive(Clone)]
pub struct Evaluation {
    /// Logit `k` corresponds to legal move `offsets[row] + k`.
    pub logits: Vec<f32>,
    pub value: f32,
}

pub trait PolicyValueEvaluator<G: GameState>: Send {
    /// `legal_moves[offsets[i]..offsets[i + 1]]` belongs to `states[i]`.
    fn evaluate(
        &mut self,
        states: &[G],
        legal_moves: &[G::Move],
        offsets: &[u32],
    ) -> Vec<Evaluation>;

    /// `None` disables the generic evaluation cache for this evaluator.
    fn evaluation_key(&self, state: &G) -> Option<u64> {
        let _ = state;
        None
    }
}
```

This trait has no tensor or `Action` dependency.

### 9.2 Encoded evaluator

Target file: `crates/alphazero/src/evaluator.rs`

Rename the current concepts:

```text
EvalBatch   -> EncodedEvalBatch
Evaluator   -> EncodedEvaluator
Evaluation  -> keep or rename EncodedEvaluation internally
```

```rust
pub struct EncodedEvalBatch {
    pub states: Vec<f32>,
    pub legal_actions: Vec<Action>,
    pub offsets: Vec<u32>,
}

pub trait EncodedEvaluator: Send {
    fn evaluate(&mut self, batch: &mut EncodedEvalBatch) -> Vec<Evaluation>;
}
```

`BatcherClient` continues implementing this low-level trait and continues handing reusable allocations across the worker channel.

### 9.3 Representation adapter

```rust
pub struct RepresentedEvaluator<G, R, E> {
    representation: R,
    encoded: E,
    batch: EncodedEvalBatch,
    _game: std::marker::PhantomData<fn() -> G>,
}
```

It implements `search::PolicyValueEvaluator<G>`:

- clears and reuses `batch`;
- encodes all supplied states with `R::encode_state`;
- converts flattened native legal moves with `R::move_to_action`;
- calls the low-level encoded evaluator;
- returns logits in the exact native legal-move order;
- returns `Some(R::encoded_state_key(state))` from `evaluation_key`.

This makes a model/game/representation mismatch a type error in statically dispatched code while preserving the current GPU gather path.

---

## 10. Search rules: exact lifecycle

### 10.1 Purpose

`SearchRules` handles additional adjudication that depends on:

- information before the MCTS root;
- the current root-to-leaf simulation path;
- persistent per-node cached metadata;
- but is not a normal local rule contained entirely in the current state.

Chess threefold repetition is the current use case. Connect Four uses a zero-cost no-op implementation.

### 10.2 Contract

Target file: `crates/search/src/rules.rs`

```rust
pub enum RuleResult {
    Continue,
    Terminal(TerminalValue),
}

pub trait SearchRules<G: GameState>: Send + 'static {
    type Context<'a>: Copy
    where
        Self: 'a,
        G: 'a;

    type PathState: Default;
    type NodeMeta: Default;

    /// Reuses the stored path allocation and initializes it for one traversal.
    fn reset_path<'a>(
        &self,
        context: Self::Context<'a>,
        root: &G,
        path: &mut Self::PathState,
    );

    /// Called after the state corresponding to a tree node has been reached.
    /// It may update state features required by evaluation and may adjudicate
    /// a terminal result external to `GameState::terminal_value`.
    fn enter_state<'a>(
        &self,
        context: Self::Context<'a>,
        state: &mut G,
        path: &mut Self::PathState,
        meta: &mut Self::NodeMeta,
    ) -> RuleResult;
}
```

### 10.3 `PathState`

`PathState` is reusable mutable data for one root-to-leaf traversal.

- It is stored once inside `MctsCore`, not allocated per simulation.
- `reset_path` clears and initializes it before every descent.
- For `NoExtraRules`, it is `()`.
- For chess, it is initially a reusable `Vec<u64>` containing repetition hashes along the current simulated path. Reserve approximately 128–256 entries once.

### 10.4 `NodeMeta`

`NodeMeta` is persistent data attached to each tree node.

- It survives across simulations during one search.
- For `NoExtraRules`, it is `()` and adds zero bytes.
- For chess, use the current cached data:

```rust
#[derive(Default)]
pub struct ChessNodeMeta {
    hash: u64,
    repetitions_before_current: u8,
    initialized: bool,
}
```

Field order may be adjusted after measuring padding.

### 10.5 Context

MCTS is reused across real game moves, while the authoritative repetition table changes each move. Therefore the borrowed root context is passed to each search rather than stored permanently in MCTS.

```rust
pub struct ChessRepetitionContext<'a> {
    tracker: &'a RepetitionTracker,
    root_hash: u64,
}
```

`ChessRepetitionRules` is a zero-sized rule algorithm. Its GAT context is `ChessRepetitionContext<'a>`.

### 10.6 Chess state capability

In `games::chess`, define a narrow capability used by Chess MCTS rules:

```rust
pub trait ChessRepetitionState: GameState<Move = chess::ChessMove> {
    fn repetition_hash(&self) -> u64;
    fn reversible_plies(&self) -> usize;

    /// No-op for current-position-only states. History-aware model states
    /// override it because repetition planes are part of their encoding.
    fn set_current_repetitions_before(&mut self, count: u8) {
        let _ = count;
    }
}
```

Do not put evaluation cache identity in this trait.

### 10.7 Chess rule algorithm

`ChessRepetitionRules::enter_state` should preserve current behavior:

1. Use cached `NodeMeta` if initialized.
2. Otherwise compute current position hash.
3. Start with occurrences before the root from `ChessRepetitionContext`.
4. Scan only the reversible suffix of `PathState`, excluding the current occurrence.
5. Cache the resulting count in `NodeMeta`.
6. Set the current repetition feature on history-aware search states.
7. If repetitions-before-current is at least two, return `Terminal(Draw)`.
8. Push the current hash into `PathState`.

The draw is returned to MCTS; the search state does not need a general `set_repetition_draw` mutation method.

### 10.8 Fifty-move rule

Keep it in `ChessPosition::terminal_value` because the halfmove clock is local state and is updated by `play`. `SearchRules` handles only history external to the compact state.

---

## 11. Target chess state ownership

### 11.1 `ChessPosition`

A compact, `Copy` rules/search position:

```rust
pub struct ChessPosition {
    board: chess::Board,
    ply: u16,
    halfmove_clock: u16,
    status: Status,
}
```

It implements `GameState<Move = chess::ChessMove>`.

Change:

```text
step_without_repetition(Action)
```

to:

```rust
fn play_with_effect(&mut self, mv: ChessMove) -> MoveEffect;
```

where `MoveEffect` records whether the move is irreversible. `GameState::play` calls it and ignores the effect; authoritative `ChessGame` uses the effect.

### 11.2 `ChessGame`

The only authoritative full-game Chess type:

```rust
pub struct ChessGame {
    position: ChessPosition,
    repetitions: RepetitionTracker,
    history: ChessHistory<8>,
}
```

It owns:

- full repetition counts since the last irreversible move;
- real terminal repetition adjudication;
- the latest eight real game frames;
- repetition count stored with each real frame;
- FEN loading and replay of real moves.

### 11.3 `ChessHistoryState<HISTORY>`

Rename current `ChessAzState<HISTORY>` to `ChessHistoryState<HISTORY>`.

It is a compact `Copy` state with a fixed array of frames and implements `GameState`.

- It contains no tensor encoding methods.
- It steps using native `ChessMove`.
- Its simulated `play` rotates frames allocation-free.
- It computes a local repetition count from retained frames; `ChessRepetitionRules` overwrites the current count with the authoritative pre-root plus path count before network evaluation.

### 11.4 Frame storage in the authoritative game

When `ChessGame::play` reaches a new real position:

1. update/reset `RepetitionTracker` according to irreversibility;
2. determine `repetitions_before = current_count - 1`;
3. push `HistoryFrame { position, repetitions_before }` into the eight-frame ring;
4. set real-game repetition terminal status if count reaches three.

This allows:

```rust
pub fn history_state<const H: usize>(&self) -> ChessHistoryState<H>;
pub fn position_state(&self) -> ChessPosition;
pub fn repetition_context(&self) -> ChessRepetitionContext<'_>;
```

### 11.5 Remove duplicated types

After migration, delete:

- `ChessAzGame<HISTORY>`;
- `ChessLegacyState` in the default build;
- app-local `ChessAzGameKind` wrappers that own separate authoritative games.

Runtime history dispatch remains a simple match that calls `game.history_state::<1|4|8>()`.

---

## 12. Target MCTS structure

```rust
pub struct Mcts<G, E, R = NoExtraRules>
where
    G: GameState,
    E: PolicyValueEvaluator<G>,
    R: SearchRules<G>,
{
    inner: MctsKind<G, E, R>,
}
```

Node:

```rust
struct Node<M, Meta> {
    parent: Option<u32>,
    first_child: u32,
    move_from_parent: Option<M>,
    visits: u32,
    virtual_loss_count: u32,
    value_sum: f32,
    prior: f32,
    logit: f32,
    terminal_value: f32,
    num_children: u16,
    flags: NodeFlags,
    meta: Meta,
}
```

Implementation guidance:

- Start with `Option<M>` for the root’s missing move.
- Immediately measure `size_of::<Node<ChessMove, ChessNodeMeta>>()` and `size_of::<Node<Connect4Move, ()>>()`.
- If `Option<M>` causes a measurable node-size/search regression, replace only the root representation, for example with a separate root record or a parallel move arena. Do not use invalid sentinel values or unsafe zero initialization without a proven safe wrapper.
- Combine booleans into flags only after the typed migration is correct; do not mix layout micro-optimization into the first generic-MCTS commit.

MCTS buffers:

```rust
nodes: Vec<Node<G::Move, R::NodeMeta>>
legal_moves: Vec<G::Move>
legal_offsets: Vec<u32>
policy_buf: Vec<(G::Move, f32, f32)>
path_state: R::PathState
pending_leaves: ...
eval_cache: Option<Arc<EvalTable<G::Move>>>
```

`SearchResult`:

```rust
pub struct SearchResult<M> {
    pub policy: Vec<(M, f32)>,
    pub selected_move: M,
    pub value: f32,
}
```

Remove generic `dense_policy`; dense/sparse action conversion belongs to AlphaZero replay/analysis code.

---

## 13. Alpha-beta compatibility target

The base architecture must support alpha-beta without importing AlphaZero concepts.

Target initial interfaces in `search/src/alphabeta/mod.rs`:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Score(i32);

pub trait StaticEvaluator<G: GameState> {
    fn evaluate(&mut self, state: &G) -> Score;
}

pub trait AlphaBetaPosition: GameState + Copy {
    fn position_key(&self) -> u64;
}
```

Initial kernel may search copied states:

```rust
fn negamax<G, E>(state: G, depth: u8, alpha: Score, beta: Score, eval: &mut E) -> Score
where
    G: AlphaBetaPosition,
    E: StaticEvaluator<G>;
```

Why copy-based first:

- current Chess and Connect Four states are compact and already copied by MCTS;
- it validates the architecture without forcing a make/unmake API into every game;
- a Chess-specific make/unmake capability can be added later if profiling proves copies are the bottleneck.

The refactor is architecturally successful only if adding this kernel requires no changes to:

- `AlphaZeroRepresentation`;
- `Action`;
- replay;
- neural evaluator batching;
- checkpoint formats;
- MCTS policy encoding.

---

# 14. Ordered implementation plan

## Phase 0 — Baseline, tests, benchmarks, and CI

### Objective

Create a trustworthy behavioral and performance baseline before changing any public abstraction.

### Changes

1. Create `docs/architecture/current-state.md` summarizing the current types and known coupling.
2. Add benchmark support. Prefer Criterion with deterministic positions and fixed seeds.
3. Add size/layout tests or a small benchmark utility that records:
   - `size_of::<ChessPosition>()`;
   - `size_of::<ChessAzState<1|4|8>>()`;
   - `size_of::<Connect4>()`;
   - current MCTS node size.
4. Record release-mode baselines for:
   - Connect Four legal generation + step;
   - Chess legal generation + step;
   - Chess v2 encode for H1/H4/H8;
   - move/action encode/decode;
   - PUCT simulations/s using a deterministic mock evaluator;
   - Gumbel simulations/s using a deterministic mock evaluator;
   - repetition-aware Chess MCTS simulations/s;
   - self-play positions/s;
   - CPU batcher throughput;
   - CUDA batcher throughput and transferred legal-logit count where available;
   - replay insertion/sample throughput.
5. Preserve/add tests for:
   - Chess perft at standard shallow depths;
   - checkmate, stalemate, fifty-move rule, repetition;
   - castling, en passant, promotions;
   - all policy round trips and collision tests;
   - v2 history/repetition planes and cache key differences;
   - leaf batching, virtual loss, duplicate-leaf coalescing;
   - PUCT/Gumbel deterministic cases;
   - eval-cache hit/miss behavior;
   - replay sparse-policy semantics;
   - Batcher allocation handoff and CPU/CUDA result equivalence.
6. Add CI commands:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

### Affected files

- new `benches/` or per-crate benchmark directories;
- existing game/MCTS/batcher/replay tests;
- workspace CI configuration;
- `docs/architecture/current-state.md`.

### Acceptance criteria

- Baseline numbers are committed to `docs/architecture/performance-baseline.md` with hardware/compiler/build flags.
- All existing tests pass before refactoring.
- No architectural changes yet.

---

## Phase 1 — Add native moves and `GameState` beside the old API

### Objective

Introduce the new rules abstraction without breaking all current consumers at once.

### Changes

1. Add `TerminalValue` and `GameState` to `engine_core`.
2. Add `Connect4Move(u8)`.
3. Use `chess::ChessMove` as Chess’s native move.
4. Change `ChessPosition` internal stepping to accept `ChessMove`:

```text
step_without_repetition(Action)
    ->
play_with_effect(ChessMove) -> MoveEffect
```

5. Implement `GameState` for:
   - `Connect4`;
   - `ChessPosition`;
   - current `ChessAzState<H>` temporarily, using native moves.
6. Keep the old `Game` trait temporarily through compatibility implementations/adapters.
7. Rename current action functions immediately for clarity:

```text
encode_move       -> encode_v1_action
decode_move       -> decode_v1_action
encode_az_move    -> encode_v2_action
decode_az_move    -> decode_v2_action
```

8. Stop using action conversion inside Chess rule stepping. Compatibility `Game::step(Action)` may decode before calling native `play`, but the rules method itself must not accept `Action`.

### Affected current files

- `crates/core/src/game.rs`
- `crates/core/src/lib.rs`
- `crates/games/src/connect4/mod.rs`
- `crates/games/src/chess/action.rs`
- `crates/games/src/chess/position.rs`
- `crates/games/src/chess/az.rs`
- `crates/games/src/chess/game.rs`
- tests using raw action IDs.

### Performance gate

- Native `play(ChessMove)` is not slower than current decode-then-step.
- `Connect4Move` remains one byte; `Option<Connect4Move>` size is recorded.
- Search code is still unchanged in this phase.

### Acceptance criteria

- All game states have a native move API.
- Rule code no longer intrinsically depends on a policy encoding.
- Old consumers still compile through temporary bridges.

---

## Phase 2 — Clean the game layer and consolidate Chess authoritative state

### Objective

Make `games` contain game rules, authoritative sessions, setup loading, notation, and compact history—but no neural tensor or policy-space ownership.

### Changes

1. Rename setup DTOs:

```text
games::position::ChessPosition    -> ChessSetup
games::position::Connect4Position -> Connect4Setup
games::position::PositionSpec     -> GameSetup
```

Move them to `games/src/setup.rs` or game-specific setup modules.

2. Remove `PositionCodec` from `engine_core`.
   - Implement `ChessGame::from_setup(&ChessSetup)` and `Connect4::from_setup(&Connect4Setup)`.
   - The app already matches `GameSetup`; a generic setup trait is unnecessary.
3. Remove `anyhow` from `engine_core/Cargo.toml`.
4. Add zero-sized notation types and move parsing/formatting out of `Game` implementations.
5. Extract `RepetitionTracker` from `ChessGame` into `chess/repetition.rs`.
6. Add fixed-capacity real history storage of eight `HistoryFrame`s to `ChessGame`.
7. Rename `ChessAzState<H>` to `ChessHistoryState<H>` and strip all encoding/action methods from it only after Phase 3 bridges are ready.
8. Add:

```rust
ChessGame::position_state() -> ChessPosition
ChessGame::history_state::<H>() -> ChessHistoryState<H>
ChessGame::repetition_context() -> ChessRepetitionContext<'_>
```

9. Migrate real-game stepping to native `ChessMove` and update repetition/history exactly once in `ChessGame::play`.
10. Replace `ChessAzGame<H>` call sites with the unified `ChessGame` plus snapshots.
11. Delete `crates/games/src/chess/az_game.rs` after all consumers migrate.

### Important history correctness

- `ChessGame::from_fen` starts with only the current frame because prior history is unknown.
- Replaying moves after FEN populates real frames.
- Irreversible moves clear repetition counts but do not clear neural history frames.
- Every stored real frame records the repetition count that existed at that time.

### Affected current files

- `crates/core/src/rules.rs`
- `crates/core/Cargo.toml`
- `crates/games/src/position.rs`
- all `crates/games/src/chess/*.rs`
- app/evaluation setup construction call sites.

### Performance gate

- `ChessGame::play` is stepped only once per real move, so the eight-frame history must not enter MCTS copies except through selected snapshot length.
- `ChessHistoryState<H>` remains fixed-size, allocation-free, and `Copy`.
- State sizes do not grow unexpectedly; record before/after values.

### Acceptance criteria

- One authoritative Chess game type exists.
- Search snapshots are constructed without heap allocation.
- `ChessAzGame` and `PositionCodec` are gone.
- Core has no error-handling dependency.

---

## Phase 3 — Introduce `Action` newtype and AlphaZero representations

### Objective

Move neural representation responsibility out of game rules while preserving current state/action layouts exactly.

### Changes

1. Add AlphaZero `Action` newtype.
2. Add `AlphaZeroRepresentation<G>` with the exact final names:
   - `STATE_SHAPE`;
   - `ACTION_SIZE`;
   - `encode_state`;
   - `move_to_action`;
   - `action_to_move`;
   - `encoded_state_key`.
3. Implement:
   - `Connect4AzRepresentation` for `Connect4`;
   - `ChessAzRepresentation<H>` for `ChessHistoryState<H>`;
   - optional `ChessV1Representation` for `ChessPosition` under `legacy-chess`.
4. Move Connect Four tensor encoding to its representation.
5. Move Chess v2 feature encoding and action-plane encoding from `games/chess/az.rs` and `action.rs` to `alphazero/representation/chess_v2.rs`.
6. Move current v2 evaluation-cache key logic to `ChessAzRepresentation<H>::encoded_state_key`.
7. Add exhaustive round-trip tests:

```text
legal move -> action -> move == original move
```

for random legal positions, promotions, castling, en passant, both colors, H1/H4/H8. 8. Add collision tests for all legal moves in one position. 9. Add golden encoding tests to prove the tensor layout is unchanged from current checkpoints. 10. Introduce temporary adapters so old evaluator/MCTS code can still consume represented states while later phases are implemented.

### Placement during migration

Before the physical crate split, add these under `crates/algorithms/src/alphazero/representation/`. They will later move mechanically into the new `alphazero` crate.

### Performance gate

- `move_to_action` matches or beats current encoders.
- No heap allocation in move/action conversion or state encoding.
- Encoded bytes/floats are identical to current implementation for all golden cases.

### Acceptance criteria

- Game states no longer own model dimensions or tensor encoding.
- `Action` is no longer exported by `engine_core`.
- All policy-index conversion is visibly located in representation modules.

---

## Phase 4 — Split encoded evaluator from typed search evaluator

### Objective

Tie game, move type, representation, and model together in the type system without losing batching performance.

### Changes

1. Rename current evaluator types to encoded-layer names.
2. Keep `BatcherClient` implementing `EncodedEvaluator` and preserve channel/allocation behavior.
3. Add `search::PolicyValueEvaluator<G>` logically inside the current algorithms crate first.
4. Add `RepresentedEvaluator<G, R, E>` with an owned reusable `EncodedEvalBatch`.
5. Move state encoding and native-move-to-action mapping into this adapter.
6. Update Batcher worker loops to use `Action::index()`/`as_u32()` instead of casts.
7. Preserve on-device legal-logit gather; only the source of legal action indices changes.
8. Move evaluation-cache identity to `PolicyValueEvaluator::evaluation_key`, implemented by the represented evaluator through `encoded_state_key`.
9. Update mock evaluators/tests to implement the typed trait directly.

### Affected current files

- `crates/algorithms/src/alphazero/evaluator.rs`
- `crates/algorithms/src/alphazero/batcher/mod.rs`
- `crates/algorithms/src/alphazero/batcher/worker.rs`
- batcher tests
- MCTS evaluation tests.

### Performance gate

- Same number of allocations per request as current Batcher client.
- Same legal-logit gather count.
- CPU/CUDA outputs equal current outputs.
- Dynamic batching throughput within measurement noise of baseline; investigate any repeatable regression over 2–3%.

### Acceptance criteria

- A represented evaluator cannot be paired with the wrong native state type without a compile error.
- Low-level Batcher remains representation-agnostic and consumes encoded buffers/actions.

---

## Phase 5 — Make MCTS generic over native moves and `SearchRules`

### Objective

Remove AlphaZero policy IDs and Chess-specific repetition fields from generic search.

### Changes

1. Change `SearchResult` to `SearchResult<M>` with native moves.
2. Change `Node` to `Node<M, Meta>`.
3. Replace common `hash/repetitions/repetition_cached` fields with `meta: R::NodeMeta`.
4. Replace `repetition_path: Vec<u64>` with `path_state: R::PathState`.
5. Make `EvalTable` generic over native move type:

```rust
EvalTable<M>
CachedEvaluation<M> { legal_moves: Vec<M>, eval: Evaluation }
```

6. Migrate evaluation buffers to flattened native legal moves and offsets.
7. MCTS calls the typed evaluator; MCTS never calls `encode_state` or sees `Action`.
8. Add `NoExtraRules` for Connect Four.
9. Implement `ChessRepetitionRules` and `ChessNodeMeta` using the exact current repetition algorithm.
10. Replace:

```text
search_with_mode
search_with_repetitions_mode
SearchDriver
StandardSearch
RepetitionSearch
```

with one typed search entry point:

```rust
mcts.search(root, rules_context, mode)
```

or two thin convenience methods where `NoExtraRules` uses `()` context. Do not retain duplicate traversal implementations. 11. Preserve PUCT/Gumbel algorithm code. Change only move/meta types and generic bounds in the first commit. 12. Preserve root exploration semantics:

- PUCT Dirichlet noise only in explore mode;
- Gumbel sample only in explore mode;
- current policy-target definitions unchanged.

13. After correctness, inspect and optimize node field layout.

### Suggested implementation slices

1. Generic `SearchResult<M>`.
2. Generic cache and policy buffers.
3. Generic node move type with `NoExtraRules` only.
4. Port standard Connect Four tests.
5. Add generic rule metadata/path state.
6. Port Chess repetition.
7. Delete `SearchDriver` and old repetition branches.
8. Port Gumbel tests.

### Performance gate

- Connect Four node size should decrease because Chess metadata disappears.
- Chess node size should not materially exceed current size after accounting for `ChessNodeMeta`.
- PUCT/Gumbel simulations/s no worse than baseline beyond 3% on repeated median measurements.
- No new allocation appears in a simulation profile.

### Acceptance criteria

- `search` code contains no AlphaZero `Action`, `STATE_SHAPE`, `ACTION_SIZE`, or tensor encoding.
- Connect Four nodes contain no Chess metadata.
- One traversal path handles both ordinary and repetition-aware search through static rule specialization.

---

## Phase 6 — Remove old core abstractions and compatibility bridges

### Objective

Finish the conceptual cutover so there is one architecture rather than old and new APIs side by side.

### Delete or replace

- old `Game` trait;
- `TensorDim`;
- `engine_core::Action`;
- `RepetitionGame`;
- `PositionCodec`;
- `legal_actions`, `step(Action)`, `reward`, `encode_state`, `parse_move`, and `format_action` implementations on game state types;
- temporary represented-game adapters.

### Global API migrations

```text
legal_actions  -> legal_moves
step            -> play
reward          -> terminal_value().map(TerminalValue::as_f32)
act_with_mode   -> select_move
best_action     -> best_move
sample_action   -> sample_move
format_action   -> notation.format_move
parse_move      -> notation.parse_move
```

### Acceptance criteria

Repository-wide searches return no default-path references to:

```text
engine_core::game::Action
trait Game
RepetitionGame
PositionCodec
Game::ACTION_SIZE
Game::STATE_SHAPE
Game::NAME
encode_state on GameState
```

---

## Phase 7 — Refactor replay, self-play, and training boundaries

### Objective

Keep replay model-oriented, keep search move-oriented, remove duplicated worker orchestration, and remove Chess-specific fields from generic configuration.

### Replay

1. `Transition.policy` remains sparse but explicitly uses AlphaZero `Action`:

```rust
pub struct Transition {
    pub state: Vec<f32>,
    pub policy: Vec<(Action, f32)>,
    pub value: f32,
}
```

2. Rename `reward` to `value` in replay/training data to distinguish terminal game reward from a training target attached to a state.
3. Convert `SearchResult<G::Move>` to replay actions in self-play with `representation.move_to_action`.
4. Keep last-write-wins duplicate canonicalization only if compatibility requires it; otherwise reject duplicate actions in debug/tests because legal moves should map injectively.
5. Keep sparse sampling and gather-based policy loss.

### Self-play configuration

Split current `SelfPlayConfig` into cohesive values:

```rust
SelfPlayWorkerConfig {
    num_games,
    threads,
    max_moves,
    progress_every,
}

SearchSchedule {
    full_profile,
    fast_profile,
    full_probability,
}

TemperatureSchedule { ... }
ResignationConfig { ... }
EvalCacheConfig { entries }
```

Remove all `chess_v2_*` prefixes from generic structures. Chess v2 defaults belong in app/model-specific configuration construction.

### Self-play orchestration

Do not introduce a large `SelfPlayDomain` trait yet. With two games, use:

- one shared worker runner that owns threads, completion claiming, replay insertion, stats, and progress;
- explicit `play_connect4_episode` and `play_chess_episode<const H: usize>` functions;
- shared helpers for search schedule, temperature selection, resignation, and backward value assignment.

This removes duplication without creating an abstraction more complicated than the domains.

### Chess episode

Each move:

1. derive `ChessHistoryState<H>` from authoritative `ChessGame`;
2. encode with `ChessAzRepresentation<H>` for replay;
3. search with typed represented evaluator and `ChessRepetitionRules` using `game.repetition_context()`;
4. receive `SearchResult<ChessMove>`;
5. convert sparse move policy to sparse `Action` policy for replay;
6. apply selected `ChessMove` to authoritative `ChessGame`.

### Connect Four episode

`Connect4` is both authoritative state and search state. It uses `NoExtraRules`.

### Training

- Trainer consumes model-shaped replay batches only.
- Network construction derives state/action dimensions from selected model/representation configuration rather than storing duplicated dimensions in game traits.
- Keep scalar-value training for Connect Four and WDL training for Chess v2.

### Affected current files

- `alphazero/replay/*`
- `alphazero/selfplay/*`
- `crates/app/src/chess_selfplay.rs`
- `crates/app/src/bin/train*.rs`
- trainer tests.

### Performance gate

- Self-play positions/s not worse than baseline beyond 3%.
- Replay insertion/sample throughput not worse than baseline.
- No extra state encoding per move: encode once for replay and evaluator as required; if duplicate encoding is observed, add an explicit encoded-state reuse path only after profiling.

### Acceptance criteria

- One worker orchestration implementation.
- Explicit game episode functions.
- Generic configs contain no Chess-v2-named fields.
- Search policies use native moves; replay policies use `Action`.

---

## Phase 8 — Replace run/model configuration with one tagged source of truth

### Objective

Remove `game + net + architecture` inconsistencies and separate reusable network types from old Chess compatibility.

### Rename reusable network

Current `LegacyAlphaZeroNet` is also the active Connect Four network. Rename it to a descriptive architecture name, for example:

```text
LegacyAlphaZeroNet -> ScalarValueResidualNet
NetConfig          -> ScalarValueNetConfig
```

Do not call ordinary Connect Four support “legacy.”

### Target config

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub format_version: u32,
    pub model: ModelConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "config", rename_all = "kebab-case")]
pub enum ModelConfig {
    Connect4AlphaZero(Connect4ModelConfig),
    ChessAlphaZeroV2(ChessModelConfig),

    #[cfg(feature = "legacy-chess")]
    ChessAlphaZeroV1(ChessV1ModelConfig),
}
```

`ChessModelConfig` should contain tunable architecture fields rather than hiding them as constants:

```rust
history: usize
channels: i64              # default 128
residual_blocks: usize     # default 12
se_hidden: i64
value_hidden: i64
```

Fixed representation facts such as 73 policy planes and `ACTION_SIZE` come from `ChessAzRepresentation<H>`, not duplicated config values.

`Connect4ModelConfig` stores network hyperparameters only. Input shape and action size come from `Connect4AzRepresentation`.

### Migration

1. Add a parser for current config files.
2. Convert current v2 runs and Connect Four runs to the new format.
3. Add a one-time CLI migration command or script if needed.
4. Write only the new format for new runs.
5. After verified migration, remove fallback parsing from default code. Keep old Chess only under `legacy-chess` if explicitly required.

### Runtime dispatch

Match `ModelConfig` once at app/training/evaluation boundaries and enter concrete generic functions:

```rust
match cfg.model {
    ModelConfig::ChessAlphaZeroV2(cfg) if cfg.history == 4 => run_chess::<4>(cfg),
    ...
}
```

Explicit `1|4|8` matches are correct and should remain.

### Acceptance criteria

- A run config has one model variant that fully determines game, representation, network, state shape, and action size.
- No ignored `net` field exists in Chess v2 configs.
- Connect Four is not represented as a generic “legacy” architecture.

---

## Phase 9 — Migrate app, analysis, UCI, evaluations, and web protocol

### 9.1 Application dispatch

Current generic `run_game<G: Game>` relies on parsing/display/model methods in one trait. Replace it with explicit game dispatch or generic helpers parameterized by state + notation + agent.

Supported engine combinations should be data, not implied by blanket implementations:

```rust
pub enum EngineKind {
    Human,
    AlphaZero,
    AlphaBeta,
    Random,
}
```

Reject unsupported combinations during app configuration.

Remove string comparisons such as `G::NAME == ChessGame::NAME`.

### 9.2 Player agents

- `HumanAgent` should receive a notation/parser value or be implemented as app-level input handling, then return a native move.
- `AlphaZeroAgent<G, R, Rules>` owns Batcher + typed MCTS and returns `G::Move`.
- Use per-game runtime enums when necessary; do not erase move types into a global integer.

### 9.3 Analysis

Move `crates/algorithms/src/analysis.rs` to AlphaZero analysis code.

Internal policy rows should be native moves. Serialized DTO:

```rust
pub struct MoveScore {
    pub mv: String,
    pub probability: f32,
    pub policy_action: Option<u32>,
}
```

Rename:

```text
best_action -> best_policy_action
```

or remove it if consumers only need `best_move`. Any numeric action in JSON must be explicitly documented as a model policy index, not a universal game move.

### 9.4 Proxy sessions

- Use one `ChessGame` session regardless of loaded model.
- Remove `ChessAzV2Session` and app-local `ChessAzGameKind` authoritative wrappers.
- At engine turn, match model history and derive the corresponding snapshot from the same `ChessGame`.
- Legal session moves are native game moves and model-independent.

### 9.5 UCI

Current UCI keeps `game`, `position_base`, `position_moves`, and an optional v2 authoritative game. Replace with one authoritative `ChessGame` that is rebuilt/replayed by the UCI `position` command and already retains history.

`bestmove`:

1. load/match model config;
2. derive `ChessPosition` or `ChessHistoryState<H>`;
3. search and receive `ChessMove`;
4. format UCI directly.

Remove all policy-action parsing/formatting from UCI.

This same UCI game state later supports alpha-beta by matching `EngineKind`/model type.

### 9.6 Evaluations

- Puzzle evaluation searches a `ChessGame` snapshot and compares formatted native moves.
- Arena/fastchess remain protocol-level and mostly unchanged.
- Remove ordinary `legacy` modules after old Chess-v1 support is removed or feature-gate them.
- Rename package `checkpoint-eval` to a broader `engine_evaluations`/`engine-evaluations` name if it now evaluates multiple engine kinds; update app dependency names.

### 9.7 Web protocol

The frontend currently declares `LegalMove { action: number; move: string }` but appears to use move strings. Change session legal moves to:

```ts
interface LegalMove {
  move: string;
}
```

or use a game-native serializable move DTO if needed. Do not expose an AlphaZero policy action as the identity of a legal game move.

Analysis policy entries may retain `policy_action` for visualization/debugging, clearly named.

### Acceptance criteria

- App/UCI/session game loops operate only on native moves.
- One authoritative Chess game is used by all model histories.
- Web legal moves do not depend on the loaded model’s policy encoding.
- Analysis distinguishes move text from optional policy index.

---

## Phase 10 — Physically split `algorithms` into `search` and `alphazero`

### Objective

Perform a mostly mechanical crate move after APIs are stable.

### New `search` crate

Move:

- generic MCTS;
- evaluator/search-rule contracts;
- evaluation table;
- alpha-beta kernel/interfaces.

Dependencies must not include:

- `tch`;
- `serde_json`;
- `anyhow` unless a public configuration parser genuinely requires it;
- concrete games outside dev-dependencies.

### New `alphazero` crate

Move:

- `Action` and representations;
- represented/encoded evaluators;
- Batcher;
- networks;
- replay;
- self-play;
- trainer;
- checkpoints/run config;
- AlphaZero analysis;
- Chess repetition-rule adapter for MCTS.

### Workspace changes

- add `crates/search` and `crates/alphazero`;
- update app/evaluation dependencies;
- delete `crates/algorithms` after imports are migrated;
- update build scripts and README.

### Build-script note

The native `libtorch` link behavior in `app` and `evaluations` may still require package-local build scripts because Cargo build-script link args are package-scoped. Do not remove the working `--no-as-needed/-ltorch` behavior merely to deduplicate ten lines. A small shared included build-script source is acceptable only if verified on the current Linux toolchain.

### Acceptance criteria

```bash
cargo check -p search
```

does not build `tch` or require `LIBTORCH`.

---

## Phase 11 — Remove or quarantine old Chess-v1 support

### Default target: remove it

Delete default-path files/branches corresponding only to old Chess policy/checkpoints:

- `games/chess/legacy.rs`;
- old Chess `64*64*5` policy encoding;
- `app/src/bin/train/legacy.rs` Chess branch;
- Chess-v1 proxy/UCI/evaluation branches;
- `evaluations/src/legacy.rs` and support if no longer needed;
- `RunArchitecture::Legacy` Chess semantics.

Do not delete the scalar-value residual network if Connect Four uses it; rename and retain it.

### If old checkpoints must remain usable

- add `legacy-chess` feature, disabled by default;
- place code under `alphazero/src/compat/v1` and evaluation compatibility modules;
- add explicit compatibility tests;
- exclude it from normal app catalogs unless the feature is enabled;
- document an eventual removal date/condition.

### Acceptance criteria

Default builds and primary source files contain no branching for obsolete Chess formats.

---

## Phase 12 — Add an alpha-beta architectural proof

### Objective

Verify the new boundaries before declaring the refactor complete.

### Implement a minimal kernel

- negamax;
- alpha-beta pruning;
- iterative deepening;
- simple transposition table;
- basic material/static evaluation for Chess, placed in a future `classical`/engine adapter module rather than in `games` rules;
- deterministic move ordering sufficient for tests.

Do not add advanced Chess-engine features yet:

- NNUE;
- null-move pruning;
- late-move reductions;
- quiescence;
- aspiration windows;
- singular extensions.

### Integration proof

Add `EngineKind::AlphaBeta` for Chess CLI/UCI behind a basic configuration. The game loop should accept the returned `ChessMove` without any representation conversion.

### Acceptance criteria

- Alpha-beta code depends on `GameState`, native move, position key, and static evaluator only.
- No AlphaZero crate import in the generic alpha-beta module.
- No architecture changes to AlphaZero are needed to add it.

---

## Phase 13 — Final performance and debt pass

### MCTS layout

Measure and consider only evidence-based changes:

- parent sentinel versus `Option<u32>`;
- root move representation;
- flags byte instead of separate booleans;
- field ordering/padding;
- whether `reward`/terminal value needs a full `f32` field;
- whether policy `logit` is needed for PUCT nodes or can remain variant-specific without harming layout;
- structure-of-arrays only if profiling proves a cache bottleneck.

### State copying

Benchmark `ChessHistoryState<1|4|8>`. Do not introduce pointer-linked history unless H8 copy cost is a measured bottleneck.

### Allocation audit

Profile one Chess search and verify no repeated allocation in:

- legal move collection;
- traversal path;
- leaf batch;
- evaluator encoded batch;
- policy buffer;
- Gumbel root action buffer;
- repetition rules;
- replay insertion.

### Error and API cleanup

- replace externally reachable panics with `Result` where input is untrusted;
- retain assertions for internal invariants/hot-path illegal calls;
- narrow visibility (`pub(crate)`/private);
- remove re-exports that exist only for old imports;
- add module-level docs for core/search/AlphaZero boundaries;
- standardize `move`, `action`, `state`, `game`, `position`, and `setup` terminology.

### Documentation

Update:

- root README architecture and supported matrix;
- training/config examples;
- UCI usage;
- checkpoint format version/migration;
- frontend API types;
- architecture diagram;
- benchmark results before/after.

### Final performance acceptance

On the same machine/compiler/build:

- no statistically repeatable >3% regression in Chess/Connect Four MCTS throughput without a documented reason;
- Batcher throughput/gather behavior unchanged within noise;
- self-play positions/s unchanged within noise;
- Connect Four node size decreases or remains equal;
- Chess node size remains controlled;
- no new per-simulation heap allocation;
- all strength/evaluation smoke tests remain consistent.

---

# 15. Current-file disposition map

## `crates/core`

| Current file   | Target action                                                                                                    |
| -------------- | ---------------------------------------------------------------------------------------------------------------- |
| `src/game.rs`  | replace broad `Game` with `GameState` + `TerminalValue`; remove `Action`, tensor shape, notation, model metadata |
| `src/agent.rs` | return `G::Move`; rename action methods to move methods                                                          |
| `src/rules.rs` | delete `PositionCodec` and `RepetitionGame`; search rules move to `search`, setup loaders to `games`             |
| `src/lib.rs`   | export only minimal core contracts                                                                               |
| `Cargo.toml`   | remove `anyhow`                                                                                                  |

## `crates/games`

| Current file        | Target action                                                                                       |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| `src/position.rs`   | rename/move setup DTOs; no generic `PositionCodec`                                                  |
| `connect4/mod.rs`   | split move/state/notation if useful; remove tensor/action encoding; implement `GameState`           |
| `chess/action.rs`   | move v2 and optional v1 policy encoding to AlphaZero representation; retain no policy code in rules |
| `chess/position.rs` | native `ChessMove` stepping; implement `GameState`; local terminal rules                            |
| `chess/game.rs`     | one authoritative game with repetition tracker + 8-frame history                                    |
| `chess/az.rs`       | rename to history state; remove encoding/action/cache methods                                       |
| `chess/az_game.rs`  | delete after unified `ChessGame` migration                                                          |
| `chess/legacy.rs`   | remove default-path old Chess representation; no wrapper needed for current-position search         |
| `chess/notation.rs` | retain SAN/PGN; add UCI notation implementation                                                     |
| `chess/zobrist.rs`  | retain optimized hasher; move repetition tracker around it as needed                                |
| tests               | split rule tests from representation tests; representation tests move to AlphaZero crate            |

## Current `crates/algorithms`

| Current area                  | Target action                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| `alphazero/mcts/*`            | move to `search/mcts`; native moves + generic rules/evaluator                                          |
| `alphazero/evaluator.rs`      | split search-facing trait and encoded AlphaZero evaluator                                              |
| `alphazero/batcher/*`         | retain in `alphazero`; update to Action newtype/represented adapter                                    |
| `alphazero/network.rs`        | split/rename reusable scalar network and Chess v2 network; derive dimensions from model/representation |
| `alphazero/network/legacy.rs` | rename reusable architecture; isolate only Chess-v1 compatibility                                      |
| `alphazero/replay/*`          | retain in `alphazero`; sparse policy uses `Action`; reward -> value                                    |
| `alphazero/selfplay/*`        | shared worker runner + explicit episodes; remove Chess-named generic fields                            |
| `alphazero/trainer.rs`        | retain in `alphazero`; consume model config and sparse replay                                          |
| `alphazero/checkpoint.rs`     | new versioned tagged `ModelConfig`; migration support                                                  |
| `analysis.rs`                 | move to `alphazero/analysis`; native moves internally, explicit policy action in DTO                   |
| `lib.rs`                      | crate deleted after split                                                                              |

## `crates/app`

| Current area           | Target action                                                                                       |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| `app.rs`               | explicit game/engine dispatch; no `G::NAME`; native moves                                           |
| `players.rs`           | typed AlphaZero agent; human parsing through notation; future alpha-beta agent                      |
| `chess_selfplay.rs`    | merge orchestration into AlphaZero self-play; retain explicit Chess episode only                    |
| `proxy.rs` + sessions  | one Chess authoritative session; remove v2 authoritative wrappers                                   |
| `uci.rs`               | one ChessGame; model match derives search snapshot; returns ChessMove                               |
| `eval_app.rs`          | setup DTO renames; updated analysis schema                                                          |
| evaluation job service | largely unchanged; update binary/package names and model config discovery only                      |
| visualization          | update setup and analysis DTO names; no architecture logic                                          |
| train binaries         | split model-specific construction cleanly; new config enum; no duplicated legacy/v2 source of truth |

## `crates/evaluations`

| Current area           | Target action                                               |
| ---------------------- | ----------------------------------------------------------- |
| arena/fastchess/report | preserve; update model config/package names                 |
| puzzle                 | one ChessGame + snapshot dispatch; native move output       |
| legacy modules         | delete or feature-gate                                      |
| model engine           | extend engine kind later; UCI options remain protocol-level |

## `web`

| Current area              | Target action                                                   |
| ------------------------- | --------------------------------------------------------------- |
| `lib/types.ts`            | legal moves are move strings/native DTOs, not AlphaZero actions |
| analysis components       | accept `policy_action` only as explicit optional model metadata |
| local game implementation | no architectural change required in this refactor               |

---

# 16. Suggested commit sequence

Each numbered item should be a compiling commit or a small group of tightly related compiling commits.

1. Add baseline docs, tests, benchmarks, and CI.
2. Add `TerminalValue` and `GameState` without deleting `Game`.
3. Add `Connect4Move`; implement native Connect Four API.
4. Change `ChessPosition` to native `ChessMove` stepping.
5. Implement `GameState` for Chess states through compatibility bridges.
6. Rename setup DTOs and remove `PositionCodec`/core `anyhow`.
7. Add notation types and migrate parsing/formatting call sites.
8. Extract `RepetitionTracker` and authoritative eight-frame Chess history.
9. Replace `ChessAzGame` with unified `ChessGame` snapshots.
10. Add AlphaZero `Action` newtype and representation trait.
11. Extract Connect Four representation.
12. Extract Chess v2 representation and golden tests.
13. Isolate/remove Chess-v1 representation.
14. Rename encoded evaluator types.
15. Add typed `PolicyValueEvaluator` and `RepresentedEvaluator`.
16. Make `SearchResult` generic over native move.
17. Make eval cache and policy buffers generic over native move.
18. Make MCTS nodes generic over move with `NoExtraRules`.
19. Add generic `SearchRules` path/meta/context support.
20. Port Chess repetition and remove common Chess node fields.
21. Delete `SearchDriver`, duplicate repetition traversal, and old MCTS APIs.
22. Remove old `Game`, `Action`, `RepetitionGame`, and bridges.
23. Convert replay policy boundary and rename target reward/value.
24. Consolidate self-play worker orchestration.
25. Split self-play configuration values.
26. Replace run config with versioned `ModelConfig` and migrate fixtures/runs.
27. Migrate app player/session/analysis code to native moves.
28. Simplify UCI to one authoritative Chess game.
29. Migrate evaluation crate and web DTOs.
30. Split `algorithms` into `search` and `alphazero` crates.
31. Remove/default-disable old Chess-v1 compatibility.
32. Add minimal alpha-beta kernel and Chess integration proof.
33. Perform node/state/allocation performance pass.
34. Update README, architecture docs, migration docs, and benchmark report.

---

# 17. Codex execution instructions

When handing a phase to Codex, include these rules:

1. Implement only the named phase and its explicitly required compatibility bridge.
2. Do not redesign unrelated modules.
3. Keep the workspace compiling after every commit.
4. Run format, clippy, tests, and the phase-specific benchmarks.
5. Preserve public behavior unless the phase explicitly renames an API or config format.
6. Do not remove an optimization because the typed version is initially harder; adapt it.
7. Do not introduce trait objects in search hot paths.
8. Do not add allocations to simulation loops.
9. Use native moves in rules/search and `Action` only in AlphaZero model boundaries.
10. Report:
    - files changed;
    - old API removed/new API added;
    - tests added/moved;
    - benchmark before/after;
    - remaining temporary compatibility debt.

Suggested prompt format:

```text
Implement Phase N of docs/architecture-refactor-plan.md.

Constraints:
- Do not proceed to Phase N+1.
- Preserve all performance invariants in section 3.
- Keep temporary compatibility only where Phase N specifies it.
- Run cargo fmt, clippy, workspace tests, and the phase benchmarks.
- Summarize exact API changes and benchmark deltas.
```

---

# 18. Definition of done

The refactor is complete when all of the following are true:

- `GameState` contains only native game-tree semantics.
- Chess and Connect Four rules use native moves.
- AlphaZero `Action` is a documented policy-index newtype outside core.
- `AlphaZeroRepresentation` owns `STATE_SHAPE`, `ACTION_SIZE`, encoding, conversion, and encoded-state key.
- MCTS stores native moves and is independent of tensors and policy indexing.
- Chess repetition is a typed search rule with reusable path state and Chess-only node metadata.
- Connect Four MCTS nodes contain no Chess repetition fields.
- One authoritative `ChessGame` supplies current-only and H1/H4/H8 snapshots.
- `ChessAzGame` and default-path `ChessLegacyState` are gone.
- Pure `search` builds without `tch`.
- Replay remains sparse and action-indexed at the model boundary.
- Batcher retains pinned memory, allocation handoff, request coalescing, and device-side legal-logit gathering.
- Self-play has one worker orchestration implementation and explicit game episodes.
- Run config has one versioned tagged model source of truth.
- App, proxy, UCI, evaluations, and web legal moves use native move identity.
- Old Chess-v1 support is removed or disabled behind an explicit feature.
- A minimal alpha-beta engine can search Chess without importing AlphaZero.
- Tests, clippy, and formatting pass.
- Performance is not materially worse than the recorded baseline, and all accepted differences are documented.

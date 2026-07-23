# Engine Zoo: a linear guide to the whole repository

> This guide describes the current refactored architecture. Where an older
> paragraph below uses the former `MctsConfig`, network-version, or dense-replay
> vocabulary, the refactored architecture section takes precedence.

This is a Rust workspace for building, training, serving, and evaluating board-game engines. Today it supports **Chess** and **Connect Four**. Its principal engine is AlphaZero: a neural network supplies a value estimate and move priors; Monte Carlo Tree Search (MCTS) improves those priors into a move decision and, during self-play, into training data.

The short version of the architecture is:

```text
rules state ──> representation ──> neural network ──> generic MCTS ──> move
     │                 │                  │                │
     └──── setup/API ──┴──── self-play ───┴── replay ──────┴── training
                                                              │
web browser <── Axum HTTP server <── sessions / analysis <────┘
```

The division is deliberate. Game rules must be correct without knowing anything about tensors or a particular network. Search must be reusable by a future alpha-beta engine without linking PyTorch. Neural concerns are isolated in `alphazero`; runtime choices (which game and which model to load) happen at the app boundary, rather than through slow dynamic dispatch in the search hot path.

## 1. Start here: workspace and repository-level files

`Cargo.toml` is the workspace manifest. It lists six crates and centralizes dependency versions. The dependency direction is one-way:

```text
engine_core → games, search → alphazero → engine_app, checkpoint-eval
```

`engine_core` has no dependencies. `games` depends on it and the `chess` rules library. `search` depends only on core and random-number libraries. `alphazero` joins games, search, and `tch` (Rust bindings for PyTorch). `engine_app` and `checkpoint-eval` are the outer executable layers. This prevents a generic search change from accidentally needing CUDA/PyTorch and makes lower layers easier to test.

The workspace uses Rust 2021. Release builds enable thin LTO, trading extra link time for better optimized binaries; development builds use optimization level 1 so the numerical/search code is less misleadingly slow while still compiling reasonably quickly. `Cargo.lock` pins the resolved dependency graph for reproducible builds.

`README.md` is the project entry point; `rustfmt.toml` contains formatting policy; `LICENSE` records the GPL-3.0-or-later license. `experiments/` holds immutable, checked-in TOML specifications for Connect Four PUCT, chess PUCT/WDL, chess Full Gumbel/WDL, a canonical scalar ablation, and classic chess. Root-level generated Fastchess data and archived upload snapshots are intentionally not repository inputs.

The historical implementation plan lives at `docs/archive/architecture-refactor-plan-2026-07.md`. The post-refactor audit is a review input, not runtime configuration. `scripts/make_lichess_easy_suite.py` and `crates/evaluations/install_stockfish.py` are opt-in tools rather than build inputs.

## 2. The vocabulary that makes the rest make sense

There are three notions that must not be conflated:

1. A **native move** is a game-rule move: `chess::ChessMove` in chess or `Connect4Move` in Connect Four. Rules, sessions, UCI, and MCTS edges use these.
2. An AlphaZero **Action** is a checked `u32` index into one specific network's policy vector. The same chess move has different indexes under the classic `64×64×5` and canonical `8×8×73` layouts.
3. A **representation** is the model-specific conversion between a game snapshot and tensors/actions. It owns state-plane layout, policy size, encoding, and cache identity.

Keeping these distinct is necessary because chess rules should not care whether a model has `64×64×5` actions or `8×8×73` actions, and because a non-neural engine has no policy vector at all.

Values are always from the **side to move's perspective**: `Win = +1`, `Draw = 0`, `Loss = -1`. Moving to a parent flips the sign. This convention makes backups and self-play targets uniform across alternating-turn games.

## 3. `crates/core`: the smallest stable contract

`crates/core/src/lib.rs` merely exports the three contracts below.

- `game.rs` defines `TerminalValue` and `GameState`. A state must be sendable, name its move type, list legal moves, mutate by playing a legal move, and report a terminal value. It deliberately says nothing about tensors, notation, or construction. `TerminalValue::as_f32` bridges rule outcomes to search/training arithmetic.
- `agent.rs` defines `PolicyMode` (`Explore` or `Deterministic`) and `Agent<G>`. An agent selects a native move for a game state. The mode lets callers request stochastic opening/self-play behavior without knowing an agent's internals.
- `notation.rs` defines `GameNotation<G>`, a small parse/format adapter. Keeping protocol text outside `GameState` avoids making rules depend on UCI, SAN, UI strings, or human input.

This crate is intentionally almost boring: any extra abstraction here would become a constraint on every future game and engine.

## 4. `crates/games`: authoritative rules and position loading

`games/src/lib.rs` exposes `chess`, `connect4`, and `setup`. `setup.rs` defines serde-friendly `ChessSetup` (`fen` plus UCI moves), `Connect4Setup` (columns), and tagged `GameSetup`. Its loader validates FEN/moves while replaying them. This is necessary at the HTTP/CLI trust boundary: accepting an arbitrary request must not manufacture an invalid game state.

### Chess

Chess is split because its authoritative game and its per-search snapshot have different jobs.

- `chess/game.rs` contains `ChessGame`, the authoritative real-game state: a `ChessPosition`, a `RepetitionTracker`, and move/history bookkeeping. `from_setup` builds it from FEN and replays moves. `play` updates repetition and clocks; `terminal_value` covers checkmate, stalemate, threefold repetition, and the fifty-move rule. `ChessRepetitionContext` lends the current repetition history to search without copying a hash map into every tree node.
- `chess/position.rs` contains `ChessPosition`, a compact `Copy` snapshot around `chess::Board` plus halfmove/fullmove clocks. It generates legal moves, applies a move, detects the move's effect (pawn movement, capture, castling-rights change), and determines ordinary board terminal status. It is what MCTS clones thousands of times, so compact value semantics are necessary for performance.
- `chess/repetition.rs` is `RepetitionTracker`, a hash-count map plus a reversible-history suffix. Irreversible moves reset the relevant history, because positions before them cannot legally contribute to a future repetition. This preserves chess correctness without scanning an entire game repeatedly.
- `chess/zobrist.rs` supplies a deterministic hasher for compact board-key maps. It is an implementation detail chosen for fast, predictable hashing rather than cryptographic security.
- `chess/mod.rs` re-exports the public types and declares `ChessRepetitionState`, the narrow capability required by repetition-aware search.
- `chess/notation.rs` implements legal UCI parsing/formatting and SAN/PGN generation. SAN disambiguation, captures, promotion, check/checkmate suffixes, move-number formatting, and PGN headers live here because they are presentation/protocol concerns, not game transitions.
- `chess/tests.rs` verifies rules that are easy to subtly break: legal count, mates/stalemates, clocks, castling, en passant, irreversible history, repetition, and perspective.

The external `chess` crate supplies legal board mechanics. Engine Zoo layers clocks, repetition history, loading, and notation policy around it because those are application-level chess semantics the network/search need.

### Connect Four

`connect4/mod.rs` is a compact complete game implementation. `Connect4Move(u8)` validates columns 0–6. `Connect4` uses two bitboards and a move count rather than a 6×7 heap array. The sentinel row and bit shifts make legal-column checks and four-in-a-row detection fast; the board is `Copy`, which is ideal for MCTS. `Status` caches ongoing/win/draw state, legal moves omit full columns and terminal states, and terminal values still use the side-to-move convention. `connect4/notation.rs` parses/formats a move as its column number. The randomized tests compare it with a naive board oracle as well as checking each win direction and invalid-move behavior.

## 5. `crates/search`: generic MCTS, not a neural engine

`search/src/lib.rs` exports evaluators, rules, and MCTS. It has no dependency on `games` in production and no dependency on `tch`; that is what makes it a reusable search kernel.

- `evaluator.rs` defines `PolicyValueEvaluator<G>`. Given game states it returns a scalar value plus logits aligned with that state’s legal native moves. Search never needs to know tensors or action indexes.
- `rules.rs` defines `SearchRules<G>` and `NoExtraRules`. Most games use the no-op implementation. Chess supplies a rule implementation that maintains path-local repetition features/terminal detection. Separating this from `GameState` avoids imposing chess-only metadata on Connect Four nodes.
- `mcts/mod.rs` defines configuration, PUCT/Gumbel variants, validation, `SearchResult`, and arena `Node`s. Children occupy contiguous ranges in one reused `Vec`; nodes refer to parents/children by `u32`. This minimizes allocations and pointer chasing in the dominant search loop. `SearchResult` keeps native moves and returns a normalized policy plus selected move/value.
- `mcts/core.rs` holds the generic `Mcts` wrapper and dispatches once to PUCT or Gumbel. It owns reusable scratch vectors, RNG, optional evaluation cache, and the typed evaluator/rules. Generic types are monomorphized, avoiding virtual calls inside selection and backup.
- `mcts/traversal.rs` descends from root, enters rule state (including root—important for repetition), selects children, adds virtual loss while a leaf is outstanding, batches leaves, coalesces duplicated leaves, evaluates unique positions once, expands, then backs up once per simulation. Virtual loss discourages all members of a leaf batch from selecting the same branch.
- `mcts/evaluation.rs` handles cache lookup/insertion and conversion from legal-move logits to normalized priors. The cache is lossy/bounded by design: collisions may replace values but never alter a key's lookup correctness within its stored entry.
- `mcts/puct.rs` implements ordinary AlphaZero selection: prior plus exploration bonus and first-play urgency. Root Dirichlet noise is enabled for self-play exploration and disabled for evaluation/match play.
- `mcts/gumbel.rs` implements Gumbel AlphaZero root sampling and sequential halving. It samples a bounded candidate set once, allocates visits by schedule, and produces the completed-Q transformed policy target. Pairing simulation count with root candidate count in `GumbelSearchProfile` prevents invalid low-budget/huge-root combinations.
- `mcts/cache.rs` is `EvalTable`, a bounded shared evaluation table with hit/miss/insert statistics.
- `mcts/tests.rs` and `mcts/tests/cases.rs` test native move support, terminal behavior, policy normalization, batching, cache ordering/coalescing, repetition/root-cycle handling, PUCT, Gumbel, and invalid configurations. One ignored release throughput harness is intentionally manual rather than a flaky unit-test performance claim.

## 6. `crates/alphazero`: model-specific inference, learning, and data

`alphazero/src/lib.rs` is the facade that exports the public model, representation, MCTS re-exports, experiment, batching, replay, self-play, trainer, and analysis APIs. This is the only layer that links `tch`.

### Representations and action layouts

`representation/mod.rs` defines `Action` and `AlphaZeroRepresentation<G>`. Every representation provides static state/action sizes, encodes a state into a caller-provided buffer, converts moves both ways, and produces an encoded-state cache key. The key must include every feature the tensor contains; otherwise the cache can return a neural result for a visually identical but semantically different state.

- `representation/connect4.rs` encodes the two player bitboards and maps seven columns directly to actions. It is small because Connect Four's state/action space is small.
- `representation/chess_v2_state.rs` defines `ChessAzState<HISTORY>`, a compact search snapshot with the authoritative board plus fixed-size history/repetition planes. `from_game` copies only the bounded information v2 needs; fixed arrays preserve `Copy`-like search efficiency rather than allocating per node.
- `representation/chess_v2.rs` is the current chess-v2 encoding and `8×8×73` move policy layout. It supports history sizes 1, 4, and 8, validates that an action is legal in the current state, handles promotions/castling/en passant, writes board/history/clock/repetition features, and keys all encoded features. It owns this mapping because the mapping is a neural-model contract, not a chess-rule contract.
- `representation/compat_v1.rs` exports `ChessClassicRepresentation`: the established 19-plane input and `64×64×5 = 20,480`-action policy mapping used by the original scalar chess checkpoints. It is an explicit model option, not part of native chess rules or the canonical-history representation.

### Networks and inference

`network.rs` makes the model contract explicit through `ModelSpec`: game, representation, residual trunk, policy head, and scalar or WDL value head. `Network` constructs the selected architecture from that spec and exposes semantic output (`RawValueOutput`); WDL is converted to the scalar expected value only at the MCTS evaluator boundary.

- `network/legacy.rs` implements the basic residual trunk selected by Connect Four and the classic chess model spec. Its filename is historical; its architecture and parameter names are deliberately retained so classic checkpoints load unchanged.
- `network/chess_v2.rs` implements the squeeze-excitation residual trunk selected by the chess canonical model spec. Separate source keeps chess-only tensor shapes from leaking into generic games.
- `evaluator.rs` wraps a representation plus batcher client as a `PolicyValueEvaluator`. It encodes states, converts legal native moves to policy actions, sends batches, then restores logits in original legal-move order. The MCTS contract remains native moves.
- `batcher/mod.rs` exposes `Batcher` and per-thread `BatcherClient`, request/result buffers, statistics, reload support, inference precision selection, and legal-logit gather interfaces.
- `batcher/tch_backend.rs` owns the production network backend. The batcher worker coalesces requests, uses grow-only staging buffers, runs inference, gathers only legal policy logits on-device, and returns client buffers for reuse. That is necessary to avoid one GPU launch/allocation or a full policy GPU-to-host copy for every leaf.

### Checkpoints, replay, and training

`experiment/` separates durable experiment intent from mutable progress. `experiment.toml` is immutable and contains the model, search/self-play, replay, training, inference settings, and experiment seed; `state.json` records iteration, generation, global step, replay count, resume kind, generated-game total, and the latest checkpoint. `train init` validates and copies a TOML experiment; `train run` cannot mutate it. `RunDir::write_latest` writes `latest.tmp.safetensors`, atomically renames it to `checkpoints/latest.safetensors`, then atomically updates state. `best.safetensors` is reserved for an actual gated evaluation result rather than incorrectly naming the continuously trained model.

`replay/mod.rs` defines `Transition`, packed `SparsePolicyBatch`, `ReplayBatch`, and a fixed-capacity, lock-protected ring `ReplayBuffer`. States are stored densely, but policies are stored only as nonzero `(Action, probability)` entries. Sampling chooses distinct positions and packs actions/probabilities/row offsets. Sparse targets are necessary for chess: allocating a dense policy target for every stored position wastes substantial memory and bandwidth.

`trainer.rs` validates training settings, samples replay, runs forward/backprop/optimizer steps, and computes policy plus value loss. Its sparse policy loss gathers only target logits rather than densifying targets. Tests compare that result to a dense reference and cover empty/invalid configurations.

### Self-play and chess rules in search

`rules.rs` implements chess repetition search rules. It resets a reusable path, enters every position along a tree descent, tracks occurrence counts/reversible suffixes, writes the representation's repetition feature, and declares a repetition terminal when appropriate. This belongs above generic MCTS because it is game/model feature policy.

`selfplay/` contains one `SelfPlayCoordinator` and game-owned worker factories. The coordinator allocates absolute game IDs, derives purpose-specific deterministic seeds, aggregates progress, propagates failures, and commits each complete trajectory to replay in one operation. `GenericSelfPlayWorker` owns ordinary `GameState` episodes; `ChessSelfPlayWorker<HISTORY>` owns the authoritative `ChessGame`, creates compact `ChessAzState<HISTORY>` search snapshots, and supplies repetition context. The configurable budget schedule keeps PUCT and Full Gumbel profiles distinct, temperature controls played-move sampling separately from search policy generation, and resignation is disabled by default until calibrated. App code only selects the concrete model/history and invokes this machinery.

`analysis.rs` exposes network-only or MCTS analysis in a display-friendly form: selected native move, value, and named policy rows. It is the bridge used by HTTP, CLI, and UCI-facing tooling.

The `batcher`, `replay`, `selfplay`, and `trainer` test modules exercise allocation reuse, mixed request splitting, precision safety, sparse policies, targets, and schedule behavior.

## 7. `crates/app`: binaries, interactive play, HTTP, UCI, and jobs

`app/src/lib.rs` exports the application modules. It is intentionally the place where strings, filesystems, sockets, CLI parsing, and runtime game/model matching occur.

### CLI binaries

- `bin/engine-zoo.rs` starts `app::run`: `play` for terminal interaction and `serve` for HTTP.
- `app.rs` defines the `play` and `serve` CLI arguments. `play` builds a human or compatible scalar AlphaZero agent, alternates sides, permits sampled openings, and prints native notation. It explicitly rejects chess v2 interactive play because v2 is served through its specialized API/UCI paths.
- `bin/train.rs` parses a new experiment, matches `ModelSpec` once to Connect Four, classic chess, or a canonical chess history size, constructs the corresponding worker factory, and invokes the common coordinator/trainer loop. `--chess-classic` creates the classic 19-plane/20,480-action scalar model; canonical chess remains the default.
- `bin/eval.rs` calls `eval_app.rs`, which offers local or remote analysis and JSON benchmark cases. Remote mode sends the same `AnalyzeRequest` used by the HTTP API.
- `bin/engine-zoo-uci.rs` starts the UCI adapter.

### Agent and UCI layers

`players.rs` parses `user`/`alphazero[:model]`, implements terminal human input, defines `InteractiveGame`, and wraps a batcher plus typed MCTS into `AlphaZeroAgent`. The trait converts authoritative games to search snapshots and supplies their rule context; chess gets `ChessPosition` plus repetition context while Connect Four is its own snapshot. `uci.rs` owns `ChessUciEngine`: option settings, model reload, FEN/start-position replay, action selection, and search limits. `uci/protocol.rs` is the small, testable parser for UCI commands, positions, `go` limits, options, and `searchmoves`. UCI exists so Fastchess and other chess tools can treat this project as a conventional chess engine.

### HTTP server and sessions

`proxy.rs` constructs an Axum router and application state. It opens a configured run, chooses CUDA if available, stores live sessions behind a mutex, allocates IDs atomically, and installs permissive CORS for the separately hosted web client. It serves health, run/checkpoint discovery, analysis, sessions, and evaluation jobs under `/api/...`, while retaining unprefixed compatibility routes.

- `proxy/http.rs` performs request-method checks, run/model resolution, JSON error responses, and public run/checkpoint metadata.
- `proxy/analysis.rs` validates `AnalyzeRequest`, loads the requested compatible model, reconstructs a `GameSetup`, chooses network or MCTS mode, and dispatches concrete representation/history sizes. This explicit match is intentional: it prevents a wrong model/game pairing and keeps hot paths statically typed.
- `proxy/sessions.rs` creates, finds, and advances chess/Connect Four sessions, including human and engine turns. A session owns the authoritative game plus UCI/SAN move history and selected model/search settings.
- `proxy/session_view.rs` serializes that session into the frontend shape: board text, legal moves/actions, SAN/PGN, terminal/reward state, and optional analysis. Chess action indexes are generated by the loaded representation, not invented by the UI.

`visualization.rs` and `visualization/assets.rs` generate the legacy/self-contained chess play HTML and embed assets. They coexist with the Svelte frontend, providing a lightweight direct server UI.

### Evaluation jobs in the server

`evaluation_jobs.rs` is an asynchronous job service: it validates requests, allocates IDs/directories, persists lifecycle state, preflights required tools, spawns suite commands, recovers interrupted jobs, and exposes artifacts. `evaluation_jobs/model.rs` contains serializable job/catalog/suite/status models. `storage.rs` performs safe ID/artifact validation and atomic-ish persistence in the run directory. `tools.rs` locates executables and parses suite outputs. The safety checks are necessary because artifact names and checkpoint identifiers cross an HTTP/filesystem boundary.

## 8. `crates/evaluations`: repeatable checkpoint assessment

The package is named `checkpoint-eval`; `lib.rs` exports its suites.

- `puzzle.rs` reads a deterministic JSONL chess puzzle suite, validates every expected legal move, loads the appropriate run/model representation, analyzes positions, calculates exact/group summaries, and writes JSONL. `suites/chess/puzzles-tactics-v1.jsonl` is the checked-in small tactical suite.
- `arena.rs` implements an in-process deterministic arena: openings, alternating colours, move limits, scores, and aggregate results. It is useful where a direct model-vs-model loop is sufficient.
- `fastchess.rs` builds, validates, runs, and parses an external Fastchess command. `fastchess/result.rs` parses WDL/score formats defensively. Fastchess is used for broad paired-opening tournaments because it is an established tournament harness.
- `match_suite.rs` drives a Fastchess run, cleans PGN, requires promised artifacts, and preserves diagnostics on failure.
- `model_engine.rs` converts a checkpoint/run into an UCI engine specification and infers normal run directory layouts.
- `report.rs` defines versioned evaluation reports and computes score fraction, Wilson interval, and smoothed local Elo. The statistics report uncertainty rather than overclaiming from a tiny match.
- `legacy.rs` and `legacy/support.rs` are retained source from the former direct Stockfish evaluator, but are not exported by the current evaluation crate. Current classic-model evaluation goes through the normal UCI/Fastchess and puzzle paths.
- `bin/eval-puzzle.rs`, `bin/eval-arena.rs`, and `bin/eval-stockfish.rs` expose explicit, validated command-line workflows. `build.rs` files in app/evaluations/alphazero provide build metadata hooks. `install_stockfish.py` downloads/installs the external engine where needed.

## 9. `web`: static SvelteKit frontend

`web/package.json` defines a Svelte 5/Vite static application. `svelte.config.js` uses the static adapter; `vite.config.ts` configures Svelte; `tsconfig.json`, `app.html`, and `app.css` are standard app shell/configuration. `+layout.ts` disables SSR and enables prerendering because this is an API-driven browser client.

The page sequence is deliberately linear:

```text
/  choose game  →  /setup choose agents/settings  →  /play play/inspect
                                                   ↘ /evaluate run jobs
```

- `routes/+page.svelte` renders game selection. `setup/+page.svelte` loads the agent registry and stores two player choices. `play/+page.svelte` manages position, local human moves, session creation, engine requests, analysis caching, and board rendering. `evaluate/+page.svelte` selects a model/suite and polls job state.
- `lib/types.ts` is the frontend contract for games, agents, sessions, policy rows, analysis, and puzzle/job objects. Keeping it centralized makes API shape changes visible to TypeScript.
- `lib/state/selection.ts` holds in-memory cloned setup selection. The README describes session-level navigation intent; it avoids a server-side account/session requirement.
- `lib/config/agents.ts` loads and minimally validates `static/config/agents.json` with cache busting. `catalog.ts` supplies development fallback games/agents/sample puzzles; `games.ts` is a narrower game catalog. The duplicated catalog is a small maintenance risk, but its purpose is fallback/offline UI rather than engine state.
- `lib/api/client.ts` centralizes JSON request/error handling and session/analysis calls; `api/evaluations.ts` adds the job endpoints.
- `lib/match/game.ts` reconstructs chess via `chess.js` or Connect Four locally to draw boards, legal highlights, and results. It is a UI convenience; the Rust backend remains authoritative for engine analysis. `match/analysis.ts` creates cache keys, turns UCI into SAN for display, and optionally temperature-samples returned policies.
- `components/ChessgroundBoard.svelte` uses Lichess Chessground for the chess board. `ChessBoard.svelte` is the simpler chess renderer, `Connect4Board.svelte` the grid. `AgentCard`, `AgentPicker`, `GameCarousel`, `PlayerStrip`, and `MatchControls` compose selection/play controls. `AnalysisPanel.svelte` accepts several compatible policy response layouts and displays raw/derived analysis. The `components/evaluations/` components render suite, model, Stockfish-level, job, result, and puzzle-insight views.

`static/config/agents.json` is the runtime agent registry. Each non-human agent names a game, server URL, model, description, and defaults. This decouples a static UI deployment from one or more running engine servers, but it also means public deployments should not expose private hostnames or model paths. `web/README.md` records install/build commands, screen flow, response shape, licensing (Chessground is GPL-3.0-or-later), and CORS expectations.

## 10. Scripts and normal workflows

The Python scripts are deliberately thin wrappers around Rust binaries, not duplicate engine implementations.

- `scripts/train_chess.py` exposes editable defaults, builds release `train`, and runs continuous chess-v2 training.
- `scripts/serve_agent.py` reads run metadata, builds release `engine-zoo`, starts `serve`, and inserts/replaces a matching entry in the UI agent registry. This is convenient locally; it intentionally edits `web/static/config/agents.json`.
- `scripts/arena_chess.py` builds UCI/evaluation binaries and launches an arena between candidate/baseline checkpoints.
- `scripts/eval_chess_puzzles.py` launches the deterministic puzzle evaluator.
- `scripts/make_lichess_easy_suite.py` is an optional suite-construction helper for Lichess-derived examples.

The expected production/research loop is: choose a model/run configuration → self-play with MCTS → append sparse trajectories to replay → train network → checkpoint/promote candidate → use puzzles for fast regression checks → use paired arenas/Stockfish for strength checks → serve the selected checkpoint for people or the web UI.

## 11. What is verified, and what is not

On 2026-07-22, `cargo test --workspace` passed: **158 tests passed, 0 failed**. This covers core rule behavior, representations, batching, replay/training, MCTS variants, UCI parsing, server/job safety, evaluation report/tool logic, and classic-chess model migration. It does not prove a trained model is strong, GPU performance is optimal, Fastchess/Stockfish are installed, or a browser build has been run on this machine. Those require the corresponding data, hardware, external binaries, and `web` dependency build.

## 12. A practical reading order

For a first deep read, follow this exact execution path:

1. `core/game.rs`, then `games/connect4/mod.rs` (smallest complete implementation).
2. `games/chess/{position,game,repetition}.rs` and its tests.
3. `search/{evaluator,rules,mcts/mod,core,traversal,evaluation,puct,gumbel}.rs`.
4. `alphazero/representation/chess_v2*.rs`, then `network*.rs`, `evaluator.rs`, and `batcher`.
5. `replay/mod.rs`, `selfplay/`, `trainer.rs`, and `experiment/`.
6. `app/proxy*.rs`, `uci*.rs`, and train binaries.
7. `evaluations`, then `web`, then operational scripts.

At each boundary, ask: “is this still game rules, generic search, model representation, or runtime/UI policy?” Almost every design decision in this repository follows from keeping those four responsibilities separate while protecting the performance-critical self-play/search loop.

## 13. The refactored architecture: the current execution path

The repository now uses explicit, typed boundaries instead of model-version
branches and bare floating-point game values.

```text
ChessGame / Connect4                 authoritative real game
        │
        ├── compact search state ──> ChessAzState<H> / Connect4
        │                                  │
        │                                  ├── representation encodes only sampled replay states
        │                                  └── native moves <-> checked policy Actions
        │
        └── repetition context ──> SearchRules ──> typed MCTS tree
                                                   │
                                           PUCT or Full Gumbel
                                                   │
                                  PolicyValueEvaluator<Result<…>>
                                                   │
                    Batcher ──> InferenceBackend ──> TchInferenceBackend
                                                   │
                         RawNetworkOutput { policy_logits, Scalar | WDL }
```

### Values, outcomes, and failures

`search::PositionValue` is the scalar used by search. It is bounded to
`[-1, 1]` and always means **the value for the player to move in the state
where it is stored**. Moving across a tree edge calls `.flipped()` explicitly.
That one rule removes the most common MCTS bug: accidentally backing up a
child's value from the parent's point of view.

Game terminals still use `engine_core::TerminalValue`; search rules convert
them at the search boundary. Training uses AlphaZero's typed `Outcome`
(`Win`, `Draw`, `Loss`) rather than a free-form float. It can become a scalar
target or a WDL class index without float equality comparisons.

Evaluation is fallible. A malformed evaluator response, queue shutdown, or
backend failure becomes an `EvaluationError`/`SearchError`; MCTS cancels all
reserved in-flight visits before returning it. The engine never invents a
value merely to keep a failed self-play game running.

### Search: two algorithms, one tree lifecycle

`SearchConfig` is an enum, not a bag of optional fields:

- `PuctConfig` owns simulation/batch counts, dynamic `pb_c`, FPU strategy,
  in-flight policy, and optional root Dirichlet noise.
- `GumbelConfig` owns simulations, considered root actions, Gumbel scale, and
  completed-Q transform settings. It has no PUCT/FPU/noise fields.

Both algorithms share tree allocation, terminal handling, evaluator/cache
calls, and reservation cleanup. They do not share selection rules.

PUCT uses completed values for Q and includes in-flight visits only in visit
denominators. That is the default unscored-virtual-visit policy: pending GPU
work changes exploration pressure but is not treated as a fake win or loss.
FPU is explicit—absolute or prior-mass-reduction—rather than hidden inside a
selection formula. Root noise changes only live root priors; it never enters
the network evaluation cache.

Full Gumbel is deliberately sequential within one tree (`leaf_batch_size =
1`). It samples root Gumbels once, uses the sequential-halving visit schedule,
computes Mctx-style completed-Q values, and uses the improved-policy deficit
at interior nodes. This is intentionally different from the older hybrid
“Gumbel root plus PUCT interior” implementation. Global GPU batching still
comes from many concurrent self-play games.

Tree nodes keep `completed_visits`, `in_flight_visits`, and
`value_sum_from_node_pov` as separate facts. The arena remains contiguous and
reused between searches, so this clarity does not introduce per-node heap
allocation.

### Models, representations, and inference

`ModelSpec` describes a valid pairing of:

1. game (`Chess` or `Connect4`);
2. representation (classic chess or canonical chess with history length); and
3. residual network trunk, policy head, and scalar/WDL value head.

It validates the shape relationship before weights load. A chess canonical
representation therefore cannot accidentally be paired with a seven-action
Connect Four policy head. The model layer returns `RawNetworkOutput` with a
flat `[batch, action_size]` policy tensor and either a scalar or WDL-logit
value output. The evaluator converts both value-head choices into the one
`PositionValue` required by search.

The dynamic `Batcher` owns queueing, bounded coalescing, request splitting,
reload ordering, response routing, and error propagation. The
`InferenceBackend` trait owns one inference pass. `TchInferenceBackend` is the
only production implementation today and retains the performance-sensitive
details: persistent model/staging buffers, pinned CUDA host buffers, FP16
device conversion, and gathering legal logits on the device before copying
them back. A mock backend makes queueing and failure behavior testable without
LibTorch.

### Replay, training, and self-play

Replay stores compact game/search states—not permanently allocated encoded
float tensors. A `ReplaySample<S>` holds a state, canonical sparse policy,
typed outcome, policy/value weights, and diagnostic metadata (search kind,
simulations, generation, game ID, ply). Sampling invokes the representation
only for selected states and writes into reusable staging storage. This is the
key memory reduction for chess replay windows.

Sparse policies are a validated value object: action indexes are in range,
probabilities are finite/non-negative, duplicate actions are summed, zero mass
is removed, the result is normalized, and an empty policy is only allowed
when its policy-training weight is zero.

The trainer shares a policy-loss implementation across scalar and WDL models.
Policy and value components have separate weighted denominators; a sample's
policy weight is counted once, not once per legal action. Microbatches scale
their numerator by the full-batch denominator before backpropagation so an
uneven final microbatch has the same update semantics as the full batch.

`SelfPlayCoordinator` allocates absolute game IDs, starts game-owned workers,
aggregates statistics, and commits a trajectory only after its whole game
succeeds. Concrete generic and chess workers own the game-specific loop. A
single experiment seed derives independent streams for worker setup, budget
choice, Gumbel, move sampling, and resignation decisions, making a fixed
configuration reproducible instead of relying on `from_os_rng()`.

### Runs and operational entry points

A run directory distinguishes immutable experiment intent from mutable
progress:

```text
run/
├── experiment.toml    immutable model/search/self-play/training specification and seed
├── state.json         iteration, generation, step, replay count, latest model
├── latest.safetensors continuously trained model
├── candidate.safetensors  optional gating candidate
├── best.safetensors       only an evaluated/gated best model
├── checkpoints/
└── metrics.jsonl
```

Writes use temporary files followed by rename; state advances only after the
corresponding model exists. Old run configuration parsing is isolated as a
migration concern rather than retained as a normal model API. Historical
classic chess configs are validated against their exact `19×8×8` input and
20,480-action policy shape, then migrated to the explicit `ChessClassic` model
specification, preserving the basic residual layer names required by weights.

## 14. Cleanup audit and next tests

This cleanup pass found no duplicate public gameplay/search implementation that
should be merged immediately. The notable consolidation already completed by
the refactor is the deletion of the old split chess self-play/train paths and
the version-named model branches. The remaining larger files are mostly
orchestrators (`batcher`, `network`, `replay`, `gumbel`, and `train`) with
separate responsibilities already extracted into submodules where a boundary
exists. Splitting them further solely to reduce line count would make control
flow harder to follow.

The highest-value additional tests are integration rather than more unit
tests:

- a deterministic tiny self-play → replay → train → reload smoke test for
  both scalar and WDL heads;
- an independent Full Gumbel trace oracle over several tiny trees and budgets;
- property tests that replay randomized legal chess games through both
  authoritative and search snapshots; and
- mock-backend tests for queue saturation and concurrent cache misses.

These complement the existing perft, terminal-boundary, sparse-policy,
batcher, and value-backup tests without duplicating them.

## 15. Is an alpha-beta engine ready to add?

Yes, with a narrow addition—not another architectural refactor.

An alpha-beta engine can live in `crates/search` as a separate module and
depend only on `engine_core::GameState` plus a small static-evaluation trait.
It can use native moves, compact `ChessPosition` snapshots, terminal values,
and existing notation/UCI/session boundaries. It must not depend on
`alphazero`, representations, `Action`, replay, `tch`, or the batcher.

The work required is normal engine work: alpha-beta/negamax, transposition
table policy, move ordering, quiescence/evaluation design, configuration, and
an app-side dispatch choice. A future NNUE evaluator would be another
evaluation implementation behind that alpha-beta-specific trait. Nothing in
the current AlphaZero API forces alpha-beta through a policy network or MCTS,
which is the architectural property that matters.

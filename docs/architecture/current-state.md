# Current architecture baseline

This document records the pre-refactor shape of the repository. It is a
description of the current implementation, not a target design.

## Workspace

The Rust workspace currently contains five crates:

- `engine_core` owns the generic `Game`, `Agent`, and repetition/encoding
  contracts.
- `games` implements Connect Four and Chess, including native positions,
  AlphaZero states, policy-action conversion, and Chess legacy support.
- `algorithms` owns AlphaZero evaluation, batching, replay, training,
  self-play, and PUCT/Gumbel MCTS.
- `engine_app` provides the engine, UCI, training, and web-facing application
  entry points.
- `checkpoint-eval` provides puzzle, arena, and model evaluation tooling.

The algorithms, application, and evaluation crates depend on `tch`/libtorch;
the game and core crates do not.

## Main runtime flow

Application and evaluation code select concrete Chess or Connect Four types and
construct an AlphaZero network/evaluator. MCTS traverses copied game states,
converts legal moves into policy indices, evaluates leaves through the local or
cross-thread batcher, and backs up values through a contiguous node arena.
Self-play and evaluation consume the resulting trajectories and replay/training
code stores sparse policy targets.

## Current contracts and coupling

- `Game` combines rules, construction, display/notation, state encoding, and
  fixed AlphaZero policy/input dimensions.
- `Action` is a numeric policy index but is also used at game boundaries where
  a native move would be clearer.
- `RepetitionGame` combines repetition bookkeeping with encoded-state cache
  identity.
- `ChessPosition`/`Connect4` are compact search states, while
  `ChessHistoryState<HISTORY>` carries AlphaZero history features. `ChessGame` and
  `ChessAzGame<HISTORY>` both retain authoritative Chess progression state.
- MCTS is generic over game state but lives under the `algorithms` crate and
  therefore shares a crate dependency on `tch` with neural-network code.
- Chess v1 compatibility remains available alongside the current Chess v2
  representation and network paths.

## Preserved behavior for Phase 0

Phase 0 makes no architectural changes. Chess and Connect Four remain
supported, AlphaZero and legacy Chess paths remain intact, and the existing
MCTS batching, sparse replay, cache, and compact-state implementations are the
baseline for later phases.

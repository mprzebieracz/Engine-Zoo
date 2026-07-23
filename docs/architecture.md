# Architecture

The workspace is layered so that adding alpha-beta is an additive change, not another refactor:

```text
engine_core
  ├─ games
  └─ search
       └─ alphazero
            ├─ engine_app
            └─ checkpoint-eval
```

`GameState` is rules-only: legal native moves, transitions, and terminal values. A neural `Action` is deliberately separate because it is an index in one model's policy layout, not a chess or Connect Four move. `AlphaZeroRepresentation` is the explicit adapter that encodes a state and maps native moves to actions.

`search` owns generic PUCT and Full Gumbel MCTS and has no `tch` dependency. `alphazero` owns the network, batcher, replay, trainer, and representations. App code performs the one-time runtime dispatch over model/game/history; hot MCTS code stays statically dispatched.

Chess has two state forms. `ChessGame` is authoritative and owns repetition bookkeeping. `ChessPosition` and representation history snapshots are compact search values. This avoids a heap-backed repetition map in every MCTS node while keeping real-game draw adjudication correct.

See [REPOSITORY_GUIDE.md](../REPOSITORY_GUIDE.md) for the linear file-by-file tour.


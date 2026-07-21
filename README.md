# engine-zoo

A Rust workspace for board-game engines and agents.

Current games:

- Chess
- Connect4

Current algorithms:

- AlphaZero

Layout:

- `crates/core/` - shared game and agent traits.
- `crates/games/` - concrete game implementations and position specs.
- `crates/search/` - generic MCTS, evaluator contracts, rules, and evaluation cache.
- `crates/alphazero/` - neural evaluation, representations, batching, replay, training, checkpoints, and analysis.
- `crates/app/` - CLI binaries and Axum API server.
- `web/` - SvelteKit TypeScript frontend.
- `scripts/` - training/evaluation helpers.
- `data/suites/` - puzzle and evaluation suites.

Common commands:

```bash
cargo test --workspace
cargo run --bin train -- --game chess
cargo run --bin engine-zoo -- serve --game chess
cargo run --bin eval -- analyze --game chess --model best
```

Remote analysis example:

```bash
cargo run --bin eval -- analyze \
  --server http://127.0.0.1:8080 \
  --game chess \
  --moves e2e4,e7e5
```

Frontend:

```bash
cd web
npm install
npm run dev
```

Serving an agent for the web UI:

```bash
python3 scripts/serve_agent.py \
  --game chess \
  --run-dir data/runs/chess-puct-real \
  --model best \
  --port 8080
```

The script starts the Rust API and registers the agent in `web/static/config/agents.json`.
The web app loads that file at runtime, so each agent entry can point at its own
`server` URL and `model` name/path.

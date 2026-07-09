# engine-zoo

A Rust workspace for board-game engines and agents.

Current games:
- Chess
- Connect4

Current algorithms:
- AlphaZero

Layout:
- `core/` - shared game and agent traits.
- `games/` - concrete game implementations and position specs.
- `algorithms/` - AlphaZero, search, training, and analysis code.
- `app/` - CLI binaries and Axum API server.
- `web/` - SvelteKit TypeScript frontend.
- `scripts/` - training/evaluation helpers.
- `suites/` - puzzle and evaluation suites.

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

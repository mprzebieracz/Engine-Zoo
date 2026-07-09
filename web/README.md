# Engine Zoo Web

SvelteKit frontend for selecting games, configuring agents, playing matches, and inspecting AlphaZero output.

## Install

```bash
npm install
npm run dev
```

The production build is created with:

```bash
npm run check
npm run build
```

## Screen flow

- `/` — animated game selection
- `/setup` — select both agents and MCTS settings
- `/play` — full game UI
- `/evaluate` — evaluation workspace

The current selection is kept in `sessionStorage` while moving between screens.

## Configure available agents

Edit:

```text
static/config/agents.json
```

Example:

```json
{
  "id": "chess-a100",
  "name": "AlphaZero · A100",
  "kind": "alphazero",
  "games": ["chess"],
  "model": "candidate",
  "server": "http://192.168.1.50:8080",
  "description": "Candidate checkpoint on the training server.",
  "badge": "Remote",
  "defaults": { "simulations": 800, "waitForCount": 1 }
}
```

`server` is the Axum server base address. `model` is sent as the model identifier in the existing session/analyze API. For a public deployment, serve this registry from the Rust backend instead so private addresses and filesystem details are not exposed.

The current backend runs one game per server process, so the example config uses port `8080` for chess and `8081` for Connect Four.

## Chess UI

The chess screen uses `@lichess-org/chessground`, the board UI developed for Lichess, plus `chess.js` to reconstruct FEN from UCI move history. The Rust backend remains authoritative for legal moves.

Chessground is GPL-3.0-or-later. Ensure the distribution of this frontend complies with that license.

## Analysis response

`AnalysisPanel.svelte` accepts several common response layouts:

- `network_policy`
- `mcts_policy`
- `root_policy`
- `policy`
- nested `network.policy` and `mcts.policy`
- array rows with `probability`, `prior`, `visits`, or `visit_count`

It displays network policy and MCTS policy separately, a normalized value gauge, the selected move, and the raw JSON response.

For the best UX, return both distributions in one analysis response:

```json
{
  "best_move": "e2e4",
  "value": 0.18,
  "network_policy": [
    { "move": "e2e4", "probability": 0.31 }
  ],
  "mcts_policy": [
    { "move": "e2e4", "visits": 281 }
  ]
}
```

## CORS

When `server` points directly at another origin, Axum must allow the frontend origin through CORS. Alternatively, leave `server` empty and proxy `/api` through Vite or your production reverse proxy.

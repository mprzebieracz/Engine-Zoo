#!/usr/bin/env bash
set -euo pipefail

GAME="${GAME:-chess}"
RUN_DIR="${RUN_DIR:-runs/chess}"
MODEL="${MODEL:-best.safetensors}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
ID="${ID:-}"
NAME="${NAME:-}"
CONFIG="${CONFIG:-web/static/config/agents.json}"
WRITE_CONFIG="${WRITE_CONFIG:-1}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/serve_agent.sh --game chess --run-dir runs/chess-puct-real --model best --port 8080

Options:
  --game          chess or connect4
  --run-dir       run directory containing run config and weights
  --model         model name/path sent by the web UI: best, candidate, ckpt_0001.safetensors, /abs/model.safetensors
  --host          bind host (default: 127.0.0.1)
  --port          bind port (default: 8080)
  --id            agent id written to web/static/config/agents.json
  --name          display name written to web/static/config/agents.json
  --config        config file path (default: web/static/config/agents.json)
  --no-config     serve only; do not update the web config
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --game) GAME="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --id) ID="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --no-config) WRITE_CONFIG=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$ID" ]]; then
  SAFE_MODEL="$(basename "$MODEL" | tr -cs '[:alnum:]_' '-' | sed 's/^-//;s/-$//')"
  ID="$GAME-$SAFE_MODEL"
fi
if [[ -z "$NAME" ]]; then
  NAME="AlphaZero · $MODEL"
fi

if [[ "$WRITE_CONFIG" == "1" ]]; then
  mkdir -p "$(dirname "$CONFIG")"
  CONFIG="$CONFIG" ID="$ID" NAME="$NAME" GAME="$GAME" MODEL="$MODEL" SERVER="http://$HOST:$PORT" node --input-type=module <<'NODE'
import fs from 'node:fs';

const path = process.env.CONFIG;
const next = {
  id: process.env.ID,
  name: process.env.NAME,
  kind: 'alphazero',
  games: [process.env.GAME],
  model: process.env.MODEL,
  server: process.env.SERVER,
  description: `${process.env.GAME} model ${process.env.MODEL}`,
  badge: 'Local',
  defaults: { simulations: 800, waitForCount: 1 }
};

let config = { agents: [] };
if (fs.existsSync(path)) {
  config = JSON.parse(fs.readFileSync(path, 'utf8'));
}
const agents = Array.isArray(config.agents) ? config.agents : [];
const without = agents.filter((agent) => agent?.id !== next.id);
if (!without.some((agent) => agent?.id === 'human')) {
  without.unshift({
    id: 'human',
    name: 'Human',
    kind: 'human',
    games: ['chess', 'connect4'],
    description: 'Moves are entered through the board.',
    badge: 'Local'
  });
}
fs.writeFileSync(path, `${JSON.stringify({ ...config, agents: [...without, next] }, null, 2)}\n`);
console.log(`registered ${next.id} -> ${next.server} (${next.model}) in ${path}`);
NODE
fi

echo "serving $GAME from $RUN_DIR on http://$HOST:$PORT/ using model '$MODEL'"
cargo build --release --bin engine-zoo

exec target/release/engine-zoo serve \
  --game "$GAME" \
  --run-dir "$RUN_DIR" \
  --bind "$HOST:$PORT"

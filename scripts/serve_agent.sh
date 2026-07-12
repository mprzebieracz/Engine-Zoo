#!/usr/bin/env bash
set -euo pipefail

GAME="${GAME:-chess}"
RUN_DIR="${RUN_DIR:-data/runs/chess-az-v2-h4}"
MODEL="${MODEL:-best}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
ID="${ID:-}"
NAME="${NAME:-}"
CONFIG="${CONFIG:-web/static/config/agents.json}"
WRITE_CONFIG="${WRITE_CONFIG:-1}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/serve_agent.sh --game chess --run-dir data/runs/chess-az-v2-h4 --model best --port 8080

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

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --game) require_value "$@"; GAME="$2"; shift 2 ;;
    --run-dir) require_value "$@"; RUN_DIR="$2"; shift 2 ;;
    --model) require_value "$@"; MODEL="$2"; shift 2 ;;
    --host) require_value "$@"; HOST="$2"; shift 2 ;;
    --port) require_value "$@"; PORT="$2"; shift 2 ;;
    --id) require_value "$@"; ID="$2"; shift 2 ;;
    --name) require_value "$@"; NAME="$2"; shift 2 ;;
    --config) require_value "$@"; CONFIG="$2"; shift 2 ;;
    --no-config) WRITE_CONFIG=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$GAME" != "chess" && "$GAME" != "connect4" ]]; then
  echo "--game must be chess or connect4" >&2
  exit 2
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "--port must be an integer from 1 to 65535" >&2
  exit 2
fi

# Check the saved run metadata before changing the web agent configuration.
# A missing architecture is a pre-v2 legacy run; all other values must be an
# architecture this server understands.
PREFLIGHT="$(RUN_DIR="$RUN_DIR" GAME="$GAME" node --input-type=module <<'NODE'
import fs from 'node:fs';
import path from 'node:path';

const runDir = process.env.RUN_DIR;
const expectedGame = process.env.GAME;
const configPath = path.join(runDir, 'config.json');

if (!fs.existsSync(configPath)) {
  throw new Error(`run config does not exist: ${configPath}`);
}

let config;
try {
  config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
} catch (error) {
  throw new Error(`cannot parse run config ${configPath}: ${error.message}`);
}

if (config.game !== expectedGame) {
  throw new Error(
    `run config game ${JSON.stringify(config.game)} does not match --game ${JSON.stringify(expectedGame)}`
  );
}

const architecture = config.architecture ?? { kind: 'legacy' };
if (architecture === null || typeof architecture !== 'object') {
  throw new Error(`unsupported run architecture in ${configPath}`);
}

switch (architecture.kind) {
  case 'legacy':
    process.stdout.write('legacy\t\n');
    break;
  case 'chess-az-v2': {
    const history = architecture.config?.history;
    if (expectedGame !== 'chess') {
      throw new Error('chess-az-v2 runs can only be served with --game chess');
    }
    if (![1, 4, 8].includes(history)) {
      throw new Error(`unsupported chess-az-v2 history ${JSON.stringify(history)} in ${configPath}`);
    }
    process.stdout.write(`chess-az-v2\t${history}\n`);
    break;
  }
  default:
    throw new Error(`unsupported run architecture ${JSON.stringify(architecture.kind)} in ${configPath}`);
}
NODE
)"
IFS=$'\t' read -r ARCHITECTURE HISTORY <<< "$PREFLIGHT"

if [[ -z "$ID" ]]; then
  SAFE_MODEL="$(basename "$MODEL" | tr -cs '[:alnum:]_' '-' | sed 's/^-//;s/-$//')"
  ID="$GAME-$SAFE_MODEL"
fi
if [[ -z "$NAME" ]]; then
  NAME="AlphaZero · $MODEL"
fi

if [[ "$WRITE_CONFIG" == "1" ]]; then
  mkdir -p "$(dirname "$CONFIG")"
  CONFIG="$CONFIG" ID="$ID" NAME="$NAME" GAME="$GAME" MODEL="$MODEL" SERVER="http://$HOST:$PORT" ARCHITECTURE="$ARCHITECTURE" HISTORY="$HISTORY" node --input-type=module <<'NODE'
import fs from 'node:fs';

const path = process.env.CONFIG;
const details = [`architecture=${process.env.ARCHITECTURE}`];
if (process.env.HISTORY) {
  details.push(`history=${process.env.HISTORY}`);
}
const next = {
  id: process.env.ID,
  name: process.env.NAME,
  kind: 'alphazero',
  games: [process.env.GAME],
  model: process.env.MODEL,
  server: process.env.SERVER,
  description: `${process.env.GAME} model ${process.env.MODEL} (${details.join(', ')})`,
  badge: 'Local',
  defaults: { simulations: 128, waitForCount: 16 }
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

echo "serving $GAME from $RUN_DIR on http://$HOST:$PORT/ using model '$MODEL' ($ARCHITECTURE${HISTORY:+, history=$HISTORY})"
cargo build --release --bin engine-zoo

exec target/release/engine-zoo serve \
  --game "$GAME" \
  --run-dir "$RUN_DIR" \
  --bind "$HOST:$PORT"

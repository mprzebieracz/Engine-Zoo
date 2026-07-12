#!/usr/bin/env bash
# Launch a reproducible, colour-paired chess checkpoint arena.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# Arena parameters. Override these with environment variables or pass CLI
# flags at invocation time (CLI flags are appended below and take precedence).
if [[ -n "${FASTCHESS_BIN:-}" ]]; then
  : "${FASTCHESS_BIN}"
else
  FASTCHESS_BIN="$(scripts/../crates/evaluations/discover_fastchess.sh 2>/dev/null | sed -n '1p' || true)"
fi
if [[ -z "$FASTCHESS_BIN" || ! -f "$FASTCHESS_BIN" || ! -x "$FASTCHESS_BIN" ]]; then
  echo "Fastchess not found. Install it with crates/evaluations/install_fastchess.sh or set FASTCHESS_BIN=/path/to/fastchess." >&2
  exit 1
fi
UCI_BIN="${UCI_BIN:-target/release/engine-zoo-uci}"

# Checkpoint paths. Each checkpoint must live in a run containing config.json.
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-data/runs/chess-az-v2-h4/best.safetensors}"
# CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-data/runs/scratch/chess_AZNetwork_0.safetensors}"
# Available architectures: legacy is the original scalar-value network;
# chess-az-v2 is the newer spatial-policy/WDL network.
CANDIDATE_ARCHITECTURE="${CANDIDATE_ARCHITECTURE:-chess-az-v2}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-data/runs/chess-az-v2-h4/checkpoints/ckpt_0450.safetensors}"
# BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-data/runs/chess/best.safetensors}"
BASELINE_ARCHITECTURE="${BASELINE_ARCHITECTURE:-chess-az-v2}"

# Initial plies sampled from each model's MCTS policy; 0 means deterministic.
OPENING_PLIES="${OPENING_PLIES:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-data/arenas/sitekarena2}"
# Total games; colors are paired, so this must be even.
GAMES="${GAMES:-4}"
# MCTS simulations per move for both engines. Higher is stronger/slower.
SIMULATIONS="${SIMULATIONS:-400}"
# Inference device: cpu, cuda, or cuda:0.
DEVICE="${DEVICE:-cuda}"
# Concurrent games; increase only if hardware has capacity.
CONCURRENCY="${CONCURRENCY:-1}"
# Maximum game length in plies; there is no practical clock limit.
MAX_MOVES="${MAX_MOVES:-512}"

# Build on demand so this script also works from a fresh checkout.
cargo build --release \
  -p checkpoint-eval --bin eval-arena \
  -p engine_app --bin engine-zoo-uci

arena_args=(
  --fastchess "$FASTCHESS_BIN"
  --uci "$UCI_BIN"
  --candidate "$CANDIDATE_CHECKPOINT"
  --candidate-architecture "$CANDIDATE_ARCHITECTURE"
  --baseline "$BASELINE_CHECKPOINT"
  --baseline-architecture "$BASELINE_ARCHITECTURE"
  --opening-plies "$OPENING_PLIES"
  --output-dir "$OUTPUT_DIR"
  --games "$GAMES"
  --simulations "$SIMULATIONS"
  --device "$DEVICE"
  --concurrency "$CONCURRENCY"
  --max-moves "$MAX_MOVES"
)

exec target/release/eval-arena "${arena_args[@]}" "$@"

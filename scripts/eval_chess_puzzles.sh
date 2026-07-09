#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-runs/chess-puct-real}"
SERVER="${SERVER:-}"
MODEL="${MODEL:-best}"
MODE="${MODE:-mcts}" # net or mcts
SIMULATIONS="${SIMULATIONS:-100}"
WAIT_FOR="${WAIT_FOR:-1}"
SUITE="${SUITE:-suites/chess_lichess_easy.jsonl}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/eval/puzzles}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
JSONL_OUT="${JSONL_OUT:-$OUT_DIR/${MODEL}_${MODE}_${SIMULATIONS}_${STAMP}.jsonl}"
HTML_OUT="${HTML_OUT:-$OUT_DIR/${MODEL}_${MODE}_${SIMULATIONS}_${STAMP}.html}"
STDERR_LOG="${STDERR_LOG:-$OUT_DIR/${MODEL}_${MODE}_${SIMULATIONS}_${STAMP}.stderr.log}"

mkdir -p "$OUT_DIR"

echo "suite: $SUITE"
echo "run dir: $RUN_DIR"
if [[ -n "$SERVER" ]]; then
  echo "server: $SERVER"
fi
echo "model: $MODEL"
echo "mode: $MODE"
echo "simulations: $SIMULATIONS"
echo "jsonl: $JSONL_OUT"
echo "html: $HTML_OUT"
echo "stderr log: $STDERR_LOG"

cargo build --release --bin eval

args=(
  bench
  --game chess
  --model "$MODEL"
  --mode "$MODE"
  --simulations "$SIMULATIONS"
  --wait-for-count "$WAIT_FOR"
  --suite "$SUITE"
  --output "$JSONL_OUT"
  --html "$HTML_OUT"
)
if [[ -n "$SERVER" ]]; then
  args+=(--server "$SERVER")
else
  args+=(--run-dir "$RUN_DIR")
fi

target/release/eval "${args[@]}" 2> "$STDERR_LOG"

echo "wrote $JSONL_OUT"
echo "wrote $HTML_OUT"
echo "wrote $STDERR_LOG"

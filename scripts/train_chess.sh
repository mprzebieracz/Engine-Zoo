#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Main run controls.
RUN_DIR="${RUN_DIR:-data/runs/chess}"
MCTS_VARIANT="${MCTS_VARIANT:-gumbel}" # puct or gumbel
GUMBEL_SAMPLED_ACTIONS="${GUMBEL_SAMPLED_ACTIONS:-16}"

# Self-play and training scale.
GAMES="${GAMES:-500}"
TRAIN_STEPS="${TRAIN_STEPS:-80}"
TRAIN_PROGRESS_EVERY="${TRAIN_PROGRESS_EVERY:-10}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-4096}"
BUFFER="${BUFFER:-500000}"

# Network architecture for new run directories. Existing run dirs keep config.json.
BLOCKS="${BLOCKS:-10}"
FILTERS="${FILTERS:-64}"

# Search and self-play throughput knobs.
SIMULATIONS="${SIMULATIONS:-400}"
MCTS_LEAF_BATCH_SIZE="${MCTS_LEAF_BATCH_SIZE:-32}"
FAST_SIMULATIONS="${FAST_SIMULATIONS:-100}"
FULL_SIMULATION_PROBABILITY="${FULL_SIMULATION_PROBABILITY:-0.25}"
THREADS="${THREADS:-32}"
WAIT_FOR="${WAIT_FOR:-128}"
BATCH_TIMEOUT_MS="${BATCH_TIMEOUT_MS:-5}"
TT_ENTRIES="${TT_ENTRIES:-1000000}"
MAX_MOVES="${MAX_MOVES:-512}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"

# Checkpoint/log cadence.
NUMBERED_CHECKPOINT_EVERY="${NUMBERED_CHECKPOINT_EVERY:-50}"
ARCHIVE_CHECKPOINT_MINUTES="${ARCHIVE_CHECKPOINT_MINUTES:-60}"
STDOUT_LOG="${STDOUT_LOG:-$RUN_DIR/train.log}"
STDERR_LOG="${STDERR_LOG:-$RUN_DIR/stderr.log}"
RESUME_LATEST_CHECKPOINT="${RESUME_LATEST_CHECKPOINT:-1}"

# Device/precision.
DEVICE="${DEVICE:-cuda}"
INFERENCE_PRECISION="${INFERENCE_PRECISION:-auto}"

# Background checkpoint evaluation. Set EVALUATE_CHECKPOINTS=0 to disable.
EVALUATE_CHECKPOINTS="${EVALUATE_CHECKPOINTS:-1}"
EVAL_EVERY="${EVAL_EVERY:-50}"
EVAL_GAMES="${EVAL_GAMES:-4}"
BASELINE_GAMES="${BASELINE_GAMES:-4}"
EVAL_SIMULATIONS="${EVAL_SIMULATIONS:-800}"
STOCKFISH_BIN="${STOCKFISH_BIN:-crates/evaluations/bin/stockfish-18/stockfish-ubuntu-x86-64}"
STOCKFISH_ELO="${STOCKFISH_ELO:-1600}"
STOCKFISH_MOVETIME_MS="${STOCKFISH_MOVETIME_MS:-200}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_BACKFILL="${EVAL_BACKFILL:-0}"

mkdir -p "$RUN_DIR"

echo "run dir: $RUN_DIR"
echo "mcts: $MCTS_VARIANT"
echo "stdout log: $STDOUT_LOG"
echo "stderr log: $STDERR_LOG"

if [[ "$RESUME_LATEST_CHECKPOINT" == "1" ]]; then
  latest_checkpoint=""
  if [[ -d "$RUN_DIR/checkpoints" ]]; then
    latest_checkpoint="$(
      find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'ckpt_*.safetensors' \
        | sort -V \
        | tail -n 1
    )"
  fi
  if [[ -n "$latest_checkpoint" ]]; then
    cp "$latest_checkpoint" "$RUN_DIR/best.safetensors"
    echo "resuming self-play from latest checkpoint: $latest_checkpoint"
  fi
fi

cargo build --release --bin train
cargo build --release -p checkpoint-eval

if [[ "$EVALUATE_CHECKPOINTS" == "1" ]]; then
  if ! command -v "$STOCKFISH_BIN" >/dev/null 2>&1 && [[ ! -x "$STOCKFISH_BIN" ]]; then
    echo "Stockfish not found: run crates/evaluations/install_stockfish.sh or set STOCKFISH_BIN" >&2
    exit 1
  fi
  mkdir -p "$RUN_DIR/evaluations"
  echo "starting checkpoint evaluator watcher" >> "$RUN_DIR/evaluations/watcher.log"
  echo "root=$ROOT_DIR run_dir=$RUN_DIR eval_every=$EVAL_EVERY" >> "$RUN_DIR/evaluations/watcher.log"
  EVAL_BIN="target/release/checkpoint-eval" \
  EVAL_EVERY="$EVAL_EVERY" EVAL_GAMES="$EVAL_GAMES" BASELINE_GAMES="$BASELINE_GAMES" EVAL_SIMULATIONS="$EVAL_SIMULATIONS" \
  STOCKFISH_BIN="$STOCKFISH_BIN" STOCKFISH_ELO="$STOCKFISH_ELO" \
  STOCKFISH_MOVETIME_MS="$STOCKFISH_MOVETIME_MS" EVAL_DEVICE="$EVAL_DEVICE" EVAL_BACKFILL="$EVAL_BACKFILL" \
  crates/evaluations/watch_checkpoints.sh "$RUN_DIR" >> "$RUN_DIR/evaluations/watcher.log" 2>&1 &
  EVAL_WATCHER_PID=$!
  trap 'kill "$EVAL_WATCHER_PID" 2>/dev/null || true' EXIT
  echo "checkpoint evaluator: every $EVAL_EVERY checkpoints, 4 games vs Stockfish and 4 vs the prior checkpoint"
fi

target/release/train \
  --game chess \
  --run-dir "$RUN_DIR" \
  --stderr-log "$STDERR_LOG" \
  --progress-every "$PROGRESS_EVERY" \
  --forever \
  --games "$GAMES" \
  --threads "$THREADS" \
  --wait-for "$WAIT_FOR" \
  --batch-timeout-ms "$BATCH_TIMEOUT_MS" \
  --simulations "$SIMULATIONS" \
  --mcts-leaf-batch-size "$MCTS_LEAF_BATCH_SIZE" \
  --fast-simulations "$FAST_SIMULATIONS" \
  --full-simulation-probability "$FULL_SIMULATION_PROBABILITY" \
  --max-moves "$MAX_MOVES" \
  --tt-entries "$TT_ENTRIES" \
  --train-steps "$TRAIN_STEPS" \
  --train-progress-every "$TRAIN_PROGRESS_EVERY" \
  --minibatch-size "$MINIBATCH_SIZE" \
  --buffer "$BUFFER" \
  --blocks "$BLOCKS" \
  --filters "$FILTERS" \
  --mode continuous \
  --mcts-variant "$MCTS_VARIANT" \
  --gumbel-sampled-actions "$GUMBEL_SAMPLED_ACTIONS" \
  --numbered-checkpoint-every "$NUMBERED_CHECKPOINT_EVERY" \
  --archive-checkpoint-minutes "$ARCHIVE_CHECKPOINT_MINUTES" \
  --device "$DEVICE" \
  --inference-precision "$INFERENCE_PRECISION" \
  | tee -a "$STDOUT_LOG"

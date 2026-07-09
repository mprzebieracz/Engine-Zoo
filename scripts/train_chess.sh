#!/usr/bin/env bash
set -euo pipefail

# Main run controls.
RUN_DIR="${RUN_DIR:-runs/chess-puct-real}"
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
SIMULATIONS="${SIMULATIONS:-800}"
FAST_SIMULATIONS="${FAST_SIMULATIONS:-100}"
FULL_SIMULATION_PROBABILITY="${FULL_SIMULATION_PROBABILITY:-0.25}"
THREADS="${THREADS:-128}"
WAIT_FOR="${WAIT_FOR:-24}"
BATCH_TIMEOUT_MS="${BATCH_TIMEOUT_MS:-5}"
TT_ENTRIES="${TT_ENTRIES:-1000000}"
MAX_MOVES="${MAX_MOVES:-512}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"

# Checkpoint/log cadence.
ARCHIVE_CHECKPOINT_MINUTES="${ARCHIVE_CHECKPOINT_MINUTES:-60}"
STDOUT_LOG="${STDOUT_LOG:-$RUN_DIR/train.log}"
STDERR_LOG="${STDERR_LOG:-$RUN_DIR/stderr.log}"
RESUME_LATEST_CHECKPOINT="${RESUME_LATEST_CHECKPOINT:-1}"

# Device/precision.
DEVICE="${DEVICE:-cuda}"
INFERENCE_PRECISION="${INFERENCE_PRECISION:-auto}"

mkdir -p "$RUN_DIR"

echo "run dir: $RUN_DIR"
echo "mcts: $MCTS_VARIANT"
echo "stdout log: $STDOUT_LOG"
echo "stderr log: $STDERR_LOG"

if [[ "$RESUME_LATEST_CHECKPOINT" == "1" ]]; then
  latest_checkpoint="$(
    find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'ckpt_*.safetensors' 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"
  if [[ -n "$latest_checkpoint" ]]; then
    cp "$latest_checkpoint" "$RUN_DIR/best.safetensors"
    echo "resuming self-play from latest checkpoint: $latest_checkpoint"
  fi
fi

cargo build --release --bin train

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
  --archive-checkpoint-minutes "$ARCHIVE_CHECKPOINT_MINUTES" \
  --device "$DEVICE" \
  --inference-precision "$INFERENCE_PRECISION" \
  | tee -a "$STDOUT_LOG"

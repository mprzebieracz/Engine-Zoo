#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Main run controls. New chess runs use the versioned AlphaZero-v2 network.
RUN_DIR="${RUN_DIR:-data/runs/chess-az-v2-h4}"
ARCHITECTURE="${ARCHITECTURE:-chess-az-v2}"
HISTORY="${HISTORY:-4}"

# Self-play and training scale.
GAMES="${GAMES:-500}"
TRAIN_STEPS="${TRAIN_STEPS:-80}"
TRAIN_PROGRESS_EVERY="${TRAIN_PROGRESS_EVERY:-10}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-4096}"
BUFFER="${BUFFER:-500000}"

# Chess-v2 search budgets: Gumbel uses paired root-candidate profiles; PUCT
# uses the full/fast simulation counts directly.
V2_FULL_SIMULATIONS="${V2_FULL_SIMULATIONS:-400}"
V2_FULL_ROOT_CANDIDATES="${V2_FULL_ROOT_CANDIDATES:-32}"
V2_FAST_SIMULATIONS="${V2_FAST_SIMULATIONS:-100}"
V2_FAST_ROOT_CANDIDATES="${V2_FAST_ROOT_CANDIDATES:-16}"
V2_FULL_SIMULATION_PROBABILITY="${V2_FULL_SIMULATION_PROBABILITY:-0.5}"
MCTS_VARIANT="${MCTS_VARIANT:-gumbel}"
case "$MCTS_VARIANT" in
  puct|gumbel) ;;
  *) echo "MCTS_VARIANT must be 'puct' or 'gumbel' (got: $MCTS_VARIANT)" >&2; exit 1 ;;
esac

# Search and self-play throughput knobs. These defaults were selected from a
# CUDA sweep on the v2 network (RTX 4080 SUPER): leaf=32, threads=16,
# wait-for=256. Override them for different GPUs or workloads.
MCTS_LEAF_BATCH_SIZE="${MCTS_LEAF_BATCH_SIZE:-24}"
THREADS="${THREADS:-24}"
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
# `auto` means FP16 self-play on CUDA and FP32 on CPU. Training is always FP32.
INFERENCE_PRECISION="${INFERENCE_PRECISION:-auto}"

# V2 evaluation adapters are not enabled yet. Set this only after selecting a
# legacy run or adding a v2-compatible evaluator.
EVALUATE_CHECKPOINTS="${EVALUATE_CHECKPOINTS:-0}"
EVAL_PROFILE="${EVAL_PROFILE:-quick}"
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
echo "architecture: $ARCHITECTURE (history=$HISTORY)"
echo "mcts: $MCTS_VARIANT"
echo "stdout log: $STDOUT_LOG"
echo "stderr log: $STDERR_LOG"

# `best` is updated every iteration, whereas numbered checkpoints may be
# sparse. Only reconstruct it from a numbered checkpoint for older runs that
# do not already have the active self-play weights.
if [[ "$RESUME_LATEST_CHECKPOINT" == "1" && ! -f "$RUN_DIR/best.safetensors" ]]; then
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
cargo build --release -p checkpoint-eval --bins
cargo build --release -p engine_app --bin engine-zoo-uci

if [[ "$EVALUATE_CHECKPOINTS" == "1" ]]; then
  if [[ "$ARCHITECTURE" == "chess-az-v2" ]]; then
    echo "checkpoint evaluation is not supported for chess-az-v2 yet; set EVALUATE_CHECKPOINTS=0" >&2
    exit 1
  fi
  if ! command -v "$STOCKFISH_BIN" >/dev/null 2>&1 && [[ ! -x "$STOCKFISH_BIN" ]]; then
    echo "Stockfish not found: run crates/evaluations/install_stockfish.sh or set STOCKFISH_BIN" >&2
    exit 1
  fi
  if [[ "$EVAL_PROFILE" == "quick" ]] && ! command -v "${FASTCHESS_BIN:-fastchess}" >/dev/null 2>&1 && [[ ! -x "${FASTCHESS_BIN:-fastchess}" ]]; then
    echo "Fastchess not found: run crates/evaluations/install_fastchess.sh or set FASTCHESS_BIN" >&2
    exit 1
  fi
  mkdir -p "$RUN_DIR/evaluations"
  echo "starting checkpoint evaluator watcher" >> "$RUN_DIR/evaluations/watcher.log"
  echo "root=$ROOT_DIR run_dir=$RUN_DIR profile=$EVAL_PROFILE eval_every=$EVAL_EVERY" >> "$RUN_DIR/evaluations/watcher.log"
  EVAL_BIN="target/release/checkpoint-eval" \
  EVAL_PROFILE="$EVAL_PROFILE" \
  EVAL_EVERY="$EVAL_EVERY" EVAL_GAMES="$EVAL_GAMES" BASELINE_GAMES="$BASELINE_GAMES" EVAL_SIMULATIONS="$EVAL_SIMULATIONS" \
  FASTCHESS_BIN="${FASTCHESS_BIN:-fastchess}" STOCKFISH_BIN="$STOCKFISH_BIN" STOCKFISH_ELO="$STOCKFISH_ELO" \
  STOCKFISH_MOVETIME_MS="$STOCKFISH_MOVETIME_MS" EVAL_DEVICE="$EVAL_DEVICE" EVAL_BACKFILL="$EVAL_BACKFILL" \
  crates/evaluations/watch_checkpoints.sh "$RUN_DIR" >> "$RUN_DIR/evaluations/watcher.log" 2>&1 &
  EVAL_WATCHER_PID=$!
  trap 'kill "$EVAL_WATCHER_PID" 2>/dev/null || true' EXIT
  echo "checkpoint evaluator: profile=$EVAL_PROFILE every $EVAL_EVERY checkpoints"
fi

target/release/train \
  --game chess \
  --architecture "$ARCHITECTURE" \
  --history "$HISTORY" \
  --run-dir "$RUN_DIR" \
  --stderr-log "$STDERR_LOG" \
  --progress-every "$PROGRESS_EVERY" \
  --forever \
  --games "$GAMES" \
  --threads "$THREADS" \
  --wait-for "$WAIT_FOR" \
  --batch-timeout-ms "$BATCH_TIMEOUT_MS" \
  --mcts-leaf-batch-size "$MCTS_LEAF_BATCH_SIZE" \
  --v2-full-simulations "$V2_FULL_SIMULATIONS" \
  --v2-full-root-candidates "$V2_FULL_ROOT_CANDIDATES" \
  --v2-fast-simulations "$V2_FAST_SIMULATIONS" \
  --v2-fast-root-candidates "$V2_FAST_ROOT_CANDIDATES" \
  --v2-full-simulation-probability "$V2_FULL_SIMULATION_PROBABILITY" \
  --mcts-variant "$MCTS_VARIANT" \
  --max-moves "$MAX_MOVES" \
  --tt-entries "$TT_ENTRIES" \
  --train-steps "$TRAIN_STEPS" \
  --train-progress-every "$TRAIN_PROGRESS_EVERY" \
  --minibatch-size "$MINIBATCH_SIZE" \
  --buffer "$BUFFER" \
  --mode continuous \
  --numbered-checkpoint-every "$NUMBERED_CHECKPOINT_EVERY" \
  --archive-checkpoint-minutes "$ARCHIVE_CHECKPOINT_MINUTES" \
  --device "$DEVICE" \
  --inference-precision "$INFERENCE_PRECISION" \
  | tee -a "$STDOUT_LOG"

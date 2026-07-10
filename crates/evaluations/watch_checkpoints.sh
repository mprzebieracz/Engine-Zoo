#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: watch_checkpoints.sh RUN_DIR}"
EVAL_BIN="${EVAL_BIN:-target/release/checkpoint-eval}"
STOCKFISH_BIN="${STOCKFISH_BIN:-crates/evaluations/bin/stockfish-18/stockfish-ubuntu-x86-64}"
EVAL_EVERY="${EVAL_EVERY:-50}"
EVAL_GAMES="${EVAL_GAMES:-4}"
BASELINE_GAMES="${BASELINE_GAMES:-4}"
EVAL_SIMULATIONS="${EVAL_SIMULATIONS:-800}"
STOCKFISH_ELO="${STOCKFISH_ELO:-1400}"
STOCKFISH_MOVETIME_MS="${STOCKFISH_MOVETIME_MS:-200}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EVAL_POLL_SECONDS="${EVAL_POLL_SECONDS:-15}"
EVAL_BACKFILL="${EVAL_BACKFILL:-0}"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
STATE_DIR="$RUN_DIR/evaluations"
STATE_FILE="$STATE_DIR/watcher.state"

mkdir -p "$STATE_DIR"
echo "watcher started: run_dir=$RUN_DIR every=$EVAL_EVERY games=$EVAL_GAMES baseline_games=$BASELINE_GAMES" 

latest_index() {
  find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'ckpt_*.safetensors' -printf '%f\n' 2>/dev/null \
    | sed -E 's/^ckpt_([0-9]+)\.safetensors$/\1/' | sort -n | tail -n 1
}

if [[ -f "$STATE_FILE" ]]; then
  last_seen="$(<"$STATE_FILE")"
  state_result="$(printf '%s/ckpt_%04d.json' "$STATE_DIR" "$last_seen")"
  if (( last_seen > 0 )) && [[ ! -f "$state_result" ]]; then
    # Older watcher versions initialized state to the newest checkpoint before
    # evaluating it. Retry that checkpoint once after upgrading.
    last_seen=$((last_seen - EVAL_EVERY))
  fi
elif [[ "$EVAL_BACKFILL" == "1" ]]; then
  last_seen=0
else
  last_seen="$(latest_index)"
  last_seen="${last_seen:-0}"
  if (( last_seen >= EVAL_EVERY )); then
    # Evaluate the latest eligible checkpoint immediately without backfilling
    # the complete history.
    last_seen=$((last_seen - EVAL_EVERY))
  fi
  printf '%s\n' "$last_seen" > "$STATE_FILE"
fi

while true; do
  while IFS= read -r checkpoint; do
    name="$(basename "$checkpoint")"
    index="${name#ckpt_}"
    index="${index%.safetensors}"
    index=$((10#$index))
    if (( index <= last_seen || index % EVAL_EVERY != 0 )); then
      continue
    fi
    echo "evaluation watcher: evaluating $checkpoint"
    baseline="$(printf '%s/checkpoints/ckpt_%04d.safetensors' "$RUN_DIR" "$((index - EVAL_EVERY))")"
    baseline_args=()
    if [[ -f "$baseline" ]]; then
      baseline_args=(--baseline "$baseline" --baseline-games "$BASELINE_GAMES")
    fi
    if "$EVAL_BIN" \
      --run-dir "$RUN_DIR" \
      --checkpoint "$checkpoint" \
      --stockfish "$STOCKFISH_BIN" \
      --stockfish-elo "$STOCKFISH_ELO" \
      --games "$EVAL_GAMES" \
      --simulations "$EVAL_SIMULATIONS" \
      --stockfish-movetime-ms "$STOCKFISH_MOVETIME_MS" \
      --device "$EVAL_DEVICE" "${baseline_args[@]}"; then
      echo "evaluation watcher: finished $checkpoint"
    else
      echo "evaluation watcher: failed $checkpoint; skipping it" >&2
    fi
    last_seen="$index"
    printf '%s\n' "$last_seen" > "$STATE_FILE"
  done < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'ckpt_*.safetensors' -print 2>/dev/null | sort -V)
  sleep "$EVAL_POLL_SECONDS"
done

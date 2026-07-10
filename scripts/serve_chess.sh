#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-data/runs/chess-puct-real}"
BIND="${BIND:-127.0.0.1:8080}"
GAME="${GAME:-chess}"

echo "serving $GAME from $RUN_DIR on http://$BIND/"

cargo build --release --bin engine-zoo

target/release/engine-zoo serve \
  --game "$GAME" \
  --run-dir "$RUN_DIR" \
  --bind "$BIND"

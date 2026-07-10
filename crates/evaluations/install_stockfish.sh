#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-crates/evaluations/bin/stockfish-18}"
URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64.tar"

mkdir -p "$DEST"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
curl -fL "$URL" -o "$tmp"
tar -xf "$tmp" -C "$DEST" --strip-components=1
echo "Installed Stockfish: $DEST/stockfish-ubuntu-x86-64"

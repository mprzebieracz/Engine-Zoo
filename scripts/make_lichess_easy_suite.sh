#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-data/suites/chess_lichess_easy.jsonl}"
CACHE="${CACHE:-scratch/lichess_db_puzzle.csv.zst}"
LIMIT="${LIMIT:-30}"
MIN_RATING="${MIN_RATING:-600}"
MAX_RATING="${MAX_RATING:-900}"
MIN_POPULARITY="${MIN_POPULARITY:-80}"
MIN_PLAYS="${MIN_PLAYS:-100}"
URL="${URL:-https://database.lichess.org/lichess_db_puzzle.csv.zst}"

mkdir -p "$(dirname "$OUT")" "$(dirname "$CACHE")"

if [[ ! -s "$CACHE" ]]; then
  echo "downloading $URL"
  curl -L "$URL" -o "$CACHE"
fi

zstd -dc "$CACHE" \
  | awk -F, -v limit="$LIMIT" -v min_rating="$MIN_RATING" -v max_rating="$MAX_RATING" -v min_pop="$MIN_POPULARITY" -v min_plays="$MIN_PLAYS" '
      NR == 1 { next }
      $4 >= min_rating && $4 <= max_rating && $6 >= min_pop && $7 >= min_plays {
        split($3, moves, " ");
        if (length(moves[1]) < 4 || length(moves[2]) < 4) next;
        gsub(/"/, "\\\"", $2);
        gsub(/"/, "\\\"", $8);
        printf("{\"name\":\"Lichess %s\",\"category\":\"%s rating %s\",\"position\":{\"game\":\"chess\",\"position\":{\"fen\":\"%s\",\"moves\":[\"%s\"]}},\"expected\":[\"%s\"]}\n", $1, $8, $4, $2, moves[1], moves[2]);
        count++;
        if (count >= limit) exit;
      }
    ' > "$OUT"

echo "wrote $OUT"

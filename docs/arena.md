# Multi-model chess arena

Keep checkpoints under `data/models/alphazero` and edit
`configs/arena.toml`. The roster names every model explicitly, including its
checkpoint path and architecture label, so incompatible model families are
not discovered or compared accidentally.

Run the arena with:

```bash
python3 scripts/arena_big.py
```

Each opponent gets a directory under the configured output directory with
cleaned PGNs, Fastchess diagnostics, `spec.json`, and `report.json`. The root
also receives `arena.json`, containing every opponent's score, smoothed
candidate-relative Elo estimate, and aggregate score.

Override the roster or use a smaller smoke run with:

```bash
python3 scripts/arena_big.py --config configs/arena.toml --debug
target/debug/eval-big-arena --config configs/arena.toml
```

The configured game count is total games per candidate/opponent pair and must
be even because colors are balanced.

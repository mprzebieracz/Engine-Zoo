#!/usr/bin/env python3
"""Round-robin checkpoint tournament via arena_big.py.

Each unique unordered pair plays ``games`` games (colors balanced). After every
pair, standings are rewritten to ``output_dir/STANDINGS.md`` and
``standings.json``. Completed pairs are skipped on resume (look for
``pairs/<a>__vs__<b>/arena.json``).

Example::

    python3 scripts/arena_round_robin.py \\
      --roster configs/arena-v6-round-robin-roster.toml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Model:
    name: str
    path: str
    architecture: str = "chess-se-h4"
    simulations: int = 400
    device: str = "cuda"
    backend: str = "auto"


def _load_roster(path: Path) -> tuple[list[Model], dict]:
    data = tomllib.loads(path.read_text())
    settings = data.get("settings", {})
    models = [
        Model(
            name=entry["name"],
            path=entry["path"],
            architecture=entry.get("architecture", "chess-se-h4"),
            simulations=int(entry.get("simulations", settings.get("simulations", 400))),
            device=entry.get("device", settings.get("device", "cuda")),
            backend=entry.get("backend", "auto"),
        )
        for entry in data["models"]
    ]
    if len(models) < 2:
        raise SystemExit("roster needs at least 2 models")
    names = [m.name for m in models]
    if len(names) != len(set(names)):
        raise SystemExit("duplicate model names in roster")
    return models, settings


def _pair_key(a: str, b: str) -> str:
    left, right = sorted([a, b])
    return f"{left}__vs__{right}"


def _pair_dir(output_dir: Path, a: str, b: str) -> Path:
    return output_dir / "pairs" / _pair_key(a, b)


def _write_pair_config(
    path: Path,
    candidate: Model,
    opponent: Model,
    *,
    games: int,
    settings: dict,
    pair_output: Path,
) -> None:
    lines = [
        f'name = "{candidate.name}"',
        f'path = "{candidate.path}"',
        f'architecture = "{candidate.architecture}"',
        f"simulations = {candidate.simulations}",
        f'device = "{candidate.device}"',
        f'backend = "{candidate.backend}"',
        "",
        "[[opponents]]",
        f'name = "{opponent.name}"',
        f'path = "{opponent.path}"',
        f'architecture = "{opponent.architecture}"',
        f"simulations = {opponent.simulations}",
        f'device = "{opponent.device}"',
        f'backend = "{opponent.backend}"',
        "",
        "[settings]",
        f"games = {games}",
        f"simulations = {int(settings.get('simulations', 400))}",
        f'device = "{settings.get("device", "cuda")}"',
        f"concurrency = {int(settings.get('concurrency', 1))}",
        f'time_control = "{settings.get("time_control", "1000000+0")}"',
        f"max_moves = {int(settings.get('max_moves', 512))}",
        f"opening_plies = {int(settings.get('opening_plies', 8))}",
        f'output_dir = "{pair_output}"',
        f'fastchess = "{settings.get("fastchess", "../crates/evaluations/bin/fastchess/fastchess")}"',
    ]
    path.write_text("[candidate]\n" + "\n".join(lines) + "\n")


def _load_pair_result(pair_output: Path, candidate: str, opponent: str) -> dict | None:
    arena = pair_output / "arena.json"
    if not arena.is_file():
        return None
    data = json.loads(arena.read_text())
    opp = data["opponents"][0]
    # Score is from candidate's perspective.
    score = opp["score"]
    return {
        "candidate": candidate,
        "opponent": opponent,
        "wins": score["wins"],
        "draws": score["draws"],
        "losses": score["losses"],
        "score_fraction": opp["score_fraction"],
        "elo_delta": opp.get("elo_delta"),
        "games": data.get("games") or data.get("total_games"),
        "arena_json": str(arena),
    }


def _aggregate(models: list[Model], results: list[dict]) -> list[dict]:
    stats = {
        m.name: {
            "name": m.name,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points": 0.0,
            "score_sum": 0.0,
        }
        for m in models
    }
    for result in results:
        a, b = result["candidate"], result["opponent"]
        aw, ad, al = result["wins"], result["draws"], result["losses"]
        for name, w, d, l in ((a, aw, ad, al), (b, al, ad, aw)):
            row = stats[name]
            row["played"] += w + d + l
            row["wins"] += w
            row["draws"] += d
            row["losses"] += l
            row["points"] += w + 0.5 * d
            row["score_sum"] += (w + 0.5 * d) / max(w + d + l, 1)
    rows = list(stats.values())
    for row in rows:
        row["score_pct"] = (
            100.0 * row["points"] / row["played"] if row["played"] else 0.0
        )
    rows.sort(key=lambda r: (r["points"], r["score_pct"], r["wins"]), reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def _write_standings(
    output_dir: Path,
    models: list[Model],
    results: list[dict],
    *,
    total_pairs: int,
) -> None:
    standings = _aggregate(models, results)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models": [m.name for m in models],
        "pairs_completed": len(results),
        "pairs_total": total_pairs,
        "results": results,
        "standings": standings,
    }
    (output_dir / "standings.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Round-robin standings",
        "",
        f"- updated: `{payload['updated_at']}`",
        f"- pairs: {len(results)} / {total_pairs}",
        f"- models: {len(models)}",
        "",
        "| Rank | Model | Points | Score % | W | D | L | Games |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in standings:
        lines.append(
            f"| {row['rank']} | `{row['name']}` | {row['points']:.1f} | "
            f"{row['score_pct']:.1f}% | {row['wins']} | {row['draws']} | "
            f"{row['losses']} | {row['played']} |"
        )
    lines.extend(["", "## Pair results", ""])
    for result in results:
        lines.append(
            f"- `{result['candidate']}` vs `{result['opponent']}`: "
            f"{result['wins']}-{result['draws']}-{result['losses']} "
            f"({100.0 * result['score_fraction']:.1f}%)"
        )
    (output_dir / "STANDINGS.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument(
        "--games",
        type=int,
        help="games per pair (even). Defaults to roster settings.games or 10.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override roster settings.output_dir",
    )
    args = parser.parse_args()

    roster_path = args.roster if args.roster.is_absolute() else ROOT / args.roster
    models, settings = _load_roster(roster_path)
    games = args.games or int(settings.get("games", 10))
    if games < 2 or games % 2 != 0:
        raise SystemExit("--games must be an even integer >= 2")

    output_dir = args.output_dir or Path(settings["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (roster_path.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pairs").mkdir(exist_ok=True)
    (output_dir / "roster.toml").write_text(roster_path.read_text())

    pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]
    results: list[dict] = []
    pending: list[tuple[Model, Model]] = []

    for left, right in pairs:
        # Prefer the lexicographically first name as candidate for stable paths.
        if left.name <= right.name:
            candidate, opponent = left, right
        else:
            candidate, opponent = right, left
        pair_output = _pair_dir(output_dir, candidate.name, opponent.name)
        existing = _load_pair_result(pair_output, candidate.name, opponent.name)
        if existing is not None:
            results.append(existing)
        else:
            pending.append((candidate, opponent))

    _write_standings(output_dir, models, results, total_pairs=len(pairs))
    print(
        f"round-robin: {len(models)} models, {len(pairs)} pairs, "
        f"{len(results)} done, {len(pending)} remaining, {games} games/pair",
        flush=True,
    )
    print(f"standings: {output_dir / 'STANDINGS.md'}", flush=True)

    for index, (candidate, opponent) in enumerate(pending, start=1):
        pair_output = _pair_dir(output_dir, candidate.name, opponent.name)
        pair_output.mkdir(parents=True, exist_ok=True)
        print(
            f"\n===== PAIR {len(results) + 1}/{len(pairs)} "
            f"({index}/{len(pending)} remaining queue): "
            f"{candidate.name} vs {opponent.name} =====",
            flush=True,
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".toml",
            prefix="arena-rr-",
            delete=False,
        ) as handle:
            config_path = Path(handle.name)
            _write_pair_config(
                config_path,
                candidate,
                opponent,
                games=games,
                settings=settings,
                pair_output=pair_output,
            )
        try:
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts/arena_big.py"),
                    "--config",
                    str(config_path),
                ],
                cwd=ROOT,
                check=True,
            )
        finally:
            config_path.unlink(missing_ok=True)

        result = _load_pair_result(pair_output, candidate.name, opponent.name)
        if result is None:
            raise SystemExit(f"pair finished without arena.json: {pair_output}")
        results.append(result)
        _write_standings(output_dir, models, results, total_pairs=len(pairs))
        print(
            f"pair result: {result['wins']}-{result['draws']}-{result['losses']} "
            f"({100.0 * result['score_fraction']:.1f}%); "
            f"standings updated",
            flush=True,
        )

    print("\n===== ROUND ROBIN COMPLETE =====", flush=True)
    print(f"standings: {output_dir / 'STANDINGS.md'}", flush=True)
    top = _aggregate(models, results)[0]
    print(
        f"leader: {top['name']} with {top['points']:.1f} points "
        f"({top['score_pct']:.1f}%)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

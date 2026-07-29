#!/usr/bin/env python3
"""Run a resumable CUDA self-play and training architecture timing matrix."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    architecture: str
    history: str
    config: Path


CASES = (
    Case(
        "chess-classic-10x64",
        "chess-classic-10x64",
        "legacy",
        Path("benchmarks/configs/architecture-matrix-chess-classic-10x64.toml"),
    ),
    Case(
        "chess-se-h1-12x128",
        "chess-se-12x128",
        "one",
        Path("benchmarks/configs/architecture-matrix-chess-se-h1-12x128.toml"),
    ),
    Case(
        "chess-se-h4-12x128",
        "chess-se-12x128",
        "four",
        Path("benchmarks/configs/architecture-matrix-chess-se-h4-12x128.toml"),
    ),
    Case(
        "chess-se-h8-12x128",
        "chess-se-12x128",
        "eight",
        Path("benchmarks/configs/architecture-matrix-chess-se-h8-12x128.toml"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results/architecture-training-matrix"),
        help="Directory for copied configs, case reports, and aggregate reports.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, help="Run at most this many incomplete cases.")
    parser.add_argument(
        "--engine-bench",
        type=Path,
        default=Path("target/release/engine-bench"),
        help="Path to an already-built engine-bench executable.",
    )
    return parser.parse_args()


def metrics(report_path: Path, case: Case) -> dict[str, object]:
    report = json.loads(report_path.read_text())
    sample = report["samples"][0]
    data = sample["metrics"]
    batch = data["batch_stats"]
    return {
        **asdict(case),
        "config": str(case.config),
        "elapsed_seconds": sample["elapsed_ns"] / 1_000_000_000,
        "positions_per_second": data["positions_per_second"],
        "inference_states_per_second": data["inference_states_per_second"],
        "average_inference_batch": batch["average_inference_batch"],
        "maximum_inference_batch": batch["maximum_inference_batch"],
        "report": str(report_path),
    }


def write_aggregate(output_dir: Path, completed: list[dict[str, object]]) -> None:
    completed.sort(key=lambda row: float(row["inference_states_per_second"]), reverse=True)
    (output_dir / "aggregate.json").write_text(json.dumps(completed, indent=2) + "\n")

    fields = list(completed[0]) if completed else [*asdict(CASES[0]), "elapsed_seconds", "positions_per_second", "inference_states_per_second", "average_inference_batch", "maximum_inference_batch", "report"]
    with (output_dir / "aggregate.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(completed)


def main() -> int:
    args = parse_args()
    if not args.dry_run and not args.engine_bench.is_file():
        raise SystemExit(
            f"engine-bench executable not found: {args.engine_bench}. Build it with "
            "cargo build --release -p engine-bench, or pass --engine-bench."
        )

    configs = args.output_dir / "configs"
    results = args.output_dir / "results"
    configs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    completed: list[dict[str, object]] = []
    pending: list[Case] = []
    for case in CASES:
        result_path = results / f"{case.name}.json"
        if result_path.is_file():
            try:
                completed.append(metrics(result_path, case))
                continue
            except (KeyError, ValueError, json.JSONDecodeError):
                print(f"re-running incomplete or invalid result: {result_path}", file=sys.stderr)
        pending.append(case)

    if args.dry_run:
        for index, case in enumerate(pending, start=1):
            print(f"{index}/{len(pending)} {case.name}")
        print(f"completed={len(completed)} pending={len(pending)} total={len(CASES)}")
        return 0

    for index, case in enumerate(pending, start=1):
        if args.limit is not None and index > args.limit:
            break

        config_path = configs / case.config.name
        result_path = results / f"{case.name}.json"
        shutil.copyfile(case.config, config_path)

        command = [
            str(args.engine_bench), "iteration", "--experiment", str(config_path),
            "--device", "cuda", "--samples", "1", "--name", case.name,
            "--output", str(result_path), "--human",
        ]
        print(f"[{index}/{len(pending)}] {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)

        row = metrics(result_path, case)
        completed.append(row)
        write_aggregate(args.output_dir, completed)
        print(
            f"  complete: {row['elapsed_seconds']:.2f}s, "
            f"{row['positions_per_second']:.1f} positions/s, "
            f"{row['inference_states_per_second']:.1f} states/s, "
            f"avg batch {row['average_inference_batch']:.1f}",
            flush=True,
        )

    write_aggregate(args.output_dir, completed)
    print(f"completed={len(completed)} pending={len(CASES) - len(completed)} total={len(CASES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

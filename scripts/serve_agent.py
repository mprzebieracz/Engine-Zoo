#!/usr/bin/env python3
"""Serve one local AlphaZero agent and register it as the only UI opponent.

Edit the defaults below, then run:

    python3 scripts/serve_agent.py

CLI flags override the defaults when you need a one-off.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Switch the served model here
# ---------------------------------------------------------------------------
RUN_DIR = "runs/chess-puct-wdl-tensorrt-v6"
MODEL = "latest"  # latest | generation-000025 | generation-000800 | ...
# Frozen v5 gen-800:
#   RUN_DIR = "runs/chess-puct-wdl-tensorrt-v5"
#   MODEL = "generation-000800"
HOST = "127.0.0.1"
PORT = 8080
GAME = "chess"
SIMULATIONS = 128
# ---------------------------------------------------------------------------

HUMAN = {
    "id": "human",
    "name": "Human",
    "kind": "human",
    "games": ["chess", "connect4"],
    "description": "Moves are entered through the board.",
    "badge": "Local",
}


def run_metadata(run_dir: Path, game: str) -> tuple[str, str | None]:
    config_path = run_dir / "experiment.toml"
    if not config_path.is_file():
        raise SystemExit(f"run experiment does not exist: {config_path}")
    model = tomllib.loads(config_path.read_text()).get("model", {})
    architecture = model.get("architecture")

    if architecture == "chess-se":
        if game != "chess":
            raise SystemExit(f"run model does not match --game {game}")
        return architecture, model.get("history")
    if architecture == "chess-classic":
        if game != "chess":
            raise SystemExit(f"run model does not match --game {game}")
        return architecture, None
    if architecture == "connect4-residual":
        if game != "connect4":
            raise SystemExit(f"run model does not match --game {game}")
        return architecture, None

    if model.get("game") != game:
        raise SystemExit(f"run model does not match --game {game}")
    representation = model.get("representation")
    if representation in ("chess-classic", "connect4-canonical"):
        return representation, None
    canonical = (
        representation.get("chess-canonical")
        if isinstance(representation, dict)
        else None
    )
    if canonical is None:
        raise SystemExit(f"unsupported model architecture: {architecture!r}")
    return "chess-canonical", canonical.get("history")


def resolve_model(run_dir: Path, model: str) -> tuple[str, Path]:
    """Map a short model name to the path string the serve API expects."""

    name = model.strip()
    if not name:
        raise SystemExit("--model must not be empty")

    aliases = {"latest", "best", "candidate"}
    if name in aliases:
        path = run_dir / "checkpoints" / f"{name}.safetensors"
        if not path.is_file():
            fallback = run_dir / f"{name}.safetensors"
            if fallback.is_file():
                return name, fallback
            raise SystemExit(_missing(run_dir, name))
        return name, path

    # Accept bare archive names: generation-000800[.safetensors]
    candidate = Path(name)
    if len(candidate.parts) == 1:
        file_name = name if name.endswith(".safetensors") else f"{name}.safetensors"
        path = run_dir / "checkpoints" / file_name
        if path.is_file():
            return f"checkpoints/{file_name}", path

    path = (run_dir / name).resolve() if not candidate.is_absolute() else candidate
    if path.is_file():
        try:
            rel = path.relative_to(run_dir.resolve()).as_posix()
        except ValueError:
            raise SystemExit(f"model is outside run dir: {path}") from None
        return rel, path

    raise SystemExit(_missing(run_dir, name))


def _missing(run_dir: Path, requested: str) -> str:
    checkpoints = run_dir / "checkpoints"
    available = sorted(p.name for p in checkpoints.glob("*.safetensors")) if checkpoints.is_dir() else []
    listing = ", ".join(available) if available else "(none)"
    return (
        f"checkpoint not found for model {requested!r} under {run_dir}\n"
        f"available: {listing}"
    )


def write_single_agent(
    config_path: Path,
    *,
    game: str,
    model: str,
    label: str,
    host: str,
    port: int,
    architecture: str,
    history: str | None,
    simulations: int,
) -> None:
    details = f"architecture={architecture}" + (f", history={history}" if history else "")
    agent = {
        "id": f"{game}-local",
        "name": label,
        "kind": "alphazero",
        "games": [game],
        "model": model,
        "server": f"http://{host}:{port}",
        "description": f"{game} · {model} ({details})",
        "badge": "Local",
        "defaults": {"simulations": simulations, "waitForCount": 16},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"agents": [HUMAN, agent]}, indent=2) + "\n")
    print(f"UI agents: human + {agent['id']} ({model}) → {config_path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--game", choices=("chess", "connect4"), default=GAME)
    p.add_argument("--run-dir", default=RUN_DIR, help=f"default: {RUN_DIR}")
    p.add_argument(
        "--model",
        default=MODEL,
        help="latest | generation-NNNNNN | path under the run dir",
    )
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--simulations", type=int, default=SIMULATIONS)
    p.add_argument("--name", help="UI display name")
    p.add_argument("--config", default="web/static/config/agents.json")
    p.add_argument("--no-config", action="store_true")
    args = p.parse_args()
    if not 1 <= args.port <= 65535:
        p.error("--port must be between 1 and 65535")

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    architecture, history = run_metadata(run_dir, args.game)
    api_model, checkpoint = resolve_model(run_dir, args.model)
    label = args.name or f"AlphaZero · {Path(api_model).name}"

    if not args.no_config:
        write_single_agent(
            Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config,
            game=args.game,
            model=api_model,
            label=label,
            host=args.host,
            port=args.port,
            architecture=architecture,
            history=history,
            simulations=args.simulations,
        )

    subprocess.run(
        ["cargo", "build", "--release", "--bin", "engine-zoo"],
        cwd=ROOT,
        check=True,
    )
    print(f"serving {checkpoint}  on http://{args.host}:{args.port}/")
    return subprocess.run(
        [
            "target/release/engine-zoo",
            "serve",
            "--game",
            args.game,
            "--run-dir",
            str(run_dir),
            "--bind",
            f"{args.host}:{args.port}",
        ],
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

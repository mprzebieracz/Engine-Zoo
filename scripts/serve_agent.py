#!/usr/bin/env python3
"""Build, register, and serve one local engine-zoo agent."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_metadata(run_dir: Path, game: str) -> tuple[str, str | None]:
    config_path = run_dir / "experiment.toml"
    if not config_path.is_file():
        raise SystemExit(f"run experiment does not exist: {config_path}")
    model = tomllib.loads(config_path.read_text()).get("model", {})
    if model.get("game") != game:
        raise SystemExit(f"run model does not match --game {game}")
    representation = model.get("representation")
    if representation == "chess-classic":
        return "chess-classic", None
    if representation == "connect4-canonical":
        return "connect4-canonical", None
    canonical = representation.get("chess-canonical") if isinstance(representation, dict) else None
    if canonical is None:
        raise SystemExit(f"unsupported representation: {representation!r}")
    return "chess-canonical", canonical.get("history")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", choices=("chess", "connect4"), default="chess")
    p.add_argument("--run-dir", default="data/runs/chess-az-v2-h4")
    p.add_argument("--model", default="best")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--id")
    p.add_argument("--name")
    p.add_argument("--config", default="web/static/config/agents.json")
    p.add_argument("--no-config", action="store_true")
    args = p.parse_args()
    if not 1 <= args.port <= 65535:
        p.error("--port must be between 1 and 65535")
    architecture, history = run_metadata(Path(args.run_dir), args.game)
    safe_model = re.sub(r"[^A-Za-z0-9_]+", "-", Path(args.model).name).strip("-")
    agent_id = args.id or f"{args.game}-{safe_model}"
    name = args.name or f"AlphaZero · {args.model}"
    if not args.no_config:
        config_path = Path(args.config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = json.loads(config_path.read_text()) if config_path.exists() else {"agents": []}
        agents = [agent for agent in config.get("agents", []) if agent.get("id") != agent_id]
        if not any(agent.get("id") == "human" for agent in agents):
            agents.insert(
                0,
                {
                    "id": "human", "name": "Human", "kind": "human",
                    "games": ["chess", "connect4"],
                    "description": "Moves are entered through the board.",
                    "badge": "Local",
                },
            )
        details = f"architecture={architecture}" + (f", history={history}" if history else "")
        agents.append(
            {
                "id": agent_id, "name": name, "kind": "alphazero",
                "games": [args.game], "model": args.model,
                "server": f"http://{args.host}:{args.port}",
                "description": f"{args.game} model {args.model} ({details})",
                "badge": "Local",
                "defaults": {"simulations": 128, "waitForCount": 16},
            }
        )
        config["agents"] = agents
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"registered {agent_id} in {config_path}")
    subprocess.run(["cargo", "build", "--release", "--bin", "engine-zoo"], cwd=ROOT, check=True)
    print(f"serving {args.game} from {args.run_dir} on http://{args.host}:{args.port}/")
    return subprocess.run(
        [
            "target/release/engine-zoo", "serve", "--game", args.game,
            "--run-dir", args.run_dir, "--bind", f"{args.host}:{args.port}",
        ],
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

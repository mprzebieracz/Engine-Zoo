#!/usr/bin/env python3
"""Install the bundled Stockfish release without shell pipelines."""

from __future__ import annotations

import argparse
import io
import tarfile
import urllib.request
from pathlib import Path

URL = "https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64.tar"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination", nargs="?", default="crates/evaluations/bin/stockfish-18")
    p.add_argument("--url", default=URL)
    args = p.parse_args()
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.url) as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:")
        members = archive.getmembers()
        root = next((member.name.split("/", 1)[0] for member in members if "/" in member.name), "")
        for member in members:
            relative = member.name[len(root) + 1:] if root and member.name.startswith(root + "/") else member.name
            if relative:
                member.name = relative
                archive.extract(member, destination)
    print(f"installed Stockfish in {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

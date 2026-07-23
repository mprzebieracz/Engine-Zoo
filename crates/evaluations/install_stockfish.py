#!/usr/bin/env python3
"""Install the bundled Stockfish release without shell pipelines."""

from __future__ import annotations

import argparse
import io
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

URL = "https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64.tar"


def safe_relative_path(member: tarfile.TarInfo, root: str) -> Path | None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
        raise ValueError(f"unsafe archive member: {member.name!r}")
    if not (member.isdir() or member.isreg()):
        raise ValueError(f"unsupported archive member type: {member.name!r}")
    parts = path.parts[1:] if root and path.parts[:1] == (root,) else path.parts
    return Path(*parts) if parts else None


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
        roots = {PurePosixPath(member.name).parts[0] for member in members if member.name}
        root = roots.pop() if len(roots) == 1 else ""
        for member in members:
            relative = safe_relative_path(member, root)
            if relative is None:
                continue
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"could not read archive member: {member.name!r}")
            with source, target.open("wb") as output:
                output.write(source.read())
    print(f"installed Stockfish in {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

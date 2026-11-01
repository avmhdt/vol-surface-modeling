"""Fail if git tracks raw data: anything under data/, any .csv.gz, or an oversized fixture."""

import subprocess
import sys
from pathlib import Path

MAX_FIXTURE_BYTES = 500_000


def violations(paths: list[str], root: Path) -> list[str]:
    """Return one message per path that breaks the data rules."""
    found = []
    for name in paths:
        if name.startswith("data/") or "/data/" in name:
            found.append(f"{name}: tracked path under data/")
        if name.endswith(".csv.gz"):
            found.append(f"{name}: .csv.gz files are never committed")
        path = root / name
        if (
            name.startswith("tests/fixtures/")
            and path.exists()
            and path.stat().st_size > MAX_FIXTURE_BYTES
        ):
            found.append(f"{name}: fixture over {MAX_FIXTURE_BYTES} bytes")
    return found


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    paths = (
        argv
        or subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    )
    found = violations(paths, root)
    for message in found:
        print(message, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

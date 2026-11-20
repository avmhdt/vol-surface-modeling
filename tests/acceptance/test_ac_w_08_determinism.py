"""AC-W-08: reproducible and clean.

The data test runs ``vsm reproduce --skip-ingest`` twice and compares every file under
``results/`` except ``experiment_log.csv`` and ``bench/``. It takes about an hour per run on
the full store, so it is marked ``data`` and runs locally only.
"""

import hashlib
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vsm import cli
from vsm.settings import REPO_ROOT

EXCLUDED = ("experiment_log.csv", "bench")
END = "2026-11-20"


def digests(root: Path) -> dict[str, str]:
    """sha256 of every result file AC-W-08 covers."""
    out = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_file() and relative.parts[0] not in EXCLUDED:
            out[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def test_no_tracked_files_under_data() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "data"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert tracked.stdout.strip() == ""


@pytest.mark.data
def test_reproduce_twice_gives_identical_results() -> None:
    runner = CliRunner()
    args = ["reproduce", "--skip-ingest", "--end", END]
    first = runner.invoke(cli.app, args)
    assert first.exit_code == 0, first.output
    before = digests(REPO_ROOT / "results")
    second = runner.invoke(cli.app, args)
    assert second.exit_code == 0, second.output
    after = digests(REPO_ROOT / "results")
    changed = sorted(
        name for name in before.keys() | after.keys() if before.get(name) != after.get(name)
    )
    assert changed == []

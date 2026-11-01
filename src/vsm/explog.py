"""Append-only experiment log: the honest trial count."""

import csv
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vsm.settings import REPO_ROOT

COLUMNS = [
    "run_id",
    "ts_utc",
    "git_sha",
    "command",
    "config_hash",
    "underlying",
    "start",
    "end",
    "params_json",
    "metrics_json",
    "notes",
]


def git_sha() -> str:
    """Short commit hash of the checkout, or ``unknown`` outside a git repo."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return done.stdout.strip()


def append_run(
    results_dir: Path,
    command: str,
    config_hash: str,
    *,
    underlying: str = "",
    start: str = "",
    end: str = "",
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    notes: str = "",
) -> str:
    """Append one run to ``experiment_log.csv`` and return its run id."""
    path = results_dir / "experiment_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    row = {
        "run_id": run_id,
        "ts_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "command": command,
        "config_hash": config_hash,
        "underlying": underlying,
        "start": start,
        "end": end,
        "params_json": json.dumps(params or {}, sort_keys=True),
        "metrics_json": json.dumps(metrics or {}, sort_keys=True),
        "notes": notes,
    }
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return run_id

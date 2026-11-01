import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from vsm import explog
from vsm.io import storage


def load_guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location("guard", Path("scripts/check_repo_guard.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_atomic_write_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "a" / "part.parquet"
    storage.write_parquet_atomic(pl.DataFrame({"x": [1]}), path)
    assert pl.read_parquet(path)["x"].to_list() == [1]
    assert [p.name for p in path.parent.iterdir()] == ["part.parquet"]


def test_failed_write_keeps_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "part.parquet"
    storage.write_parquet_atomic(pl.DataFrame({"x": [1]}), path)

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(storage.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        storage.write_parquet_atomic(pl.DataFrame({"x": [2]}), path)
    assert pl.read_parquet(path)["x"].to_list() == [1]
    assert [p.name for p in tmp_path.iterdir()] == ["part.parquet"]


def test_upsert_new_rows_win_and_first_seen_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "t.parquet"
    storage.upsert_parquet(
        pl.DataFrame({"k": ["a"], "v": [1], "first": [10]}), path, ["k"], keep_first=("first",)
    )
    merged = storage.upsert_parquet(
        pl.DataFrame({"k": ["a", "b"], "v": [2, 3], "first": [20, 20]}),
        path,
        ["k"],
        keep_first=("first",),
    )
    assert merged.to_dicts() == [
        {"k": "a", "v": 2, "first": 10},
        {"k": "b", "v": 3, "first": 20},
    ]


def test_snapshot_paths_follow_the_partition_layout(tmp_path: Path) -> None:
    path = storage.snapshot_path(tmp_path, "option_quotes", "BTC", "2026-11-01", "id")
    assert path.relative_to(tmp_path).as_posix() == (
        "raw/deribit/option_quotes/venue=deribit/underlying=BTC/date=2026-11-01/part-id.parquet"
    )


def test_experiment_log_appends_one_row_per_run(tmp_path: Path) -> None:
    first = explog.append_run(tmp_path, "build", "sha256:x", underlying="BTC", metrics={"a": 1})
    second = explog.append_run(tmp_path, "check", "sha256:x", notes="again")
    with (tmp_path / "experiment_log.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == explog.COLUMNS
    assert [r["run_id"] for r in rows] == [first, second]
    assert json.loads(rows[0]["metrics_json"]) == {"a": 1}
    assert rows[0]["git_sha"]


def test_git_sha_outside_a_repo_is_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(explog, "REPO_ROOT", tmp_path)
    assert explog.git_sha() == "unknown"


def test_repo_guard_flags_data_and_fixtures(tmp_path: Path) -> None:
    guard = load_guard()
    big = tmp_path / "tests" / "fixtures" / "big.json"
    big.parent.mkdir(parents=True)
    big.write_bytes(b"x" * 500_001)
    found = guard.violations(
        ["data/raw/x.csv", "tests/fixtures/big.json", "a.csv.gz", "src/vsm/cli.py"], tmp_path
    )
    assert len(found) == 3
    assert guard.main(["src/vsm/cli.py"]) == 0

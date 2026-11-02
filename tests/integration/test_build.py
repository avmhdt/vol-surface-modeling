"""``vsm build`` on a synthetic snapshot written in the raw layout."""

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from fixtures.make_fixtures import build_chain, build_futures, tick_table
from vsm import cli
from vsm.config import AppConfig, load_config
from vsm.io import storage
from vsm.pipeline import Snapshot, forward_gap_metrics, run_day, select_snapshot

DAY = date(2026, 11, 2)
LABEL = "20261102T0815Z"
runner = CliRunner()


def write_snapshot(data: Path, label: str = LABEL, *, futures: bool = True) -> None:
    snapshot = f"deribit-BTC-{label}"
    day = f"{label[:4]}-{label[4:6]}-{label[6:8]}"
    chain = build_chain().with_columns(pl.lit(snapshot).alias("snapshot_id"))
    storage.write_parquet_atomic(
        chain, storage.snapshot_path(data, storage.OPTION_QUOTES, "BTC", day, snapshot)
    )
    if futures:
        storage.write_parquet_atomic(
            build_futures().with_columns(pl.lit(snapshot).alias("snapshot_id")),
            storage.snapshot_path(data, storage.FUTURE_QUOTES, "BTC", day, snapshot),
        )
    storage.write_parquet_atomic(tick_table(chain), storage.instruments_path(data, "BTC"))


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    loaded = load_config()
    loaded.paths.results_dir = tmp_path / "results"
    loaded.paths.reports_dir = tmp_path / "reports"
    return loaded


def test_exact_snapshot_is_preferred(tmp_path: Path) -> None:
    write_snapshot(tmp_path)
    write_snapshot(tmp_path, "20261102T0830Z")
    assert select_snapshot(tmp_path, "BTC", DAY, "08:15") == Snapshot(LABEL, False)


def test_nearest_snapshot_within_30_minutes_is_flagged(tmp_path: Path) -> None:
    write_snapshot(tmp_path, "20261102T0840Z")
    assert select_snapshot(tmp_path, "BTC", DAY, "08:15") == Snapshot("20261102T0840Z", True)


def test_no_snapshot_means_no_surface(tmp_path: Path, config: AppConfig) -> None:
    write_snapshot(tmp_path, "20261102T0915Z")
    assert select_snapshot(tmp_path, "BTC", DAY, "08:15") is None
    assert run_day(tmp_path, config, "BTC", DAY) is None


def test_missing_futures_file_is_not_built(tmp_path: Path, config: AppConfig) -> None:
    write_snapshot(tmp_path, futures=False)
    assert run_day(tmp_path, config, "BTC", DAY) is None


def test_run_day_writes_interim_tables_and_report(tmp_path: Path, config: AppConfig) -> None:
    write_snapshot(tmp_path)
    report = run_day(tmp_path, config, "BTC", DAY)
    assert report is not None
    assert report["snapshot"] == LABEL
    assert report["snapshot_flagged"] is False
    assert report["kept"] > 250
    interim = tmp_path / "interim"
    quotes = pl.read_parquet(
        interim / "clean_quotes" / "underlying=BTC" / "date=2026-11-02" / f"part-{LABEL}.parquet"
    )
    forwards = pl.read_parquet(
        interim / "forwards" / "underlying=BTC" / "date=2026-11-02" / f"part-{LABEL}.parquet"
    )
    assert quotes.height == report["rows"]
    assert dict(forwards.schema) == storage.FORWARDS_SCHEMA
    saved = json.loads((config.results_dir() / "dq" / "BTC_2026-11-02.json").read_text())
    assert saved["mark_iv_check"] == report["mark_iv_check"]


def test_report_is_byte_identical_on_a_rerun(tmp_path: Path, config: AppConfig) -> None:
    write_snapshot(tmp_path)
    run_day(tmp_path, config, "BTC", DAY)
    path = config.results_dir() / "dq" / "BTC_2026-11-02.json"
    first = path.read_bytes()
    run_day(tmp_path, config, "BTC", DAY)
    assert path.read_bytes() == first


def test_forward_gap_metrics_cover_listed_expiries(tmp_path: Path, config: AppConfig) -> None:
    write_snapshot(tmp_path)
    run_day(tmp_path, config, "BTC", DAY)
    metrics = forward_gap_metrics(tmp_path, config, "BTC")
    assert metrics["n"] == 4
    assert metrics["median_abs_bps"] < 15.0
    assert metrics["p95_abs_bps"] >= metrics["median_abs_bps"]


def test_forward_gap_metrics_with_nothing_built(tmp_path: Path, config: AppConfig) -> None:
    assert forward_gap_metrics(tmp_path, config, "BTC") == {
        "underlying": "BTC",
        "days": 0,
        "n": 0,
    }


def test_cli_build_logs_the_run(
    tmp_path: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    write_snapshot(data)
    monkeypatch.setenv("VSM_DATA_DIR", str(data))
    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    args = ["build", "--underlying", "BTC", "--start", "2026-11-01", "--end", "2026-11-02"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "2026-11-01: no snapshot" in result.output
    log = (config.results_dir() / "experiment_log.csv").read_text(encoding="utf-8")
    assert log.count("\n") == 2
    assert ",build," in log


def test_cli_build_fails_with_nothing_to_build(
    tmp_path: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VSM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    args = ["build", "--underlying", "BTC", "--start", "2026-11-01", "--end", "2026-11-01"]
    assert runner.invoke(cli.app, args).exit_code == 1


def test_cli_build_refuses_unknown_steps() -> None:
    args = ["build", "--underlying", "BTC", "--start", "2026-11-01", "--end", "2026-11-01"]
    result = runner.invoke(cli.app, [*args, "--steps", "clean,bogus"])
    assert result.exit_code == 2
    assert "bogus" in result.output

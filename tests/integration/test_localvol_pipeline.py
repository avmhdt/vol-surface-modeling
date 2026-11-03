"""Local vol grids and barrier tables on a synthetic stored day."""

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from test_build import DAY, write_snapshot
from typer.testing import CliRunner

from vsm import cli, pipeline
from vsm.config import AppConfig, load_config
from vsm.localvol import runner
from vsm.models.runner import calibrate_heston_day

cli_runner = CliRunner()


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    loaded = load_config()
    loaded.paths.results_dir = tmp_path / "results"
    loaded.paths.reports_dir = tmp_path / "reports"
    loaded.mc.paths = 2_000
    loaded.mc.steps_per_day = 2
    return loaded


@pytest.fixture
def stored(tmp_path: Path, config: AppConfig) -> Path:
    data = tmp_path / "data"
    write_snapshot(data)
    assert pipeline.run_day(data, config, "BTC", DAY) is not None
    return data


def test_grid_is_stored_and_checked(stored: Path, config: AppConfig) -> None:
    entry = runner.run_localvol_day(stored, config, "BTC", DAY)
    assert entry is not None
    assert entry["finite"]
    assert entry["positive"]
    assert len(entry["repricing"]["points"]) == 15
    grid = pl.read_parquet(
        stored / "processed" / "localvol" / "underlying=BTC" / f"date={DAY}" / "grid.parquet"
    )
    assert grid.height == 121 * 8
    saved = json.loads((config.results_dir() / "metrics" / "localvol_BTC.json").read_text())
    assert saved[DAY.isoformat()]["snapshot_id"].startswith("deribit-BTC-")


def test_barrier_needs_a_heston_fit(stored: Path, config: AppConfig) -> None:
    assert runner.run_barrier_day(stored, config, "BTC", DAY) is None


def test_barrier_table_prices_every_product_and_replaces_its_day(
    stored: Path, config: AppConfig
) -> None:
    calibrate_heston_day(stored, config, "BTC", DAY)
    rows = runner.run_barrier_day(stored, config, "BTC", DAY)
    assert rows is not None
    assert {row["product"] for row in rows} == {"down_and_out_call", "up_and_out_call"}
    assert all(row["local_vol_price"] > 0.0 and row["heston_price"] > 0.0 for row in rows)
    runner.run_barrier_day(stored, config, "BTC", DAY, paths=1_000, seed=5)
    path = config.results_dir() / "tables" / "barrier_lv_vs_heston_BTC.csv"
    table = pl.read_csv(path)
    assert table.height == 2
    assert set(table["paths"].to_list()) == {1_000}


def test_days_without_a_surface_are_skipped(stored: Path, config: AppConfig) -> None:
    other = date(2026, 1, 1)
    assert runner.run_localvol_day(stored, config, "BTC", other) is None
    assert runner.run_barrier_day(stored, config, "BTC", other) is None


def test_a_surface_that_fails_the_gate_is_refused(
    stored: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = runner.check_surface(
        runner.gated_surface(stored, config, "BTC", DAY), config.arbitrage
    )
    monkeypatch.setattr(runner, "check_surface", lambda *_: replace(report, passed=False))
    assert runner.gated_surface(stored, config, "BTC", DAY) is None
    assert runner.run_localvol_day(stored, config, "BTC", DAY) is None


def test_commands_report_and_fail_cleanly(
    stored: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VSM_DATA_DIR", str(stored))
    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    args = ["--underlying", "BTC", "--date", DAY.isoformat()]
    assert "worst repricing" in cli_runner.invoke(cli.app, ["localvol", *args]).output
    assert cli_runner.invoke(cli.app, ["barrier", *args]).exit_code == 1
    calibrate_heston_day(stored, config, "BTC", DAY)
    result = cli_runner.invoke(cli.app, ["barrier", *args, "--paths", "1000"])
    assert result.exit_code == 0, result.output
    assert "Heston" in result.output
    missing = ["--underlying", "BTC", "--date", "2026-01-01"]
    assert cli_runner.invoke(cli.app, ["localvol", *missing]).exit_code == 1

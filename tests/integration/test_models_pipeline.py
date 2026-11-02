"""Calibration, comparison, export blocks and benchmarks on a synthetic stored day."""

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from test_build import DAY, LABEL, write_snapshot
from typer.testing import CliRunner

from vsm import bench, cli, pipeline
from vsm.config import AppConfig, load_config
from vsm.export.reader import read_surface_bundle
from vsm.models import runner
from vsm.surface import store

cli_runner = CliRunner()


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    loaded = load_config()
    loaded.paths.results_dir = tmp_path / "results"
    loaded.paths.reports_dir = tmp_path / "reports"
    return loaded


@pytest.fixture
def stored(tmp_path: Path, config: AppConfig) -> Path:
    data = tmp_path / "data"
    write_snapshot(data)
    assert pipeline.run_day(data, config, "BTC", DAY) is not None
    return data


def test_heston_day_stores_both_variants(stored: Path, config: AppConfig) -> None:
    result = runner.calibrate_heston_day(stored, config, "BTC", DAY)
    assert result is not None
    assert result["n_quotes"] > 0
    rows = pl.read_parquet(store.heston_path(stored, "BTC"))
    assert set(rows["variant"].to_list()) == {"all", "min_7d"}


def test_sabr_day_stores_both_betas(stored: Path, config: AppConfig) -> None:
    result = runner.calibrate_sabr_day(stored, config, "BTC", DAY)
    assert result is not None
    assert result["converged"]
    rows = pl.read_parquet(store.sabr_path(stored, "BTC"))
    assert set(rows["beta"].to_list()) == {0.5, 1.0}


def test_a_day_without_a_surface_is_skipped(stored: Path, config: AppConfig) -> None:
    other = date(2026, 1, 1)
    assert runner.calibrate_heston_day(stored, config, "BTC", other) is None
    assert runner.calibrate_sabr_day(stored, config, "BTC", other) is None


def test_export_gains_the_model_blocks(stored: Path, config: AppConfig) -> None:
    bundle_path = stored / "exports" / "BTC" / LABEL / "surface.json"
    assert read_surface_bundle(str(bundle_path)).document["heston"] is None
    runner.calibrate_heston_day(stored, config, "BTC", DAY)
    runner.calibrate_sabr_day(stored, config, "BTC", DAY)
    document = read_surface_bundle(str(bundle_path)).document
    assert document["heston"]["converged"] in (True, False)
    assert {"v0", "kappa", "theta", "xi", "rho"} <= document["heston"].keys()
    assert len(document["sabr"]) >= 5
    assert {entry["beta"] for entry in document["sabr"]} == {1.0}


def test_comparison_table_covers_all_models(stored: Path, config: AppConfig) -> None:
    runner.calibrate_heston_day(stored, config, "BTC", DAY)
    runner.calibrate_sabr_day(stored, config, "BTC", DAY)
    summary = runner.refresh_comparison(stored, config, "BTC")
    assert [row["model"] for row in summary] == ["SVI", "SSVI", "Heston", "SABR"]
    tables = config.results_dir() / "tables"
    table = pl.read_csv(tables / "model_comparison_BTC.csv")
    assert set(table.columns) >= {
        "rmse_iv_bp",
        "within_spread_frac",
        "max_error_bp",
        "calibration_seconds",
        "median_daily_param_change",
    }
    assert (tables / "sabr_beta_sensitivity_BTC.csv").exists()
    assert pl.read_csv(tables / "heston_variants_BTC.csv").height == 2
    figure = config.reports_dir() / "figures" / "model_fit_comparison_BTC.png"
    assert figure.stat().st_size > 1_000


def test_comparison_tables_are_identical_on_a_rerun(stored: Path, config: AppConfig) -> None:
    runner.calibrate_sabr_day(stored, config, "BTC", DAY)
    runner.refresh_comparison(stored, config, "BTC")
    path = config.results_dir() / "tables" / "model_comparison_BTC.csv"
    first = path.read_bytes()
    runner.refresh_comparison(stored, config, "BTC")
    assert path.read_bytes() == first


def test_comparison_with_nothing_stored(tmp_path: Path, config: AppConfig) -> None:
    assert runner.refresh_comparison(tmp_path, config, "BTC") == []


def test_bench_writes_times_and_the_table_reads_them(stored: Path, config: AppConfig) -> None:
    runner.calibrate_sabr_day(stored, config, "BTC", DAY)
    report = bench.run_bench(stored, config)
    entry = report["heston"]["BTC"]
    assert entry["days"] == 1
    assert entry["enough_days"] is False
    assert entry["median_seconds"] > 0.0
    saved = json.loads((config.results_dir() / "bench" / "heston_calib.json").read_text())
    assert saved["jacobian"] == "analytic"
    assert saved["underlyings"]["BTC"]["details"][0]["n_quotes"] > 0
    summary = runner.refresh_comparison(stored, config, "BTC")
    assert all(row["calibration_seconds"] is not None for row in summary)


def test_quick_bench_skips_stored_days(tmp_path: Path, config: AppConfig) -> None:
    report = bench.run_bench(tmp_path, config, quick=True)
    assert set(report) == {"iv"}
    assert report["iv"]["failed"] == 0
    assert (config.results_dir() / "bench" / "hardware.json").exists()


def test_cli_calibrate_logs_the_run(
    stored: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VSM_DATA_DIR", str(stored))
    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    span = ["--underlying", "BTC", "--start", "2026-11-02", "--end", "2026-11-02"]
    result = cli_runner.invoke(cli.app, ["calibrate", "--model", "sabr", *span])
    assert result.exit_code == 0, result.output
    log = (config.results_dir() / "experiment_log.csv").read_text(encoding="utf-8")
    assert "calibrate sabr" in log


def test_cli_calibrate_refuses_an_unknown_model() -> None:
    span = ["--underlying", "BTC", "--start", "2026-11-02", "--end", "2026-11-02"]
    assert cli_runner.invoke(cli.app, ["calibrate", "--model", "bates", *span]).exit_code == 2


def test_cli_calibrate_fails_with_nothing_stored(
    tmp_path: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VSM_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    span = ["--underlying", "BTC", "--start", "2026-11-02", "--end", "2026-11-02"]
    assert cli_runner.invoke(cli.app, ["calibrate", "--model", "sabr", *span]).exit_code == 1


def test_cli_bench_quick(
    tmp_path: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VSM_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    result = cli_runner.invoke(cli.app, ["bench", "--quick"])
    assert result.exit_code == 0, result.output
    assert "vols_per_second" in result.output

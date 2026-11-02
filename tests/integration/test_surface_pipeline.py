"""``vsm build`` through fit, gate and export on a synthetic snapshot, and the stored history."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_build import DAY, LABEL, write_snapshot
from typer.testing import CliRunner

from fixtures.make_fixtures import SSVI as TRUE_SSVI
from vsm import cli, pipeline
from vsm.config import AppConfig, load_config
from vsm.export.reader import read_surface_bundle
from vsm.io import storage
from vsm.surface import store
from vsm.surface.arbitrage import check_surface
from vsm.surface.fit import fit_slice_chain, slice_data
from vsm.surface.ssvi import fit_ssvi
from vsm.surface.svi import RawSVI

runner = CliRunner()


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    loaded = load_config()
    loaded.paths.results_dir = tmp_path / "results"
    loaded.paths.reports_dir = tmp_path / "reports"
    return loaded


@pytest.fixture
def built(tmp_path: Path, config: AppConfig) -> Path:
    data = tmp_path / "data"
    write_snapshot(data)
    report = pipeline.run_day(data, config, "BTC", DAY)
    assert report is not None
    assert report["surface"]["passed"]
    return data


def test_build_recovers_the_true_surface(built: Path) -> None:
    params = pl.read_parquet(store.params_path(built, "BTC"))
    assert params["ssvi_rho"][0] == pytest.approx(TRUE_SSVI["rho"], abs=0.15)
    assert params["ssvi_gamma"][0] == pytest.approx(TRUE_SSVI["gamma"], abs=0.15)
    assert dict(params.schema) == storage.SURFACE_PARAMS_SCHEMA


def test_export_validates_and_only_passing_rows_reach_consumers(built: Path) -> None:
    bundle = read_surface_bundle(str(built / "exports" / "BTC" / LABEL / "surface.json"))
    assert bundle.document["arbitrage"]["passed"] is True
    exported = pl.read_parquet(store.export_params_path(built, "BTC"))
    assert exported["arb_free"].all()


def test_stored_surface_rebuilds_exactly(built: Path, config: AppConfig) -> None:
    rows = pl.read_parquet(store.params_path(built, "BTC"))
    surface = store.load_surface(rows)
    report = check_surface(surface, config.arbitrage)
    assert report.passed
    again = store.surface_rows(store.with_report(surface, report), "s", "c")
    assert again["svi_a"].to_list() == rows["svi_a"].to_list()
    assert again["ssvi_theta"].to_list() == rows["ssvi_theta"].to_list()


def test_rebuild_is_a_row_update_not_a_duplicate(built: Path, config: AppConfig) -> None:
    first = pl.read_parquet(store.params_path(built, "BTC")).height
    pipeline.run_day(built, config, "BTC", DAY)
    assert pl.read_parquet(store.params_path(built, "BTC")).height == first


def test_clean_only_steps_skip_the_surface(tmp_path: Path, config: AppConfig) -> None:
    write_snapshot(tmp_path)
    report = pipeline.run_day(tmp_path, config, "BTC", DAY, steps=pipeline.CLEAN_STEPS)
    assert report is not None
    assert "surface" not in report
    assert not store.params_path(tmp_path, "BTC").exists()


def test_no_export_step_writes_no_bundle(tmp_path: Path, config: AppConfig) -> None:
    write_snapshot(tmp_path)
    steps = (*pipeline.CLEAN_STEPS, "fit", "check")
    report = pipeline.run_day(tmp_path, config, "BTC", DAY, steps=steps)
    assert report is not None
    assert report["surface"]["exported"] is False
    assert not (tmp_path / "exports" / "BTC" / LABEL).exists()


def test_failing_gate_is_recorded_and_not_exported(
    tmp_path: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_snapshot(tmp_path)
    real = pipeline.check_surface

    def failing(surface: object, cfg: object) -> object:
        report = real(surface, cfg)  # type: ignore[arg-type]
        return type(report)(**{**report.__dict__, "passed": False})

    monkeypatch.setattr(pipeline, "check_surface", failing)
    report = pipeline.run_day(tmp_path, config, "BTC", DAY)
    assert report is not None
    assert report["surface"] == {
        "passed": False,
        "slices": 7,
        "sources": report["surface"]["sources"],
        "exported": False,
    }
    assert not (tmp_path / "exports" / "BTC" / LABEL).exists()
    assert not store.export_params_path(tmp_path, "BTC").exists()
    assert store.params_path(tmp_path, "BTC").exists()
    saved = json.loads((config.results_dir() / "arbcheck" / "BTC_2026-11-02.json").read_text())
    assert saved["passed"] is False


def test_empty_day_gives_no_surface(
    tmp_path: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_snapshot(tmp_path)
    monkeypatch.setattr(pipeline, "build_slices", lambda *_: None)
    report = pipeline.run_day(tmp_path, config, "BTC", DAY)
    assert report is not None
    assert report["surface"] == {"passed": False, "slices": 0, "exported": False}


def test_outputs_are_written_and_figures_exist(built: Path, config: AppConfig) -> None:
    outputs = pipeline.refresh_outputs(built, config, "BTC")
    assert outputs["within_spread"]["pooled_svi"] > 0.9
    results = config.results_dir()
    for name in ("fit_within_spread_BTC", "stability_BTC", "spline_baseline_BTC"):
        assert (results / "metrics" / f"{name}.json").exists()
    assert (results / "arbcheck" / "summary_BTC.csv").exists()
    table = pl.read_csv(results / "tables" / "rnd_skew_BTC.csv")
    assert table.height == 7
    assert (table["skewness"] < 0).all()
    assert (table["rr25"] < 0).all()
    figures = config.reports_dir() / "figures"
    assert (figures / "rnd_skew_BTC.png").stat().st_size > 1_000
    assert (figures / "spline_vs_svi_density.png").stat().st_size > 1_000


def test_outputs_are_identical_on_a_rerun(built: Path, config: AppConfig) -> None:
    pipeline.refresh_outputs(built, config, "BTC")
    results = config.results_dir()
    names = ["arbcheck/summary_BTC.csv", "metrics/fit_within_spread_BTC.json"]
    first = [(results / name).read_bytes() for name in names]
    pipeline.refresh_outputs(built, config, "BTC")
    assert [(results / name).read_bytes() for name in names] == first


def test_outputs_with_nothing_built(tmp_path: Path, config: AppConfig) -> None:
    assert set(pipeline.refresh_outputs(tmp_path, config, "BTC")) == {"forward_gap"}


def test_recheck_and_export_day(built: Path, config: AppConfig) -> None:
    report = pipeline.recheck_day(built, config, "BTC", DAY)
    assert report is not None
    assert report.passed
    path = pipeline.export_day(built, config, "BTC", DAY)
    assert path is not None
    assert Path(path).exists()
    other = date(2026, 1, 1)
    assert pipeline.recheck_day(built, config, "BTC", other) is None
    assert pipeline.export_day(built, config, "BTC", other) is None


def test_canonical_snapshot_is_preferred_among_stored(built: Path, config: AppConfig) -> None:
    rows = pl.read_parquet(store.params_path(built, "BTC"))
    later = rows.with_columns(
        pl.lit("deribit-BTC-20261102T0915Z").alias("snapshot_id"),
        (pl.col("as_of_ns") + 3_600_000_000_000).alias("as_of_ns"),
    )
    storage.write_parquet_atomic(pl.concat([rows, later]), store.params_path(built, "BTC"))
    chosen = pipeline._stored_rows(built, config, "BTC", DAY)
    assert chosen is not None
    assert chosen["snapshot_id"][0] == f"deribit-BTC-{LABEL}"


def test_fallback_chain_uses_ssvi_when_refits_are_rejected(
    tmp_path: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_snapshot(tmp_path)
    report = pipeline.run_day(tmp_path, config, "BTC", DAY, steps=pipeline.CLEAN_STEPS)
    assert report is not None
    quotes = pl.read_parquet(
        tmp_path
        / "interim"
        / "clean_quotes"
        / "underlying=BTC"
        / "date=2026-11-02"
        / f"part-{LABEL}.parquet"
    )
    groups = slice_data(quotes, config)
    ssvi = fit_ssvi([data for _, data in groups]).ssvi
    data = groups[0][1]

    from vsm.surface import fit

    monkeypatch.setattr(fit, "_best_fit", lambda *_: None)
    result = fit_slice_chain(data, ssvi, None, None, config)
    assert result is not None
    assert result[1] == "ssvi_fallback"

    high = RawSVI(a=10.0, b=0.1, rho=0.0, m=0.0, sigma=0.1)
    quoted = np.array([-0.1, 0.1])
    assert fit_slice_chain(data, ssvi, high, quoted, config) is None


def test_cli_check_export_and_validate(
    built: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VSM_DATA_DIR", str(built))
    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    span = ["--underlying", "BTC", "--start", "2026-11-01", "--end", "2026-11-02"]
    result = runner.invoke(cli.app, ["check", *span])
    assert result.exit_code == 0, result.output
    assert "2026-11-01: no stored surface" in result.output
    assert "passed=True" in result.output

    result = runner.invoke(cli.app, ["export", "--underlying", "BTC", "--date", "2026-11-02"])
    assert result.exit_code == 0, result.output

    path = built / "exports" / "BTC" / LABEL / "surface.json"
    result = runner.invoke(cli.app, ["export", "validate", str(path)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("valid: BTC")

    bad = built / "bad.json"
    bad.write_text('{"schema": "nope"}', encoding="utf-8")
    assert runner.invoke(cli.app, ["export", "validate", str(bad)]).exit_code == 1
    assert runner.invoke(cli.app, ["export"]).exit_code == 2


def test_cli_check_fails_on_a_failed_gate(
    built: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VSM_DATA_DIR", str(built))
    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    real = pipeline.recheck_day

    def failing(*args: object) -> object:
        report = real(*args)  # type: ignore[arg-type]
        return type(report)(**{**report.__dict__, "passed": False})

    monkeypatch.setattr(cli, "recheck_day", failing)
    span = ["--underlying", "BTC", "--start", "2026-11-02", "--end", "2026-11-02"]
    assert runner.invoke(cli.app, ["check", *span]).exit_code == 1

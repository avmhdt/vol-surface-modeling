"""Temporal regularization and the stability study on two synthetic recorder days."""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_build import DAY, write_snapshot

from vsm import pipeline
from vsm.config import AppConfig, load_config
from vsm.io import storage
from vsm.surface import regularized
from vsm.surface.svi import RawSVI, fit_raw_svi

NEXT = DAY + timedelta(days=1)


DAY_NS = 86_400_000_000_000


def make_config(root: Path) -> AppConfig:
    loaded = load_config()
    loaded.paths.results_dir = root / "results"
    loaded.paths.reports_dir = root / "reports"
    loaded.stability.selection_days = 2
    loaded.stability.lambda_grid = [0.0, 1.0e-2]
    return loaded


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return make_config(tmp_path)


def shift_snapshot(data: Path, label: str) -> None:
    """The fixture chain, received one day later, so the two days are distinct as-ofs."""
    day = f"{label[:4]}-{label[4:6]}-{label[6:8]}"
    snapshot = f"deribit-BTC-{label}"
    for dataset in (storage.OPTION_QUOTES, storage.FUTURE_QUOTES):
        path = storage.snapshot_path(data, dataset, "BTC", day, snapshot)
        frame = pl.read_parquet(path).with_columns(
            pl.col("ts_recv") + DAY_NS, pl.col("ts_exchange") + DAY_NS
        )
        storage.write_parquet_atomic(frame, path)


@pytest.fixture(scope="module")
def stored(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("regularized")
    data = root / "data"
    build_config = make_config(root)
    for day in (DAY, NEXT):
        label = f"{day:%Y%m%d}T0815Z"
        write_snapshot(data, label)
        if day == NEXT:
            shift_snapshot(data, label)
        assert pipeline.run_day(data, build_config, "BTC", day) is not None
    return data


def test_penalty_pulls_a_slice_toward_its_prior() -> None:
    truth = RawSVI(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    prior = RawSVI(a=0.03, b=0.15, rho=-0.1, m=0.05, sigma=0.25)
    k = np.linspace(-0.5, 0.5, 15)
    weights = np.ones(k.size)
    free = fit_raw_svi(k, truth.w(k), weights).svi
    pulled = fit_raw_svi(k, truth.w(k), weights, prior=prior, lam=10.0).svi
    assert abs(free.rho - truth.rho) < 1e-3
    assert abs(pulled.rho - prior.rho) < abs(free.rho - prior.rho)
    same = fit_raw_svi(k, truth.w(k), weights, prior=prior, lam=0.0).svi
    assert same == pytest.approx(free)


def test_holdout_leaves_out_every_other_liquid_strike(stored: Path) -> None:
    quotes = pipeline.clean_quotes_of(stored, "BTC", f"{DAY:%Y%m%d}T0815Z")
    assert quotes is not None
    training, held = regularized.holdout_split(quotes)
    usable = quotes.filter(pl.col("drop_reason").is_null() & pl.col("is_otm") & pl.col("is_liquid"))
    assert 0 < held.height < usable.height
    assert training.filter(pl.col("drop_reason") == regularized.HOLDOUT).height == held.height
    assert training.height == quotes.height


def test_selection_scores_the_grid_and_logs_it(stored: Path, config: AppConfig) -> None:
    days = regularized.recorder_days(stored, config, "BTC")
    assert [d.label for d in days] == [f"{DAY:%Y%m%d}T0815Z", f"{NEXT:%Y%m%d}T0815Z"]
    config.underlyings = ["BTC"]
    result = regularized.select_lambda(stored, config)
    assert result["chosen_lambda"] in config.stability.lambda_grid
    assert result["by_underlying"]["BTC"][0]["holdout_points"] > 0
    saved = json.loads((config.results_dir() / "metrics" / "stability_lambda.json").read_text())
    assert saved["chosen_lambda"] == result["chosen_lambda"]


def test_selection_needs_enough_days(stored: Path, config: AppConfig) -> None:
    config.stability.selection_days = 5
    with pytest.raises(ValueError, match="needs 5"):
        regularized.select_lambda(stored, config)


def test_study_needs_a_frozen_lambda_then_compares_both(stored: Path, config: AppConfig) -> None:
    config.stability.lam = None
    with pytest.raises(ValueError, match="not frozen"):
        regularized.run_stability(stored, config, "BTC")
    config.stability.lam = 1.0e-2
    metrics = regularized.run_stability(stored, config, "BTC")
    assert metrics["ac_h_01"]["recorder_days"] == 2
    assert metrics["ac_h_01"]["enough_days"] is False
    assert metrics["ac_h_01"]["passed"] is False
    assert metrics["regularized_gate_passed"] == 2
    assert set(metrics["jw_median_reduction"]) == {"jw_v", "jw_psi", "jw_p", "jw_c", "jw_vtilde"}
    assert (config.reports_dir() / "figures" / "param_history_BTC.png").exists()

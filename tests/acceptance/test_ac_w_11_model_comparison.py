"""AC-W-11 and AC-H-04, on the real store: the comparison exists and Heston is fast."""

import json
from pathlib import Path

import polars as pl
import pytest

REQUIRED = {
    "rmse_iv_bp",
    "within_spread_frac",
    "max_error_bp",
    "calibration_seconds",
    "median_daily_param_change",
}


@pytest.mark.data
@pytest.mark.parametrize("underlying", ["BTC", "ETH"])
def test_comparison_covers_every_model(underlying: str) -> None:
    table = pl.read_csv(f"results/tables/model_comparison_{underlying}.csv")
    assert set(table.columns) >= REQUIRED
    assert set(table["model"].to_list()) == {"SVI", "SSVI", "Heston", "SABR"}
    assert table["calibration_seconds"].null_count() == 0


@pytest.mark.data
def test_comparison_figure_exists() -> None:
    assert Path("reports/figures/model_fit_comparison_BTC.png").stat().st_size > 1_000


@pytest.mark.data
@pytest.mark.parametrize("underlying", ["BTC", "ETH"])
def test_heston_calibrates_under_a_second_over_twenty_days(underlying: str) -> None:
    report = json.loads(Path("results/bench/heston_calib.json").read_text(encoding="utf-8"))
    entry = report["underlyings"][underlying]
    assert report["jacobian"] == "analytic"
    assert entry["median_seconds"] <= report["target_seconds"]
    if not entry["enough_days"]:
        pytest.xfail(f"AC-H-04 needs {report['min_days']} days, {entry['days']} exist")

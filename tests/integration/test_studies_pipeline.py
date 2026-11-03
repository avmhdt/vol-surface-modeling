"""VRP, straddle and sticky studies on a synthetic stored day and a synthetic market."""

import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_build import DAY, LABEL, write_snapshot
from typer.testing import CliRunner

from vsm import cli, pipeline
from vsm.config import AppConfig, load_config
from vsm.conventions import datetime_to_ns
from vsm.io import storage
from vsm.studies import sticky, straddle, vrp

cli_runner = CliRunner()
HOUR_NS = 3_600_000_000_000
START = datetime(2025, 10, 1, tzinfo=UTC)
END = datetime(2027, 2, 1, tzinfo=UTC)


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    loaded = load_config()
    loaded.paths.results_dir = tmp_path / "results"
    loaded.paths.reports_dir = tmp_path / "reports"
    loaded.straddle.bootstrap.n = 200
    return loaded


def write_market(data: Path, forward: float) -> None:
    """Hourly perpetual, funding and DVOL, and daily delivery prices, for BTC."""
    raw = storage.raw_dir(data)
    ts = np.arange(datetime_to_ns(START), datetime_to_ns(END), HOUR_NS, dtype=np.int64)
    rng = np.random.default_rng(5)
    steps = 0.6 * math.sqrt(1 / 8760) * rng.standard_normal(ts.size)
    close = forward * np.exp(np.cumsum(steps) - np.cumsum(steps)[np.searchsorted(ts, _day_ns())])
    perp = pl.DataFrame({"ts": ts, "close": close})
    (raw / "perp_ohlc" / "symbol=BTC-PERPETUAL").mkdir(parents=True)
    perp.write_parquet(raw / "perp_ohlc" / "symbol=BTC-PERPETUAL" / "part-2026.parquet")
    funding = pl.DataFrame({"ts": ts, "interest_1h": np.full(ts.size, 1e-6)})
    (raw / "funding" / "symbol=BTC-PERPETUAL").mkdir(parents=True)
    funding.write_parquet(raw / "funding" / "symbol=BTC-PERPETUAL" / "part-2026.parquet")
    dvol = pl.DataFrame({"ts": ts, "open": np.full(ts.size, 60.0), "close": 60.0 - 5.0 * steps})
    (raw / "dvol" / "underlying=BTC").mkdir(parents=True)
    dvol.write_parquet(raw / "dvol" / "underlying=BTC" / "part-2026.parquet")
    days = [START.date() + timedelta(days=i) for i in range((END - START).days)]
    hours = [datetime_to_ns(datetime(d.year, d.month, d.day, 7, tzinfo=UTC)) for d in days]
    delivery = pl.DataFrame(
        {
            "date": days,
            "index_name": "btc_usd",
            "delivery_price": close[np.searchsorted(ts, hours)],
        }
    )
    (raw / "delivery" / "index_name=btc_usd").mkdir(parents=True)
    delivery.write_parquet(raw / "delivery" / "index_name=btc_usd" / "delivery.parquet")


def _day_ns() -> int:
    return datetime_to_ns(datetime(DAY.year, DAY.month, DAY.day, 8, 15, tzinfo=UTC))


@pytest.fixture
def stored(tmp_path: Path, config: AppConfig) -> Path:
    data = tmp_path / "data"
    write_snapshot(data)
    assert pipeline.run_day(data, config, "BTC", DAY) is not None
    forward = float(pipeline.read_params(data, "BTC")["forward"][0])
    write_market(data, forward)
    return data


def test_vrp_from_surfaces_and_from_dvol(stored: Path, config: AppConfig) -> None:
    table = vrp.run_vrp(stored, config, "BTC", DAY, DAY, "surfaces")
    assert table.height == 1
    assert 0.0 < table["implied_var"][0] < 4.0
    assert table["realized_var"][0] == pytest.approx(0.36, rel=0.5)
    dvol = vrp.run_vrp(stored, config, "BTC", date(2025, 11, 1), date(2026, 6, 1), "dvol")
    assert dvol.height > 200
    saved = json.loads((config.results_dir() / "metrics" / "vrp_dvol_BTC.json").read_text())
    assert saved["overlap_lags"] == 29
    assert set(saved["by_year"]) == {"2025", "2026"}
    with pytest.raises(ValueError, match="source"):
        vrp.run_vrp(stored, config, "BTC", DAY, DAY, "other")


def test_realized_variance_needs_the_whole_window(stored: Path) -> None:
    closes = vrp.perp_closes(stored, "BTC")
    late = datetime_to_ns(END) - 10 * 24 * HOUR_NS
    assert math.isnan(vrp.realized_variance(closes, late, 30.0))


@pytest.mark.parametrize("hedge", straddle.HEDGES)
def test_straddle_trade_is_priced_and_summarized(
    stored: Path, config: AppConfig, hedge: str
) -> None:
    table = straddle.run_straddles(stored, config, "BTC", "recorder_daily", hedge)
    assert table.height == 1
    row = table.row(0, named=True)
    assert 20.0 <= row["days"] <= 45.0
    assert row["premium_fill"] < row["premium_mid"]
    assert row["net_pnl"] < row["gross_pnl"]
    assert row["option_fees"] > 0.0 and row["hedge_costs"] > 0.0
    name = f"recorder_daily_{hedge}_BTC"
    saved = json.loads((config.results_dir() / "metrics" / f"straddle_{name}.json").read_text())
    assert saved["trades"] == 1
    assert (config.reports_dir() / "figures" / f"straddle_pnl_{name}.png").exists()
    if hedge == "min_variance":
        assert row["smile_slope"] != 0.0


def test_straddle_rejects_unknown_choices(stored: Path, config: AppConfig) -> None:
    with pytest.raises(ValueError, match="hedge"):
        straddle.run_straddles(stored, config, "BTC", "recorder_daily", "other")
    with pytest.raises(ValueError, match="D-02"):
        straddle.run_straddles(stored, config, "BTC", "tardis_daily", "entry_iv")


def test_monthly_source_takes_only_tardis_snapshots(stored: Path, config: AppConfig) -> None:
    table = straddle.run_straddles(stored, config, "BTC", "tardis_monthly", "entry_iv")
    assert table.is_empty()
    saved = json.loads(
        (config.results_dir() / "metrics" / "straddle_tardis_monthly_entry_iv_BTC.json").read_text()
    )
    assert saved["trades"] == 0


def test_sticky_says_when_there_are_too_few_days(stored: Path, config: AppConfig) -> None:
    metrics = sticky.run_sticky(stored, config, "BTC", "2026-11-01", "2026-11-30")
    assert metrics["pairs"] == 0
    assert "not enough" in metrics["conclusion"]


def test_sticky_regression_and_conclusions() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(0.0, 0.03, 200)
    flat = sticky.slope_with_ci(x, rng.normal(0.0, 0.002, 200))
    moving = sticky.slope_with_ci(x, -0.5 * x + rng.normal(0.0, 0.002, 200))
    assert moving["slope"] == pytest.approx(-0.5, abs=0.05)
    assert sticky.conclusion(flat, moving).startswith("sticky strike")
    assert sticky.conclusion(moving, flat).startswith("sticky delta")
    assert sticky.conclusion(flat, flat).startswith("undecided")
    assert sticky.conclusion(moving, moving).startswith("neither")


def test_sticky_pairs_need_consecutive_days(stored: Path, config: AppConfig) -> None:
    surface = sticky.daily_surfaces(stored, config, "BTC", "2026-11-02", "2026-11-02")[0]
    later = type(surface)(**{**surface.__dict__, "as_of_ns": surface.as_of_ns + 86_400_000_000_000})
    gap = type(surface)(
        **{**surface.__dict__, "as_of_ns": surface.as_of_ns + 3 * 86_400_000_000_000}
    )
    pairs = sticky.sticky_pairs([surface, later, gap])
    assert pairs.height == 1
    assert pairs["log_return"][0] == pytest.approx(0.0, abs=1e-12)
    assert pairs["d_iv_fixed_strike"][0] == pytest.approx(0.0, abs=1e-12)


def test_study_commands(stored: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VSM_DATA_DIR", str(stored))
    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    span = ["--start", DAY.isoformat(), "--end", DAY.isoformat()]
    result = cli_runner.invoke(cli.app, ["vrp", "--underlying", "BTC", *span])
    assert "1 windows" in result.output, result.output
    assert (
        cli_runner.invoke(cli.app, ["vrp", "--underlying", "BTC", *span, "--source", "x"]).exit_code
        == 2
    )
    args = ["straddle", "--underlying", "BTC", "--source", "recorder_daily"]
    result = cli_runner.invoke(cli.app, args)
    assert "1 trades" in result.output, result.output
    assert cli_runner.invoke(cli.app, [*args, "--hedge", "x"]).exit_code == 2
    result = cli_runner.invoke(cli.app, ["sticky", "--underlying", "BTC", *span])
    assert "0 daily pairs" in result.output
    assert LABEL in pipeline.read_params(stored, "BTC")["snapshot_id"][0]

"""Variance risk premium: 30-day implied against realized variance (FR-21).

Implied variance is the model-free variance-swap rate of each gated surface,

    sigma^2 T = 2 * integral of OTM(K) / K^2 dK

in forward units, which is 2 * integral of b(k) e^-k dk in log-moneyness for the undiscounted
OTM price b. Each slice gives its own rate; 30 days is reached by linear interpolation in total
variance between the two slices around it, so no rate is extrapolated. DVOL squared is the long
cross-check. Realized variance sums squared hourly log returns of the perpetual over
(t, t + 30 days], annualized by the window's length.
"""

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.integrate import quad

from vsm.config import AppConfig, config_hash
from vsm.conventions import F64
from vsm.explog import append_run
from vsm.io import storage
from vsm.iv.black import black76_price
from vsm.pipeline import read_params
from vsm.studies.stats import summarize
from vsm.surface.arbitrage import check_surface
from vsm.surface.store import load_surface
from vsm.surface.svi import RawSVI
from vsm.surface.types import Surface

DAYS_PER_YEAR = 365.0
HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
K_WIDTH_SD = 12.0
SOURCES = ("surfaces", "dvol")


def slice_variance(svi: RawSVI, t: float) -> float:
    """Model-free total variance sigma^2 T of one slice, in forward units.

    The integrand has a kink at k = 0, where puts give way to calls, so each side is integrated
    on its own.
    """
    atm = max(float(svi.w(np.array(0.0))), 1e-12)
    width = K_WIDTH_SD * math.sqrt(atm) + 0.5

    def integrand(k: float) -> float:
        w = max(float(svi.w(np.array(k))), 1e-16)
        price = black76_price(1.0, math.exp(k), t, math.sqrt(w / t), k >= 0.0)
        return float(price) * math.exp(-k)

    puts, _ = quad(integrand, -width, 0.0, limit=200, epsabs=1e-12, epsrel=1e-10)
    calls, _ = quad(integrand, 0.0, width, limit=200, epsabs=1e-12, epsrel=1e-10)
    return 2.0 * float(puts + calls)


def variance_swap_rate(surface: Surface, t: float) -> float:
    """Annualized variance-swap rate at exactly ``t``, or nan outside the quoted expiries."""
    times = [item.t for item in surface.slices]
    if not times or t < times[0] or t > times[-1]:
        return math.nan
    totals = [slice_variance(item.svi, item.t) for item in surface.slices]
    return float(np.interp(t, times, totals)) / t


def _series(folder: Path, column: str) -> pl.DataFrame:
    files = sorted(folder.glob("part-*.parquet"))
    if not files:
        return pl.DataFrame({"ts": [], column: []}, schema={"ts": pl.Int64, column: pl.Float64})
    frame = pl.concat([pl.read_parquet(path, columns=["ts", column]) for path in files])
    return frame.unique("ts").sort("ts")


def perp_closes(data_dir: Path, underlying: str) -> pl.DataFrame:
    """Hourly perpetual closes; ``ts`` is the bar's open, so a close is known at ts + 1 h."""
    folder = storage.raw_dir(data_dir) / "perp_ohlc" / f"symbol={underlying}-PERPETUAL"
    return _series(folder, "close")


def realized_variance(closes: pl.DataFrame, start_ns: int, days: float) -> float:
    """Annualized realized variance over (start, start + days], or nan if not all observed."""
    end_ns = start_ns + int(days * DAY_NS)
    known = closes.with_columns((pl.col("ts") + HOUR_NS).alias("known"))
    # The first price is the last close known by the start.
    window = known.filter((pl.col("known") > start_ns - HOUR_NS) & (pl.col("known") <= end_ns))
    expected = round(days * 24) + 1
    if window.height < expected:
        return math.nan
    prices = window["close"].to_numpy()
    returns = np.diff(np.log(prices))
    return float(np.sum(returns * returns) / (days / DAYS_PER_YEAR))


def _surface_rows(
    data_dir: Path, config: AppConfig, underlying: str, start: date, end: date
) -> list[tuple[int, float]]:
    """(as-of, implied variance) for each canonical gated surface in the range."""
    params = read_params(data_dir, underlying)
    canonical = config.recorder.canonical_time_utc.replace(":", "")
    rows = []
    t = config.vrp.tenor_days / DAYS_PER_YEAR
    for (snapshot,), frame in params.partition_by("snapshot_id", as_dict=True).items():
        label = str(snapshot).rsplit("-", 1)[1]
        day = datetime.strptime(label[:8], "%Y%m%d").date()
        if not start <= day <= end or not label.endswith(f"T{canonical}Z"):
            continue
        surface = load_surface(frame)
        if not check_surface(surface, config.arbitrage).passed:
            continue
        rows.append((surface.as_of_ns, variance_swap_rate(surface, t)))
    return sorted(rows)


def _dvol_rows(data_dir: Path, underlying: str, start: date, end: date) -> list[tuple[int, float]]:
    """(time, DVOL squared) once a day, from the hourly bar that opens at 08:00 UTC."""
    dvol = _series(storage.raw_dir(data_dir) / "dvol" / f"underlying={underlying}", "open")
    daily = dvol.filter(((pl.col("ts") // HOUR_NS) % 24) == 8)
    rows = []
    for ts, value in daily.iter_rows():
        day = datetime.fromtimestamp(ts / 1e9, UTC).date()
        if start <= day <= end:
            rows.append((int(ts), (float(value) / 100.0) ** 2))
    return rows


def run_vrp(
    data_dir: Path, config: AppConfig, underlying: str, start: date, end: date, source: str
) -> pl.DataFrame:
    """One row per observation date, then summary statistics in ``results/metrics``."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    days = float(config.vrp.tenor_days)
    closes = perp_closes(data_dir, underlying)
    if source == "surfaces":
        implied = _surface_rows(data_dir, config, underlying, start, end)
    else:
        implied = _dvol_rows(data_dir, underlying, start, end)
    records = []
    for as_of_ns, iv2 in implied:
        rv2 = realized_variance(closes, as_of_ns, days)
        records.append(
            {
                "date": datetime.fromtimestamp(as_of_ns / 1e9, UTC).date().isoformat(),
                "as_of_ns": as_of_ns,
                "implied_var": iv2,
                "realized_var": rv2,
                "vrp": iv2 - rv2,
            }
        )
    schema = {
        "date": pl.String,
        "as_of_ns": pl.Int64,
        "implied_var": pl.Float64,
        "realized_var": pl.Float64,
        "vrp": pl.Float64,
    }
    table = pl.DataFrame(records, schema=schema).filter(pl.col("vrp").is_finite())
    tables = config.results_dir() / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    table.write_csv(tables / f"vrp_{source}_{underlying}.csv")
    metrics = vrp_metrics(table, config, spacing_days=_spacing(table))
    metrics.update(
        {
            "underlying": underlying,
            "source": source,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "tenor_days": days,
        }
    )
    path = config.results_dir() / "metrics" / f"vrp_{source}_{underlying}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_run(
        config.results_dir(),
        f"vrp {source}",
        config_hash(config),
        underlying=underlying,
        start=start.isoformat(),
        end=end.isoformat(),
        metrics={key: metrics["all"].get(key) for key in ("n", "mean", "nw_t")},
    )
    return table


def _spacing(table: pl.DataFrame) -> float:
    if table.height < 2:
        return float("inf")
    gaps = np.diff(table["as_of_ns"].to_numpy()) / DAY_NS
    return float(np.median(gaps))


def vrp_metrics(table: pl.DataFrame, config: AppConfig, *, spacing_days: float) -> dict[str, Any]:
    """Whole-sample and by-year summaries; HAC lags cover the windows' overlap."""
    tenor = float(config.vrp.tenor_days)
    overlap = max(math.ceil(tenor / spacing_days) - 1, 0) if spacing_days > 0 else 0
    boot = config.straddle.bootstrap

    def stats(values: F64) -> dict[str, Any]:
        return summarize(
            values, overlap=overlap, draws=boot.n, mean_block=boot.mean_block, seed=boot.seed
        )

    years = table.with_columns(pl.col("date").str.slice(0, 4).alias("year"))
    by_year = {
        str(year): stats(part["vrp"].to_numpy())
        for (year,), part in sorted(years.partition_by("year", as_dict=True).items())
    }
    implied = table["implied_var"].to_numpy()
    realized = table["realized_var"].to_numpy()
    return {
        "all": stats(table["vrp"].to_numpy()),
        "by_year": by_year,
        "implied_vol_mean": float(np.sqrt(implied).mean()) if implied.size else None,
        "realized_vol_mean": float(np.sqrt(realized).mean()) if realized.size else None,
        "overlap_lags": overlap,
    }

"""Sticky strike or sticky delta, from daily surfaces (FR-23, AC-H-07).

For each pair of consecutive daily canonical surfaces, the 30-day IV is read at a fixed strike
(the first day's 30-day forward) and at a fixed log-moneyness, which is fixed delta to first
order. Both changes are regressed on the log return of the 30-day forward. Under sticky strike
the fixed-strike slope is 0 and the fixed-moneyness slope is minus the skew dsigma/dk; under
sticky delta it is the other way round. The skew is the 30-day SSVI slope at the money.
"""

import itertools
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from vsm.config import AppConfig, config_hash
from vsm.conventions import F64
from vsm.explog import append_run
from vsm.localvol.runner import forward_at
from vsm.pipeline import read_params
from vsm.studies.stats import newey_west_se
from vsm.surface.arbitrage import check_surface
from vsm.surface.store import load_surface
from vsm.surface.types import Surface
from vsm.viz import figures

TENOR_DAYS = 30.0
MIN_PAIRS = 10
SKEW_STEP = 1e-4
T_CRITICAL = 1.96


def _day(surface: Surface) -> date:
    return datetime.fromtimestamp(surface.as_of_ns / 1e9, UTC).date()


def _vol(surface: Surface, k: float, t: float) -> float:
    return math.sqrt(float(surface.ssvi.w(np.array(k), np.array(t))) / t)


def daily_surfaces(
    data_dir: Path, config: AppConfig, underlying: str, start: str, end: str
) -> list[Surface]:
    """Gated canonical surfaces in the date range, oldest first."""
    params = read_params(data_dir, underlying)
    canonical = config.recorder.canonical_time_utc.replace(":", "")
    found = []
    for (snapshot,), frame in params.partition_by("snapshot_id", as_dict=True).items():
        label = str(snapshot).rsplit("-", 1)[1]
        day = f"{label[:4]}-{label[4:6]}-{label[6:8]}"
        if not label.endswith(f"T{canonical}Z") or not start <= day <= end:
            continue
        surface = load_surface(frame)
        if check_surface(surface, config.arbitrage).passed:
            found.append(surface)
    return sorted(found, key=lambda item: item.as_of_ns)


def sticky_pairs(surfaces: list[Surface]) -> pl.DataFrame:
    """One row per pair of surfaces exactly one day apart."""
    t = TENOR_DAYS / 365.0
    rows = []
    for before, after in itertools.pairwise(surfaces):
        # As-of is the latest receive time, so consecutive days are never exactly 24 h apart.
        if (_day(after) - _day(before)).days != 1:
            continue
        f0 = forward_at(before, t)
        f1 = forward_at(after, t)
        strike_k = math.log(f0 / f1)
        skew = (_vol(before, SKEW_STEP, t) - _vol(before, -SKEW_STEP, t)) / (2.0 * SKEW_STEP)
        rows.append(
            {
                "date": datetime.fromtimestamp(after.as_of_ns / 1e9, UTC).date().isoformat(),
                "log_return": math.log(f1 / f0),
                "d_iv_fixed_strike": _vol(after, strike_k, t) - _vol(before, 0.0, t),
                "d_iv_fixed_moneyness": _vol(after, 0.0, t) - _vol(before, 0.0, t),
                "skew": skew,
            }
        )
    schema = {
        "date": pl.String,
        "log_return": pl.Float64,
        "d_iv_fixed_strike": pl.Float64,
        "d_iv_fixed_moneyness": pl.Float64,
        "skew": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def slope_with_ci(x: F64, y: F64) -> dict[str, float]:
    """OLS slope of y on x with a Newey-West standard error and a 95% interval."""
    centred = x - x.mean()
    slope = float(centred @ (y - y.mean()) / (centred @ centred))
    residual = y - y.mean() - slope * centred
    # The slope is the mean of n * centred * residual / sum(centred^2), so its HAC error follows.
    score = x.size * centred * residual / float(centred @ centred)
    lags = math.floor(4.0 * (x.size / 100.0) ** (2.0 / 9.0))
    error = newey_west_se(score, lags)
    return {
        "slope": slope,
        "se": error,
        "ci95_low": slope - T_CRITICAL * error,
        "ci95_high": slope + T_CRITICAL * error,
    }


def conclusion(fixed_strike: dict[str, float], fixed_moneyness: dict[str, float]) -> str:
    """Which rule the data is closer to, stated plainly, including when it cannot tell."""
    strike_zero = fixed_strike["ci95_low"] <= 0.0 <= fixed_strike["ci95_high"]
    moneyness_zero = fixed_moneyness["ci95_low"] <= 0.0 <= fixed_moneyness["ci95_high"]
    if strike_zero and not moneyness_zero:
        return "sticky strike: fixed-strike IVs do not move with spot, fixed-moneyness IVs do"
    if moneyness_zero and not strike_zero:
        return "sticky delta: fixed-moneyness IVs do not move with spot, fixed-strike IVs do"
    if strike_zero and moneyness_zero:
        return "undecided: neither slope is distinguishable from zero"
    return "neither: both slopes differ from zero"


def run_sticky(
    data_dir: Path, config: AppConfig, underlying: str, start: str, end: str
) -> dict[str, Any]:
    """Regressions, metrics file and figure; with too few pairs, says so and stops there."""
    pairs = sticky_pairs(daily_surfaces(data_dir, config, underlying, start, end))
    results = config.results_dir()
    metrics: dict[str, Any] = {
        "underlying": underlying,
        "start": start,
        "end": end,
        "pairs": pairs.height,
        "min_pairs": MIN_PAIRS,
        "tenor_days": TENOR_DAYS,
    }
    if pairs.height >= MIN_PAIRS:
        returns = pairs["log_return"].to_numpy()
        strike = slope_with_ci(returns, pairs["d_iv_fixed_strike"].to_numpy())
        moneyness = slope_with_ci(returns, pairs["d_iv_fixed_moneyness"].to_numpy())
        metrics.update(
            {
                "fixed_strike": strike,
                "fixed_moneyness": moneyness,
                "mean_skew_dsigma_dk": float(pairs["skew"].mean()),  # type: ignore[arg-type]
                "conclusion": conclusion(strike, moneyness),
            }
        )
        figures.sticky_regression(
            returns,
            pairs["d_iv_fixed_strike"].to_numpy(),
            pairs["d_iv_fixed_moneyness"].to_numpy(),
            f"{underlying}: 30-day IV changes against spot moves",
            config.reports_dir() / "figures" / f"sticky_regression_{underlying}.png",
        )
    else:
        metrics["conclusion"] = f"not enough consecutive daily surfaces ({pairs.height})"
    hedge_errors = {}
    for hedge in ("entry_iv", "min_variance"):
        for source in ("tardis_monthly", "recorder_daily"):
            path = results / "metrics" / f"straddle_{source}_{hedge}_{underlying}.json"
            if path.exists():
                hedge_errors[f"{source}_{hedge}"] = json.loads(path.read_text()).get(
                    "hedge_error_var"
                )
    metrics["straddle_hedge_error_var"] = hedge_errors
    (results / "tables").mkdir(parents=True, exist_ok=True)
    pairs.write_csv(results / "tables" / f"sticky_pairs_{underlying}.csv")
    path = results / "metrics" / f"sticky_{underlying}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_run(
        results,
        "sticky",
        config_hash(config),
        underlying=underlying,
        start=start,
        end=end,
        metrics={"pairs": pairs.height, "conclusion": metrics["conclusion"]},
    )
    return metrics

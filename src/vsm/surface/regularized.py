"""Temporal regularization of the daily recorder surfaces, and its stability study (FR-14, AC-H-01).

Each day is refitted with the previous day's fit as both start and prior: lambda times the
squared change of the SSVI shape (rho, eta, gamma) and of each slice's scaled raw SVI
parameters is added to the fit error. Lambda is chosen once, on the first
``stability.selection_days`` recorder days, by holdout error: every other liquid strike of each
expiry is left out of the fit and scored afterwards. The choice is written to
``results/metrics/stability_lambda.json`` and then frozen in config (rule 14).

Only recorder days are used, because day-over-day changes mean little between monthly Tardis
snapshots. The stored surfaces stay unregularized; this module never overwrites them.
"""

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

from vsm.config import AppConfig, config_hash
from vsm.explog import append_run, git_sha
from vsm.forwards import ForwardEstimate
from vsm.io import storage
from vsm.io.tardis import snapshot_path
from vsm.pipeline import clean_quotes_of, read_params
from vsm.surface.arbitrage import check_surface
from vsm.surface.fit import build_slices
from vsm.surface.ssvi import SSVI
from vsm.surface.stability import daily_shape, stability_metrics, within_spread_metrics
from vsm.surface.store import load_surface, surface_rows, with_report
from vsm.surface.svi import RawSVI
from vsm.surface.types import Surface
from vsm.viz import figures

HOLDOUT = "holdout"
JW_REDUCTION = 0.5  # AC-H-01
MAX_SPREAD_DROP = 0.02  # AC-H-01, as a share
MAX_P95_D_RHO = 0.10  # AC-H-01
MAX_P95_REL_D_ETA = 0.20  # AC-H-01
MIN_RECORDER_DAYS = 30  # AC-H-01

log = structlog.get_logger("vsm.stability")


@dataclass(frozen=True)
class Day:
    """What one recorder day needs to be refitted."""

    label: str
    stored: Surface
    quotes: pl.DataFrame
    forwards: list[ForwardEstimate]


def _forwards(data_dir: Path, underlying: str, label: str) -> list[ForwardEstimate]:
    day = f"{label[:4]}-{label[4:6]}-{label[6:8]}"
    path = (
        data_dir
        / "interim"
        / "forwards"
        / f"underlying={underlying}"
        / f"date={day}"
        / f"part-{label}.parquet"
    )
    frame = pl.read_parquet(path)
    return [
        ForwardEstimate(
            expiry_ns=row["expiry_ns"],
            t=row["t"],
            forward=row["forward"],
            forward_se=row["forward_se"],
            df=row["df"],
            method=row["method"],
            n_pairs=row["n_pairs"],
            future_px=row["future_px"],
            gap_bps=row["gap_bps"],
            basis_ann=row["basis_ann"],
        )
        for row in frame.iter_rows(named=True)
    ]


def recorder_days(data_dir: Path, config: AppConfig, underlying: str) -> list[Day]:
    """Canonical recorder days with a stored surface that passes the gate, oldest first."""
    params = read_params(data_dir, underlying)
    canonical = config.recorder.canonical_time_utc.replace(":", "")
    days = []
    for (snapshot,), frame in params.partition_by("snapshot_id", as_dict=True).items():
        label = str(snapshot).rsplit("-", 1)[1]
        if not label.endswith(f"T{canonical}Z"):
            continue
        day = datetime.strptime(label[:8], "%Y%m%d").date()
        if snapshot_path(data_dir, underlying, day, label).exists():
            continue
        stored = load_surface(frame)
        quotes = clean_quotes_of(data_dir, underlying, label)
        if quotes is None or not check_surface(stored, config.arbitrage).passed:
            continue
        days.append(Day(label, stored, quotes, _forwards(data_dir, underlying, label)))
    return sorted(days, key=lambda d: d.label)


def holdout_split(quotes: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(training quotes, held-out liquid rows): every other liquid OTM strike of each expiry."""
    usable = pl.col("drop_reason").is_null() & pl.col("is_otm") & pl.col("is_liquid")
    ranked = quotes.with_row_index("_row").with_columns(
        pl.when(usable)
        .then(pl.col("strike").rank("ordinal").over("expiry_ns", usable))
        .otherwise(None)
        .alias("_rank")
    )
    held = (ranked["_rank"].fill_null(0) % 2 == 0) & ranked["_rank"].is_not_null()
    training = ranked.with_columns(
        pl.when(held).then(pl.lit(HOLDOUT)).otherwise(pl.col("drop_reason")).alias("drop_reason")
    ).drop("_row", "_rank")
    return training, quotes.filter(held)


def _surface(day: Day, built: tuple[Any, ...], underlying: str) -> Surface:
    ssvi, slices, dropped, sources = built
    return Surface(
        underlying=underlying,
        venue=storage.VENUE,
        as_of_ns=day.stored.as_of_ns,
        snapshot_id=day.stored.snapshot_id,
        spot_index=day.stored.spot_index,
        ssvi=ssvi,
        slices=slices,
        dropped_expiries=dropped,
        fallback_counts=sources,
    )


def _fit(
    day: Day,
    quotes: pl.DataFrame,
    config: AppConfig,
    prior: Surface | None,
    lam: float,
) -> Surface | None:
    use_prior = prior is not None and lam > 0.0
    prior_ssvi: SSVI | None = prior.ssvi if use_prior and prior is not None else None
    prior_slices: dict[int, RawSVI] = (
        {item.expiry_ns: item.svi for item in prior.slices} if use_prior and prior else {}
    )
    built = build_slices(
        quotes, day.forwards, config, prior_ssvi=prior_ssvi, prior_slices=prior_slices, lam=lam
    )
    if built is None:
        return None
    return _surface(day, built, day.stored.underlying)


def _holdout_error(surface: Surface, held: pl.DataFrame) -> tuple[float, int]:
    """(sum of squared IV errors, count) on held-out rows whose expiry has a slice."""
    total = 0.0
    count = 0
    for item in surface.slices:
        rows = held.filter(pl.col("expiry_ns") == item.expiry_ns)
        if rows.is_empty():
            continue
        k = rows["k"].to_numpy()
        model = np.sqrt(np.maximum(item.svi.w(k), 0.0) / item.t)
        error = model - rows["target_iv"].to_numpy()
        total += float(np.sum(error * error))
        count += int(error.size)
    return total, count


def chain(days: list[Day], config: AppConfig, lam: float) -> list[Surface]:
    """Refit days in order, each with the previous refit as prior; failures keep the prior."""
    out: list[Surface] = []
    prior: Surface | None = None
    for day in days:
        fitted = _fit(day, day.quotes, config, prior, lam)
        if fitted is None:
            log.warning("regularized_fit_failed", label=day.label, lam=lam)
            continue
        out.append(fitted)
        prior = fitted
    return out


def holdout_score(days: list[Day], config: AppConfig, lam: float) -> dict[str, float]:
    """Holdout RMSE in vol points, over every day after the first."""
    total = 0.0
    count = 0
    prior: Surface | None = None
    for index, day in enumerate(days):
        if index:
            training, held = holdout_split(day.quotes)
            trial = _fit(day, training, config, prior, lam)
            if trial is not None:
                error, n = _holdout_error(trial, held)
                total += error
                count += n
        fitted = _fit(day, day.quotes, config, prior, lam)
        if fitted is not None:
            prior = fitted
    rmse = math.sqrt(total / count) if count else math.nan
    return {"lambda": lam, "holdout_rmse_iv_bp": 1e4 * rmse, "holdout_points": count}


def select_lambda(data_dir: Path, config: AppConfig) -> dict[str, Any]:
    """Score the grid on the first selection days of every underlying; the lowest total wins."""
    cfg = config.stability
    scores: dict[str, list[dict[str, float]]] = {}
    used: dict[str, list[str]] = {}
    for underlying in config.underlyings:
        days = recorder_days(data_dir, config, underlying)[: cfg.selection_days]
        if len(days) < cfg.selection_days:
            raise ValueError(
                f"{underlying} has {len(days)} recorder days, needs {cfg.selection_days}"
            )
        used[underlying] = [day.label for day in days]
        scores[underlying] = [holdout_score(days, config, lam) for lam in cfg.lambda_grid]
    totals = {
        lam: sum(entry[i]["holdout_rmse_iv_bp"] for entry in scores.values())
        for i, lam in enumerate(cfg.lambda_grid)
    }
    chosen = min(totals, key=lambda lam: (totals[lam], -lam))
    result = {
        "chosen_lambda": chosen,
        "selection_days": cfg.selection_days,
        "grid": cfg.lambda_grid,
        "total_holdout_rmse_iv_bp": {str(lam): value for lam, value in totals.items()},
        "by_underlying": scores,
        "days_used": used,
    }
    path = config.results_dir() / "metrics" / "stability_lambda.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_run(
        config.results_dir(),
        "stability select",
        config_hash(config),
        params={"grid": cfg.lambda_grid, "selection_days": cfg.selection_days},
        metrics={"chosen_lambda": chosen},
    )
    return result


def _params_of(surfaces: list[Surface], config: AppConfig) -> pl.DataFrame:
    sha = git_sha()
    digest = config_hash(config)
    frames = [
        surface_rows(with_report(s, check_surface(s, config.arbitrage)), sha, digest)
        for s in surfaces
    ]
    return pl.concat(frames) if frames else pl.DataFrame(schema=storage.SURFACE_PARAMS_SCHEMA)


def run_stability(data_dir: Path, config: AppConfig, underlying: str) -> dict[str, Any]:
    """AC-H-01 on the recorder days, at the frozen lambda, against the stored surfaces."""
    lam = config.stability.lam
    if lam is None:
        raise ValueError("stability.lambda is not frozen yet; run the selection first")
    days = recorder_days(data_dir, config, underlying)
    plain = _params_of([day.stored for day in days], config)
    regularized_surfaces = chain(days, config, lam)
    regularized = _params_of(regularized_surfaces, config)
    gate = [check_surface(s, config.arbitrage).passed for s in regularized_surfaces]
    before = stability_metrics(plain)
    after = stability_metrics(regularized)
    spread_before = within_spread_metrics(plain, underlying)
    spread_after = within_spread_metrics(regularized, underlying)
    reductions = {
        name: (1.0 - after["jw_median_abs"][name] / value) if value else None
        for name, value in (before["jw_median_abs"] or {}).items()
    }
    drop = spread_before.get("pooled_svi", math.nan) - spread_after.get("pooled_svi", math.nan)
    checks = {
        "recorder_days": len(days),
        "enough_days": len(days) >= MIN_RECORDER_DAYS,
        "p95_abs_d_rho_ok": (after["p95_abs_d_rho"] or math.inf) <= MAX_P95_D_RHO,
        "p95_rel_d_eta_ok": (after["p95_rel_d_eta"] or math.inf) <= MAX_P95_REL_D_ETA,
        "jw_reduced_by_half": all(r is not None and r >= JW_REDUCTION for r in reductions.values()),
        "spread_drop_ok": drop <= MAX_SPREAD_DROP,
    }
    checks["passed"] = all(checks[k] for k in checks if k != "recorder_days")
    metrics = {
        "underlying": underlying,
        "lambda": lam,
        "days": [day.label for day in days],
        "unregularized": {**before, "within_spread": spread_before},
        "regularized": {**after, "within_spread": spread_after},
        "regularized_gate_passed": int(sum(gate)),
        "jw_median_reduction": reductions,
        "pooled_within_spread_drop": drop,
        "ac_h_01": checks,
    }
    path = config.results_dir() / "metrics" / f"stability_regularized_{underlying}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _history_figure(plain, regularized, config, underlying)
    append_run(
        config.results_dir(),
        "stability",
        config_hash(config),
        underlying=underlying,
        params={"lambda": lam},
        metrics=checks,
    )
    return metrics


def _shape(params: pl.DataFrame) -> pl.DataFrame:
    return daily_shape(params) if params.height else pl.DataFrame()


def _history_figure(
    plain: pl.DataFrame, regularized: pl.DataFrame, config: AppConfig, underlying: str
) -> None:
    before = _shape(plain)
    after = _shape(regularized)
    if before.height < 2:
        return
    figures.param_history(
        before["as_of_ns"].to_numpy(),
        {name: before[name].to_numpy() for name in ("rho", "eta", "theta_30d")},
        after["as_of_ns"].to_numpy(),
        {name: after[name].to_numpy() for name in ("rho", "eta", "theta_30d")},
        f"{underlying}: SSVI parameters by recorder day",
        config.reports_dir() / "figures" / f"param_history_{underlying}.png",
    )

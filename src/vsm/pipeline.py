"""Daily build: snapshot selection, cleaning, surface fit, arbitrage gate and export."""

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

from vsm.clean import CleanResult, clean_snapshot
from vsm.config import AppConfig, config_hash
from vsm.conventions import ns_to_datetime
from vsm.explog import git_sha
from vsm.export.bundle import write_surface_bundle
from vsm.io import recorder, storage
from vsm.surface import spline_baseline
from vsm.surface.arbitrage import check_surface
from vsm.surface.density import moments, risk_reversal_25
from vsm.surface.fit import build_slices
from vsm.surface.stability import stability_metrics, within_spread_metrics
from vsm.surface.store import (
    DROPPED,
    load_surface,
    params_path,
    store_surface,
    with_report,
)
from vsm.surface.svi import RawSVI
from vsm.surface.types import ArbReport, Surface
from vsm.viz import figures

CLEAN_STEPS = ("clean", "forwards", "iv")
ALL_STEPS = (*CLEAN_STEPS, "fit", "check", "export")
REFERENCE_TENOR = 30.0 / 365.0

CANONICAL_TOLERANCE = timedelta(minutes=recorder.CANONICAL_TOLERANCE_MIN)

log = structlog.get_logger("vsm.pipeline")


@dataclass(frozen=True)
class Snapshot:
    """The snapshot used for one as-of date; ``flagged`` means it was off the exact time."""

    label: str
    flagged: bool


def select_snapshot(data_dir: Path, underlying: str, day: date, time_utc: str) -> Snapshot | None:
    """The snapshot at ``time_utc``, else the nearest within 30 minutes, else None (DQ-13)."""
    hour, minute = (int(part) for part in time_utc.split(":"))
    target = datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)
    captured = recorder.captured_labels(data_dir, storage.OPTION_QUOTES, underlying)
    exact = target.strftime(recorder.LABEL_FORMAT)
    if exact in captured:
        return Snapshot(exact, False)
    near = [
        (abs(when - target), label)
        for label in captured
        if abs((when := recorder.label_time(label)) - target) <= CANONICAL_TOLERANCE
    ]
    return Snapshot(min(near)[1], True) if near else None


def _read_snapshot(
    data_dir: Path, dataset: str, underlying: str, label: str
) -> pl.DataFrame | None:
    day = recorder.label_time(label).strftime("%Y-%m-%d")
    snapshot = f"{storage.VENUE}-{underlying}-{label}"
    path = storage.snapshot_path(data_dir, dataset, underlying, day, snapshot)
    return pl.read_parquet(path) if path.exists() else None


def _tick_sizes(data_dir: Path, underlying: str) -> pl.DataFrame:
    path = storage.instruments_path(data_dir, underlying)
    if not path.exists():
        return pl.DataFrame(schema={"symbol": pl.String, "tick_size": pl.Float64})
    return pl.read_parquet(path, columns=["symbol", "tick_size"])


def forwards_frame(result: CleanResult, underlying: str, spot_index: float) -> pl.DataFrame:
    """``forwards`` table of one snapshot."""
    as_of_ns = int(result.quotes["as_of_ns"][0])
    rows = [
        {
            **asdict(estimate),
            "as_of_ns": as_of_ns,
            "underlying": underlying,
            "spot_index": spot_index,
        }
        for estimate in result.forwards
    ]
    return pl.DataFrame(rows, schema=storage.FORWARDS_SCHEMA).select(list(storage.FORWARDS_SCHEMA))


def run_day(
    data_dir: Path,
    config: AppConfig,
    underlying: str,
    day: date,
    *,
    at_utc: str | None = None,
    steps: tuple[str, ...] = ALL_STEPS,
) -> dict[str, Any] | None:
    """Build one as-of date. Returns its DQ report, or None if no snapshot exists (DQ-13)."""
    time_utc = at_utc or config.recorder.canonical_time_utc
    snapshot = select_snapshot(data_dir, underlying, day, time_utc)
    if snapshot is None:
        log.warning("no_snapshot", underlying=underlying, day=day.isoformat(), time_utc=time_utc)
        return None
    options = _read_snapshot(data_dir, storage.OPTION_QUOTES, underlying, snapshot.label)
    futures = _read_snapshot(data_dir, storage.FUTURE_QUOTES, underlying, snapshot.label)
    if options is None or futures is None:
        log.warning("incomplete_snapshot", underlying=underlying, label=snapshot.label)
        return None

    as_of_ns = max(int(options["ts_recv"].max()), int(futures["ts_recv"].max()))  # type: ignore[arg-type]
    spot_index = float(futures["index_px"][0])
    result = clean_snapshot(
        options, futures, spot_index, as_of_ns, _tick_sizes(data_dir, underlying), config
    )

    stamp = day.isoformat()
    part = f"part-{snapshot.label}.parquet"
    interim = data_dir / "interim"
    storage.write_parquet_atomic(
        result.quotes,
        interim / "clean_quotes" / f"underlying={underlying}" / f"date={stamp}" / part,
    )
    storage.write_parquet_atomic(
        forwards_frame(result, underlying, spot_index),
        interim / "forwards" / f"underlying={underlying}" / f"date={stamp}" / part,
    )
    report = {
        "underlying": underlying,
        "date": stamp,
        "snapshot": snapshot.label,
        "snapshot_flagged": snapshot.flagged,
        **result.report,
    }
    out = config.results_dir() / "dq" / f"{underlying}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if "fit" in steps:
        report["surface"] = _fit_day(
            data_dir, config, underlying, stamp, snapshot.label, result, spot_index, steps
        )
    return report


def _finite(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else value


def _arbcheck(surface: Surface, stamp: str, label: str, spline: dict[str, Any]) -> dict[str, Any]:
    report = surface.report
    assert report is not None
    return {
        "underlying": surface.underlying,
        "date": stamp,
        "snapshot": label,
        "passed": report.passed,
        "min_g": _finite(report.min_g),
        "min_calendar_dw": _finite(report.min_calendar_dw),
        "density_mass_max_err": _finite(report.density_mass_max_err),
        "density_mean_max_err": _finite(report.density_mean_max_err),
        "sources": dict(sorted(surface.fallback_counts.items())),
        "dropped": len(surface.dropped_expiries),
        "ssvi": {
            "rho": surface.ssvi.rho,
            "eta": surface.ssvi.eta,
            "gamma": surface.ssvi.gamma,
        },
        "slices": [
            {
                "expiry": ns_to_datetime(item.expiry_ns).strftime("%Y-%m-%dT%H:%MZ"),
                "source": item.source,
                "passed": check.passed,
                "min_g": _finite(check.min_g),
                "min_calendar_dw": _finite(check.min_calendar_dw),
                "density_mass_err": _finite(check.density_mass_err),
                "density_mean_err": _finite(check.density_mean_err),
                "price_ok": check.price_ok,
            }
            for item, check in zip(surface.slices, report.slices, strict=True)
        ],
        "spline": spline,
    }


def _write_arbcheck(config: AppConfig, content: dict[str, Any]) -> None:
    out = config.results_dir() / "arbcheck" / f"{content['underlying']}_{content['date']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fit_day(
    data_dir: Path,
    config: AppConfig,
    underlying: str,
    stamp: str,
    label: str,
    result: CleanResult,
    spot_index: float,
    steps: tuple[str, ...],
) -> dict[str, Any]:
    """Fit, gate, store and export one day. A failed gate is recorded and never exported."""
    built = build_slices(result.quotes, result.forwards, config)
    if built is None:
        log.warning("no_surface", underlying=underlying, date=stamp)
        return {"passed": False, "slices": 0, "exported": False}
    ssvi, slices, dropped, sources = built
    surface = Surface(
        underlying=underlying,
        venue=storage.VENUE,
        as_of_ns=int(result.quotes["as_of_ns"][0]),
        snapshot_id=f"{storage.VENUE}-{underlying}-{label}",
        spot_index=spot_index,
        ssvi=ssvi,
        slices=slices,
        dropped_expiries=dropped,
        fallback_counts=sources,
    )
    gated = with_report(surface, check_surface(surface, config.arbitrage))
    report = gated.report
    assert report is not None
    sha = git_sha()
    digest = config_hash(config)
    store_surface(data_dir, gated, sha, digest)
    _write_arbcheck(config, _arbcheck(gated, stamp, label, spline_baseline.evaluate(result.quotes)))
    exported = report.passed and "export" in steps
    if exported:
        write_surface_bundle(
            gated,
            str(data_dir / "exports"),
            cfg=config.arbitrage,
            coin_rate=config.conventions.coin_rate,
            git_sha=sha,
            config_hash=digest,
        )
    if not report.passed:
        log.warning("gate_failed", underlying=underlying, date=stamp)
    return {
        "passed": report.passed,
        "slices": len(slices),
        "sources": sources,
        "exported": exported,
    }


def read_params(data_dir: Path, underlying: str) -> pl.DataFrame:
    """Stored surface history of one underlying, empty if none."""
    path = params_path(data_dir, underlying)
    if not path.exists():
        return pl.DataFrame(schema=storage.SURFACE_PARAMS_SCHEMA)
    return pl.read_parquet(path)


def _stored_rows(
    data_dir: Path, config: AppConfig, underlying: str, day: date
) -> pl.DataFrame | None:
    """Stored rows of one date: the canonical snapshot if there, else the latest."""
    params = read_params(data_dir, underlying)
    stamp = day.strftime("%Y%m%d")
    rows = params.filter(pl.col("snapshot_id").str.contains(f"-{stamp}T"))
    if rows.is_empty():
        return None
    canonical = config.recorder.canonical_time_utc.replace(":", "")
    wanted = f"{storage.VENUE}-{underlying}-{stamp}T{canonical}Z"
    snapshots = rows["snapshot_id"].unique().sort().to_list()
    chosen = wanted if wanted in snapshots else snapshots[-1]
    return rows.filter(pl.col("snapshot_id") == chosen)


def recheck_day(data_dir: Path, config: AppConfig, underlying: str, day: date) -> ArbReport | None:
    """Re-run the gate on a stored surface, without refitting."""
    rows = _stored_rows(data_dir, config, underlying, day)
    if rows is None:
        return None
    return check_surface(load_surface(rows), config.arbitrage)


def export_day(data_dir: Path, config: AppConfig, underlying: str, day: date) -> str | None:
    """Re-write a stored surface's bundle. Raises if it does not pass the gate."""
    rows = _stored_rows(data_dir, config, underlying, day)
    if rows is None:
        return None
    surface = load_surface(rows)
    gated = with_report(surface, check_surface(surface, config.arbitrage))
    return write_surface_bundle(
        gated,
        str(data_dir / "exports"),
        cfg=config.arbitrage,
        coin_rate=config.conventions.coin_rate,
        git_sha=str(rows["git_sha"][0]),
        config_hash=str(rows["config_hash"][0]),
    )


def _arbcheck_files(config: AppConfig, underlying: str) -> list[dict[str, Any]]:
    root = config.results_dir() / "arbcheck"
    files = sorted(root.glob(f"{underlying}_*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def _arbcheck_summary(checks: list[dict[str, Any]], config: AppConfig, underlying: str) -> None:
    """One row per day: slices needing each fallback, then violations left after them."""
    rows = []
    for check in checks:
        slices = check["slices"]
        sources = check["sources"]
        rows.append(
            {
                "date": check["date"],
                "snapshot": check["snapshot"],
                "passed": check["passed"],
                "slices": len(slices),
                "fit_svi": sources.get("svi", 0),
                "fallback_constrained": sources.get("svi_constrained", 0),
                "fallback_ssvi": sources.get("ssvi_fallback", 0),
                "dropped": check["dropped"],
                "failed_slices_after_fallback": sum(not item["passed"] for item in slices),
                "min_g": check["min_g"],
                "min_calendar_dw": check["min_calendar_dw"],
                "density_mass_max_err": check["density_mass_max_err"],
            }
        )
    out = config.results_dir() / "arbcheck" / f"summary_{underlying}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out)


def _write_metric(config: AppConfig, name: str, metrics: dict[str, Any]) -> None:
    out = config.results_dir() / "metrics" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _spline_metrics(checks: list[dict[str, Any]], underlying: str) -> dict[str, Any]:
    """AC-H-06: how often a spline surface breaks the checks exported surfaces pass."""
    days = len(checks)
    violated = sum(
        c["spline"]["butterfly_violations"] + c["spline"]["calendar_violations"] > 0 for c in checks
    )
    return {
        "underlying": underlying,
        "days": days,
        "days_with_spline_violation": violated,
        "share_of_days": violated / days if days else None,
        "exported_days_with_violation": sum(bool(c["passed"]) and c["min_g"] < 0.0 for c in checks),
        "butterfly_violations": sum(c["spline"]["butterfly_violations"] for c in checks),
        "calendar_violations": sum(c["spline"]["calendar_violations"] for c in checks),
    }


def _rnd_table(params: pl.DataFrame, config: AppConfig, underlying: str) -> pl.DataFrame:
    """Density moments against 25-delta risk reversal and SVI-JW skew, per slice (FR-28)."""
    rows = []
    usable = params.filter(pl.col("slice_source") != DROPPED).sort("as_of_ns", "expiry_ns")
    for row in usable.iter_rows(named=True):
        svi = RawSVI(row["svi_a"], row["svi_b"], row["svi_rho"], row["svi_m"], row["svi_sigma"])
        stats = moments(svi)
        rows.append(
            {
                "snapshot_id": row["snapshot_id"],
                "t_days": row["t"] * 365.0,
                "variance": stats["variance"],
                "skewness": stats["skewness"],
                "kurtosis": stats["kurtosis"],
                "rr25": risk_reversal_25(svi, row["t"]),
                "jw_psi": row["jw_psi"],
            }
        )
    table = pl.DataFrame(rows)
    out = config.results_dir() / "tables" / f"rnd_skew_{underlying}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out)
    return table


def _latest_clean(data_dir: Path, underlying: str, label: str) -> pl.DataFrame | None:
    day = recorder.label_time(label).strftime("%Y-%m-%d")
    path = (
        data_dir
        / "interim"
        / "clean_quotes"
        / f"underlying={underlying}"
        / f"date={day}"
        / f"part-{label}.parquet"
    )
    return pl.read_parquet(path) if path.exists() else None


def _draft_figures(
    data_dir: Path, config: AppConfig, underlying: str, params: pl.DataFrame, table: pl.DataFrame
) -> None:
    """AC-H-06 drafts from the latest stored day, at the slice nearest 30 days."""
    reports = config.reports_dir() / "figures"
    figures.rnd_skew(
        table["rr25"].to_numpy(),
        table["skewness"].to_numpy(),
        table["t_days"].to_numpy(),
        f"{underlying}: density skewness and the 25-delta risk reversal",
        reports / f"rnd_skew_{underlying}.png",
    )
    if underlying != config.underlyings[0]:
        return  # the spec has one spline figure, for the first underlying
    latest = params.filter(pl.col("as_of_ns") == pl.col("as_of_ns").max())
    label = str(latest["snapshot_id"][0]).rsplit("-", 1)[1]
    quotes = _latest_clean(data_dir, underlying, label)
    if quotes is None:
        return
    splines = spline_baseline.spline_slices(quotes)
    fitted = latest.filter(pl.col("slice_source") != DROPPED)
    if not splines or fitted.is_empty():
        return
    spline = min(splines, key=lambda s: abs(s.t - REFERENCE_TENOR))
    nearest = fitted.with_columns((pl.col("t") - spline.t).abs().alias("gap")).sort("gap")
    row = nearest.row(0, named=True)
    svi = RawSVI(row["svi_a"], row["svi_b"], row["svi_rho"], row["svi_m"], row["svi_sigma"])
    figures.spline_vs_svi_density(
        spline,
        svi,
        f"{underlying} {spline.t * 365:.0f}-day slice, {label}: implied density of k",
        reports / "spline_vs_svi_density.png",
    )


def refresh_outputs(data_dir: Path, config: AppConfig, underlying: str) -> dict[str, Any]:
    """Rebuild every summary, metric, table and figure from the stored days."""
    gap = forward_gap_metrics(data_dir, config, underlying)
    params = read_params(data_dir, underlying)
    checks = _arbcheck_files(config, underlying)
    if not checks or params.is_empty():
        return {"forward_gap": gap}
    _arbcheck_summary(checks, config, underlying)
    spread = within_spread_metrics(params, underlying)
    _write_metric(config, f"fit_within_spread_{underlying}", spread)
    stability = {"underlying": underlying, **stability_metrics(params)}
    _write_metric(config, f"stability_{underlying}", stability)
    spline = _spline_metrics(checks, underlying)
    _write_metric(config, f"spline_baseline_{underlying}", spline)
    table = _rnd_table(params, config, underlying)
    _draft_figures(data_dir, config, underlying, params, table)
    return {"forward_gap": gap, "within_spread": spread, "spline": spline}


def forward_gap_metrics(data_dir: Path, config: AppConfig, underlying: str) -> dict[str, Any]:
    """AC-W-05: |ln(F_parity / F_future)| over all built days, written to results/metrics."""
    root = data_dir / "interim" / "forwards" / f"underlying={underlying}"
    files = sorted(root.glob("date=*/part-*.parquet"))
    metrics: dict[str, Any] = {"underlying": underlying, "days": 0, "n": 0}
    if files:
        frame = pl.concat([pl.read_parquet(path) for path in files])
        gaps = frame["gap_bps"].drop_nulls().abs().to_numpy()
        metrics["days"] = len(files)
        metrics["n"] = int(gaps.size)
        if gaps.size:
            metrics["median_abs_bps"] = float(np.median(gaps))
            metrics["p95_abs_bps"] = float(np.percentile(gaps, 95))
            metrics["max_abs_bps"] = float(gaps.max())
    out = config.results_dir() / "metrics" / f"forward_gap_{underlying}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics

"""Run calibrations over stored days and build the model comparison (FR-18)."""

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

from vsm.config import AppConfig, config_hash
from vsm.conventions import F64
from vsm.explog import git_sha
from vsm.forwards import ForwardEstimate
from vsm.io import storage
from vsm.io.recorder import label_time
from vsm.models.heston_calib import HestonFit, calibrate_heston, model_fit_stats, select_quotes
from vsm.models.heston_cf import HestonParams
from vsm.models.heston_cos import heston_price_cos
from vsm.models.sabr import calibrate_sabr, sabr_hagan_vol
from vsm.pipeline import clean_quotes_of, export_day, stored_rows
from vsm.surface.fit import build_slices
from vsm.surface.ssvi import SSVI
from vsm.surface.store import DROPPED, heston_path, params_path, sabr_path
from vsm.surface.svi import RawSVI
from vsm.viz import figures

SHORT_EXPIRY_DAYS = 7.0  # the Heston variant of R-06
SENSITIVITY_BETA = 0.5
MODELS = ("SVI", "SSVI", "Heston", "SABR")
JW_COLUMNS = ("jw_v", "jw_psi", "jw_p", "jw_c", "jw_vtilde")

log = structlog.get_logger("vsm.models")


def _init(config: AppConfig) -> HestonParams:
    init = config.heston.init
    return HestonParams(init["v0"], init["kappa"], init["theta"], init["xi"], init["rho"])


def _liquid_by_expiry(quotes: pl.DataFrame) -> list[pl.DataFrame]:
    rows = quotes.filter(pl.col("is_liquid")).sort("expiry_ns", "strike")
    return rows.partition_by("expiry_ns", maintain_order=True)


def _heston_row(
    fit: HestonFit, variant: str, snapshot: str, as_of_ns: int, sha: str, digest: str
) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot,
        "as_of_ns": as_of_ns,
        "variant": variant,
        "v0": fit.params.v0,
        "kappa": fit.params.kappa,
        "theta": fit.params.theta,
        "xi": fit.params.xi,
        "rho": fit.params.rho,
        "rmse_iv_bp": fit.rmse_iv_bp,
        "within_spread_frac": fit.within_spread_frac,
        "max_error_bp": fit.max_error_bp,
        "n_quotes": fit.n_quotes,
        "evaluations": fit.evaluations,
        "converged": fit.converged,
        "git_sha": sha,
        "config_hash": digest,
    }


def calibrate_heston_day(
    data_dir: Path, config: AppConfig, underlying: str, day: date
) -> dict[str, Any] | None:
    """Fit Heston on all expiries and without short ones; store both."""
    rows = stored_rows(data_dir, config, underlying, day)
    if rows is None:
        return None
    snapshot = str(rows["snapshot_id"][0])
    quotes = clean_quotes_of(data_dir, underlying, snapshot.rsplit("-", 1)[1])
    if quotes is None:
        return None
    sha = git_sha()
    digest = config_hash(config)
    as_of_ns = int(rows["as_of_ns"][0])
    fit = calibrate_heston(select_quotes(quotes), _init(config), config.heston)
    variant = calibrate_heston(
        select_quotes(quotes, min_days=SHORT_EXPIRY_DAYS), _init(config), config.heston
    )
    frame = pl.DataFrame(
        [
            _heston_row(fit, "all", snapshot, as_of_ns, sha, digest),
            _heston_row(variant, "min_7d", snapshot, as_of_ns, sha, digest),
        ],
        schema=storage.HESTON_PARAMS_SCHEMA,
    )
    storage.upsert_parquet(frame, heston_path(data_dir, underlying), ["snapshot_id", "variant"])
    export_day(data_dir, config, underlying, day)
    if not fit.converged:
        log.warning("heston_not_converged", snapshot=snapshot, message=fit.message)
    return {
        "converged": fit.converged,
        "rmse_iv_bp": fit.rmse_iv_bp,
        "within_spread_frac": fit.within_spread_frac,
        "n_quotes": fit.n_quotes,
        "evaluations": fit.evaluations,
        "rmse_iv_bp_min_7d": variant.rmse_iv_bp,
    }


def calibrate_sabr_day(
    data_dir: Path, config: AppConfig, underlying: str, day: date
) -> dict[str, Any] | None:
    """Fit SABR per expiry at the configured beta and at the sensitivity beta; store both."""
    rows = stored_rows(data_dir, config, underlying, day)
    if rows is None:
        return None
    snapshot = str(rows["snapshot_id"][0])
    quotes = clean_quotes_of(data_dir, underlying, snapshot.rsplit("-", 1)[1])
    if quotes is None:
        return None
    records = []
    for beta in (config.sabr.beta, SENSITIVITY_BETA):
        for group in _liquid_by_expiry(quotes):
            if group.height < 3:
                continue
            fit = calibrate_sabr(group, beta, obloj=config.sabr.obloj_correction)
            records.append(
                {
                    "snapshot_id": snapshot,
                    "as_of_ns": int(rows["as_of_ns"][0]),
                    "expiry_ns": int(group["expiry_ns"][0]),
                    "beta": beta,
                    "alpha": fit.alpha,
                    "rho": fit.rho,
                    "nu": fit.nu,
                    "rmse_iv_bp": fit.rmse_iv_bp,
                    "within_spread_frac": fit.within_spread_frac,
                    "max_error_bp": fit.max_error_bp,
                    "n_quotes": fit.n_quotes,
                    "converged": fit.converged,
                }
            )
    frame = pl.DataFrame(records, schema=storage.SABR_PARAMS_SCHEMA)
    storage.upsert_parquet(
        frame, sabr_path(data_dir, underlying), ["snapshot_id", "expiry_ns", "beta"]
    )
    export_day(data_dir, config, underlying, day)
    primary = frame.filter(pl.col("beta") == config.sabr.beta)
    return {
        "expiries": primary.height,
        "converged": bool(primary["converged"].all()),
        "median_rmse_iv_bp": float(np.median(primary["rmse_iv_bp"].to_numpy())),
    }


def _stats(model_iv: F64, rows: pl.DataFrame) -> dict[str, float]:
    error = model_iv - rows["target_iv"].to_numpy()
    inside = (model_iv >= rows["bid_iv"].to_numpy()) & (model_iv <= rows["ask_iv"].to_numpy())
    return {
        "rmse_iv_bp": 1e4 * float(np.sqrt(np.mean(error * error))),
        "within_spread_frac": float(inside.mean()),
        "max_error_bp": 1e4 * float(np.max(np.abs(error))),
    }


def _surface_iv(rows: pl.DataFrame, liquid: pl.DataFrame) -> tuple[F64, F64]:
    """Model IVs of the liquid quotes under the SVI slices and under SSVI."""
    k = liquid["k"].to_numpy()
    t = liquid["t"].to_numpy()
    expiry = liquid["expiry_ns"].to_numpy()
    svi_iv = np.full(k.size, np.nan)
    ssvi_iv = np.full(k.size, np.nan)
    for row in rows.iter_rows(named=True):
        mask = expiry == row["expiry_ns"]
        if row["slice_source"] == DROPPED or not mask.any():
            continue
        svi = RawSVI(row["svi_a"], row["svi_b"], row["svi_rho"], row["svi_m"], row["svi_sigma"])
        svi_iv[mask] = np.sqrt(svi.w(k[mask]) / t[mask])
        anchor = _ssvi_slice(row)
        ssvi_iv[mask] = np.sqrt(anchor.w(k[mask]) / t[mask])
    return svi_iv, ssvi_iv


def _ssvi_slice(row: dict[str, Any]) -> RawSVI:
    surface = SSVI(
        row["ssvi_rho"], row["ssvi_eta"], row["ssvi_gamma"], (row["t"],), (row["ssvi_theta"],)
    )
    return surface.slice_at(row["t"])


def _sabr_iv(frame: pl.DataFrame, liquid: pl.DataFrame) -> F64:
    iv = np.full(liquid.height, np.nan)
    expiry = liquid["expiry_ns"].to_numpy()
    strike = liquid["strike"].to_numpy()
    for row in frame.iter_rows(named=True):
        mask = expiry == row["expiry_ns"]
        if not mask.any():
            continue
        first = liquid.filter(pl.col("expiry_ns") == row["expiry_ns"]).row(0, named=True)
        iv[mask] = sabr_hagan_vol(
            first["forward"],
            strike[mask],
            first["t"],
            row["alpha"],
            row["beta"],
            row["rho"],
            row["nu"],
        )
    return iv


def _read(path: Path) -> pl.DataFrame | None:
    return pl.read_parquet(path) if path.exists() else None


def _daily_rows(
    data_dir: Path, config: AppConfig, underlying: str, params: pl.DataFrame
) -> list[dict[str, Any]]:
    """One row per model per day, all on the same liquid quotes."""
    heston = _read(heston_path(data_dir, underlying))
    sabr = _read(sabr_path(data_dir, underlying))
    daily: list[dict[str, Any]] = []
    snapshots = params["snapshot_id"].unique().sort().to_list()
    for snapshot in snapshots:
        quotes = clean_quotes_of(data_dir, underlying, str(snapshot).rsplit("-", 1)[1])
        if quotes is None:
            continue
        rows = params.filter(pl.col("snapshot_id") == snapshot)
        liquid = quotes.filter(pl.col("is_liquid"))
        svi_iv, ssvi_iv = _surface_iv(rows, liquid)
        ivs: dict[str, tuple[F64, pl.DataFrame]] = {}
        usable = np.isfinite(svi_iv)
        kept = liquid.filter(pl.Series(usable))
        ivs["SVI"] = (svi_iv[usable], kept)
        ivs["SSVI"] = (ssvi_iv[usable], kept)
        if sabr is not None:
            fits = sabr.filter(
                (pl.col("snapshot_id") == snapshot) & (pl.col("beta") == config.sabr.beta)
            )
            iv = _sabr_iv(fits, liquid)
            finite = np.isfinite(iv)
            ivs["SABR"] = (iv[finite], liquid.filter(pl.Series(finite)))
        for model, (model_iv, subset) in ivs.items():
            if subset.height:
                daily.append({"snapshot_id": snapshot, "model": model, **_stats(model_iv, subset)})
        if heston is not None:
            fits = heston.filter((pl.col("snapshot_id") == snapshot) & (pl.col("variant") == "all"))
            for row in fits.iter_rows(named=True):
                params_h = HestonParams(
                    row["v0"], row["kappa"], row["theta"], row["xi"], row["rho"]
                )
                selected = select_quotes(quotes)
                is_call = selected["cp"].to_numpy() == 1
                price = heston_price_cos(
                    params_h,
                    selected["forward"].to_numpy(),
                    selected["strike"].to_numpy(),
                    selected["t"].to_numpy(),
                    is_call,
                    config.heston.cos_terms,
                    config.heston.cos_truncation_L,
                )
                rmse, inside, worst = model_fit_stats(price, selected, is_call)
                daily.append(
                    {
                        "snapshot_id": snapshot,
                        "model": "Heston",
                        "rmse_iv_bp": rmse,
                        "within_spread_frac": inside,
                        "max_error_bp": worst,
                    }
                )
    return daily


def _param_changes(
    data_dir: Path, underlying: str, params: pl.DataFrame
) -> dict[str, float | None]:
    usable = params.filter(pl.col("slice_source") != DROPPED)
    shape = usable.unique("as_of_ns").sort("as_of_ns")

    def diffs(frame: pl.DataFrame, columns: tuple[str, ...]) -> float | None:
        if frame.height < 2:
            return None
        changes = [np.abs(np.diff(frame[c].to_numpy())) for c in columns]
        return float(np.median(np.concatenate(changes)))

    jw = None
    if usable["as_of_ns"].n_unique() > 1:
        ordered = usable.sort("expiry_ns", "as_of_ns")
        jw_changes = ordered.select(
            [pl.col(c).diff().over("expiry_ns").abs().alias(c) for c in JW_COLUMNS]
        )
        jw = float(np.nanmedian(jw_changes.to_numpy().astype(np.float64)))
    result: dict[str, float | None] = {
        "SVI": jw,
        "SSVI": diffs(shape, ("ssvi_rho", "ssvi_eta", "ssvi_gamma")),
        "Heston": None,
        "SABR": None,
    }
    heston = _read(heston_path(data_dir, underlying))
    if heston is not None:
        rows = heston.filter(pl.col("variant") == "all").sort("as_of_ns")
        result["Heston"] = diffs(rows, ("v0", "kappa", "theta", "xi", "rho"))
    sabr = _read(sabr_path(data_dir, underlying))
    if sabr is not None and sabr["as_of_ns"].n_unique() > 1:
        ordered = sabr.sort("beta", "expiry_ns", "as_of_ns")
        changes = ordered.select(
            [
                pl.col(c).diff().over("beta", "expiry_ns").abs().alias(c)
                for c in ("alpha", "rho", "nu")
            ]
        )
        result["SABR"] = float(np.nanmedian(changes.to_numpy().astype(np.float64)))
    return result


def _times(config: AppConfig, underlying: str) -> dict[str, float]:
    """Measured by ``vsm bench``, so that re-running a study leaves its tables byte-identical."""
    path = config.results_dir() / "bench" / f"calibration_times_{underlying}.json"
    if not path.exists():
        return {}
    times: dict[str, float] = json.loads(path.read_text(encoding="utf-8"))
    return times


def refresh_comparison(data_dir: Path, config: AppConfig, underlying: str) -> list[dict[str, Any]]:
    """Write the comparison table, the daily table, sensitivity tables and the figure."""
    path = params_path(data_dir, underlying)
    if not path.exists():
        return []
    params = pl.read_parquet(path)
    daily = _daily_rows(data_dir, config, underlying, params)
    tables = config.results_dir() / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if not daily:
        return []
    frame = pl.DataFrame(daily)
    frame.write_csv(tables / f"model_fit_daily_{underlying}.csv")
    changes = _param_changes(data_dir, underlying, params)
    times = _times(config, underlying)
    summary: list[dict[str, Any]] = []
    for model in MODELS:
        rows = frame.filter(pl.col("model") == model)
        if rows.is_empty():
            continue
        summary.append(
            {
                "model": model,
                "days": rows.height,
                "rmse_iv_bp": float(rows["rmse_iv_bp"].mean()),  # type: ignore[arg-type]
                "within_spread_frac": float(rows["within_spread_frac"].mean()),  # type: ignore[arg-type]
                "max_error_bp": float(rows["max_error_bp"].max()),  # type: ignore[arg-type]
                "calibration_seconds": times.get(model),
                "median_daily_param_change": changes.get(model),
            }
        )
    pl.DataFrame(summary).write_csv(tables / f"model_comparison_{underlying}.csv")
    _sensitivity_tables(data_dir, config, underlying)
    if underlying == config.underlyings[0]:
        figures.model_fit_comparison(
            [row["model"] for row in summary],
            [row["rmse_iv_bp"] for row in summary],
            [row["within_spread_frac"] for row in summary],
            f"{underlying}: IV fit error by model, liquid quotes",
            config.reports_dir() / "figures" / f"model_fit_comparison_{underlying}.png",
        )
    return summary


def _sensitivity_tables(data_dir: Path, config: AppConfig, underlying: str) -> None:
    tables = config.results_dir() / "tables"
    sabr = _read(sabr_path(data_dir, underlying))
    if sabr is not None:
        by_beta = (
            sabr.group_by("beta")
            .agg(
                pl.col("rmse_iv_bp").mean(),
                pl.col("within_spread_frac").mean(),
                pl.col("max_error_bp").max(),
                pl.len().alias("slices"),
            )
            .sort("beta")
        )
        by_beta.write_csv(tables / f"sabr_beta_sensitivity_{underlying}.csv")
    heston = _read(heston_path(data_dir, underlying))
    if heston is not None:
        by_variant = (
            heston.group_by("variant")
            .agg(
                pl.col("rmse_iv_bp").mean(),
                pl.col("within_spread_frac").mean(),
                pl.col("max_error_bp").max(),
                pl.col("n_quotes").mean(),
                pl.len().alias("days"),
            )
            .sort("variant")
        )
        by_variant.write_csv(tables / f"heston_variants_{underlying}.csv")


def measure_times(
    data_dir: Path, config: AppConfig, underlying: str, days: list[date]
) -> dict[str, Any]:
    """Median wall time per day of each fit, JIT warm, for AC-H-04 and the comparison table."""
    heston_seconds: list[float] = []
    sabr_seconds: list[float] = []
    surface_seconds: list[float] = []
    details: list[dict[str, Any]] = []
    for day in days:
        rows = stored_rows(data_dir, config, underlying, day)
        if rows is None:
            continue
        snapshot = str(rows["snapshot_id"][0])
        quotes = clean_quotes_of(data_dir, underlying, snapshot.rsplit("-", 1)[1])
        if quotes is None:
            continue
        selected = select_quotes(quotes)
        calibrate_heston(selected, _init(config), config.heston)  # warm caches
        fit = calibrate_heston(selected, _init(config), config.heston)
        heston_seconds.append(fit.seconds)
        details.append(
            {
                "snapshot": snapshot,
                "seconds": round(fit.seconds, 4),
                "evaluations": fit.evaluations,
                "jacobians": fit.jacobians,
                "n_quotes": fit.n_quotes,
                "rmse_iv_bp": round(fit.rmse_iv_bp, 1),
                "converged": fit.converged,
            }
        )
        started = time.perf_counter()
        for group in _liquid_by_expiry(quotes):
            if group.height >= 3:
                calibrate_sabr(group, config.sabr.beta)
        sabr_seconds.append(time.perf_counter() - started)
        forwards = _stored_forwards(data_dir, underlying, snapshot)
        started = time.perf_counter()
        build_slices(quotes, forwards, config)
        surface_seconds.append(time.perf_counter() - started)
    return {
        "heston_seconds": heston_seconds,
        "sabr_seconds": sabr_seconds,
        "surface_seconds": surface_seconds,
        "details": details,
    }


def _stored_forwards(data_dir: Path, underlying: str, snapshot: str) -> list[ForwardEstimate]:
    label = snapshot.rsplit("-", 1)[1]
    day = label_time(label).strftime("%Y-%m-%d")
    path = (
        data_dir
        / "interim"
        / "forwards"
        / f"underlying={underlying}"
        / f"date={day}"
        / f"part-{label}.parquet"
    )
    return [
        ForwardEstimate(
            r["expiry_ns"],
            r["t"],
            r["forward"],
            r["forward_se"],
            r["df"],
            r["method"],
            r["n_pairs"],
            r["future_px"],
            r["gap_bps"],
            r["basis_ann"],
        )
        for r in pl.read_parquet(path).iter_rows(named=True)
    ]

"""Day-over-day parameter stability from the surface history (FR-14, metrics only).

Selecting the regularization strength lambda waits for M6, once 14 days exist.
"""

from typing import Any

import numpy as np
import polars as pl

JW_COLUMNS = ("jw_v", "jw_psi", "jw_p", "jw_c", "jw_vtilde")
REFERENCE_DAYS = 30.0
YEAR_DAYS = 365.0


def daily_shape(params: pl.DataFrame) -> pl.DataFrame:
    """One row per as-of: SSVI shape parameters and theta at 30 days."""
    rows = []
    for key, group in params.sort("as_of_ns", "t").group_by("as_of_ns", maintain_order=True):
        rows.append(
            {
                "as_of_ns": key[0],
                "rho": group["ssvi_rho"][0],
                "eta": group["ssvi_eta"][0],
                "gamma": group["ssvi_gamma"][0],
                "theta_30d": float(
                    np.interp(
                        REFERENCE_DAYS / YEAR_DAYS,
                        group["t"].to_numpy(),
                        group["ssvi_theta"].to_numpy(),
                    )
                ),
            }
        )
    return pl.DataFrame(rows)


def stability_metrics(params: pl.DataFrame) -> dict[str, Any]:
    """AC-H-01 inputs. Changes need at least two days; until then they are None."""
    shape = daily_shape(params) if params.height else pl.DataFrame()
    metrics: dict[str, Any] = {"days": shape.height}
    if shape.height < 2:
        return {**metrics, "p95_abs_d_rho": None, "p95_rel_d_eta": None, "jw_median_abs": None}
    rho = shape["rho"].to_numpy()
    eta = shape["eta"].to_numpy()
    theta = shape["theta_30d"].to_numpy()
    gamma = shape["gamma"].to_numpy()
    metrics["p95_abs_d_rho"] = float(np.percentile(np.abs(np.diff(rho)), 95))
    metrics["p95_rel_d_eta"] = float(np.percentile(np.abs(np.diff(eta)) / eta[:-1], 95))
    metrics["median_abs_d_gamma"] = float(np.median(np.abs(np.diff(gamma))))
    metrics["median_rel_d_theta_30d"] = float(np.median(np.abs(np.diff(theta)) / theta[:-1]))

    usable = params.filter(pl.col("slice_source") != "dropped").sort("expiry_ns", "as_of_ns")
    changes = usable.with_columns(
        [pl.col(c).diff().over("expiry_ns").abs().alias(c) for c in JW_COLUMNS]
    )
    metrics["jw_median_abs"] = {c: changes[c].drop_nulls().median() for c in JW_COLUMNS}
    return metrics


def within_spread_metrics(params: pl.DataFrame, underlying: str) -> dict[str, Any]:
    """AC-W-03: pooled and median-per-day share of liquid quotes inside the spread."""
    rows = params.filter(pl.col("slice_source") != "dropped")
    metrics: dict[str, Any] = {"underlying": underlying, "days": 0, "n_liquid": 0}
    if rows.is_empty():
        return metrics
    n = rows["n_liquid"].cast(pl.Float64).to_numpy()
    svi = rows["within_spread_frac"].fill_null(0.0).fill_nan(0.0).to_numpy()
    ssvi = rows["ssvi_within_spread_frac"].fill_null(0.0).fill_nan(0.0).to_numpy()
    daily = (
        rows.with_columns(
            pl.Series("inside", svi * n),
            pl.Series("n", n),
        )
        .group_by("as_of_ns")
        .agg(pl.col("inside").sum(), pl.col("n").sum())
        .with_columns((pl.col("inside") / pl.col("n")).alias("frac"))
        .sort("as_of_ns")
    )
    metrics["days"] = daily.height
    metrics["n_liquid"] = int(n.sum())
    metrics["pooled_svi"] = float(np.sum(svi * n) / np.sum(n))
    metrics["pooled_ssvi"] = float(np.sum(ssvi * n) / np.sum(n))
    metrics["median_day_svi"] = float(daily["frac"].median())  # type: ignore[arg-type]
    return metrics

"""Forward and coin discount factor per expiry from put-call parity on coin premiums.

Parity for inverse options is C_coin - P_coin = DF_coin * (1 - K/F), a line in K with
intercept alpha = DF_coin and slope beta = -DF_coin/F, so F = -alpha/beta (FR-08, DQ-09).
"""

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from vsm.config import ForwardsConfig
from vsm.conventions import F64, YEAR_SECONDS, year_fraction

HUBER_ITERATIONS = 20
MAD_TO_SIGMA = 0.6745
SCALE_FLOOR = 1e-13


@dataclass(frozen=True)
class ForwardEstimate:
    """One expiry's forward. ``method`` is parity_wls, future or deribit_underlying."""

    expiry_ns: int
    t: float
    forward: float
    forward_se: float
    df: float
    method: str
    n_pairs: int
    future_px: float | None
    gap_bps: float | None
    basis_ann: float


def huber_wls(
    strike: F64, y: F64, weight: F64, huber_c: float
) -> tuple[float, float, float] | None:
    """Fit y = alpha + beta * K by Huber-weighted least squares.

    Returns (alpha, beta, standard error of F = -alpha/beta), or None if the slope cannot
    give a positive forward.
    """
    scale_k = float(np.mean(strike))
    design = np.column_stack([np.ones_like(strike), strike / scale_k])
    root_w = np.sqrt(weight)
    robust = np.ones_like(y)
    coef = np.zeros(2)
    for _ in range(HUBER_ITERATIONS):
        w = root_w * np.sqrt(robust)
        coef, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
        residual = (y - design @ coef) * root_w
        sigma = float(np.median(np.abs(residual))) / MAD_TO_SIGMA
        if sigma <= SCALE_FLOOR:
            break
        updated = np.minimum(1.0, huber_c * sigma / np.maximum(np.abs(residual), 1e-300))
        converged = bool(np.max(np.abs(updated - robust)) < 1e-8)
        robust = updated
        if converged:
            break

    alpha = float(coef[0])
    beta = float(coef[1]) / scale_k
    if beta >= 0.0 or alpha <= 0.0:
        return None
    n = strike.size
    w2 = weight * robust
    resid = y - design @ coef
    dof = max(n - 2, 1)
    variance = float(np.sum(w2 * resid * resid)) / dof
    covariance = variance * np.linalg.pinv((design * w2[:, None]).T @ design)
    gradient = np.array([-1.0 / beta, (alpha / (beta * beta)) / scale_k])
    forward_var = float(gradient @ covariance @ gradient)
    return alpha, beta, math.sqrt(max(forward_var, 0.0))


def _parity_pairs(expiry_rows: pl.DataFrame, k_window: float) -> pl.DataFrame:
    """Call and put at one strike, both two-sided, inside the moneyness window."""
    usable = expiry_rows.filter(
        pl.col("drop_reason").is_null()
        & pl.col("bid_px").is_not_null()
        & pl.col("ask_px").is_not_null()
    ).with_columns(
        ((pl.col("bid_px") + pl.col("ask_px")) / 2).alias("mid"),
        (pl.col("ask_px") - pl.col("bid_px")).alias("spread"),
    )
    calls = usable.filter(pl.col("cp") == 1).select("strike", "mid", "spread", "underlying_px")
    puts = usable.filter(pl.col("cp") == -1).select("strike", "mid", "spread")
    pairs = calls.join(puts, on="strike", suffix="_put")
    if pairs.is_empty():
        return pairs
    reference = float(np.median(pairs["underlying_px"].to_numpy()))
    return pairs.filter((pl.col("strike") / reference).log().abs() <= k_window)


def _future_price(futures: pl.DataFrame, expiry_ns: int) -> float | None:
    rows = futures.filter(pl.col("expiry_ns") == expiry_ns)
    if rows.is_empty():
        return None
    row = rows.row(0, named=True)
    if row["bid_px"] is not None and row["ask_px"] is not None:
        return float(row["bid_px"] + row["ask_px"]) / 2.0
    return None if row["mark_px"] is None else float(row["mark_px"])


def implied_forwards(
    quotes: pl.DataFrame,
    futures: pl.DataFrame,
    spot_index: float,
    cfg: ForwardsConfig,
    *,
    year_seconds: int = YEAR_SECONDS,
) -> list[ForwardEstimate]:
    """One estimate per expiry in ``quotes``, in expiry order.

    ``quotes`` needs as_of_ns, expiry_ns, strike, cp, bid_px, ask_px, underlying_px and
    drop_reason; rows with a drop_reason are left out of the regression. ``future_px`` and
    ``gap_bps`` compare with a dated future of the same expiry where one is listed.
    """
    estimates = []
    as_of_ns = int(quotes["as_of_ns"][0])
    for expiry_key, group in quotes.sort("expiry_ns").group_by("expiry_ns", maintain_order=True):
        expiry_ns = int(expiry_key[0])
        t = float(year_fraction(np.array([expiry_ns]), as_of_ns, year_seconds)[0])
        future_px = _future_price(futures, expiry_ns)
        pairs = _parity_pairs(group, cfg.k_window)
        fit = None
        if pairs.height >= cfg.min_pairs:
            spread = (pairs["spread"] + pairs["spread_put"]).to_numpy()
            fit = huber_wls(
                pairs["strike"].to_numpy(),
                (pairs["mid"] - pairs["mid_put"]).to_numpy(),
                1.0 / (spread * spread),
                cfg.huber_c,
            )
        if fit is not None:
            alpha, beta, se = fit
            forward = -alpha / beta
            method, df = "parity_wls", alpha
        elif future_px is not None:
            forward, se, method, df = future_px, math.nan, "future", 1.0
        else:
            underlying = group["underlying_px"].drop_nulls()
            if underlying.is_empty():
                continue
            median = float(np.median(underlying.to_numpy()))
            forward, se, method, df = median, math.nan, "deribit_underlying", 1.0
        gap_bps = None
        if method == "parity_wls" and future_px is not None:
            gap_bps = 1e4 * math.log(forward / future_px)
        estimates.append(
            ForwardEstimate(
                expiry_ns=expiry_ns,
                t=t,
                forward=forward,
                forward_se=se,
                df=df,
                method=method,
                n_pairs=pairs.height,
                future_px=future_px,
                gap_bps=gap_bps,
                basis_ann=math.log(forward / spot_index) / t if t > 0.0 else math.nan,
            )
        )
    return estimates

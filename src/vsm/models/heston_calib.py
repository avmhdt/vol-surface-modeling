"""Heston calibration on vega-scaled price residuals, with analytic Jacobian (FR-15)."""

import math
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import least_squares

from vsm.config import HestonConfig
from vsm.conventions import F64
from vsm.iv.solver import OK, implied_vol
from vsm.models.heston_cf import HestonParams
from vsm.models.heston_cos import heston_price_and_grad_cos, heston_price_cos

MAX_QUOTES = 400
MAX_EVALUATIONS = 300
TOLERANCE = 1e-7
NAMES = ("v0", "kappa", "theta", "xi", "rho")


@dataclass(frozen=True)
class HestonFit:
    """A calibration. ``converged`` is False when the optimizer did not report success."""

    params: HestonParams
    converged: bool
    evaluations: int
    jacobians: int
    rmse_iv_bp: float
    within_spread_frac: float
    max_error_bp: float
    n_quotes: int
    seconds: float
    message: str


def select_quotes(quotes: pl.DataFrame, *, min_days: float = 0.0) -> pl.DataFrame:
    """Liquid quotes of the day, at most 400, thinned evenly across expiries and strikes."""
    rows = quotes.filter(pl.col("is_liquid") & (pl.col("t") >= min_days / 365.0))
    rows = rows.sort("expiry_ns", "strike")
    if rows.height <= MAX_QUOTES:
        return rows
    keep = np.unique(np.linspace(0, rows.height - 1, MAX_QUOTES).astype(np.int64))
    thinned: pl.DataFrame = rows[keep.tolist()]
    return thinned


def _bounds_arrays(cfg: HestonConfig) -> tuple[F64, F64]:
    low = np.array([cfg.bounds[name][0] for name in NAMES])
    high = np.array([cfg.bounds[name][1] for name in NAMES])
    return low, high


def model_fit_stats(
    price: F64, rows: pl.DataFrame, is_call: np.typing.NDArray[np.bool_]
) -> tuple[float, float, float]:
    """(RMSE, share inside the spread, max error) in true IV terms, in vol bp."""
    vol, status = implied_vol(
        price,
        rows["forward"].to_numpy(),
        rows["strike"].to_numpy(),
        rows["t"].to_numpy(),
        is_call,
    )
    usable = status == OK
    error = vol[usable] - rows["target_iv"].to_numpy()[usable]
    inside = (vol[usable] >= rows["bid_iv"].to_numpy()[usable]) & (
        vol[usable] <= rows["ask_iv"].to_numpy()[usable]
    )
    if not error.size:
        return math.nan, math.nan, math.nan
    return (
        1e4 * float(np.sqrt(np.mean(error * error))),
        float(inside.mean()),
        1e4 * float(np.max(np.abs(error))),
    )


def calibrate_heston(quotes: pl.DataFrame, init: HestonParams, cfg: HestonConfig) -> HestonFit:
    """Fit all expiries jointly from ``init``; ``quotes`` are those already selected.

    Residuals are (model - mid) / vega * weight, close to IV errors in units of the spread.
    ``lm`` has no bounds, so it works in logistic coordinates that cannot leave them.
    """
    forward = quotes["forward"].to_numpy()
    strike = quotes["strike"].to_numpy()
    t = quotes["t"].to_numpy()
    is_call = quotes["cp"].to_numpy() == 1
    market = quotes["mid_fwd"].to_numpy()
    scale = quotes["weight"].to_numpy() / quotes["vega_fwd"].to_numpy()
    low, high = _bounds_arrays(cfg)
    span = high - low
    logistic = cfg.optimizer == "lm"

    def to_params(x: F64) -> F64:
        coordinates: F64 = low + span / (1.0 + np.exp(-x)) if logistic else x
        return coordinates

    cache: dict[bytes, tuple[F64, F64]] = {}

    def evaluate(x: F64) -> tuple[F64, F64]:
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = heston_price_and_grad_cos(
                HestonParams(*to_params(x)),
                forward,
                strike,
                t,
                is_call,
                cfg.cos_terms,
                cfg.cos_truncation_L,
            )
        return cache[key]

    def residuals(x: F64) -> F64:
        residual: F64 = scale * (evaluate(x)[0] - market)
        return residual

    def jacobian(x: F64) -> F64:
        jac: F64 = scale[:, None] * evaluate(x)[1]
        if logistic:
            share = (to_params(x) - low) / span
            jac = jac * (span * share * (1.0 - share))
        return jac

    start = init.as_array()
    started = time.perf_counter()
    if logistic:
        share = np.clip((start - low) / span, 1e-6, 1.0 - 1e-6)
        solved = least_squares(
            residuals,
            np.log(share / (1.0 - share)),
            jac=jacobian,
            method="lm",
            max_nfev=MAX_EVALUATIONS,
            xtol=TOLERANCE,
            ftol=TOLERANCE,
            gtol=TOLERANCE,
        )
    else:
        solved = least_squares(
            residuals,
            start,
            jac=jacobian,
            bounds=(low, high),
            method="trf",
            max_nfev=MAX_EVALUATIONS,
            xtol=TOLERANCE,
            ftol=TOLERANCE,
            gtol=TOLERANCE,
        )
    seconds = time.perf_counter() - started
    params = HestonParams(*to_params(solved.x))
    price = heston_price_cos(
        params, forward, strike, t, is_call, cfg.cos_terms, cfg.cos_truncation_L
    )
    rmse, inside, worst = model_fit_stats(price, quotes, is_call)
    return HestonFit(
        params=params,
        converged=bool(solved.success),
        evaluations=int(solved.nfev),
        jacobians=int(solved.njev or 0),
        rmse_iv_bp=rmse,
        within_spread_frac=inside,
        max_error_bp=worst,
        n_quotes=quotes.height,
        seconds=seconds,
        message=str(solved.message),
    )

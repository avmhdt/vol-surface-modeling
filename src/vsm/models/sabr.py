"""SABR lognormal vol expansion (Hagan et al. 2002), calibrated per expiry (FR-17)."""

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import least_squares

from vsm.conventions import F64

SMALL_Z = 1e-7
RHO_MAX = 0.999
NU_MIN = 1e-4
NU_MAX = 50.0


@dataclass(frozen=True)
class SABRFit:
    """One expiry. ``converged`` is False when the optimizer did not report success."""

    alpha: float
    beta: float
    rho: float
    nu: float
    converged: bool
    rmse_iv_bp: float
    within_spread_frac: float
    max_error_bp: float
    n_quotes: int


def _z_over_x(z: F64, rho: float) -> F64:
    """z / x(z), with its series near z = 0, where both vanish."""
    root = np.sqrt(1.0 - 2.0 * rho * z + z * z)
    safe = np.where(np.abs(z) < SMALL_Z, 1.0, z)
    ratio = safe / np.log((root + safe - rho) / (1.0 - rho))
    result: F64 = np.where(np.abs(z) < SMALL_Z, 1.0 - 0.5 * rho * z, ratio)
    return result


def sabr_hagan_vol(
    forward: float,
    strike: F64,
    t: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    *,
    obloj: bool = False,
) -> F64:
    """Hagan's lognormal implied vol; ``obloj`` uses Oblój's (2008) leading term instead."""
    k = np.asarray(strike, dtype=np.float64)
    log_fk = np.log(forward / k)
    power = forward * k
    one_minus = 1.0 - beta
    mid = power ** (0.5 * one_minus)
    correction = (
        one_minus**2 / 24.0 * alpha**2 / power**one_minus
        + rho * beta * nu * alpha / (4.0 * mid)
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    )
    if obloj:
        if beta == 1.0:
            z = nu * log_fk / alpha
        else:
            z = nu * (forward**one_minus - k**one_minus) / (alpha * one_minus)
        small = np.abs(z) < SMALL_Z
        safe = np.where(small, 1.0, z)
        leading = np.where(small, alpha / mid, nu * log_fk / safe) * _z_over_x(z, rho)
        result: F64 = leading * (1.0 + correction * t)
        return result
    z = nu / alpha * mid * log_fk
    denominator = mid * (1.0 + one_minus**2 / 24.0 * log_fk**2 + one_minus**4 / 1920.0 * log_fk**4)
    vol: F64 = alpha / denominator * _z_over_x(z, rho) * (1.0 + correction * t)
    return vol


def calibrate_sabr(slice_quotes: pl.DataFrame, beta: float, *, obloj: bool = False) -> SABRFit:
    """Fit (alpha, rho, nu) on one expiry's quotes, weighted by inverse IV spread."""
    forward = float(slice_quotes["forward"][0])
    t = float(slice_quotes["t"][0])
    strike = slice_quotes["strike"].to_numpy()
    target = slice_quotes["target_iv"].to_numpy()
    weight = slice_quotes["weight"].to_numpy()
    atm = float(target[np.argmin(np.abs(np.log(strike / forward)))])

    def residuals(x: F64) -> F64:
        model = sabr_hagan_vol(
            forward, strike, t, float(x[0]), beta, float(x[1]), float(x[2]), obloj=obloj
        )
        residual: F64 = weight * (model - target)
        return residual

    start = np.array([atm * forward ** (1.0 - beta), -0.1, 1.0])
    solved = least_squares(
        residuals,
        start,
        bounds=([1e-6, -RHO_MAX, NU_MIN], [math.inf, RHO_MAX, NU_MAX]),
        method="trf",
        x_scale=np.array([start[0], 1.0, 1.0]),
    )
    alpha, rho, nu = (float(v) for v in solved.x)
    model = sabr_hagan_vol(forward, strike, t, alpha, beta, rho, nu, obloj=obloj)
    error = model - target
    liquid = slice_quotes["is_liquid"].to_numpy()
    inside = (model >= slice_quotes["bid_iv"].to_numpy()) & (
        model <= slice_quotes["ask_iv"].to_numpy()
    )
    return SABRFit(
        alpha=alpha,
        beta=beta,
        rho=rho,
        nu=nu,
        converged=bool(solved.success),
        rmse_iv_bp=1e4 * float(np.sqrt(np.mean(error * error))),
        within_spread_frac=float(inside[liquid].mean()) if liquid.any() else math.nan,
        max_error_bp=1e4 * float(np.max(np.abs(error))),
        n_quotes=slice_quotes.height,
    )

"""The static-arbitrage gate (FR-13): every surface passes here before anything can use it."""

import math

import numpy as np

from vsm.config import ArbitrageConfig
from vsm.conventions import F64
from vsm.iv.black import black76_price
from vsm.surface.density import moments
from vsm.surface.ssvi import SSVI
from vsm.surface.svi import RawSVI, durrleman_g, slice_g
from vsm.surface.types import ArbReport, SliceCheck, Surface

__all__ = [
    "calendar_margin",
    "check_grid",
    "check_slice",
    "check_surface",
    "check_svi_conditions",
    "durrleman_g",
]


def check_grid(svi: RawSVI, quoted: F64, cfg: ArbitrageConfig) -> F64:
    """z = k / sqrt(theta) on the configured grid, plus the whole quoted range.

    A short slice can be quoted well past z = 6, and g can dip between the quotes, so the
    range is covered at the same density rather than only at its ends.
    """
    points = np.asarray(quoted, dtype=np.float64)
    parts = []
    if points.size:
        parts.append(np.linspace(float(points.min()), float(points.max()), cfg.grid_z.n))
    theta = float(svi.w(np.array(0.0)))
    if theta > 0.0:
        z = np.linspace(cfg.grid_z.min, cfg.grid_z.max, cfg.grid_z.n)
        parts.append(z * math.sqrt(theta))
    return np.unique(np.concatenate(parts)) if parts else points


def _prices_ok(svi: RawSVI, k: F64, tol: float) -> bool:
    """Call prices in units of F are non-increasing and convex in strike.

    A slope between close strikes is only known to one ulp of price over the gap, so that
    rounding is added to ``tol``. It is what the division itself introduces, not slack.
    """
    strike, first = np.unique(np.exp(k), return_index=True)
    w = svi.w(k[first])
    price = black76_price(1.0, strike, 1.0, np.sqrt(w), True)
    gap = np.diff(strike)
    slope = np.diff(price) / gap
    noise = (np.spacing(price[:-1]) + np.spacing(price[1:])) / gap
    if np.any(slope > tol + noise):
        return False
    return not bool(np.any(np.diff(slope) < -(tol + noise[:-1] + noise[1:])))


def check_slice(
    svi: RawSVI,
    quoted: F64,
    cfg: ArbitrageConfig,
    *,
    lower: RawSVI | None = None,
    lower_quoted: F64 | None = None,
) -> SliceCheck:
    """All gate checks of one slice, with calendar against ``lower`` when given."""
    k = check_grid(svi, quoted, cfg)
    w = svi.w(k)
    if np.any(w <= 0.0):
        return SliceCheck(-math.inf, math.inf, math.inf, False, None, False)
    min_g = float(np.min(slice_g(svi, k)))
    stats = moments(svi)
    mass_err = abs(stats["mass"] - 1.0)
    mean_err = abs(stats["mean_ratio"] - 1.0)
    price_ok = _prices_ok(svi, k, cfg.tol_price)
    min_dw = None
    calendar_ok = True
    if lower is not None:
        reference = lower_quoted if lower_quoted is not None else quoted
        min_dw = calendar_margin(lower, svi, reference, quoted, cfg)
        calendar_ok = min_dw >= -cfg.tol_calendar
    passed = (
        min_g >= -cfg.tol_g
        and mass_err <= cfg.tol_density_mass
        and mean_err <= cfg.tol_density_mean
        and price_ok
        and calendar_ok
    )
    return SliceCheck(min_g, mass_err, mean_err, price_ok, min_dw, passed)


def calendar_margin(
    lower: RawSVI, upper: RawSVI, lower_quoted: F64, upper_quoted: F64, cfg: ArbitrageConfig
) -> float:
    """min(w_upper - w_lower) over both slices' check grids."""
    k = np.concatenate([check_grid(lower, lower_quoted, cfg), check_grid(upper, upper_quoted, cfg)])
    return float(np.min(upper.w(k) - lower.w(k)))


def check_svi_conditions(ssvi: SSVI, cfg: ArbitrageConfig) -> tuple[SliceCheck, ...]:
    """The gate on each SSVI node slice, in order."""
    checks = []
    lower: RawSVI | None = None
    no_quotes = np.empty(0)
    for t in ssvi.t_nodes:
        current = ssvi.slice_at(t)
        checks.append(check_slice(current, no_quotes, cfg, lower=lower, lower_quoted=no_quotes))
        lower = current
    return tuple(checks)


def check_surface(surface: Surface, cfg: ArbitrageConfig) -> ArbReport:
    """Gate a whole surface: every SVI slice, every SSVI node and the term structure."""
    checks = []
    lower: RawSVI | None = None
    lower_quoted: F64 | None = None
    for item in surface.slices:
        quoted = np.array([item.k_min, item.k_max])
        checks.append(check_slice(item.svi, quoted, cfg, lower=lower, lower_quoted=lower_quoted))
        lower, lower_quoted = item.svi, quoted
    ssvi_checks = check_svi_conditions(surface.ssvi, cfg)
    everything = [*checks, *ssvi_checks]
    margins = [c.min_calendar_dw for c in everything if c.min_calendar_dw is not None]
    return ArbReport(
        passed=bool(checks) and all(c.passed for c in everything),
        min_g=min(c.min_g for c in everything),
        min_calendar_dw=min(margins) if margins else 0.0,
        density_mass_max_err=max(c.density_mass_err for c in everything),
        density_mean_max_err=max(c.density_mean_err for c in everything),
        slices=tuple(checks),
        ssvi_slices=ssvi_checks,
    )

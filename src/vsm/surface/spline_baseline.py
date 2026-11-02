"""Cubic-spline baseline (FR-27): what the arbitrage gate protects against.

A natural cubic spline of mid IV in k per slice, linear in total variance across expiries,
goes through the same checks as the SVI slices on each slice's quoted range. Nothing in a
spline knows about no-arbitrage, so any violation it shows comes from the interpolation.
"""

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

from vsm.conventions import F64
from vsm.surface.svi import durrleman_g

MIN_STRIKES = 4
GRID_POINTS = 401


@dataclass(frozen=True)
class SplineSlice:
    """A natural cubic spline of IV in k, for one expiry."""

    t: float
    spline: CubicSpline
    k_min: float
    k_max: float

    def grid(self) -> F64:
        """The quoted range: the only place a spline means anything."""
        return np.linspace(self.k_min, self.k_max, GRID_POINTS)

    def w(self, k: F64) -> F64:
        """Total variance."""
        return np.asarray(self.spline(k)) ** 2 * self.t

    def g(self, k: F64) -> F64:
        """Durrleman's g from the spline's own derivatives."""
        iv = np.asarray(self.spline(k))
        d1 = np.asarray(self.spline(k, 1))
        d2 = np.asarray(self.spline(k, 2))
        w = iv * iv * self.t
        dw = 2.0 * iv * d1 * self.t
        d2w = 2.0 * (d1 * d1 + iv * d2) * self.t
        return durrleman_g(k, w, dw, d2w)


def spline_slices(quotes: pl.DataFrame) -> list[SplineSlice]:
    """One spline per expiry of the kept OTM mid IVs, shortest first."""
    kept = quotes.filter(pl.col("drop_reason").is_null() & pl.col("is_otm"))
    slices = []
    for _, rows in kept.sort("expiry_ns", "k").group_by("expiry_ns", maintain_order=True):
        unique = rows.unique("k", keep="first", maintain_order=True)
        if unique.height < MIN_STRIKES:
            continue
        k = unique["k"].to_numpy()
        spline = CubicSpline(k, unique["mid_iv"].to_numpy(), bc_type="natural")
        slices.append(SplineSlice(float(rows["t"][0]), spline, float(k[0]), float(k[-1])))
    return slices


def evaluate(quotes: pl.DataFrame) -> dict[str, Any]:
    """Butterfly and calendar violations of the spline surface of one day."""
    slices = spline_slices(quotes)
    butterfly = [s for s in slices if float(np.min(s.g(s.grid()))) < 0.0]
    calendar = 0
    for lower, upper in pairwise(slices):
        low = max(lower.k_min, upper.k_min)
        high = min(lower.k_max, upper.k_max)
        if high > low:
            k = np.linspace(low, high, GRID_POINTS)
            calendar += int(float(np.min(upper.w(k) - lower.w(k))) < 0.0)
    return {
        "slices": len(slices),
        "butterfly_violations": len(butterfly),
        "calendar_violations": calendar,
    }

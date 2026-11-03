"""Knock-out calls: the product definition and the closed form under flat Black."""

import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm

BarrierKind = Literal["down_and_out_call", "up_and_out_call"]


@dataclass(frozen=True)
class BarrierSpec:
    """A knock-out call with strike and barrier as fractions of the forward."""

    kind: BarrierKind
    strike_pct: float
    barrier_pct: float
    tenor_days: float

    @property
    def is_down(self) -> bool:
        """True for a barrier below the forward."""
        return self.kind == "down_and_out_call"


def black_barrier_price(spec: BarrierSpec, forward: float, vol: float) -> float:
    """Continuously monitored knock-out call under flat Black, undiscounted.

    These are Reiner and Rubinstein's terms A, B, C and D. Zero rate and zero carry give
    mu = -1/2, so the reflected terms scale F by H / F and K by F / H.
    """
    root = vol * math.sqrt(spec.tenor_days / 365.0)
    strike = forward * spec.strike_pct
    barrier = forward * spec.barrier_pct
    ratio = barrier / forward
    eta = 1.0 if spec.is_down else -1.0

    def vanilla(x: float) -> float:
        return float(forward * norm.cdf(x) - strike * norm.cdf(x - root))

    def reflected(y: float) -> float:
        return float(
            forward * ratio * norm.cdf(eta * y) - strike / ratio * norm.cdf(eta * (y - root))
        )

    a = vanilla(math.log(forward / strike) / root + 0.5 * root)
    b = vanilla(math.log(forward / barrier) / root + 0.5 * root)
    c = reflected(math.log(barrier * barrier / (forward * strike)) / root + 0.5 * root)
    d = reflected(math.log(barrier / forward) / root + 0.5 * root)
    if spec.is_down:
        return a - c if strike >= barrier else b - d
    return 0.0 if strike >= barrier else a - b + c - d

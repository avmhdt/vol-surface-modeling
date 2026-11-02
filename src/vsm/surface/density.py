"""Risk-neutral density from an SVI slice (Breeden-Litzenberger, in closed form via g(k))."""

import math

import numpy as np
from scipy.special import ndtr

from vsm.conventions import F64
from vsm.surface.svi import RawSVI, slice_g

INTEGRATION_POINTS = 16_001
INTEGRATION_Z = 30.0
SQRT_2PI = math.sqrt(2.0 * math.pi)


def density_k(k: F64, svi: RawSVI) -> F64:
    """Density of ln(K/F): g(k) / sqrt(2 pi w) * exp(-d2^2 / 2)."""
    log_moneyness = np.asarray(k, dtype=np.float64)
    w = svi.w(log_moneyness)
    root = np.sqrt(w)
    d2 = -log_moneyness / root - 0.5 * root
    return slice_g(svi, log_moneyness) / (SQRT_2PI * root) * np.exp(-0.5 * d2 * d2)


def bl_density(k: F64, slice_: RawSVI, forward: float) -> F64:
    """Density in strike K = F exp(k), per unit of strike."""
    return density_k(k, slice_) / (forward * np.exp(np.asarray(k, dtype=np.float64)))


def integration_grid(svi: RawSVI) -> F64:
    """Log-moneyness grid wide enough to hold the mass of the density."""
    root = math.sqrt(max(float(svi.w(np.array(0.0))), 1e-12))
    limit = INTEGRATION_Z * root
    return np.linspace(-limit, limit, INTEGRATION_POINTS)


def risk_reversal_25(svi: RawSVI, t: float) -> float:
    """25-delta risk reversal in decimal vol: IV of the 25-delta call minus the put's."""
    root = math.sqrt(max(float(svi.w(np.array(0.0))), 1e-12))
    k = np.linspace(-6.0 * root, 6.0 * root, 2401)
    total = np.sqrt(svi.w(k))
    call_delta = ndtr(-k / total + 0.5 * total)
    order = np.argsort(call_delta)
    ranked_delta = call_delta[order]
    iv = total / math.sqrt(t)
    return float(
        np.interp(0.25, ranked_delta, iv[order]) - np.interp(0.75, ranked_delta, iv[order])
    )


def moments(svi: RawSVI) -> dict[str, float]:
    """Moments of ln(S_T/F) and E[S_T]/F by quadrature.

    ``mass`` should be 1 and ``mean_ratio`` should be 1; the gate tests both.
    """
    k = integration_grid(svi)
    p = density_k(k, svi)
    mass = float(np.trapezoid(p, k))
    mean_ratio = float(np.trapezoid(p * np.exp(k), k))
    mean = float(np.trapezoid(p * k, k)) / mass
    variance = float(np.trapezoid(p * (k - mean) ** 2, k)) / mass
    root = math.sqrt(variance)
    skewness = float(np.trapezoid(p * ((k - mean) / root) ** 3, k)) / mass
    kurtosis = float(np.trapezoid(p * ((k - mean) / root) ** 4, k)) / mass
    return {
        "mass": mass,
        "mean_ratio": mean_ratio,
        "mean_log": mean,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }

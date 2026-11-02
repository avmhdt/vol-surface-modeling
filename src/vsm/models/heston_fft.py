"""Carr-Madan FFT pricing of Heston, an independent cross-check of COS (FR-16)."""

import math

import numpy as np
from scipy.interpolate import CubicSpline

from vsm.conventions import F64
from vsm.models.heston_cf import HestonParams, heston_cf

DAMPING = 1.5
GRID_POINTS = 2**18
ETA = 0.05


def heston_price_fft(p: HestonParams, forward: float, strikes: F64, t: float) -> F64:
    """Undiscounted call prices, damped by alpha = 1.5, on an FFT grid in log-strike."""
    n = GRID_POINTS
    v = ETA * np.arange(n)
    step = 2.0 * math.pi / (n * ETA)
    half_width = 0.5 * n * step
    alpha = DAMPING
    argument = v - (alpha + 1.0) * 1j
    psi = heston_cf(argument, t, p) / (alpha * alpha + alpha - v * v + 1j * (2.0 * alpha + 1.0) * v)
    simpson = (3.0 + (-1.0) ** (np.arange(n) + 1)) / 3.0
    simpson[0] = 1.0 / 3.0
    transformed = np.fft.fft(np.exp(1j * v * half_width) * psi * ETA * simpson)
    log_strikes = -half_width + step * np.arange(n)
    calls = np.exp(-alpha * log_strikes) / math.pi * np.real(transformed)
    window = np.abs(log_strikes) < 8.0
    spline = CubicSpline(log_strikes[window], calls[window])
    requested = np.log(np.asarray(strikes, dtype=np.float64) / forward)
    prices: F64 = forward * spline(requested)
    return prices

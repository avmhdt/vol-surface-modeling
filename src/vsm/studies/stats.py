"""Inference for the studies: Newey-West standard errors and the stationary bootstrap.

Both target the mean of a series whose terms may be serially correlated. Overlapping windows
make that correlation certain, so the lag count never falls below the overlap.
"""

import math
from typing import Any

import numpy as np

from vsm.conventions import F64


def auto_lags(n: int) -> int:
    """Newey and West's (1994) rule of thumb, floor(4 (n / 100)^(2/9))."""
    return math.floor(4.0 * math.pow(n / 100.0, 2.0 / 9.0))


def hac_lags(n: int, overlap: int = 0) -> int:
    """The rule of thumb, or the number of overlapping neighbours if that is larger."""
    return max(auto_lags(n), overlap)


def newey_west_se(values: F64, lags: int) -> float:
    """Standard error of the mean with Bartlett weights."""
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    centred = x - x.mean()
    variance = float(centred @ centred) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * float(centred[lag:] @ centred[:-lag]) / n
    return math.sqrt(max(variance, 0.0) / n)


def stationary_bootstrap_means(values: F64, draws: int, mean_block: float, seed: int) -> F64:
    """Means of ``draws`` resamples by Politis and Romano's (1994) stationary bootstrap.

    Blocks start at uniform positions and have geometric lengths with mean ``mean_block``,
    wrapping around the end of the sample.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    rng = np.random.Generator(np.random.PCG64(seed))
    restart = rng.random((draws, n)) < 1.0 / mean_block
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(draws, n))
    index = np.where(restart, starts, 0)
    for column in range(1, n):
        index[:, column] = np.where(
            restart[:, column], starts[:, column], (index[:, column - 1] + 1) % n
        )
    means: F64 = x[index].mean(axis=1)
    return means


def summarize(
    values: F64,
    *,
    overlap: int = 0,
    draws: int = 10_000,
    mean_block: float = 3.0,
    seed: int = 7,
) -> dict[str, Any]:
    """Mean, median, Newey-West t-stat, share positive and a 95% bootstrap interval."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        return {"n": n}
    lags = hac_lags(n, overlap)
    error = newey_west_se(x, lags)
    means = stationary_bootstrap_means(x, draws, mean_block, seed)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n": n,
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "nw_se": error,
        "nw_t": float(x.mean() / error) if error > 0.0 else math.nan,
        "nw_lags": lags,
        "frac_positive": float(np.mean(x > 0.0)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_draws": draws,
        "bootstrap_mean_block": mean_block,
    }

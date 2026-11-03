"""AC-W-10: the VRP and straddle machinery is unbiased on synthetic data.

GBM paths, hedged daily at zero cost. With implied equal to realized vol, the hedged short
straddle earns nothing on average. With implied 20% above realized, it earns the theoretical
gamma P&L. A flat surface's variance-swap rate is its variance.
"""

import math

import numpy as np
import pytest

from vsm.iv.black import black76_price
from vsm.studies.straddle import hedge_path
from vsm.studies.vrp import slice_variance, variance_swap_rate
from vsm.surface.ssvi import SSVI
from vsm.surface.svi import RawSVI
from vsm.surface.types import Surface, SurfaceSlice

DAY_NS = 86_400_000_000_000
PATHS = 2_000
DAYS = 30
HOURS = 24 * DAYS
SIGMA_REAL = 0.6


def short_straddles(sigma_imp: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(P&L, theoretical gamma P&L) of each path."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / (365.0 * 24.0)
    shocks = rng.standard_normal((PATHS, HOURS))
    logs = np.cumsum(-0.5 * SIGMA_REAL**2 * dt + SIGMA_REAL * math.sqrt(dt) * shocks, axis=1)
    hourly = 100.0 * np.exp(np.concatenate([np.zeros((PATHS, 1)), logs], axis=1))
    daily = hourly[:, ::24]
    times = np.arange(DAYS + 1, dtype=np.int64) * DAY_NS
    expiry = int(times[-1])
    t = DAYS / 365.0
    premium = float(
        black76_price(100.0, 100.0, t, sigma_imp, True)
        + black76_price(100.0, 100.0, t, sigma_imp, False)
    )
    pnl = np.empty(PATHS)
    theory = np.empty(PATHS)
    vols = np.full(DAYS, sigma_imp)
    funding = np.zeros(DAYS)
    for path in range(PATHS):
        result = hedge_path(times, daily[path], funding, vols, 100.0, expiry)
        pnl[path] = premium - abs(daily[path, -1] - 100.0) + result.hedge_pnl
        theory[path] = result.gamma_pnl
    return pnl, theory


def test_fair_implied_vol_earns_nothing() -> None:
    pnl, _ = short_straddles(SIGMA_REAL, 11)
    assert abs(pnl.mean()) <= 3.0 * pnl.std(ddof=1) / math.sqrt(PATHS)


def test_rich_implied_vol_earns_the_gamma_pnl() -> None:
    pnl, theory = short_straddles(1.2 * SIGMA_REAL, 12)
    assert theory.mean() > 0.0
    assert pnl.mean() == pytest.approx(theory.mean(), rel=0.05)


def flat_slice(t: float, variance: float) -> SurfaceSlice:
    svi = RawSVI(a=variance * t, b=0.0, rho=0.0, m=0.0, sigma=1.0)
    return SurfaceSlice(
        expiry_ns=int(t * 365.0 * DAY_NS),
        t=t,
        forward=100.0,
        forward_se=0.0,
        forward_method="synthetic",
        df=1.0,
        basis_ann=0.0,
        svi=svi,
        source="svi",
        k_min=-1.0,
        k_max=1.0,
        n_quotes=10,
        n_liquid=10,
        rmse_iv_bp=0.0,
        within_spread_frac=1.0,
        ssvi_within_spread_frac=1.0,
    )


def test_flat_surface_variance_swap_rate_is_its_variance() -> None:
    variance = 0.6**2
    slices = tuple(flat_slice(days / 365.0, variance) for days in (7, 21, 45, 90))
    ssvi = SSVI(0.0, 0.0, 0.5, tuple(s.t for s in slices), tuple(variance * s.t for s in slices))
    surface = Surface("BTC", "deribit", 0, "synthetic", 100.0, ssvi, slices)
    assert variance_swap_rate(surface, 30 / 365.0) == pytest.approx(variance, abs=1e-6)
    assert slice_variance(slices[0].svi, slices[0].t) == pytest.approx(variance * 7 / 365, abs=1e-8)
    assert math.isnan(variance_swap_rate(surface, 2 / 365.0))
    assert math.isnan(variance_swap_rate(surface, 200 / 365.0))

"""AC-W-06: pricers and diagnostics against known results."""

import numpy as np
import pytest
from scipy.integrate import quad

from vsm.iv.black import black76_price
from vsm.models.heston_cf import HestonParams, heston_cf
from vsm.models.heston_cos import heston_price_cos
from vsm.models.heston_fft import heston_price_fft
from vsm.models.sabr import sabr_hagan_vol

FANG_OOSTERLEE = HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
PUBLISHED_CALL = 5.785155450
CRYPTO = HestonParams(v0=0.3, kappa=2.0, theta=0.36, xi=1.2, rho=-0.2)
TENORS = (0.1, 0.25, 0.5, 1.0, 2.0)


def lewis_call(p: HestonParams, forward: float, strike: float, t: float) -> float:
    """Lewis's (2001) single integral, by adaptive quadrature: shares nothing with COS."""
    x = np.log(forward / strike)

    def integrand(u: float) -> float:
        phi = heston_cf(np.array([complex(u, -0.5)]), t, p)[0]
        return float((np.exp(1j * u * x) * phi).real / (u * u + 0.25))

    value, _ = quad(integrand, 0.0, 1000.0, limit=4000, epsabs=1e-12, epsrel=1e-12)
    return float(forward - np.sqrt(forward * strike) / np.pi * value)


def test_cos_matches_the_published_reference() -> None:
    assert float(heston_price_cos(FANG_OOSTERLEE, 100.0, 100.0, 1.0, True)) == pytest.approx(
        PUBLISHED_CALL, abs=1e-7
    )


def test_published_reference_matches_independent_quadrature() -> None:
    assert lewis_call(FANG_OOSTERLEE, 100.0, 100.0, 1.0) == pytest.approx(PUBLISHED_CALL, abs=1e-7)


@pytest.mark.parametrize(("p", "scale"), [(FANG_OOSTERLEE, 0.04), (CRYPTO, 0.36)])
def test_cos_matches_fft_on_the_grid(p: HestonParams, scale: float) -> None:
    worst = 0.0
    for t in TENORS:
        strikes = 100.0 * np.exp(np.linspace(-2.0, 2.0, 9) * np.sqrt(scale * t))
        calls = strikes >= 100.0
        cos = heston_price_cos(p, 100.0, strikes, t, calls)
        fft_calls = heston_price_fft(p, 100.0, strikes, t)
        fft = np.where(calls, fft_calls, fft_calls - 100.0 + strikes)
        worst = max(worst, float(np.max(np.abs(cos - fft) / cos)))
    assert worst <= 1e-6


def test_put_call_parity_holds_in_cos() -> None:
    strikes = np.array([70.0, 100.0, 140.0])
    calls = heston_price_cos(CRYPTO, 100.0, strikes, 0.5, True)
    puts = heston_price_cos(CRYPTO, 100.0, strikes, 0.5, False)
    assert calls - puts == pytest.approx(100.0 - strikes, abs=1e-9)


def test_wide_strike_on_a_short_expiry_is_priced() -> None:
    # The usual shifted-interval form cannot reach this strike at all.
    price = float(heston_price_cos(CRYPTO, 100.0, 300.0, 0.01, True))
    assert -1e-12 <= price < 1e-3


def test_cos_within_three_standard_errors_of_monte_carlo() -> None:
    p = CRYPTO
    t = 0.5
    paths, steps = 200_000, 200
    dt = t / steps
    rng = np.random.default_rng(20261102)
    log_s = np.zeros(paths)
    v = np.full(paths, p.v0)
    for _ in range(steps):
        z1 = rng.standard_normal(paths)
        z2 = p.rho * z1 + np.sqrt(1.0 - p.rho**2) * rng.standard_normal(paths)
        positive = np.maximum(v, 0.0)
        root = np.sqrt(positive * dt)
        log_s += -0.5 * positive * dt + root * z1
        v += p.kappa * (p.theta - positive) * dt + p.xi * root * z2
    for strike in (80.0, 100.0, 125.0):
        payoff = np.maximum(100.0 * np.exp(log_s) - strike, 0.0)
        estimate = float(payoff.mean())
        error = float(payoff.std(ddof=1)) / np.sqrt(paths)
        assert abs(float(heston_price_cos(p, 100.0, strike, t, True)) - estimate) <= 3.0 * error


def test_heston_with_no_vol_of_vol_is_black() -> None:
    p = HestonParams(v0=0.25, kappa=1.0, theta=0.25, xi=1e-4, rho=0.0)
    strikes = np.array([80.0, 100.0, 120.0])
    black = black76_price(100.0, strikes, 1.0, 0.5, True)
    assert heston_price_cos(p, 100.0, strikes, 1.0, True) == pytest.approx(black, rel=1e-6)


def test_sabr_without_vol_of_vol_is_black_at_beta_one() -> None:
    strikes = np.array([60.0, 90.0, 100.0, 110.0, 160.0])
    vol = sabr_hagan_vol(100.0, strikes, 0.5, 0.6, 1.0, -0.3, 1e-12)
    assert vol == pytest.approx(np.full(5, 0.6), abs=1e-10)


def test_sabr_atm_limit_at_other_betas() -> None:
    alpha = 6.0
    beta = 0.5
    expected = alpha * 100.0 ** (beta - 1.0)
    vol = float(sabr_hagan_vol(100.0, np.array(100.0), 1e-12, alpha, beta, 0.0, 1e-12))
    assert vol == pytest.approx(expected, abs=1e-10)

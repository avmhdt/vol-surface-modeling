import numpy as np
import pytest

from vsm.surface.svi import (
    RawSVI,
    default_grid,
    durrleman_g,
    fit_raw_svi,
    quasi_explicit,
    slice_g,
)

GOOD = RawSVI(a=0.02, b=0.1, rho=-0.3, m=0.05, sigma=0.2)
# Axel Vogt's example from Gatheral and Jacquier (2014): looks fine, has butterfly arbitrage.
VOGT = RawSVI(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
K = np.linspace(-1.5, 1.5, 61)


def test_derivatives_match_finite_differences() -> None:
    h = 1e-5
    assert GOOD.dw(K) == pytest.approx((GOOD.w(K + h) - GOOD.w(K - h)) / (2 * h), rel=1e-7)
    assert GOOD.d2w(K) == pytest.approx((GOOD.dw(K + h) - GOOD.dw(K - h)) / (2 * h), rel=1e-6)


def test_minimum_variance_is_where_the_formula_says() -> None:
    k = np.linspace(-2.0, 2.0, 200_001)
    root = GOOD.a + GOOD.b * GOOD.sigma * np.sqrt(1 - GOOD.rho**2)
    assert float(GOOD.w(k).min()) == pytest.approx(root, rel=1e-9)


def test_jump_wing_parameters() -> None:
    t = 0.25
    jw = GOOD.to_jw(t)
    w_atm = float(GOOD.w(np.array(0.0)))
    assert jw.v == pytest.approx(w_atm / t)
    h = 1e-6
    slope = float(GOOD.w(np.array(h)) - GOOD.w(np.array(-h))) / (2 * h)
    assert jw.psi == pytest.approx(slope / (2 * np.sqrt(w_atm)), rel=1e-8)
    assert jw.p == pytest.approx(GOOD.b * (1 - GOOD.rho) / np.sqrt(w_atm))
    assert jw.c == pytest.approx(GOOD.b * (1 + GOOD.rho) / np.sqrt(w_atm))
    assert jw.vtilde < jw.v


def test_g_is_positive_on_a_good_slice() -> None:
    assert float(slice_g(GOOD, K).min()) > 0.0


def test_g_detects_vogt_butterfly_arbitrage() -> None:
    assert float(slice_g(VOGT, np.linspace(-1.5, 1.5, 3001)).min()) < 0.0


def test_durrleman_g_on_a_flat_surface() -> None:
    w = np.full(5, 0.04)
    k = np.linspace(-1.0, 1.0, 5)
    assert durrleman_g(k, w, np.zeros(5), np.zeros(5)) == pytest.approx(np.ones(5))


def test_quasi_explicit_recovers_exact_data() -> None:
    start = quasi_explicit(K, GOOD.w(K), np.ones_like(K) / K.size)
    assert start.w(K) == pytest.approx(GOOD.w(K), rel=1e-4)


def test_fit_recovers_a_clean_slice() -> None:
    fit = fit_raw_svi(K, GOOD.w(K), np.ones_like(K))
    assert fit.converged
    assert fit.svi.w(K) == pytest.approx(GOOD.w(K), rel=1e-5)


def test_fit_repairs_vogt_arbitrage() -> None:
    fit = fit_raw_svi(K, VOGT.w(K), np.ones_like(K), grid=np.linspace(-1.5, 1.5, 601))
    assert fit.converged
    assert float(slice_g(fit.svi, np.linspace(-1.5, 1.5, 601)).min()) >= 0.0
    # Between grid points g can dip below the margin; the gate's denser grid catches that.
    assert float(slice_g(fit.svi, np.linspace(-1.5, 1.5, 30_001)).min()) >= -1e-5
    assert fit.rmse_w > 0.0


def test_unconstrained_fit_keeps_vogt_arbitrage() -> None:
    fit = fit_raw_svi(K, VOGT.w(K), np.ones_like(K), enforce_butterfly=False)
    assert float(slice_g(fit.svi, np.linspace(-1.5, 1.5, 3001)).min()) < 0.0


def test_lower_slice_is_a_floor() -> None:
    lower = RawSVI(a=0.03, b=0.1, rho=-0.3, m=0.05, sigma=0.2)
    grid = np.linspace(-1.5, 1.5, 61)
    fit = fit_raw_svi(K, GOOD.w(K), np.ones_like(K), lower_slice=lower, grid=grid)
    assert float((fit.svi.w(grid) - lower.w(grid)).min()) >= -1e-9


def test_lee_bound_holds_on_steep_data() -> None:
    steep = 0.01 + 3.0 * np.abs(K)
    fit = fit_raw_svi(K, steep, np.ones_like(K), enforce_butterfly=False)
    assert fit.svi.b * (1 + abs(fit.svi.rho)) <= 2.0 + 1e-6


def test_explicit_start_is_used() -> None:
    fit = fit_raw_svi(K, GOOD.w(K), np.ones_like(K), init=GOOD)
    assert fit.rmse_w < 1e-6


def test_default_grid_runs_past_the_quotes() -> None:
    k = np.array([-0.1, 0.2])
    grid = default_grid(k)
    assert grid.min() < -0.1
    assert grid.max() > 0.2

"""AC-W-07: the Heston analytic gradient is correct.

The COS price depends on the parameters through phi and, weakly, through the truncation
interval, which the gradient holds fixed on purpose. Finite differences hold it fixed too, so
both differentiate the same function.

At the spec's step of 1e-6 relative, the reference is limited by rounding in the prices it
subtracts, which is about ulp(price) / step. At extreme parameters on some builds that exceeds
1e-5 of a small sensitivity, so the comparison allows for that rounding. A step ten times longer
cuts the rounding tenfold, and there the 1e-5 limit applies with no allowance.
"""

import numpy as np
import pytest

from vsm.config import load_config
from vsm.conventions import F64
from vsm.models import heston_cos
from vsm.models.heston_cf import HestonParams, heston_cf, heston_cf_and_grad
from vsm.models.heston_cos import heston_price_and_grad_cos, heston_price_cos

NAMES = ("v0", "kappa", "theta", "xi", "rho")
SETS = 50
STRIKES = 100.0 * np.exp(np.linspace(-0.3, 0.3, 9))
CALLS = STRIKES >= 100.0
TOLERANCE = 1e-5
SPEC_STEP = 1e-6
ROUNDING_ULPS = 16.0


def random_parameters() -> list[tuple[F64, float]]:
    """Inside the configured bounds, away from their extreme corners."""
    rng = np.random.default_rng(7)
    low = np.array([1e-3, 0.05, 1e-3, 0.05, -0.95])
    high = np.array([4.0, 50.0, 4.0, 10.0, 0.95])
    return [(rng.uniform(low, high), float(rng.uniform(0.05, 2.0))) for _ in range(SETS)]


def central_difference(x: F64, t: float, j: int, relative_step: float) -> tuple[F64, F64]:
    """The difference, and the rounding bound on it, per strike."""
    step = relative_step * abs(x[j])
    up, down = x.copy(), x.copy()
    up[j] += step
    down[j] -= step
    high = heston_price_cos(HestonParams(*up), 100.0, STRIKES, t, CALLS)
    low = heston_price_cos(HestonParams(*down), 100.0, STRIKES, t, CALLS)
    rounding = ROUNDING_ULPS * np.spacing(np.maximum(np.abs(high), np.abs(low))) / (2.0 * step)
    return (high - low) / (2.0 * step), rounding


def test_defaults_match_the_config() -> None:
    cfg = load_config().heston
    assert cfg.cos_terms == heston_cos.DEFAULT_TERMS
    assert cfg.cos_truncation_L == heston_cos.DEFAULT_L


def worst_relative_error(
    j: int, relative_step: float, monkeypatch: pytest.MonkeyPatch
) -> tuple[float, float]:
    """Worst error over the sets: plain, and after the reference's rounding is taken off."""
    real = heston_cos._bounds
    plain = 0.0
    allowed = 0.0
    for x, t in random_parameters():
        p = HestonParams(*x)
        _, grad = heston_price_and_grad_cos(p, 100.0, STRIKES, t, CALLS)
        frozen = real(t, p, heston_cos.DEFAULT_L)
        monkeypatch.setattr(heston_cos, "_bounds", lambda *_, b=frozen: b)
        difference, rounding = central_difference(x, t, j, relative_step)
        monkeypatch.setattr(heston_cos, "_bounds", real)
        floor = np.maximum(np.abs(difference), 1e-6)
        error = np.abs(grad[:, j] - difference)
        plain = max(plain, float((error / floor).max()))
        allowed = max(allowed, float((np.maximum(error - rounding, 0.0) / floor).max()))
    return plain, allowed


@pytest.mark.parametrize("j", range(5), ids=NAMES)
def test_gradient_at_the_spec_step(j: int, monkeypatch: pytest.MonkeyPatch) -> None:
    assert worst_relative_error(j, SPEC_STEP, monkeypatch)[1] <= TOLERANCE


@pytest.mark.parametrize("j", range(5), ids=NAMES)
def test_gradient_at_a_longer_step_needs_no_allowance(
    j: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert worst_relative_error(j, 10.0 * SPEC_STEP, monkeypatch)[0] <= TOLERANCE


def test_cf_gradient_at_complex_arguments() -> None:
    p = HestonParams(0.3, 2.0, 0.36, 1.2, -0.2)
    u = np.linspace(0.1, 30.0, 40) - 1j
    x = p.as_array()
    _, grad = heston_cf_and_grad(u, 0.5, p)
    for j in range(5):
        step = 1e-6 * x[j]
        up, down = x.copy(), x.copy()
        up[j] += step
        down[j] -= step
        difference = (
            heston_cf(u, 0.5, HestonParams(*up)) - heston_cf(u, 0.5, HestonParams(*down))
        ) / (2.0 * step)
        assert grad[j] == pytest.approx(difference, rel=1e-5, abs=1e-8)


def test_cf_stays_accurate_with_small_xi_and_large_kappa() -> None:
    p = HestonParams(0.5, 40.0, 0.5, 0.01, 0.2)
    u = np.linspace(0.1, 3.0, 25) + 0j
    # With almost no vol of vol the variance stays at 0.5, so phi is close to Gaussian.
    expected = np.exp(-0.5 * 0.5 * 1.0 * (u * u + 1j * u))
    assert heston_cf(u, 1.0, p) == pytest.approx(expected, rel=1e-3)

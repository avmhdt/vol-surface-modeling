"""AC-W-07: the Heston analytic gradient is correct.

The COS price depends on the parameters through phi and, weakly, through the truncation
interval, which the gradient holds fixed on purpose. Finite differences hold it fixed too, so
both differentiate the same function.
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


def random_parameters() -> list[tuple[F64, float]]:
    """Inside the configured bounds, away from their extreme corners."""
    rng = np.random.default_rng(7)
    low = np.array([1e-3, 0.05, 1e-3, 0.05, -0.95])
    high = np.array([4.0, 50.0, 4.0, 10.0, 0.95])
    return [(rng.uniform(low, high), float(rng.uniform(0.05, 2.0))) for _ in range(SETS)]


def central_difference(x: F64, t: float, j: int) -> F64:
    step = 1e-6 * x[j]
    up, down = x.copy(), x.copy()
    up[j] += step
    down[j] -= step
    high = heston_price_cos(HestonParams(*up), 100.0, STRIKES, t, CALLS)
    low = heston_price_cos(HestonParams(*down), 100.0, STRIKES, t, CALLS)
    return (high - low) / (2.0 * step)


def test_defaults_match_the_config() -> None:
    cfg = load_config().heston
    assert cfg.cos_terms == heston_cos.DEFAULT_TERMS
    assert cfg.cos_truncation_L == heston_cos.DEFAULT_L


@pytest.mark.parametrize("j", range(5), ids=NAMES)
def test_gradient_matches_central_differences(j: int, monkeypatch: pytest.MonkeyPatch) -> None:
    real = heston_cos._bounds
    worst = 0.0
    for x, t in random_parameters():
        p = HestonParams(*x)
        _, grad = heston_price_and_grad_cos(p, 100.0, STRIKES, t, CALLS)
        frozen = real(t, p, heston_cos.DEFAULT_L)
        monkeypatch.setattr(heston_cos, "_bounds", lambda *_, b=frozen: b)
        difference = central_difference(x, t, j)
        monkeypatch.setattr(heston_cos, "_bounds", real)
        relative = np.abs(grad[:, j] - difference) / np.maximum(np.abs(difference), 1e-6)
        worst = max(worst, float(relative.max()))
    assert worst <= TOLERANCE


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

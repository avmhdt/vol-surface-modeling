"""Dupire local vol from SSVI against finite differences of the surface itself."""

import numpy as np
import pytest

from vsm.conventions import F64
from vsm.localvol.dupire import LOCAL_VOL_CAP, dupire_local_vol, flat_grid, local_vol_grid
from vsm.surface.ssvi import SSVI

SURFACE = SSVI(
    rho=-0.25,
    eta=1.1,
    gamma=0.4,
    t_nodes=(0.02, 0.1, 0.5, 1.0),
    theta_nodes=(0.008, 0.04, 0.2, 0.42),
)


def by_differences(ssvi: SSVI, k: F64, t: float) -> F64:
    """The spec's formula, with every derivative taken numerically."""
    step_k = 1e-4
    step_t = 1e-6
    time = np.full_like(k, t)
    w = ssvi.w(k, time)
    dw = (ssvi.w(k + step_k, time) - ssvi.w(k - step_k, time)) / (2.0 * step_k)
    d2w = (ssvi.w(k + step_k, time) - 2.0 * w + ssvi.w(k - step_k, time)) / step_k**2
    dt = (ssvi.w(k, time + step_t) - ssvi.w(k, time - step_t)) / (2.0 * step_t)
    ratio = k / w
    g = 1.0 - ratio * dw + 0.25 * (-0.25 - 1.0 / w + ratio * ratio) * dw * dw + 0.5 * d2w
    return np.sqrt(dt / g)


@pytest.mark.parametrize("t", [0.01, 0.05, 0.3, 0.75, 2.0])
def test_analytic_form_matches_finite_differences(t: float) -> None:
    k = np.linspace(-0.8, 0.8, 17)
    assert dupire_local_vol(SURFACE, k, np.array(t)) == pytest.approx(
        by_differences(SURFACE, k, t), rel=1e-5
    )


def test_slope_at_a_node_is_the_right_hand_one() -> None:
    at_node = float(dupire_local_vol(SURFACE, np.array(0.0), np.array(0.1)))
    just_after = float(dupire_local_vol(SURFACE, np.array(0.0), np.array(0.1 + 1e-9)))
    assert at_node == pytest.approx(just_after, rel=1e-6)


def test_single_node_surface() -> None:
    ssvi = SSVI(rho=0.0, eta=0.0, gamma=0.5, t_nodes=(0.5,), theta_nodes=(0.18,))
    vol = dupire_local_vol(ssvi, np.array([-0.5, 0.5]), np.array([0.1, 1.0]))
    assert vol == pytest.approx(np.full(2, 0.6), abs=1e-12)


def test_grid_shape_and_cap() -> None:
    k = np.linspace(-1.5, 1.5, 7)
    t = np.array([1e-6, 0.1, 0.5])
    grid = local_vol_grid(SURFACE, k, t)
    assert grid.vol.shape == (3, 7)
    assert np.all(np.isfinite(grid.vol))
    assert np.all(grid.vol > 0.0)
    assert grid.vol.max() <= LOCAL_VOL_CAP


def test_flat_grid_is_flat() -> None:
    grid = flat_grid(0.7)
    assert np.all(grid.vol == 0.7)
    assert grid.k[1] - grid.k[0] == grid.k[2] - grid.k[1]

"""AC-W-06 and FR-20: local vol and the Monte Carlo against closed forms.

The coverage job runs with Numba off, where a path costs thousands of times more, so sizes shrink
there. Three standard errors is then a looser test, which the JIT jobs make up for.
"""

import math
import os

import numpy as np
import pytest

from vsm.iv.black import black76_price
from vsm.localvol.barrier import BarrierSpec, black_barrier_price
from vsm.localvol.dupire import dupire_local_vol, flat_grid, local_vol_grid
from vsm.localvol.mc import pair_mean, price_barrier_mc, simulate
from vsm.models.heston_cf import HestonParams
from vsm.models.heston_cos import heston_price_cos
from vsm.surface.ssvi import SSVI

NO_JIT = os.environ.get("NUMBA_DISABLE_JIT") == "1"
PATHS = 2_000 if NO_JIT else 200_000
STEPS_PER_DAY = 2 if NO_JIT else 24
SEED = 20261103
DOWN = BarrierSpec("down_and_out_call", 1.0, 0.85, 30.0)
UP = BarrierSpec("up_and_out_call", 1.0, 1.20, 30.0)
SURFACE = SSVI(
    rho=-0.2,
    eta=0.9,
    gamma=0.4,
    t_nodes=(7 / 365, 30 / 365, 90 / 365),
    theta_nodes=(0.006, 0.027, 0.09),
)


def test_flat_surface_gives_flat_local_vol() -> None:
    flat = SSVI(rho=0.0, eta=0.0, gamma=0.5, t_nodes=(0.1, 1.0), theta_nodes=(0.04, 0.4))
    k, t = np.meshgrid(np.linspace(-1.5, 1.5, 61), np.array([0.02, 0.1, 0.5, 1.0, 2.0]))
    assert dupire_local_vol(flat, k, t) == pytest.approx(np.full(k.shape, math.sqrt(0.4)), abs=1e-8)


@pytest.mark.parametrize("spec", [DOWN, UP])
def test_flat_local_vol_matches_the_barrier_closed_form(spec: BarrierSpec) -> None:
    price, error = price_barrier_mc(flat_grid(0.6), spec, 100.0, PATHS, STEPS_PER_DAY, SEED)
    assert abs(price - black_barrier_price(spec, 100.0, 0.6)) <= 3.0 * error


def test_knock_out_is_cheaper_than_the_vanilla() -> None:
    vanilla = float(black76_price(100.0, 100.0, 30 / 365, 0.6, True))
    assert 0.0 < black_barrier_price(UP, 100.0, 0.6) < vanilla
    assert 0.0 < black_barrier_price(DOWN, 100.0, 0.6) < vanilla


def test_barrier_inside_the_strike_closed_form_cases() -> None:
    above = BarrierSpec("up_and_out_call", 1.25, 1.10, 30.0)
    assert black_barrier_price(above, 100.0, 0.6) == 0.0
    deep = BarrierSpec("down_and_out_call", 0.9, 0.95, 30.0)
    in_the_money = black_barrier_price(deep, 100.0, 0.6)
    price, error = price_barrier_mc(flat_grid(0.6), deep, 100.0, PATHS, STEPS_PER_DAY, SEED)
    assert abs(price - in_the_money) <= 3.0 * error


def test_local_vol_monte_carlo_reprices_its_surface() -> None:
    tenors = [14.0, 30.0, 60.0]
    dt = 1.0 / (365.0 * STEPS_PER_DAY)
    middle = (np.arange(round(60.0 * STEPS_PER_DAY)) + 0.5) * dt
    grid = local_vol_grid(SURFACE, np.linspace(-1.5, 1.5, 121), middle)
    result = simulate(grid, tenors, PATHS, STEPS_PER_DAY, SEED)
    for index, days in enumerate(tenors):
        t = days / 365.0
        terminal = np.exp(result.ends[index])
        for strike in (0.8, 0.9, 1.0, 1.1, 1.2):
            sign = 1.0 if strike >= 1.0 else -1.0
            w = float(SURFACE.w(np.array(math.log(strike)), np.array(t)))
            target = float(black76_price(1.0, strike, t, math.sqrt(w / t), strike >= 1.0))
            price, error = pair_mean(np.maximum(sign * (terminal - strike), 0.0))
            assert abs(price - target) <= 3.5 * error, (days, strike)


@pytest.mark.parametrize(
    "params",
    [
        HestonParams(v0=0.3, kappa=2.0, theta=0.36, xi=1.2, rho=-0.2),
        HestonParams(
            v0=0.1, kappa=1.0, theta=0.5, xi=2.5, rho=-0.6
        ),  # low variance, QE's hard case
    ],
)
def test_heston_quadratic_exponential_matches_cos(params: HestonParams) -> None:
    result = simulate(params, [60.0], PATHS, STEPS_PER_DAY, SEED)
    terminal = 100.0 * np.exp(result.ends[0])
    assert pair_mean(terminal - 100.0)[0] == pytest.approx(0.0, abs=4.0 * pair_mean(terminal)[1])
    for strike in (80.0, 100.0, 125.0):
        price, error = pair_mean(np.maximum(terminal - strike, 0.0))
        target = float(heston_price_cos(params, 100.0, strike, 60 / 365, True))
        assert abs(price - target) <= 3.0 * error, strike


def test_same_seed_same_paths_different_seed_different_paths() -> None:
    params = flat_grid(0.5)
    first = simulate(params, [5.0], 6_000, 2, SEED)
    second = simulate(params, [5.0], 6_000, 2, SEED)
    other = simulate(params, [5.0], 6_000, 2, SEED + 1)
    assert np.array_equal(first.ends, second.ends)
    assert not np.array_equal(first.ends, other.ends)


def test_paths_are_antithetic_and_block_sizes_do_not_matter_for_the_count() -> None:
    result = simulate(flat_grid(0.5), [1.0, 2.0], 5_001, 4, SEED)
    assert result.ends.shape == (2, 5_000)
    # Antithetic normals give a sum of exactly -sigma^2 t per pair under a flat vol.
    pair_sum = result.ends[1, :2].sum()
    assert pair_sum == pytest.approx(-0.25 * 2.0 / 365.0, abs=1e-12)


def test_horizons_must_increase() -> None:
    with pytest.raises(ValueError, match="increasing"):
        simulate(flat_grid(0.5), [2.0, 1.0], 100, 2, SEED)


def test_grid_with_coarse_times_is_interpolated() -> None:
    grid = local_vol_grid(SURFACE, np.linspace(-1.5, 1.5, 61), np.array([1 / 365, 40 / 365]))
    result = simulate(grid, [3.0], 2_000, 2, SEED)
    assert np.all(np.isfinite(result.ends))

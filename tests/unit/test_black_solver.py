import math

import numpy as np
import pytest
from scipy.special import erfcx as scipy_erfcx
from scipy.special import log_ndtr, ndtr

from vsm.iv import black, solver
from vsm.iv.black import black76_price, black76_vega
from vsm.iv.solver import implied_vol, implied_vol_detail


def textbook_call(forward: float, strike: float, total: float) -> float:
    d1 = math.log(forward / strike) / total + 0.5 * total
    return float(forward * ndtr(d1) - strike * ndtr(d1 - total))


@pytest.mark.parametrize("y", [0.0, 1e-8, 0.1, 0.46875, 0.47, 1.0, 3.9, 4.0, 4.1, 30.0, 1e4])
def test_erfcx_matches_scipy(y: float) -> None:
    assert black.erfcx(y) == pytest.approx(float(scipy_erfcx(y)), rel=2e-15)


@pytest.mark.parametrize("z", [-37.0, -8.0, -1.0, 0.0, 1.0, 8.0])
def test_norm_cdf_matches_scipy_in_the_tail(z: float) -> None:
    assert math.log(black.norm_cdf(z) + 1e-320) == pytest.approx(
        float(log_ndtr(z)), rel=1e-13, abs=1e-13
    )


@pytest.mark.parametrize(
    ("forward", "strike", "total"),
    [
        (100.0, 100.0, 0.2),
        (100.0, 120.0, 0.3),
        (100.0, 80.0, 0.3),
        (1.0, 3.0, 1.5),
        (5.0, 1.0, 2.0),
    ],
)
def test_call_matches_the_textbook_formula(forward: float, strike: float, total: float) -> None:
    price = float(black76_price(forward, strike, 1.0, total, True))
    assert price == pytest.approx(textbook_call(forward, strike, total), rel=1e-12)


def test_put_call_parity_and_discount() -> None:
    call = black76_price(100.0, 110.0, 0.5, 0.4, True, 0.9)
    put = black76_price(100.0, 110.0, 0.5, 0.4, False, 0.9)
    assert float(call - put) == pytest.approx(0.9 * (100.0 - 110.0), rel=1e-12)


def test_deep_wing_keeps_relative_precision() -> None:
    # The plain formula subtracts two equal numbers here and returns noise or zero.
    a, s = 3.0, 0.5
    h, t = -a / s, s / 2
    expected = (
        np.exp(-(h * h + t * t) / 2)
        * 0.5
        * (scipy_erfcx(-(h + t) / np.sqrt(2)) - scipy_erfcx(-(h - t) / np.sqrt(2)))
    )
    assert black.norm_otm_price(a, s) == pytest.approx(float(expected), rel=1e-13)


def test_zero_vol_or_time_is_intrinsic_and_bad_input_is_nan() -> None:
    assert float(black76_price(110.0, 100.0, 1.0, 0.0, True)) == 10.0
    assert float(black76_price(110.0, 100.0, 0.0, 0.5, False)) == 0.0
    assert math.isnan(float(black76_price(-1.0, 100.0, 1.0, 0.5, True)))
    assert math.isnan(float(black76_price(100.0, 100.0, 1.0, -0.5, True)))


def test_vega_matches_a_finite_difference() -> None:
    bump = 1e-6
    up = black76_price(100.0, 110.0, 0.5, 0.4 + bump, True)
    down = black76_price(100.0, 110.0, 0.5, 0.4 - bump, True)
    assert float(black76_vega(100.0, 110.0, 0.5, 0.4)) == pytest.approx(
        float(up - down) / (2 * bump), rel=1e-8
    )


def test_vega_edge_cases() -> None:
    assert float(black76_vega(100.0, 100.0, 1.0, 0.0)) == 0.0
    assert math.isnan(float(black76_vega(-1.0, 100.0, 1.0, 0.5)))


def test_arrays_broadcast_and_keep_shape() -> None:
    strikes = np.array([[90.0, 100.0], [110.0, 120.0]])
    price = black76_price(100.0, strikes, 1.0, 0.3, True)
    vol, status = implied_vol(price, 100.0, strikes, 1.0, True)
    assert price.shape == (2, 2)
    assert vol.shape == (2, 2)
    assert np.all(status == solver.OK)
    assert vol == pytest.approx(np.full((2, 2), 0.3), rel=1e-12)


@pytest.mark.parametrize(
    ("price", "is_call", "expected"),
    [
        (9.0, True, solver.BELOW_INTRINSIC),
        (101.0, True, solver.ABOVE_UPPER_BOUND),
        (100.0, True, solver.ABOVE_UPPER_BOUND),
        (91.0, False, solver.ABOVE_UPPER_BOUND),
        (float("nan"), True, solver.NO_CONVERGENCE),
    ],
)
def test_out_of_bounds_prices_return_nan_and_a_status(
    price: float, is_call: bool, expected: int
) -> None:
    vol, status = implied_vol(price, 100.0, 90.0, 1.0, is_call)
    assert math.isnan(float(vol))
    assert int(status) == expected


def test_price_at_intrinsic_is_zero_vol() -> None:
    vol, status = implied_vol(10.0, 100.0, 90.0, 1.0, True)
    assert float(vol) == 0.0
    assert int(status) == solver.OK


@pytest.mark.parametrize(
    ("forward", "strike", "t"), [(-1.0, 100.0, 1.0), (100.0, 0.0, 1.0), (100.0, 100.0, 0.0)]
)
def test_invalid_inputs_are_flagged(forward: float, strike: float, t: float) -> None:
    vol, status = implied_vol(5.0, forward, strike, t, True)
    assert math.isnan(float(vol))
    assert int(status) == solver.NO_CONVERGENCE


def test_discounted_price_round_trips() -> None:
    price = black76_price(100.0, 105.0, 0.25, 0.7, True, 0.97)
    vol, status = implied_vol(price, 100.0, 105.0, 0.25, True, 0.97)
    assert int(status) == solver.OK
    assert float(vol) == pytest.approx(0.7, rel=1e-13)


@pytest.mark.parametrize(("a", "s"), [(0.0, 0.3), (0.5, 0.2), (2.0, 0.6), (3.0, 2.5), (1.0, 8.0)])
def test_householder_converges_in_a_few_iterations(a: float, s: float) -> None:
    beta = black.norm_otm_price(a, s)
    solved, used = solver.solve_normalized(beta, a, 50)
    assert solved == pytest.approx(s, rel=1e-13)
    assert 0 < used <= 5


@pytest.mark.parametrize(("a", "s"), [(0.5, 0.2), (2.0, 0.6), (0.0, 0.3)])
def test_brent_fallback_finishes_a_stalled_solve(a: float, s: float) -> None:
    strike = math.exp(a)
    price = black76_price(1.0, strike, 1.0, s, True)
    vol, status, iterations = implied_vol_detail(price, 1.0, strike, 1.0, True, 1.0, 1)
    assert int(status) == solver.OK
    assert int(iterations) == 2  # the max_iter + 1 marker
    assert float(vol) == pytest.approx(s, rel=1e-12)


def test_brent_rejects_a_bracket_without_a_root() -> None:
    beta = black.norm_otm_price(1.0, 0.5)
    assert solver._brent(beta, 1.0, 1.0, 2.0)[1] == solver.NO_CONVERGENCE
    assert math.isnan(solver._brent(beta, 1.0, 1.0, 2.0)[0])


def test_brent_accepts_a_root_on_the_bracket() -> None:
    beta = black.norm_otm_price(1.0, 0.5)
    assert solver._brent(beta, 1.0, 0.5, 2.0) == (0.5, solver.OK)
    assert solver._brent(beta, 1.0, 0.1, 0.5) == (0.5, solver.OK)

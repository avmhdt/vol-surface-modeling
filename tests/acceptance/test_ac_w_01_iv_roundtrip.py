"""AC-W-01: price -> vol -> price to 1e-10 across moneyness, deep wings included.

The owner amended the vol round-trip on 2026-11-01. An ITM price is intrinsic plus a much
smaller time value, so the time value is known only to about one ulp of the whole price. Where
vega is tiny, that ulp alone moves vol by more than 1e-10, and no solver can recover what the
input no longer holds. ITM points now count only where 2 ulp(p) / vega <= 1e-10.
``test_literal_vol_round_trip_on_itm_points`` keeps the original wording visible as a strict
xfail, so the shortfall cannot disappear unnoticed.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vsm.iv.black import black76_price, black76_vega
from vsm.iv.solver import OK, implied_vol

FORWARD = 1.0
TOLERANCE = 1e-10
MIN_PRICE = 1e-12
MIN_VEGA = 1e-8

K_GRID = np.linspace(-3.0, 3.0, 121)
VOL_GRID = np.logspace(np.log10(0.005), np.log10(3.0), 60)


def grid(is_call: bool) -> dict[str, np.ndarray]:
    k, vol = np.meshgrid(K_GRID, VOL_GRID, indexing="ij")
    strike = FORWARD * np.exp(k.ravel())
    vol = vol.ravel()
    calls = np.full(strike.size, is_call)
    price = black76_price(FORWARD, strike, 1.0, vol, calls)
    solved, status = implied_vol(price, FORWARD, strike, 1.0, calls)
    otm = strike >= FORWARD if is_call else strike < FORWARD
    return {
        "strike": strike,
        "vol": vol,
        "calls": calls,
        "price": price,
        "solved": solved,
        "status": status,
        "repriced": black76_price(FORWARD, strike, 1.0, solved, calls),
        "vega": black76_vega(FORWARD, strike, 1.0, vol),
        "otm": otm,
    }


@pytest.fixture(scope="module", params=[True, False], ids=["call", "put"])
def points(request: pytest.FixtureRequest) -> dict[str, np.ndarray]:
    return grid(request.param)


def test_grid_is_the_specified_size(points: dict[str, np.ndarray]) -> None:
    assert points["strike"].size == 121 * 60


def test_price_round_trip_at_every_point(points: dict[str, np.ndarray]) -> None:
    inside = points["price"] / FORWARD >= MIN_PRICE
    assert inside.sum() > 5_000
    assert np.all(points["status"][inside] == OK)
    relative = (
        np.abs(points["repriced"][inside] - points["price"][inside]) / points["price"][inside]
    )
    assert relative.max() <= TOLERANCE


def test_vol_round_trip_at_every_otm_point(points: dict[str, np.ndarray]) -> None:
    inside = points["otm"] & (points["vega"] / FORWARD >= MIN_VEGA)
    assert inside.sum() > 1_500
    assert np.abs(points["solved"][inside] - points["vol"][inside]).max() <= TOLERANCE


def test_vol_round_trip_on_itm_points_the_input_can_resolve(
    points: dict[str, np.ndarray],
) -> None:
    inside = ~points["otm"] & (points["vega"] / FORWARD >= MIN_VEGA)
    resolvable = np.zeros_like(inside)
    resolvable[inside] = (
        2.0 * np.spacing(points["price"][inside]) / points["vega"][inside] <= TOLERANCE
    )
    assert resolvable.sum() > 1_000
    assert np.abs(points["solved"][resolvable] - points["vol"][resolvable]).max() <= TOLERANCE


def test_itm_misses_stay_within_the_rounding_of_the_input(
    points: dict[str, np.ndarray],
) -> None:
    inside = ~points["otm"] & (points["vega"] / FORWARD >= MIN_VEGA)
    limit = 4.0 * np.spacing(points["price"][inside]) / points["vega"][inside]
    assert np.all(np.abs(points["solved"][inside] - points["vol"][inside]) <= limit + 1e-13)


def test_same_itm_points_round_trip_from_the_otm_price(points: dict[str, np.ndarray]) -> None:
    inside = ~points["otm"] & (points["vega"] / FORWARD >= MIN_VEGA)
    strike = points["strike"][inside]
    other = ~points["calls"][inside]
    price = black76_price(FORWARD, strike, 1.0, points["vol"][inside], other)
    solved, status = implied_vol(price, FORWARD, strike, 1.0, other)
    assert np.all(status == OK)
    assert np.abs(solved - points["vol"][inside]).max() <= 1e-13


@pytest.mark.xfail(
    strict=True,
    reason="Original wording: double rounding of the ITM input; amended, see docs/decisions.md",
)
def test_literal_vol_round_trip_on_itm_points(points: dict[str, np.ndarray]) -> None:
    inside = ~points["otm"] & (points["vega"] / FORWARD >= MIN_VEGA)
    assert np.abs(points["solved"][inside] - points["vol"][inside]).max() <= TOLERANCE


@settings(max_examples=10_000, deadline=None)
@given(
    k=st.floats(-3.0, 3.0),
    total_vol=st.floats(0.005, 3.0),
    is_call=st.booleans(),
)
def test_random_cases_round_trip_in_price(k: float, total_vol: float, is_call: bool) -> None:
    strike = FORWARD * np.exp(k)
    price = black76_price(FORWARD, strike, 1.0, total_vol, is_call)
    if price / FORWARD < MIN_PRICE:
        return
    solved, status = implied_vol(price, FORWARD, strike, 1.0, is_call)
    assert status == OK
    repriced = black76_price(FORWARD, strike, 1.0, solved, is_call)
    assert abs(repriced - price) / price <= TOLERANCE


@settings(max_examples=2_000, deadline=None)
@given(
    k=st.floats(-3.0, 3.0),
    total_vol=st.floats(0.005, 3.0),
    is_call=st.booleans(),
    fraction=st.floats(0.5, 0.999999),
)
def test_prices_below_intrinsic_or_above_the_bound_are_flagged(
    k: float, total_vol: float, is_call: bool, fraction: float
) -> None:
    strike = FORWARD * np.exp(k)
    intrinsic = max(FORWARD - strike, 0.0) if is_call else max(strike - FORWARD, 0.0)
    upper = FORWARD if is_call else strike
    if intrinsic > 0.0:
        assert implied_vol(intrinsic * fraction, FORWARD, strike, 1.0, is_call)[1] != OK
    assert implied_vol(upper / fraction, FORWARD, strike, 1.0, is_call)[1] != OK

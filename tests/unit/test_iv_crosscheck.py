"""Independent reference: py_vollib's Let's Be Rational, on 10,000 random points."""

import numpy as np
import pytest
from py_vollib.black import black as vollib_black
from py_vollib.black.implied_volatility import implied_volatility

from vsm.iv.black import black76_price
from vsm.iv.solver import OK, implied_vol

N = 10_000


@pytest.fixture(scope="module")
def points() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261102)
    moneyness = rng.uniform(-1.5, 1.5, N)
    total_vol = np.exp(rng.uniform(np.log(0.05), np.log(2.0), N))
    return {
        "strike": 100.0 * np.exp(moneyness),
        "t": rng.uniform(0.02, 2.0, N),
        "total": total_vol,
        "is_call": moneyness >= 0.0,  # OTM, the well-conditioned side
    }


def test_prices_agree_with_py_vollib(points: dict[str, np.ndarray]) -> None:
    vol = points["total"] / np.sqrt(points["t"])
    ours = black76_price(100.0, points["strike"], points["t"], vol, points["is_call"])
    for i in range(0, N, 10):
        flag = "c" if points["is_call"][i] else "p"
        theirs = vollib_black(flag, 100.0, points["strike"][i], points["t"][i], 0.0, vol[i])
        assert ours[i] == pytest.approx(theirs, rel=1e-10, abs=1e-13)


def test_vols_agree_with_py_vollib_to_1e_8(points: dict[str, np.ndarray]) -> None:
    vol = points["total"] / np.sqrt(points["t"])
    prices = black76_price(100.0, points["strike"], points["t"], vol, points["is_call"])
    ours, status = implied_vol(prices, 100.0, points["strike"], points["t"], points["is_call"])
    compared = 0
    for i in range(N):
        if status[i] != OK:
            continue
        flag = "c" if points["is_call"][i] else "p"
        try:
            theirs = implied_volatility(
                prices[i], 100.0, points["strike"][i], 0.0, points["t"][i], flag
            )
        except Exception:
            continue  # only compare where both converge
        compared += 1
        assert abs(ours[i] - theirs) <= 1e-8
    assert compared > 0.9 * N

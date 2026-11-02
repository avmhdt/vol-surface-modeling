import numpy as np
import polars as pl
import pytest

from vsm.config import HestonConfig
from vsm.iv.black import black76_vega
from vsm.iv.solver import implied_vol
from vsm.models.heston_calib import MAX_QUOTES, calibrate_heston, select_quotes
from vsm.models.heston_cf import HestonParams, cumulants, heston_cf
from vsm.models.heston_cos import heston_price_cos
from vsm.models.sabr import calibrate_sabr, sabr_hagan_vol

TRUE = HestonParams(v0=0.32, kappa=3.0, theta=0.42, xi=1.4, rho=-0.35)
INIT = HestonParams(v0=0.36, kappa=2.0, theta=0.36, xi=1.0, rho=-0.2)


def heston_quotes(p: HestonParams) -> pl.DataFrame:
    rows = []
    for t in (0.05, 0.1, 0.25, 0.5, 1.0):
        forward = 100.0
        k = np.linspace(-0.3, 0.3, 13) * np.sqrt(t) * 2.0
        strike = forward * np.exp(k)
        is_call = strike >= forward
        price = heston_price_cos(p, forward, strike, t, is_call)
        iv, _ = implied_vol(price, forward, strike, t, is_call)
        vega = black76_vega(forward, strike, t, iv)
        for i in range(k.size):
            rows.append(
                {
                    "expiry_ns": int(t * 1e9),
                    "forward": forward,
                    "strike": float(strike[i]),
                    "t": t,
                    "cp": 1 if is_call[i] else -1,
                    "mid_fwd": float(price[i]),
                    "vega_fwd": float(vega[i]),
                    "weight": 20.0,
                    "target_iv": float(iv[i]),
                    "bid_iv": float(iv[i]) - 0.01,
                    "ask_iv": float(iv[i]) + 0.01,
                    "is_liquid": True,
                }
            )
    return pl.DataFrame(rows)


@pytest.mark.parametrize("optimizer", ["lm", "trf"])
def test_calibration_recovers_known_parameters(optimizer: str) -> None:
    cfg = HestonConfig(optimizer=optimizer)  # type: ignore[arg-type]
    fit = calibrate_heston(heston_quotes(TRUE), INIT, cfg)
    assert fit.converged
    assert fit.params.as_array() == pytest.approx(TRUE.as_array(), rel=1e-3)
    assert fit.rmse_iv_bp < 0.5
    assert fit.within_spread_frac == 1.0
    assert fit.evaluations < 100
    assert fit.message


def test_lm_stays_inside_the_bounds() -> None:
    cfg = HestonConfig(optimizer="lm", bounds={**HestonConfig().bounds, "xi": (0.001, 0.5)})
    fit = calibrate_heston(heston_quotes(TRUE), HestonParams(0.3, 2.0, 0.3, 0.4, -0.2), cfg)
    assert fit.params.xi <= 0.5


def test_select_quotes_keeps_liquid_and_limits_the_count() -> None:
    quotes = heston_quotes(TRUE)
    many = pl.concat([quotes] * 8)
    assert select_quotes(many).height == MAX_QUOTES
    mixed = quotes.with_columns((pl.col("strike") > 100.0).alias("is_liquid"))
    assert select_quotes(mixed)["strike"].min() > 100.0
    assert select_quotes(quotes, min_days=40.0)["t"].min() >= 0.1


def test_cf_is_a_martingale_and_starts_at_one() -> None:
    assert heston_cf(np.array([0.0, -1j]), 0.7, TRUE) == pytest.approx(np.array([1.0, 1.0]))


def test_second_cumulant_is_the_curvature_of_the_cf() -> None:
    h = 1e-3
    log_phi = np.log(heston_cf(np.array([h + 0j]), 0.5, TRUE))
    assert cumulants(0.5, TRUE)[1] == pytest.approx(-2.0 * log_phi.real[0] / h**2, rel=1e-2)


def test_sabr_is_continuous_through_the_money() -> None:
    near = sabr_hagan_vol(
        100.0, np.array([99.9999999, 100.0, 100.0000001]), 0.5, 0.6, 1.0, -0.3, 1.0
    )
    assert near == pytest.approx(np.full(3, near[1]), abs=1e-7)


def test_sabr_skew_follows_rho() -> None:
    strikes = np.array([80.0, 120.0])
    low, high = sabr_hagan_vol(100.0, strikes, 0.5, 0.6, 1.0, -0.6, 1.0)
    assert low > high


def test_obloj_agrees_at_the_money_and_differs_in_the_wings() -> None:
    args = (0.5, 6.0, 0.5, -0.3, 1.0)
    atm = np.array([100.0])
    assert sabr_hagan_vol(100.0, atm, *args, obloj=True) == pytest.approx(
        sabr_hagan_vol(100.0, atm, *args), rel=1e-12
    )
    wing = np.array([50.0])
    assert sabr_hagan_vol(100.0, wing, *args, obloj=True) != pytest.approx(
        sabr_hagan_vol(100.0, wing, *args), rel=1e-4
    )


def test_obloj_and_hagan_coincide_at_beta_one() -> None:
    args = (0.5, 0.6, 1.0, -0.3, 1.0)
    wing = np.array([50.0, 150.0])
    assert sabr_hagan_vol(100.0, wing, *args, obloj=True) == pytest.approx(
        sabr_hagan_vol(100.0, wing, *args), rel=1e-12
    )


def sabr_quotes(alpha: float, beta: float, rho: float, nu: float) -> pl.DataFrame:
    strike = 100.0 * np.exp(np.linspace(-0.4, 0.4, 15))
    iv = sabr_hagan_vol(100.0, strike, 0.25, alpha, beta, rho, nu)
    return pl.DataFrame(
        {
            "forward": 100.0,
            "t": 0.25,
            "strike": strike,
            "target_iv": iv,
            "weight": 10.0,
            "bid_iv": iv - 0.005,
            "ask_iv": iv + 0.005,
            "is_liquid": True,
        }
    )


@pytest.mark.parametrize(("beta", "alpha"), [(1.0, 0.6), (0.5, 6.0)])
def test_sabr_calibration_recovers_parameters(beta: float, alpha: float) -> None:
    fit = calibrate_sabr(sabr_quotes(alpha, beta, -0.3, 1.2), beta)
    assert fit.converged
    assert fit.alpha == pytest.approx(alpha, rel=1e-4)
    assert fit.rho == pytest.approx(-0.3, abs=1e-3)
    assert fit.nu == pytest.approx(1.2, rel=1e-3)
    assert fit.within_spread_frac == 1.0


def test_sabr_with_no_liquid_quotes_reports_nan() -> None:
    quotes = sabr_quotes(0.6, 1.0, -0.3, 1.2).with_columns(pl.lit(False).alias("is_liquid"))
    assert np.isnan(calibrate_sabr(quotes, 1.0).within_spread_frac)

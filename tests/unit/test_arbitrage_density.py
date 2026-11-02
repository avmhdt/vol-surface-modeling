import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from vsm.config import ArbitrageConfig
from vsm.surface.arbitrage import (
    _prices_ok,
    calendar_margin,
    check_grid,
    check_slice,
    check_surface,
)
from vsm.surface.density import bl_density, density_k, moments, risk_reversal_25
from vsm.surface.spline_baseline import evaluate, spline_slices
from vsm.surface.ssvi import SSVI
from vsm.surface.stability import stability_metrics, within_spread_metrics
from vsm.surface.svi import RawSVI
from vsm.surface.types import Surface, SurfaceSlice

CFG = ArbitrageConfig()
QUOTED = np.array([-0.4, 0.4])
GOOD = RawSVI(a=0.02, b=0.1, rho=-0.3, m=0.05, sigma=0.2)
VOGT = RawSVI(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
# Raw SVI with b tiny and sigma large is flat in total variance over any finite range.
FLAT_W = 0.04
FLAT = RawSVI(a=FLAT_W - 1e-9 * 1e6, b=1e-9, rho=0.0, m=0.0, sigma=1e6)


def surface_of(*slices: RawSVI) -> Surface:
    ssvi = SSVI(-0.2, 1.0, 0.4, (0.1, 0.2), (0.02, 0.04))
    items = tuple(
        SurfaceSlice(
            expiry_ns=i,
            t=0.1 * (i + 1),
            forward=100.0,
            forward_se=0.1,
            forward_method="parity_wls",
            df=1.0,
            basis_ann=0.0,
            svi=svi,
            source="svi",
            k_min=-0.4,
            k_max=0.4,
            n_quotes=10,
            n_liquid=8,
            rmse_iv_bp=10.0,
            within_spread_frac=1.0,
            ssvi_within_spread_frac=0.5,
        )
        for i, svi in enumerate(slices)
    )
    return Surface("BTC", "deribit", 0, "deribit-BTC-20261102T0815Z", 100.0, ssvi, items)


def test_grid_has_the_spec_points_and_the_quotes() -> None:
    k = check_grid(GOOD, np.array([-3.0]), CFG)
    assert k.size == CFG.grid_z.n + 1
    assert k.min() == -3.0


def test_grid_covers_the_whole_quoted_range() -> None:
    k = check_grid(GOOD, np.array([-3.0, 3.0]), CFG)
    assert np.max(np.diff(k)) <= 6.0 / (CFG.grid_z.n - 1) + 1e-12


def test_gate_catches_arbitrage_inside_the_quotes_past_z_6() -> None:
    theta = float(VOGT.w(np.array(0.0)))
    edge = 6.0 * np.sqrt(theta)
    assert edge < 0.9  # the dip is near k = 0.92, past the z grid and between the end points
    repaired = RawSVI(a=-0.0503, b=0.13403, rho=0.26289, m=0.31710, sigma=0.5050)
    assert not check_slice(VOGT, np.array([-1.5, 1.5]), CFG).passed
    assert check_slice(VOGT, np.array([-1.5, 1.5]), CFG).min_g < -CFG.tol_g
    assert check_slice(repaired, np.array([-0.5, 0.5]), CFG).min_g > 0.0


def test_price_check_allows_rounding_between_close_strikes() -> None:
    # A real surface, inside the SSVI conditions, whose strikes are 7e-8 apart at the low end.
    surface = SSVI(0.625, 1.0, 0.5, (0.5, 1.0), (1.6875, 4.5))
    assert check_slice(surface.slice_at(1.0), np.empty(0), CFG).price_ok


def test_price_check_still_catches_a_real_violation() -> None:
    assert not _prices_ok(VOGT, np.linspace(0.7, 1.3, 601), CFG.tol_price)


def test_good_slice_passes() -> None:
    check = check_slice(GOOD, QUOTED, CFG)
    assert check.passed
    assert check.min_g > 0.0
    assert check.density_mass_err < 1e-3
    assert check.price_ok
    assert check.min_calendar_dw is None


def test_vogt_slice_fails_on_g() -> None:
    check = check_slice(VOGT, QUOTED, CFG)
    assert check.min_g < -CFG.tol_g
    assert not check.passed


def test_negative_variance_fails_outright() -> None:
    check = check_slice(RawSVI(-0.5, 0.1, 0.0, 0.0, 0.1), QUOTED, CFG)
    assert not check.passed
    assert check.min_calendar_dw is None


def test_calendar_inversion_fails() -> None:
    higher = RawSVI(a=0.05, b=0.1, rho=-0.3, m=0.05, sigma=0.2)
    assert calendar_margin(GOOD, higher, QUOTED, QUOTED, CFG) > 0.0
    assert not check_slice(GOOD, QUOTED, CFG, lower=higher, lower_quoted=QUOTED).passed
    assert check_slice(higher, QUOTED, CFG, lower=GOOD, lower_quoted=QUOTED).passed


def test_check_surface_reports_the_worst_case() -> None:
    higher = RawSVI(a=0.05, b=0.1, rho=-0.3, m=0.05, sigma=0.2)
    report = check_surface(surface_of(GOOD, higher), CFG)
    assert report.passed
    assert len(report.slices) == 2
    assert len(report.ssvi_slices) == 2
    assert not check_surface(surface_of(higher, GOOD), CFG).passed


def test_empty_surface_does_not_pass() -> None:
    assert not check_surface(surface_of(), CFG).passed


def test_density_integrates_to_one_with_the_right_mean() -> None:
    stats = moments(GOOD)
    assert stats["mass"] == pytest.approx(1.0, abs=1e-6)
    assert stats["mean_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert stats["skewness"] < 0.0


def test_flat_surface_density_is_lognormal() -> None:
    k = np.linspace(-1.0, 1.0, 201)
    root = np.sqrt(FLAT_W)
    expected = norm.pdf((k + 0.5 * FLAT_W) / root) / root
    assert density_k(k, FLAT) == pytest.approx(expected, abs=1e-6)


def test_density_in_strike_is_density_in_k_over_k() -> None:
    k = np.array([-0.2, 0.0, 0.3])
    forward = 100.0
    assert bl_density(k, GOOD, forward) == pytest.approx(density_k(k, GOOD) / (forward * np.exp(k)))


def test_flat_surface_has_no_risk_reversal() -> None:
    assert risk_reversal_25(FLAT, 1.0) == pytest.approx(0.0, abs=1e-9)


def test_negative_skew_gives_a_negative_risk_reversal() -> None:
    assert risk_reversal_25(GOOD, 0.25) < 0.0


def quotes_from(vols: np.ndarray, k: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "drop_reason": [None] * k.size,
            "is_otm": [True] * k.size,
            "expiry_ns": [1] * k.size,
            "k": k,
            "t": [0.1] * k.size,
            "mid_iv": vols,
        },
        schema_overrides={"drop_reason": pl.String},
    )


def test_spline_on_a_smooth_smile_has_no_violations() -> None:
    k = np.linspace(-0.4, 0.4, 15)
    assert evaluate(quotes_from(np.sqrt(GOOD.w(k) / 0.1), k))["butterfly_violations"] == 0


def test_spline_on_a_noisy_smile_breaks_butterfly() -> None:
    k = np.linspace(-0.4, 0.4, 15)
    noise = np.array([0.0, 0.02] * 8)[: k.size]
    assert evaluate(quotes_from(np.sqrt(GOOD.w(k) / 0.1) + noise, k))["butterfly_violations"] == 1


def test_spline_detects_calendar_inversion() -> None:
    k = np.linspace(-0.4, 0.4, 15)
    near = quotes_from(np.full(k.size, 0.9), k)
    far = quotes_from(np.full(k.size, 0.5), k).with_columns(
        pl.lit(2, dtype=pl.Int64).alias("expiry_ns"), pl.lit(0.2).alias("t")
    )
    assert evaluate(pl.concat([near, far]))["calendar_violations"] == 1


def test_spline_skips_thin_slices() -> None:
    k = np.array([-0.1, 0.0, 0.1])
    assert spline_slices(quotes_from(np.full(3, 0.5), k)) == []


def params_of(days: int) -> pl.DataFrame:
    rows = []
    for day in range(days):
        for t, theta in ((0.05, 0.02), (0.1, 0.04)):
            rows.append(
                {
                    "as_of_ns": day,
                    "expiry_ns": 100,
                    "t": t,
                    "ssvi_rho": -0.2 + 0.01 * day,
                    "ssvi_eta": 1.0 + 0.01 * day,
                    "ssvi_gamma": 0.4,
                    "ssvi_theta": theta,
                    "slice_source": "svi",
                    "jw_v": 0.5 + 0.01 * day,
                    "jw_psi": -0.1,
                    "jw_p": 0.4,
                    "jw_c": 0.3,
                    "jw_vtilde": 0.45,
                    "n_liquid": 10,
                    "within_spread_frac": 0.9,
                    "ssvi_within_spread_frac": 0.6,
                }
            )
    return pl.DataFrame(rows)


def test_stability_needs_two_days() -> None:
    assert stability_metrics(params_of(1))["p95_abs_d_rho"] is None


def test_stability_measures_daily_changes() -> None:
    metrics = stability_metrics(params_of(5))
    assert metrics["days"] == 5
    assert metrics["p95_abs_d_rho"] == pytest.approx(0.01)
    assert metrics["jw_median_abs"]["jw_v"] == pytest.approx(0.0, abs=0.0101)


def test_within_spread_pools_by_liquid_count() -> None:
    metrics = within_spread_metrics(params_of(3), "BTC")
    assert metrics["days"] == 3
    assert metrics["n_liquid"] == 60
    assert metrics["pooled_svi"] == pytest.approx(0.9)
    assert metrics["pooled_ssvi"] == pytest.approx(0.6)
    assert metrics["median_day_svi"] == pytest.approx(0.9)


def test_within_spread_with_nothing_stored() -> None:
    empty = params_of(1).filter(pl.col("as_of_ns") < 0)
    assert within_spread_metrics(empty, "BTC")["days"] == 0

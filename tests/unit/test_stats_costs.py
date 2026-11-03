"""Inference helpers and the fee table."""

from datetime import date

import numpy as np
import pytest

from vsm.studies.costs import load_fees, option_fee_coin
from vsm.studies.stats import (
    auto_lags,
    hac_lags,
    newey_west_se,
    stationary_bootstrap_means,
    summarize,
)


def test_newey_west_without_lags_is_the_usual_standard_error() -> None:
    x = np.random.default_rng(1).standard_normal(500)
    assert newey_west_se(x, 0) == pytest.approx(x.std(ddof=0) / np.sqrt(500), rel=1e-12)


def test_newey_west_widens_for_overlapping_sums() -> None:
    noise = np.random.default_rng(2).standard_normal(2_000)
    overlapping = np.convolve(noise, np.ones(10), mode="valid")
    assert newey_west_se(overlapping, 9) > 2.0 * newey_west_se(overlapping, 0)


def test_lag_rules() -> None:
    assert auto_lags(100) == 4
    assert hac_lags(100, overlap=29) == 29
    assert hac_lags(100, overlap=0) == 4


def test_bootstrap_is_seeded_and_centred() -> None:
    x = np.random.default_rng(3).standard_normal(200) + 1.0
    first = stationary_bootstrap_means(x, 2_000, 3.0, 7)
    assert np.array_equal(first, stationary_bootstrap_means(x, 2_000, 3.0, 7))
    assert first.mean() == pytest.approx(x.mean(), abs=0.02)


def test_summary_of_a_short_or_normal_series() -> None:
    assert summarize(np.array([1.0])) == {"n": 1}
    x = np.random.default_rng(4).standard_normal(300) + 0.5
    result = summarize(x, draws=1_000)
    assert result["ci95_low"] < result["mean"] < result["ci95_high"]
    assert result["nw_t"] > 3.0
    assert 0.0 < result["frac_positive"] < 1.0


def test_fee_table_has_every_period_in_order() -> None:
    fees = load_fees("configs/fees/deribit.yaml")
    assert fees.source.startswith("https://support.deribit.com/")
    assert fees.on(date(2020, 3, 1)).option_taker == 0.0004
    assert fees.on(date(2020, 8, 24)).option_taker == 0.0003
    assert fees.on(date(2026, 7, 31)).perpetual_taker["BTC"] == 0.0005
    assert fees.on(date(2026, 8, 1)).perpetual_taker["ETH"] == 0.00035
    with pytest.raises(ValueError, match="no fee schedule"):
        fees.on(date(2019, 1, 1))


def test_option_fee_is_capped_at_a_share_of_premium() -> None:
    assert option_fee_coin(0.05, 0.0003, 0.125) == 0.0003
    assert option_fee_coin(0.0001, 0.0003, 0.125) == pytest.approx(0.0000125)

"""AC-W-05: cleaning and forwards are correct and point-in-time."""

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fixtures.make_fixtures import (
    AS_OF_NS,
    ATM_VOL,
    SPOT,
    build_chain,
    build_futures,
    expiry_of,
    forward_of,
    tick_table,
)
from vsm.clean import clean_snapshot
from vsm.config import AppConfig, ForwardsConfig, load_config
from vsm.conventions import NS_PER_S, datetime_to_ns
from vsm.forwards import huber_wls, implied_forwards


def tagged(chain: pl.DataFrame) -> pl.DataFrame:
    return chain.with_columns(
        pl.lit(AS_OF_NS, dtype=pl.Int64).alias("as_of_ns"),
        pl.lit(None, dtype=pl.String).alias("drop_reason"),
    )


@pytest.fixture
def config() -> AppConfig:
    return load_config()


def test_synthetic_parity_recovers_the_forward_to_1e_10() -> None:
    estimates = implied_forwards(
        tagged(build_chain(noise=False)), build_futures(), SPOT, ForwardsConfig()
    )
    assert len(estimates) == len(ATM_VOL)
    for days, estimate in zip(ATM_VOL, estimates, strict=True):
        assert estimate.method == "parity_wls"
        assert estimate.expiry_ns == datetime_to_ns(expiry_of(days))
        assert abs(estimate.forward / forward_of(days) - 1.0) <= 1e-10
        assert estimate.df == pytest.approx(1.0, abs=1e-10)
        expected_basis = math.log(forward_of(days) / SPOT) / estimate.t
        assert estimate.basis_ann == pytest.approx(expected_basis, abs=1e-9)


def test_noisy_chain_stays_close_and_reports_a_standard_error() -> None:
    estimates = implied_forwards(tagged(build_chain()), build_futures(), SPOT, ForwardsConfig())
    for days, estimate in zip(ATM_VOL, estimates, strict=True):
        if days >= 7:
            assert abs(math.log(estimate.forward / forward_of(days))) * 1e4 < 15.0
            assert estimate.forward_se > 0.0


def test_huber_resists_an_outlier() -> None:
    strike = np.linspace(70_000.0, 90_000.0, 21)
    y = 1.0 - strike / 80_000.0
    y[3] += 0.05
    fit = huber_wls(strike, y, np.ones_like(y), 1.345)
    assert fit is not None
    assert -fit[0] / fit[1] == pytest.approx(80_000.0, rel=1e-4)


def test_a_rising_parity_line_is_refused() -> None:
    strike = np.array([70_000.0, 80_000.0, 90_000.0, 95_000.0])
    y = strike / 80_000.0 - 1.0
    assert huber_wls(strike, y, np.ones_like(y), 1.345) is None


def test_few_pairs_fall_back_to_the_future_then_underlying() -> None:
    chain = build_chain(noise=False)
    keep_strikes = chain.group_by("expiry_ns").agg(pl.col("strike").min().alias("low"))
    thin = chain.join(keep_strikes, on="expiry_ns").filter(pl.col("strike") == pl.col("low"))
    estimates = implied_forwards(tagged(thin.drop("low")), build_futures(), SPOT, ForwardsConfig())
    by_days = dict(zip(ATM_VOL, estimates, strict=True))
    assert by_days[30].method == "future"
    assert by_days[30].forward == pytest.approx(forward_of(30))
    assert by_days[30].df == 1.0
    assert math.isnan(by_days[30].forward_se)
    assert by_days[3].method == "deribit_underlying"
    assert by_days[3].forward == pytest.approx(forward_of(3))


def test_future_falls_back_to_mark_when_the_book_is_empty() -> None:
    chain = build_chain(noise=False)
    keep = chain.group_by("expiry_ns").agg(pl.col("strike").min().alias("low"))
    thin = chain.join(keep, on="expiry_ns").filter(pl.col("strike") == pl.col("low")).drop("low")
    futures = build_futures().with_columns(pl.lit(None, dtype=pl.Float64).alias("bid_px"))
    estimates = implied_forwards(tagged(thin), futures, SPOT, ForwardsConfig())
    assert dict(zip(ATM_VOL, estimates, strict=True))[30].forward == pytest.approx(forward_of(30))


def test_gap_to_the_listed_future_is_reported_and_large_gaps_are_flagged(
    config: AppConfig,
) -> None:
    chain = build_chain(noise=False)
    expiry_ns = datetime_to_ns(expiry_of(30))
    futures = build_futures().with_columns(
        pl.when(pl.col("expiry_ns") == expiry_ns)
        .then(pl.col("bid_px") * 1.01)
        .otherwise(pl.col("bid_px"))
        .alias("bid_px"),
        pl.when(pl.col("expiry_ns") == expiry_ns)
        .then(pl.col("ask_px") * 1.01)
        .otherwise(pl.col("ask_px"))
        .alias("ask_px"),
    )
    result = clean_snapshot(chain, futures, SPOT, AS_OF_NS, tick_table(chain), config)
    gap = next(e for e in result.forwards if e.expiry_ns == expiry_ns)
    assert gap.gap_bps == pytest.approx(-1e4 * math.log(1.01), abs=0.01)
    assert len(result.report["forward_gap_flagged_expiries"]) == 1


def test_every_dropped_row_has_a_drop_reason(config: AppConfig) -> None:
    chain = build_chain()
    result = clean_snapshot(chain, build_futures(), SPOT, AS_OF_NS, tick_table(chain), config)
    unusable = result.quotes.filter(pl.col("mid_iv").is_null())
    assert unusable.height > 0
    assert unusable["drop_reason"].null_count() == 0


def test_injected_future_rows_never_change_outputs(config: AppConfig) -> None:
    chain = build_chain()
    futures = build_futures()
    baseline = clean_snapshot(chain, futures, SPOT, AS_OF_NS, tick_table(chain), config)

    later = AS_OF_NS + 60 * NS_PER_S
    poisoned_quotes = pl.concat(
        [
            chain,
            chain.with_columns(
                pl.lit(later).alias("ts_recv"),
                (pl.col("ask_px") * 3).alias("ask_px"),
                pl.lit("deribit-BTC-20261102T0915Z").alias("snapshot_id"),
            ),
        ],
        how="vertical_relaxed",
    )
    poisoned_futures = pl.concat(
        [futures, futures.with_columns(pl.lit(later).alias("ts_recv"), pl.lit(1.0).alias("bid_px"))]
    )
    injected = clean_snapshot(
        poisoned_quotes, poisoned_futures, SPOT, AS_OF_NS, tick_table(chain), config
    )
    assert injected.report == baseline.report
    assert injected.forwards[3].forward == baseline.forwards[3].forward
    assert injected.quotes.sort("symbol").equals(baseline.quotes.sort("symbol"), null_equal=True)


@pytest.mark.data
@pytest.mark.parametrize("underlying", ["BTC", "ETH"])
def test_real_data_gap_and_mark_iv(underlying: str) -> None:
    metrics = json.loads(
        Path(f"results/metrics/forward_gap_{underlying}.json").read_text(encoding="utf-8")
    )
    assert metrics["n"] > 0
    assert metrics["median_abs_bps"] <= 15.0
    reports = sorted(Path("results/dq").glob(f"{underlying}_*.json"))
    assert reports
    for path in reports:
        check = json.loads(path.read_text(encoding="utf-8"))["mark_iv_check"]
        assert check["median_abs_diff_vol_pts"] <= 0.25

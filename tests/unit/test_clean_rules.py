"""Each cleaning rule on a small table built for it."""

from typing import Any

import numpy as np
import polars as pl
import pytest

from fixtures.make_fixtures import (
    AS_OF_NS,
    SPOT,
    build_chain,
    build_futures,
    tick_table,
)
from vsm.clean import clean_snapshot, mark_iv_check, point_in_time, tag_quotes
from vsm.config import AppConfig, CleaningConfig, load_config
from vsm.conventions import NS_PER_S

SECOND = NS_PER_S


@pytest.fixture
def config() -> AppConfig:
    return load_config()


def clean(chain: pl.DataFrame, config: AppConfig) -> Any:
    ticks = tick_table(chain).unique("symbol")
    return clean_snapshot(chain, build_futures(), SPOT, AS_OF_NS, ticks, config)


def edit(chain: pl.DataFrame, symbol: str, **values: Any) -> pl.DataFrame:
    return chain.with_columns(
        pl.when(pl.col("symbol") == symbol)
        .then(pl.lit(value, dtype=chain.schema[name]))
        .otherwise(pl.col(name))
        .alias(name)
        for name, value in values.items()
    )


def reason_of(result: Any, symbol: str) -> str | None:
    return result.quotes.filter(pl.col("symbol") == symbol)["drop_reason"][0]


def fit_symbol(chain: pl.DataFrame) -> str:
    """An OTM call a little above the 30-day forward, kept by every rule."""
    rows = chain.filter(pl.col("cp") == 1).filter(pl.col("symbol").str.contains("2DEC26"))
    return rows.filter(pl.col("strike") > pl.col("underlying_px")).sort("strike")["symbol"][2]


def test_the_clean_chain_keeps_most_rows(config: AppConfig) -> None:
    result = clean(build_chain(), config)
    assert result.report["kept"] > 250
    assert result.report["liquid"] > 80
    assert result.quotes.height == build_chain().height


def test_every_drop_has_a_reason_and_no_kept_row_has_a_null_iv(config: AppConfig) -> None:
    quotes = clean(build_chain(), config).quotes
    kept = quotes.filter(pl.col("drop_reason").is_null())
    for column in ("mid_iv", "bid_iv", "ask_iv", "target_iv", "vega_fwd", "weight"):
        assert kept[column].null_count() == 0
    assert quotes.filter(pl.col("mid_iv").is_null())["drop_reason"].null_count() == 0


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"bid_px": None}, "zero_bid"),
        ({"bid_px": 0.0}, "zero_bid"),
        ({"ask_px": None}, "no_ask"),
        ({"ask_px": 0.001, "bid_px": 0.002}, "crossed"),
        ({"bid_px": 0.0001}, "wide_market"),
        ({"bid_px": 0.0010, "ask_px": 0.0035}, "wide_market"),
        ({"ts_recv": AS_OF_NS - 500 * SECOND}, "stale"),
    ],
)
def test_single_rules(config: AppConfig, change: dict[str, Any], reason: str) -> None:
    chain = build_chain()
    symbol = fit_symbol(chain)
    assert reason_of(clean(edit(chain, symbol, **change), config), symbol) == reason


def test_wide_iv_spread_is_dropped(config: AppConfig) -> None:
    chain = build_chain()
    symbol = fit_symbol(chain)
    row = chain.filter(pl.col("symbol") == symbol).row(0, named=True)
    wide = edit(chain, symbol, bid_px=row["bid_px"] * 0.8, ask_px=row["ask_px"] * 1.2)
    result = clean(wide, config)
    quote = result.quotes.filter(pl.col("symbol") == symbol).row(0, named=True)
    assert quote["ask_iv"] - quote["bid_iv"] > config.cleaning.max_iv_spread
    assert quote["drop_reason"] == "wide_market"


def test_short_expiries_are_recorded_not_fitted(config: AppConfig) -> None:
    result = clean(build_chain(), config)
    one_day = result.quotes.filter(pl.col("t") < 2 / 365)
    assert one_day.height > 0
    assert set(one_day["drop_reason"].to_list()) == {"short_expiry"}
    assert not one_day["is_liquid"].any()


def test_expired_rows_are_tagged_and_do_not_break_the_run(config: AppConfig) -> None:
    chain = build_chain()
    past = chain.with_columns(pl.lit(AS_OF_NS - SECOND).alias("expiry_ns")).head(4)
    result = clean(pl.concat([chain, past]), config)
    assert result.quotes.filter(pl.col("t") < 0)["drop_reason"].to_list() == ["short_expiry"] * 4


def test_bound_violations_and_ask_below_intrinsic_are_counted(config: AppConfig) -> None:
    chain = build_chain()
    itm = (
        chain.filter((pl.col("cp") == 1) & (pl.col("strike") < 0.9 * pl.col("underlying_px")))
        .filter(pl.col("symbol").str.contains("2DEC26"))
        .sort("strike")["symbol"][0]
    )
    result = clean(edit(chain, itm, bid_px=0.01, ask_px=0.02), config)
    assert reason_of(result, itm) == "bound_violation"
    assert result.report["ask_below_intrinsic"] == 1


def test_is_otm_follows_the_forward(config: AppConfig) -> None:
    quotes = clean(build_chain(), config).quotes
    calls = quotes.filter(pl.col("cp") == 1)
    puts = quotes.filter(pl.col("cp") == -1)
    assert (calls["is_otm"] == (calls["strike"] >= calls["forward"])).all()
    assert (puts["is_otm"] == (puts["strike"] < puts["forward"])).all()


def test_thin_slice_is_dropped_whole(config: AppConfig) -> None:
    chain = build_chain()
    code = "-2DEC26-"
    inside = pl.col("symbol").str.contains(code)
    nearest = (pl.col("strike") / pl.col("underlying_px")).log().abs()
    otm = pl.when(pl.col("cp") == 1).then(pl.col("strike") >= pl.col("underlying_px"))
    otm = otm.otherwise(pl.col("strike") < pl.col("underlying_px"))
    keep = chain.filter(inside & otm).sort(nearest).head(4).select("symbol").to_series().to_list()
    thin = chain.filter(~inside | pl.col("symbol").is_in(keep))
    quotes = clean(thin, config).quotes.filter(inside)
    assert quotes.height == 4
    assert set(quotes["drop_reason"].to_list()) == {"thin_slice"}


def test_slice_needs_a_quote_on_each_side_of_the_forward(config: AppConfig) -> None:
    chain = build_chain()
    one_sided = chain.filter(~(pl.col("symbol").str.contains("2DEC26") & (pl.col("cp") == -1)))
    quotes = clean(one_sided, config).quotes.filter(pl.col("symbol").str.contains("2DEC26"))
    assert quotes["drop_reason"].null_count() == 0


def test_weights_are_inverse_iv_spread_and_clipped(config: AppConfig) -> None:
    kept = clean(build_chain(), config).quotes.filter(pl.col("drop_reason").is_null())
    expected = (1.0 / (kept["ask_iv"] - kept["bid_iv"])).clip(1.0, 1000.0)
    assert np.allclose(kept["weight"].to_numpy(), expected.to_numpy())
    assert kept["weight"].min() >= 1.0
    assert kept["weight"].max() <= 1000.0


def test_liquid_quotes_meet_the_definition(config: AppConfig) -> None:
    liquid = clean(build_chain(), config).quotes.filter(pl.col("is_liquid"))
    assert liquid["is_otm"].all()
    assert (liquid["ask_iv"] - liquid["bid_iv"]).max() <= config.liquid.max_iv_spread
    assert liquid["t"].min() >= 2 / 365


def test_target_defaults_to_mid(config: AppConfig) -> None:
    kept = clean(build_chain(), config).quotes.filter(pl.col("drop_reason").is_null())
    assert kept["target_iv"].to_list() == kept["mid_iv"].to_list()


def test_microprice_target_uses_sizes_where_present(config: AppConfig) -> None:
    config.iv.target = "microprice"
    chain = build_chain().with_columns(
        pl.when(pl.col("cp") == 1).then(9.0).otherwise(None).alias("bid_qty"),
        pl.when(pl.col("cp") == 1).then(1.0).otherwise(None).alias("ask_qty"),
    )
    kept = clean(chain, config).quotes.filter(pl.col("drop_reason").is_null())
    calls = kept.filter(pl.col("cp") == 1)
    puts = kept.filter(pl.col("cp") == -1)
    assert ((calls["target_iv"] > calls["mid_iv"]) & (calls["target_iv"] < calls["ask_iv"])).all()
    assert puts["target_iv"].to_list() == puts["mid_iv"].to_list()


def test_implied_coin_rate_changes_the_conversion(config: AppConfig) -> None:
    config.conventions.coin_rate = "implied"
    forwards_zero = clean(build_chain(), load_config()).quotes
    forwards_implied = clean(build_chain(), config).quotes
    assert (
        forwards_implied["mid_fwd"].drop_nulls().len()
        == forwards_zero["mid_fwd"].drop_nulls().len()
    )


def test_point_in_time_drops_future_rows_and_duplicates() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["a", "a", "b"],
            "snapshot_id": ["s", "s", "s"],
            "ts_recv": [1, 2, 3],
            "x": [10, 20, 30],
        }
    )
    assert point_in_time(frame, 2).to_dicts() == [
        {"symbol": "a", "snapshot_id": "s", "ts_recv": 2, "x": 20}
    ]


def test_tag_keeps_the_first_reason() -> None:
    frame = pl.DataFrame(
        {
            "ts_recv": [AS_OF_NS - 500 * SECOND],
            "expiry_ns": [AS_OF_NS + 86_400 * SECOND],
            "bid_px": [None],
            "ask_px": [0.1],
        },
        schema_overrides={"bid_px": pl.Float64},
    )
    assert tag_quotes(frame, AS_OF_NS, CleaningConfig())["drop_reason"][0] == "stale"


def test_mark_iv_check_passes_on_consistent_marks(config: AppConfig) -> None:
    report = clean(build_chain(), config).report["mark_iv_check"]
    assert report["n"] > 100
    assert report["median_abs_diff_vol_pts"] < 0.01
    assert report["warning"] is False


def test_mark_iv_check_warns_on_a_unit_error(config: AppConfig) -> None:
    wrong = build_chain().with_columns(pl.col("mark_iv_pct") * 1.05)
    report = clean(wrong, config).report["mark_iv_check"]
    assert report["warning"] is True


def test_mark_iv_check_on_an_empty_table() -> None:
    empty = pl.DataFrame(
        schema={
            "mark_px": pl.Float64,
            "mark_iv_pct": pl.Float64,
            "underlying_px": pl.Float64,
            "t": pl.Float64,
            "cp": pl.Int8,
            "strike": pl.Float64,
        }
    )
    assert mark_iv_check(empty, 0.0) == {
        "n": 0,
        "median_abs_diff_vol_pts": None,
        "warning": False,
    }


def test_report_counts_add_up(config: AppConfig) -> None:
    report = clean(build_chain(), config).report
    assert sum(report["by_reason"].values()) == report["rows"]
    per_expiry = sum(sum(v.values()) for v in report["per_expiry"].values())
    assert per_expiry == report["rows"]
    assert report["forward_methods"] == {"deribit_underlying": 1, "parity_wls": 7}

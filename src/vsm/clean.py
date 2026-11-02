"""Quote cleaning: point-in-time selection, DQ rules, IVs, weights and the daily DQ report.

Cleaning runs in two passes because the forward sits between them. ``tag_quotes`` applies the
rules that need no forward; ``implied_forwards`` uses the surviving quotes, in or out of the
money; ``apply_forwards`` then converts prices, solves IVs and applies the remaining rules.
Every dropped row stays in the table with a ``drop_reason``.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.special import ndtr

from vsm.config import AppConfig, CleaningConfig
from vsm.conventions import (
    F64,
    NS_PER_S,
    YEAR_SECONDS,
    inverse_to_forward_usd,
    log_moneyness,
    ns_to_datetime,
    percent_to_decimal,
    year_fraction,
)
from vsm.forwards import ForwardEstimate, implied_forwards
from vsm.iv.black import black76_vega
from vsm.iv.solver import OK, implied_vol

MARK_IV_WARNING_VOL_POINTS = 0.25  # DQ-17
WEIGHT_CLIP = (1.0, 1000.0)  # FR-10
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class CleanResult:
    """One underlying's snapshot after cleaning."""

    quotes: pl.DataFrame
    forwards: list[ForwardEstimate]
    report: dict[str, Any]


def point_in_time(rows: pl.DataFrame, as_of_ns: int) -> pl.DataFrame:
    """Rows known at ``as_of_ns``, one per (symbol, snapshot_id) (DQ-01, DQ-12)."""
    return (
        rows.filter(pl.col("ts_recv") <= as_of_ns)
        .sort("ts_recv")
        .unique(subset=["symbol", "snapshot_id"], keep="last", maintain_order=True)
    )


def _first_reason(*rules: tuple[pl.Expr, str]) -> pl.Expr:
    """``drop_reason`` kept if set, else the first rule that fires, in order."""
    chain: Any = pl.when(pl.col("drop_reason").is_not_null()).then(pl.col("drop_reason"))
    for condition, reason in rules:
        chain = chain.when(condition).then(pl.lit(reason))
    result: pl.Expr = chain.otherwise(pl.lit(None, dtype=pl.String)).alias("drop_reason")
    return result


def tag_quotes(
    rows: pl.DataFrame, as_of_ns: int, cfg: CleaningConfig, year_seconds: int = YEAR_SECONDS
) -> pl.DataFrame:
    """First pass: staleness, short expiries, zero and crossed quotes (DQ-01, 02, 05)."""
    t = year_fraction(rows["expiry_ns"].to_numpy(), as_of_ns, year_seconds)
    min_t = cfg.min_days_to_expiry * SECONDS_PER_DAY / year_seconds
    stale_before = as_of_ns - cfg.staleness_s * NS_PER_S
    tagged = rows.with_columns(
        pl.lit(as_of_ns, dtype=pl.Int64).alias("as_of_ns"),
        pl.Series("t", t),
        pl.lit(None, dtype=pl.String).alias("drop_reason"),
    )
    return tagged.with_columns(
        _first_reason(
            (pl.col("ts_recv") < stale_before, "stale"),
            (pl.col("t") < min_t, "short_expiry"),
            (pl.col("bid_px").is_null() | (pl.col("bid_px") <= 0.0), "zero_bid"),
            (pl.col("ask_px").is_null(), "no_ask"),
            (pl.col("ask_px") <= pl.col("bid_px"), "crossed"),
        )
    )


def _solve_ivs(
    frame: pl.DataFrame, column: str, mask: NDArray[np.bool_], max_iter: int
) -> tuple[F64, NDArray[np.int8]]:
    """Implied vols of ``column`` on ``mask``; other rows are NaN with status 0."""
    n = frame.height
    vol = np.full(n, np.nan)
    status = np.zeros(n, dtype=np.int8)
    if mask.any():
        solved, codes = implied_vol(
            frame[column].to_numpy()[mask],
            frame["forward"].to_numpy()[mask],
            frame["strike"].to_numpy()[mask],
            frame["t"].to_numpy()[mask],
            frame["cp"].to_numpy()[mask] == 1,
            max_iter=max_iter,
        )
        vol[mask] = solved
        status[mask] = codes
    return vol, status


def _microprice(frame: pl.DataFrame) -> pl.Expr:
    size = pl.col("bid_qty") + pl.col("ask_qty")
    weighted = pl.col("bid_fwd") * pl.col("ask_qty") + pl.col("ask_fwd") * pl.col("bid_qty")
    return (weighted / size).alias("micro_fwd")


def apply_forwards(
    tagged: pl.DataFrame,
    forwards: list[ForwardEstimate],
    tick_size: pl.DataFrame,
    config: AppConfig,
) -> pl.DataFrame:
    """Second pass: forward-space prices, IVs, DQ-03, 04, 06, 07 and the liquid flag (DQ-08)."""
    cfg = config.cleaning
    use_implied_df = config.conventions.coin_rate == "implied"
    table = pl.DataFrame(
        {
            "expiry_ns": [f.expiry_ns for f in forwards],
            "forward": [f.forward for f in forwards],
            "df": [f.df for f in forwards],
        },
        schema={"expiry_ns": pl.Int64, "forward": pl.Float64, "df": pl.Float64},
    )
    frame = tagged.join(table, on="expiry_ns", how="left").join(tick_size, on="symbol", how="left")
    frame = frame.with_columns(
        _first_reason((pl.col("forward").is_null(), "no_forward")),
        pl.col("tick_size").fill_null(0.0),
    )
    discount = frame["df"].to_numpy() if use_implied_df else 1.0
    forward = frame["forward"].to_numpy()
    strike = frame["strike"].to_numpy()
    bid = inverse_to_forward_usd(frame["bid_px"].cast(pl.Float64).to_numpy(), forward, discount)
    ask = inverse_to_forward_usd(frame["ask_px"].cast(pl.Float64).to_numpy(), forward, discount)
    frame = frame.with_columns(
        pl.Series("k", log_moneyness(strike, forward)),
        pl.Series("bid_fwd", bid),
        pl.Series("ask_fwd", ask),
        pl.Series("mid_fwd", 0.5 * (bid + ask)),
    )

    is_call = pl.col("cp") == 1
    intrinsic = (
        pl.when(is_call)
        .then((pl.col("forward") - pl.col("strike")).clip(lower_bound=0.0))
        .otherwise((pl.col("strike") - pl.col("forward")).clip(lower_bound=0.0))
    )
    upper = pl.when(is_call).then(pl.col("forward")).otherwise(pl.col("strike"))
    frame = frame.with_columns(
        (pl.col("ask_fwd") < intrinsic).fill_null(False).alias("ask_below_intrinsic"),
        _first_reason(
            ((pl.col("mid_fwd") <= intrinsic) | (pl.col("mid_fwd") >= upper), "bound_violation"),
        ),
    )

    kept = frame["drop_reason"].is_null().to_numpy()
    mid_iv, mid_status = _solve_ivs(frame, "mid_fwd", kept, config.iv.max_iter)
    bid_iv, bid_status = _solve_ivs(frame, "bid_fwd", kept, config.iv.max_iter)
    ask_iv, ask_status = _solve_ivs(frame, "ask_fwd", kept, config.iv.max_iter)
    iv_failed = (mid_status != OK) | (bid_status != OK) | (ask_status != OK)
    spread = ask_iv - bid_iv
    frame = frame.with_columns(
        pl.Series("mid_iv", mid_iv).fill_nan(None),
        pl.Series("bid_iv", bid_iv).fill_nan(None),
        pl.Series("ask_iv", ask_iv).fill_nan(None),
        pl.Series("iv_failed", iv_failed),
        pl.Series("iv_spread", spread).fill_nan(None),
    )
    relative_spread = (pl.col("ask_px") - pl.col("bid_px")) / (
        0.5 * (pl.col("ask_px") + pl.col("bid_px"))
    )
    frame = frame.with_columns(
        _first_reason(
            (pl.col("iv_failed"), "iv_fail"),
            (pl.col("bid_px") <= cfg.min_bid_ticks * pl.col("tick_size"), "wide_market"),
            (pl.col("iv_spread") > cfg.max_iv_spread, "wide_market"),
            (relative_spread > cfg.max_rel_spread, "wide_market"),
        ),
        (
            pl.when(is_call)
            .then(pl.col("strike") >= pl.col("forward"))
            .otherwise(pl.col("strike") < pl.col("forward"))
        ).alias("is_otm"),
    )
    return _finish(frame, config)


def _finish(frame: pl.DataFrame, config: AppConfig) -> pl.DataFrame:
    """Slice adequacy, target IV, weights, vega and the liquid flag."""
    cfg = config.cleaning
    fit_rows = pl.col("drop_reason").is_null() & pl.col("is_otm")
    below = fit_rows & (pl.col("strike") < pl.col("forward"))
    at_or_above = fit_rows & (pl.col("strike") >= pl.col("forward"))
    adequate = (
        (fit_rows.sum().over("expiry_ns") >= cfg.min_strikes_per_slice)
        & (below.sum().over("expiry_ns") >= 1)
        & (at_or_above.sum().over("expiry_ns") >= 1)
    )
    frame = frame.with_columns(_first_reason((~adequate, "thin_slice")))

    target = np.array(frame["mid_iv"].to_numpy(), dtype=np.float64)
    if config.iv.target == "microprice":
        frame = frame.with_columns(_microprice(frame))
        has_sizes = (
            frame["micro_fwd"].is_not_null().to_numpy() & frame["drop_reason"].is_null().to_numpy()
        )
        micro, status = _solve_ivs(frame, "micro_fwd", has_sizes, config.iv.max_iter)
        usable = has_sizes & (status == OK)
        target[usable] = micro[usable]

    forward = frame["forward"].to_numpy()
    strike = frame["strike"].to_numpy()
    t = frame["t"].to_numpy()
    mid_iv = np.array(frame["mid_iv"].to_numpy(), dtype=np.float64)
    vega = np.full(frame.height, np.nan)
    call_delta = np.full(frame.height, np.nan)
    solved = np.isfinite(mid_iv) & (t > 0.0)
    if solved.any():
        total = mid_iv[solved] * np.sqrt(t[solved])
        vega[solved] = black76_vega(forward[solved], strike[solved], t[solved], mid_iv[solved])
        call_delta[solved] = ndtr(np.log(forward[solved] / strike[solved]) / total + 0.5 * total)

    liquid = config.liquid
    min_t = liquid.min_days_to_expiry * SECONDS_PER_DAY / config.conventions.year_seconds
    spread = pl.col("ask_iv") - pl.col("bid_iv")
    weight = (1.0 / spread).clip(*WEIGHT_CLIP)
    return frame.with_columns(
        pl.Series("target_iv", target).fill_nan(None),
        pl.Series("vega_fwd", vega).fill_nan(None),
        pl.Series("call_delta", call_delta).fill_nan(None),
        pl.when(pl.col("drop_reason").is_null()).then(weight).otherwise(None).alias("weight"),
    ).with_columns(
        (
            pl.col("drop_reason").is_null()
            & pl.col("is_otm")
            & (pl.col("call_delta") >= liquid.min_abs_delta)
            & (pl.col("call_delta") <= liquid.max_abs_delta)
            & (spread <= liquid.max_iv_spread)
            & (pl.col("t") >= min_t)
        )
        .fill_null(False)
        .alias("is_liquid")
    )


def mark_iv_check(quotes: pl.DataFrame, min_t: float) -> dict[str, Any]:
    """DQ-17: IV from Deribit's mark price against its mark IV, on OTM quotes.

    Median absolute difference in vol points; above 0.25 it warns, since a gap that large
    points to a convention error rather than noise.
    """
    rows = quotes.filter(
        pl.col("mark_px").is_not_null()
        & (pl.col("mark_px") > 0.0)
        & pl.col("mark_iv_pct").is_not_null()
        & (pl.col("mark_iv_pct") > 0.0)
        & pl.col("underlying_px").is_not_null()
        & (pl.col("t") >= min_t)
        & (
            pl.when(pl.col("cp") == 1)
            .then(pl.col("strike") >= pl.col("underlying_px"))
            .otherwise(pl.col("strike") < pl.col("underlying_px"))
        )
    )
    if rows.is_empty():
        return {"n": 0, "median_abs_diff_vol_pts": None, "warning": False}
    forward = rows["underlying_px"].to_numpy()
    t = rows["t"].to_numpy()
    rate = rows["interest_rate"].fill_null(0.0).to_numpy()
    discount = np.exp(-rate * t)
    price = inverse_to_forward_usd(rows["mark_px"].to_numpy(), forward)
    vol, status = implied_vol(
        price, forward, rows["strike"].to_numpy(), t, rows["cp"].to_numpy() == 1, discount
    )
    usable = status == OK
    decimal_mark = percent_to_decimal(rows["mark_iv_pct"].to_numpy())
    diff = 100.0 * np.abs(vol[usable] - decimal_mark[usable])
    median = float(np.median(diff)) if diff.size else math.nan
    return {
        "n": int(diff.size),
        "median_abs_diff_vol_pts": median,
        "warning": bool(diff.size and median > MARK_IV_WARNING_VOL_POINTS),
    }


def _iso(expiry_ns: int) -> str:
    return ns_to_datetime(expiry_ns).strftime("%Y-%m-%dT%H:%MZ")


def dq_report(
    quotes: pl.DataFrame, forwards: list[ForwardEstimate], config: AppConfig
) -> dict[str, Any]:
    """Counts per drop reason and per expiry, forward methods, and the DQ-09 and DQ-17 flags."""
    counts = quotes.group_by("drop_reason").len()
    by_reason = {
        row["drop_reason"] or "kept": int(row["len"]) for row in counts.iter_rows(named=True)
    }
    per_expiry: dict[str, dict[str, int]] = {}
    grouped = quotes.group_by("expiry_ns", "drop_reason").len().sort("expiry_ns", "drop_reason")
    for row in grouped.iter_rows(named=True):
        entry = per_expiry.setdefault(_iso(row["expiry_ns"]), {})
        entry[row["drop_reason"] or "kept"] = int(row["len"])
    methods: dict[str, int] = {}
    for estimate in forwards:
        methods[estimate.method] = methods.get(estimate.method, 0) + 1
    limit = config.forwards.max_futures_gap_bps
    flagged = [
        _iso(f.expiry_ns) for f in forwards if f.gap_bps is not None and abs(f.gap_bps) > limit
    ]
    min_t = config.cleaning.min_days_to_expiry * SECONDS_PER_DAY / config.conventions.year_seconds
    return {
        "rows": quotes.height,
        "kept": by_reason.get("kept", 0),
        "liquid": int(quotes["is_liquid"].sum()),
        "by_reason": dict(sorted(by_reason.items())),
        "per_expiry": per_expiry,
        "ask_below_intrinsic": int(quotes["ask_below_intrinsic"].sum()),
        "forward_methods": dict(sorted(methods.items())),
        "forward_gap_flagged_expiries": flagged,
        "mark_iv_check": mark_iv_check(quotes, min_t),
    }


CLEAN_COLUMNS = [
    "snapshot_id",
    "ts_recv",
    "ts_exchange",
    "venue",
    "symbol",
    "underlying",
    "expiry_ns",
    "strike",
    "cp",
    "bid_px",
    "ask_px",
    "mark_px",
    "mark_iv_pct",
    "underlying_px",
    "as_of_ns",
    "t",
    "forward",
    "df",
    "k",
    "bid_fwd",
    "ask_fwd",
    "mid_fwd",
    "bid_iv",
    "ask_iv",
    "mid_iv",
    "target_iv",
    "vega_fwd",
    "weight",
    "is_otm",
    "is_liquid",
    "drop_reason",
]


def clean_snapshot(
    options: pl.DataFrame,
    futures: pl.DataFrame,
    spot_index: float,
    as_of_ns: int,
    tick_size: pl.DataFrame,
    config: AppConfig,
    *,
    staleness_s: int | None = None,
) -> CleanResult:
    """Run one snapshot from raw quotes to ``clean_quotes``, ``forwards`` and the DQ report.

    ``staleness_s`` overrides the configured limit, for Tardis-derived snapshots (DQ-01).
    """
    known = point_in_time(options, as_of_ns)
    known_futures = point_in_time(futures, as_of_ns)
    cleaning = config.cleaning
    if staleness_s is not None:
        cleaning = cleaning.model_copy(update={"staleness_s": staleness_s})
    tagged = tag_quotes(known, as_of_ns, cleaning, config.conventions.year_seconds)
    live = tagged.filter(pl.col("t") > 0.0)
    forwards = implied_forwards(
        live,
        known_futures,
        spot_index,
        config.forwards,
        year_seconds=config.conventions.year_seconds,
    )
    frame = apply_forwards(tagged, forwards, tick_size, config)
    return CleanResult(
        quotes=frame.select(CLEAN_COLUMNS),
        forwards=forwards,
        report=dq_report(frame, forwards, config),
    )

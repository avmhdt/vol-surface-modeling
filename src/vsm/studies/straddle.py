"""Delta-hedged ATM-forward straddles with costs (FR-22, AC-H-02, AC-W-10).

One trade per entry snapshot:

- Sell (or buy) one call and one put on one coin, at the listed strike nearest the forward, on
  the expiry nearest 30 days inside 20 to 45 days. Fills are at the bid (or ask), in forward-space
  USD, which is the coin premium sold forward at F.
- Hedge daily at the entry time with the perpetual, at the last close known by then, using a
  Black-76 delta at the entry IV (variants: the current surface's IV where one is stored, or a
  minimum-variance delta). The perpetual is treated as linear, which is exact if its coin P&L is
  converted to USD as it accrues. Funding is paid on the hedge every hour, at Deribit's 1-hour
  rate, and is part of both gross and net P&L.
- Settle at Deribit's delivery price on the expiry date, and close the hedge at the last close
  known at 08:00 UTC.

Gross P&L fills at mid and pays no fees or slippage. Net P&L fills at the touch, pays the option
fee, the delivery fee and the perpetual taker fee by date, and pays ``hedge_slippage_bps`` on
every hedge trade. P&L is in USD per coin and, for the statistics, as a fraction of the entry
forward, so that years with different price levels weigh the same.

The theoretical gamma P&L of a short straddle is -sum 1/2 Gamma P^2 (r^2 - sigma^2 dt) over the
hedge intervals, with r the log return of the hedge price.
"""

import itertools
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.stats import norm

from vsm.config import AppConfig, StraddleConfig, config_hash
from vsm.conventions import F64, year_fraction
from vsm.explog import append_run
from vsm.io import storage
from vsm.io.tardis import snapshot_path
from vsm.pipeline import clean_quotes_of, read_params
from vsm.studies.costs import FeeTable, load_fees, option_fee_coin
from vsm.studies.stats import summarize
from vsm.studies.vrp import perp_closes
from vsm.surface.arbitrage import check_surface
from vsm.surface.store import load_surface
from vsm.surface.types import Surface
from vsm.viz import figures

HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
HEDGES = ("entry_iv", "current_iv", "min_variance")
SOURCE_TIME = {"tardis_monthly": "tardis", "recorder_daily": "recorder"}  # where entries come from
BETA_WINDOW_DAYS = 365
MIN_BETA_DAYS = 60


@dataclass(frozen=True)
class Entry:
    """What is known at entry about one straddle."""

    snapshot_id: str
    as_of_ns: int
    expiry_ns: int
    strike: float
    forward: float
    df: float
    entry_vol: float
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float
    call_bid_coin: float
    call_ask_coin: float
    put_bid_coin: float
    put_ask_coin: float


@dataclass(frozen=True)
class HedgeResult:
    """The hedge's P&L over one trade, per coin of straddle notional, in USD."""

    hedge_pnl: float
    funding_pnl: float
    trading_cost: float
    gamma_pnl: float
    final_price: float
    rebalances: int


def black_delta_gamma(price: float, strike: float, tau: float, vol: float) -> tuple[float, float]:
    """(delta, gamma) of a long straddle under undiscounted Black-76."""
    root = vol * math.sqrt(tau)
    d1 = math.log(price / strike) / root + 0.5 * root
    delta = 2.0 * float(norm.cdf(d1)) - 1.0
    gamma = 2.0 * float(norm.pdf(d1)) / (price * root)
    return delta, gamma


def black_vega(price: float, strike: float, tau: float, vol: float) -> float:
    """Vega of a long straddle per unit of vol."""
    root = vol * math.sqrt(tau)
    d1 = math.log(price / strike) / root + 0.5 * root
    return 2.0 * price * float(norm.pdf(d1)) * math.sqrt(tau)


def hedge_path(
    times_ns: NDArray[np.int64],
    prices: F64,
    funding: F64,
    vols: F64,
    strike: float,
    expiry_ns: int,
    *,
    side: float = -1.0,
    taker_fee: float = 0.0,
    slippage_bps: float = 0.0,
    smile_slope: float = 0.0,
) -> HedgeResult:
    """Delta-hedge a straddle along one path of hedge prices.

    ``times_ns`` and ``prices`` run from entry to expiry. ``funding[i]`` is the summed 1-hour
    funding rate paid by a long perpetual over interval i, and ``vols[i]`` the vol used for the
    delta at its start. ``side`` is -1 for a short straddle. ``smile_slope`` is d sigma / d ln P,
    which turns the delta into the minimum-variance one.
    """
    hedge = 0.0
    hedge_pnl = 0.0
    funding_pnl = 0.0
    cost = 0.0
    gamma_pnl = 0.0
    rate = taker_fee + slippage_bps * 1e-4
    years = year_fraction(np.full(times_ns.size, expiry_ns, dtype=np.int64), 0) - year_fraction(
        times_ns.astype(np.int64), 0
    )
    for i in range(times_ns.size - 1):
        tau = float(years[i])
        delta, gamma = black_delta_gamma(float(prices[i]), strike, tau, float(vols[i]))
        if smile_slope != 0.0:
            vega = black_vega(float(prices[i]), strike, tau, float(vols[i]))
            delta += vega * smile_slope / float(prices[i])
        target = -side * delta
        cost += abs(target - hedge) * float(prices[i]) * rate
        hedge = target
        step = float(prices[i + 1]) - float(prices[i])
        hedge_pnl += hedge * step
        funding_pnl -= hedge * float(prices[i]) * float(funding[i])
        dt = float(years[i] - years[i + 1])
        r = math.log(float(prices[i + 1]) / float(prices[i]))
        gamma_pnl += side * 0.5 * gamma * float(prices[i]) ** 2 * (r * r - float(vols[i]) ** 2 * dt)
    cost += abs(hedge) * float(prices[-1]) * rate
    return HedgeResult(
        hedge_pnl=hedge_pnl,
        funding_pnl=funding_pnl,
        trading_cost=cost,
        gamma_pnl=gamma_pnl,
        final_price=float(prices[-1]),
        rebalances=int(times_ns.size - 1),
    )


def choose_entry(surface: Surface, quotes: pl.DataFrame, cfg: StraddleConfig) -> Entry | None:
    """The ATM-forward straddle on the expiry nearest the target, if both legs are two-sided."""
    days = [(item, item.t * 365.0) for item in surface.slices]
    inside = [(item, d) for item, d in days if cfg.min_days <= d <= cfg.max_days]
    if not inside:
        return None
    chosen = min(inside, key=lambda pair: abs(pair[1] - cfg.target_days))[0]
    legs = quotes.filter(
        (pl.col("expiry_ns") == chosen.expiry_ns)
        & (pl.col("bid_px") > 0.0)
        & (pl.col("ask_px") > 0.0)
        & pl.col("bid_fwd").is_finite()
        & pl.col("ask_fwd").is_finite()
    )
    strikes = legs.group_by("strike").agg(pl.col("cp").n_unique().alias("sides"))
    both = strikes.filter(pl.col("sides") == 2)["strike"].to_list()
    if not both:
        return None
    strike = min(both, key=lambda value: abs(value - chosen.forward))
    pair = legs.filter(pl.col("strike") == strike)
    call = pair.filter(pl.col("cp") == 1).row(0, named=True)
    put = pair.filter(pl.col("cp") == -1).row(0, named=True)
    k = math.log(strike / chosen.forward)
    vol = math.sqrt(float(chosen.svi.w(np.array(k))) / chosen.t)
    return Entry(
        snapshot_id=surface.snapshot_id,
        as_of_ns=surface.as_of_ns,
        expiry_ns=chosen.expiry_ns,
        strike=float(strike),
        forward=chosen.forward,
        df=chosen.df,
        entry_vol=vol,
        call_bid=float(call["bid_fwd"]),
        call_ask=float(call["ask_fwd"]),
        put_bid=float(put["bid_fwd"]),
        put_ask=float(put["ask_fwd"]),
        call_bid_coin=float(call["bid_px"]),
        call_ask_coin=float(call["ask_px"]),
        put_bid_coin=float(put["bid_px"]),
        put_ask_coin=float(put["ask_px"]),
    )


@dataclass(frozen=True)
class Market:
    """Hourly hedge prices, funding, DVOL and delivery prices of one underlying."""

    closes: pl.DataFrame
    funding: pl.DataFrame
    dvol: pl.DataFrame
    delivery: dict[date, float]

    def price_known_at(self, ts_ns: int) -> float | None:
        """The last perpetual close known by ``ts_ns``: bars are stamped at their open."""
        known = self.closes.filter(pl.col("ts") + HOUR_NS <= ts_ns).tail(1)
        if known.is_empty() or int(known["ts"][0]) + 2 * HOUR_NS < ts_ns:
            return None
        return float(known["close"][0])

    def funding_between(self, start_ns: int, end_ns: int) -> float:
        """Summed 1-hour funding rates of the hours in [start, end)."""
        rows = self.funding.filter((pl.col("ts") >= start_ns) & (pl.col("ts") < end_ns))
        return float(rows["interest_1h"].sum())

    def smile_slope(self, as_of_ns: int) -> float | None:
        """d sigma / d ln P from DVOL over the year before ``as_of_ns``, or None if too short.

        The slope of daily DVOL changes on daily perpetual log returns, both at 08:00 UTC; it uses
        no data after entry.
        """
        start = as_of_ns - BETA_WINDOW_DAYS * DAY_NS
        daily = (
            self.dvol.filter(
                (pl.col("ts") >= start)
                & (pl.col("ts") + HOUR_NS <= as_of_ns)
                & (((pl.col("ts") // HOUR_NS) % 24) == 8)
            )
            .join(self.closes.rename({"close": "price"}), on="ts", how="inner")
            .sort("ts")
        )
        if daily.height < MIN_BETA_DAYS:
            return None
        vol = daily["close"].to_numpy() / 100.0
        price = daily["price"].to_numpy()
        returns = np.diff(np.log(price))
        changes = np.diff(vol)
        variance = float(np.var(returns))
        if variance <= 0.0:
            return None
        return float(np.cov(returns, changes, ddof=0)[0, 1] / variance)


def load_market(data_dir: Path, underlying: str) -> Market:
    """Everything the hedge needs, read once."""
    raw = storage.raw_dir(data_dir)

    def read(folder: Path, columns: list[str]) -> pl.DataFrame:
        files = sorted(folder.glob("part-*.parquet"))
        if not files:
            return pl.DataFrame({name: [] for name in columns})
        return (
            pl.concat([pl.read_parquet(f, columns=columns) for f in files]).unique("ts").sort("ts")
        )

    funding = read(raw / "funding" / f"symbol={underlying}-PERPETUAL", ["ts", "interest_1h"])
    dvol = read(raw / "dvol" / f"underlying={underlying}", ["ts", "close"])
    path = raw / "delivery" / f"index_name={underlying.lower()}_usd" / "delivery.parquet"
    delivery: dict[date, float] = {}
    if path.exists():
        frame = pl.read_parquet(path)
        delivery = dict(
            zip(frame["date"].to_list(), frame["delivery_price"].to_list(), strict=True)
        )
    return Market(perp_closes(data_dir, underlying), funding, dvol, delivery)


def _surfaces(
    data_dir: Path, config: AppConfig, underlying: str, source: str
) -> dict[str, Surface]:
    """Gated canonical surfaces of the source, by snapshot label."""
    params = read_params(data_dir, underlying)
    canonical = config.straddle.entry_time_utc.replace(":", "")
    found: dict[str, Surface] = {}
    for (snapshot,), frame in params.partition_by("snapshot_id", as_dict=True).items():
        label = str(snapshot).rsplit("-", 1)[1]
        if not label.endswith(f"T{canonical}Z"):
            continue
        day = datetime.strptime(label[:8], "%Y%m%d").date()
        origin = (
            "tardis" if snapshot_path(data_dir, underlying, day, label).exists() else "recorder"
        )
        if origin != SOURCE_TIME[source]:
            continue
        surface = load_surface(frame)
        if check_surface(surface, config.arbitrage).passed:
            found[label] = surface
    return dict(sorted(found.items()))


def run_trade(
    entry: Entry,
    market: Market,
    fees: FeeTable,
    cfg: StraddleConfig,
    underlying: str,
    hedge: str,
    current_vols: dict[int, float],
) -> dict[str, Any] | None:
    """One trade's P&L, or None if the market data does not cover it."""
    expiry_day = datetime.fromtimestamp(entry.expiry_ns / 1e9, UTC).date()
    settle = market.delivery.get(expiry_day)
    if settle is None:
        return None
    times = [*list(range(entry.as_of_ns, entry.expiry_ns, DAY_NS)), entry.expiry_ns]
    prices = [market.price_known_at(ts) for ts in times]
    if any(price is None for price in prices):
        return None
    times_ns = np.array(times, dtype=np.int64)
    path = np.array(prices, dtype=np.float64)
    funding = np.array([market.funding_between(a, b) for a, b in itertools.pairwise(times)])
    vols = np.full(times_ns.size - 1, entry.entry_vol)
    used_current = 0
    slope = 0.0
    if hedge == "current_iv":
        for i, ts in enumerate(times[:-1]):
            if i and ts in current_vols:
                vols[i] = current_vols[ts]
                used_current += 1
    elif hedge == "min_variance":
        slope = market.smile_slope(entry.as_of_ns) or 0.0
    period = fees.on(datetime.fromtimestamp(entry.as_of_ns / 1e9, UTC).date())
    side = -1.0 if cfg.side == "short" else 1.0
    taker = period.perpetual_taker[underlying]
    gross_hedge = hedge_path(
        times_ns, path, funding, vols, entry.strike, entry.expiry_ns, side=side, smile_slope=slope
    )
    net_hedge = hedge_path(
        times_ns,
        path,
        funding,
        vols,
        entry.strike,
        entry.expiry_ns,
        side=side,
        taker_fee=taker,
        slippage_bps=cfg.costs.hedge_slippage_bps,
        smile_slope=slope,
    )
    usd_per_coin = entry.forward / entry.df
    if side < 0:
        fill = entry.call_bid + entry.put_bid
        coins = (entry.call_bid_coin, entry.put_bid_coin)
    else:
        fill = entry.call_ask + entry.put_ask
        coins = (entry.call_ask_coin, entry.put_ask_coin)
    if cfg.costs.option_fill == "mid":
        fill = 0.5 * (entry.call_bid + entry.call_ask + entry.put_bid + entry.put_ask)
    mid = 0.5 * (entry.call_bid + entry.call_ask + entry.put_bid + entry.put_ask)
    option_fee = usd_per_coin * sum(
        option_fee_coin(coin, period.option_taker, period.option_cap) for coin in coins
    )
    payoff = abs(settle - entry.strike)
    delivery_fee = settle * option_fee_coin(
        payoff / settle, period.delivery_rate, period.delivery_cap
    )
    gross = side * (payoff - mid) + gross_hedge.hedge_pnl + gross_hedge.funding_pnl
    net = (
        side * (payoff - fill)
        - option_fee
        - delivery_fee
        + net_hedge.hedge_pnl
        + net_hedge.funding_pnl
        - net_hedge.trading_cost
    )
    return {
        "snapshot_id": entry.snapshot_id,
        "entry_date": datetime.fromtimestamp(entry.as_of_ns / 1e9, UTC).date().isoformat(),
        "expiry_date": expiry_day.isoformat(),
        "days": (entry.expiry_ns - entry.as_of_ns) / DAY_NS,
        "hedge": hedge,
        "strike": entry.strike,
        "forward": entry.forward,
        "entry_vol": entry.entry_vol,
        "premium_mid": mid,
        "premium_fill": fill,
        "settle": settle,
        "payoff": payoff,
        "hedge_pnl": gross_hedge.hedge_pnl,
        "funding_pnl": gross_hedge.funding_pnl,
        "option_fees": option_fee,
        "delivery_fee": delivery_fee,
        "hedge_costs": net_hedge.trading_cost,
        "gamma_pnl": gross_hedge.gamma_pnl,
        "gross_pnl": gross,
        "net_pnl": net,
        "gross_frac": gross / entry.forward,
        "net_frac": net / entry.forward,
        "rebalances": gross_hedge.rebalances,
        "current_iv_days": used_current,
        "smile_slope": slope,
    }


def run_straddles(
    data_dir: Path, config: AppConfig, underlying: str, source: str, hedge: str
) -> pl.DataFrame:
    """Every settled trade of a source and hedge variant, plus summary statistics."""
    if hedge not in HEDGES:
        raise ValueError(f"hedge must be one of {HEDGES}")
    if source not in SOURCE_TIME:
        raise ValueError("source must be tardis_monthly or recorder_daily; tardis_daily needs D-02")
    cfg = config.straddle
    fees = load_fees(cfg.costs.fee_table)
    market = load_market(data_dir, underlying)
    surfaces = _surfaces(data_dir, config, underlying, source)
    current = _current_vols(data_dir, config, underlying)
    rows = []
    for label, surface in surfaces.items():
        quotes = clean_quotes_of(data_dir, underlying, label)
        if quotes is None:
            continue
        entry = choose_entry(surface, quotes.filter(pl.col("drop_reason").is_null()), cfg)
        if entry is None:
            continue
        vols = {ts: value.get(entry.expiry_ns, math.nan) for ts, value in current.items()}
        finite = {ts: value for ts, value in vols.items() if math.isfinite(value)}
        row = run_trade(entry, market, fees, cfg, underlying, hedge, finite)
        if row is not None:
            rows.append(row)
    table = pl.DataFrame(rows) if rows else pl.DataFrame()
    results = config.results_dir()
    name = f"{source}_{hedge}_{underlying}"
    (results / "tables").mkdir(parents=True, exist_ok=True)
    table.write_csv(results / "tables" / f"straddles_{name}.csv")
    metrics = straddle_metrics(table, config, source)
    metrics.update(
        {"underlying": underlying, "source": source, "hedge": hedge, "fee_source": fees.source}
    )
    path = results / "metrics" / f"straddle_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if table.height:
        dates = [datetime.fromisoformat(day) for day in table["entry_date"].to_list()]
        years = np.array([d.year + (d.timetuple().tm_yday - 1) / 365.0 for d in dates])
        figures.straddle_pnl(
            years,
            table["gross_frac"].to_numpy(),
            table["net_frac"].to_numpy(),
            f"{underlying}: {cfg.side} straddles, {source}, hedge {hedge}",
            config.reports_dir() / "figures" / f"straddle_pnl_{name}.png",
        )
    append_run(
        results,
        f"straddle {source} {hedge}",
        config_hash(config),
        underlying=underlying,
        params={"source": source, "hedge": hedge, "side": cfg.side},
        metrics={
            "trades": metrics["trades"],
            "net_mean_frac": metrics.get("net", {}).get("mean"),
            "net_nw_t": metrics.get("net", {}).get("nw_t"),
        },
    )
    return table


def _current_vols(
    data_dir: Path, config: AppConfig, underlying: str
) -> dict[int, dict[int, float]]:
    """ATM vol by expiry of every stored canonical surface, keyed by as-of."""
    out: dict[int, dict[int, float]] = {}
    for surface in _surfaces(data_dir, config, underlying, "recorder_daily").values():
        out[surface.as_of_ns] = {
            item.expiry_ns: math.sqrt(float(item.svi.w(np.array(0.0))) / item.t)
            for item in surface.slices
        }
    return out


def straddle_metrics(table: pl.DataFrame, config: AppConfig, source: str) -> dict[str, Any]:
    """Gross and net summaries as a fraction of the forward, gamma R^2 and a by-year table."""
    boot = config.straddle.bootstrap
    if table.is_empty():
        return {"trades": 0}
    spacing = 30.0 if source == "tardis_monthly" else 1.0
    overlap = max(math.ceil(config.straddle.target_days / spacing) - 1, 0)

    def stats(values: F64) -> dict[str, Any]:
        return summarize(
            values, overlap=overlap, draws=boot.n, mean_block=boot.mean_block, seed=boot.seed
        )

    gross = table["gross_frac"].to_numpy()
    theory = (table["gamma_pnl"] / table["forward"]).to_numpy()
    with_funding = (table["gross_pnl"] / table["forward"]).to_numpy()
    # Funding is carry, not gamma, so the decomposition keeps it as its own term.
    realized = ((table["gross_pnl"] - table["funding_pnl"]) / table["forward"]).to_numpy()
    residual = realized - theory
    total = float(np.sum((realized - realized.mean()) ** 2))
    identity = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else math.nan
    slope, intercept = np.polyfit(theory, realized, 1) if theory.size > 1 else (math.nan, math.nan)
    years = table.with_columns(pl.col("entry_date").str.slice(0, 4).alias("year"))
    missing = missing_months(table, source)
    by_year = {
        str(year): {
            "trades": part.height,
            "gross_mean": float(part["gross_frac"].mean()),  # type: ignore[arg-type]
            "net_mean": float(part["net_frac"].mean()),  # type: ignore[arg-type]
            "months_missing": [month for month in missing if month.startswith(str(year))],
        }
        for (year,), part in sorted(years.partition_by("year", as_dict=True).items())
    }
    for month in missing:
        by_year.setdefault(
            month[:4], {"trades": 0, "months_missing": [m for m in missing if m[:4] == month[:4]]}
        )
    return {
        "trades": table.height,
        "first_entry": str(table["entry_date"].min()),
        "last_entry": str(table["entry_date"].max()),
        "months_missing": missing,
        "gross": stats(gross),
        "net": stats(table["net_frac"].to_numpy()),
        "gamma_r2": _r2(theory, realized),
        "gamma_r2_with_funding": _r2(theory, with_funding),
        "gamma_r2_identity": identity,
        "gamma_slope": float(slope),
        "gamma_intercept": float(intercept),
        "funding_mean": float(np.mean(with_funding - realized)),
        "hedge_error_var": float(np.var(residual, ddof=1)) if residual.size > 1 else math.nan,
        "gross_var": float(np.var(gross, ddof=1)) if gross.size > 1 else math.nan,
        "by_year": dict(sorted(by_year.items())),
        "overlap_lags": overlap,
    }


def _r2(x: F64, y: F64) -> float:
    """R^2 of the least-squares line of y on x."""
    if x.size < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def missing_months(table: pl.DataFrame, source: str) -> list[str]:
    """For monthly entries, every month between the first and last entry that has none.

    The spec asks for entries from 2020-03 to the latest month, so a hole is reported, never
    filled or hidden.
    """
    if source != "tardis_monthly" or table.is_empty():
        return []
    have = {day[:7] for day in table["entry_date"].to_list()}
    first = date.fromisoformat(str(table["entry_date"].min()))
    last = date.fromisoformat(str(table["entry_date"].max()))
    months = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        label = f"{year:04d}-{month:02d}"
        if label not in have:
            months.append(label)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months

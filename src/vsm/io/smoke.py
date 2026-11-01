"""Call each Deribit public endpoint once and check the fields the schemas rely on."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from vsm.conventions import datetime_to_ns, ns_to_ms
from vsm.io.deribit import DeribitClient

DAY_MS = 86_400_000

OPTION_SUMMARY_FIELDS = {
    "instrument_name",
    "base_currency",
    "creation_timestamp",
    "bid_price",
    "ask_price",
    "mid_price",
    "mark_price",
    "last",
    "mark_iv",
    "underlying_index",
    "underlying_price",
    "interest_rate",
    "open_interest",
    "volume",
}
FUTURE_SUMMARY_FIELDS = {
    "instrument_name",
    "base_currency",
    "creation_timestamp",
    "bid_price",
    "ask_price",
    "mark_price",
    "current_funding",
    "funding_8h",
    "open_interest",
}
INSTRUMENT_FIELDS = {
    "instrument_name",
    "base_currency",
    "kind",
    "expiration_timestamp",
    "tick_size",
    "min_trade_amount",
    "contract_size",
    "settlement_currency",
    "is_active",
    "settlement_period",
}
TICKER_FIELDS = {
    "instrument_name",
    "timestamp",
    "best_bid_amount",
    "best_ask_amount",
    "bid_iv",
    "ask_iv",
    "greeks",
}
CHART_FIELDS = {"ticks", "open", "high", "low", "close", "volume", "status"}
FUNDING_FIELDS = {"timestamp", "interest_1h", "interest_8h", "index_price", "prev_index_price"}
DELIVERY_FIELDS = {"date", "delivery_price"}


@dataclass(frozen=True)
class SmokeResult:
    """Outcome of one endpoint check."""

    endpoint: str
    ok: bool
    detail: str


def _rows_have(rows: list[dict[str, Any]], required: set[str]) -> str:
    if not rows:
        raise ValueError("empty result")
    missing = required - set().union(*(row.keys() for row in rows))
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    return f"{len(rows)} rows"


def _check(name: str, probe: Callable[[], str]) -> SmokeResult:
    try:
        return SmokeResult(name, True, probe())
    except Exception as exc:
        return SmokeResult(name, False, repr(exc)[:200])


def _dvol_detail(result: dict[str, Any]) -> str:
    data = result["data"]
    if not data or len(data[0]) != 5:
        raise ValueError("expected [ts, open, high, low, close] rows")
    return f"{len(data)} candles"


def run_smoke(
    client: DeribitClient, currencies: list[str], *, now: datetime | None = None
) -> list[SmokeResult]:
    """Run every endpoint check for ``currencies``."""
    end_ms = ns_to_ms(datetime_to_ns(now or datetime.now(UTC)))
    start_ms = end_ms - DAY_MS
    results: list[SmokeResult] = []
    for ccy in currencies:
        index = f"{ccy.lower()}_usd"
        perp = f"{ccy}-PERPETUAL"
        options = client.book_summary(ccy, "option") if ccy else []
        results.append(
            _check(
                f"get_book_summary_by_currency {ccy} option",
                lambda opts=options: _rows_have(opts, OPTION_SUMMARY_FIELDS),
            )
        )
        checks: list[tuple[str, Callable[[], str]]] = [
            (
                f"get_book_summary_by_currency {ccy} future",
                lambda c=ccy: _rows_have(client.book_summary(c, "future"), FUTURE_SUMMARY_FIELDS),
            ),
            (
                f"get_instruments {ccy} option",
                lambda c=ccy: _rows_have(client.instruments(c, "option"), INSTRUMENT_FIELDS),
            ),
            (
                f"get_instruments {ccy} future",
                lambda c=ccy: _rows_have(client.instruments(c, "future"), INSTRUMENT_FIELDS),
            ),
            (
                f"get_instruments {ccy} option expired",
                lambda c=ccy: _rows_have(
                    client.instruments(c, "option", expired=True), INSTRUMENT_FIELDS
                ),
            ),
            (f"get_index_price {index}", lambda i=index: f"{client.index_price(i):.2f}"),
            (
                f"ticker {ccy}",
                lambda opts=options: _rows_have(
                    [client.ticker(opts[0]["instrument_name"])], TICKER_FIELDS
                ),
            ),
            (
                f"get_volatility_index_data {ccy}",
                lambda c=ccy: _dvol_detail(client.volatility_index(c, start_ms, end_ms, "3600")),
            ),
            (
                f"get_tradingview_chart_data {perp}",
                lambda p=perp: _rows_have(
                    [client.chart_data(p, start_ms, end_ms, "60")], CHART_FIELDS
                ),
            ),
            (
                f"get_funding_rate_history {perp}",
                lambda p=perp: _rows_have(
                    client.funding_history(p, start_ms, end_ms), FUNDING_FIELDS
                ),
            ),
            (
                f"get_delivery_prices {index}",
                lambda i=index: _rows_have(
                    client.delivery_prices(i, 0, 3)["data"], DELIVERY_FIELDS
                ),
            ),
        ]
        results += [_check(name, probe) for name, probe in checks]
    return results

"""Shared fixtures: a mocked Deribit transport replaying trimmed JSON responses."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from vsm.io.deribit import DeribitClient

FIXTURES = Path(__file__).parent / "fixtures" / "deribit"
Handler = Callable[[httpx.Request], httpx.Response | None]


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def ok(result: Any) -> httpx.Response:
    return httpx.Response(200, json={"jsonrpc": "2.0", "result": result})


def replay(request: httpx.Request) -> httpx.Response:
    """Answer a public method from the fixture files; ETH reuses the BTC rows renamed."""
    method = request.url.path.rsplit("/", 1)[1]
    params = dict(request.url.params)
    ccy = params.get("currency", "BTC")

    def rename(rows: Any) -> Any:
        text = json.dumps(rows)
        return json.loads(text.replace("BTC", ccy)) if ccy != "BTC" else rows

    if method == "get_book_summary_by_currency":
        return ok(rename(load_fixture(f"book_{params['kind']}")))
    if method == "get_instruments":
        name = "expired" if params.get("expired") == "true" else params["kind"]
        return ok(rename(load_fixture(f"instruments_{name}")))
    if method == "get_index_price":
        return ok(load_fixture("index_price"))
    if method == "ticker":
        return ok({**load_fixture("ticker"), "instrument_name": params["instrument_name"]})
    if method == "get_volatility_index_data":
        start = int(params["start_timestamp"])
        return ok({"data": [[start, 40.0, 41.0, 39.0, 40.5]], "continuation": None})
    if method == "get_tradingview_chart_data":
        start = int(params["start_timestamp"])
        return ok(
            {
                "status": "ok",
                "ticks": [start],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [7.0],
                "cost": [700.0],
            }
        )
    if method == "get_funding_rate_history":
        return ok(
            [
                {
                    "timestamp": int(params["start_timestamp"]),
                    "index_price": 100.0,
                    "prev_index_price": 99.0,
                    "interest_8h": 1e-5,
                    "interest_1h": 1e-6,
                }
            ]
        )
    if method == "get_delivery_prices":
        return ok(
            {
                "data": [
                    {"date": "2026-11-01", "delivery_price": 77784.69},
                    {"date": "2026-10-31", "delivery_price": 77219.0},
                ],
                "records_total": 2,
            }
        )
    return httpx.Response(400, json={"error": {"code": -32601, "message": "Method not found"}})


@pytest.fixture
def make_client() -> Iterator[Callable[..., DeribitClient]]:
    """Factory for clients on the mock transport; ``override`` can intercept requests."""
    clients: list[DeribitClient] = []

    def build(override: Handler | None = None, **kwargs: Any) -> DeribitClient:
        def handler(request: httpx.Request) -> httpx.Response:
            answer = override(request) if override else None
            return answer if answer is not None else replay(request)

        kwargs.setdefault("max_rps", 1_000_000.0)
        kwargs.setdefault("sleep", lambda _: None)
        client = DeribitClient(
            "https://test.invalid/api/v2", transport=httpx.MockTransport(handler), **kwargs
        )
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.close()

from collections.abc import Callable

import httpx
import pytest

from vsm.io.deribit import DeribitClient, DeribitError, RateLimiter

Factory = Callable[..., DeribitClient]


def test_private_methods_are_refused(make_client: Factory) -> None:
    client = make_client()
    with pytest.raises(ValueError, match="public"):
        client.call("private/get_account_summary")


def test_booleans_are_lowercase_and_none_is_dropped(make_client: Factory) -> None:
    seen: list[httpx.Request] = []

    def spy(request: httpx.Request) -> None:
        seen.append(request)

    client = make_client(spy)
    client.instruments("BTC", "option", expired=True)
    client.call("public/get_index_price", index_name="btc_usd", extra=None)
    assert seen[0].url.params["expired"] == "true"
    assert "extra" not in seen[1].url.params


def test_retries_rate_limit_then_succeeds(make_client: Factory) -> None:
    attempts = {"n": 0}

    def limited(request: httpx.Request) -> httpx.Response | None:
        attempts["n"] += 1
        if attempts["n"] <= 2:
            return httpx.Response(
                200, json={"error": {"code": 10028, "message": "too_many_requests"}}
            )
        return None

    client = make_client(limited)
    assert client.index_price("btc_usd") > 0
    assert client.retries_used == 2


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_retries_http_errors(make_client: Factory, status: int) -> None:
    attempts = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response | None:
        attempts["n"] += 1
        return httpx.Response(status) if attempts["n"] == 1 else None

    client = make_client(flaky)
    assert client.index_price("btc_usd") > 0
    assert client.retries_used == 1


def test_retries_dropped_connection(make_client: Factory) -> None:
    attempts = {"n": 0}

    def dropped(request: httpx.Request) -> httpx.Response | None:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return None

    assert make_client(dropped).index_price("btc_usd") > 0


def test_invalid_json_is_transient(make_client: Factory) -> None:
    attempts = {"n": 0}

    def garbled(request: httpx.Request) -> httpx.Response | None:
        attempts["n"] += 1
        return httpx.Response(200, text="<html>") if attempts["n"] == 1 else None

    assert make_client(garbled).index_price("btc_usd") > 0


def test_gives_up_after_configured_retries(make_client: Factory) -> None:
    attempts = {"n": 0}

    def down(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503)

    client = make_client(down, retries=5)
    with pytest.raises(DeribitError):
        client.index_price("btc_usd")
    assert attempts["n"] == 6


def test_method_errors_do_not_retry(make_client: Factory) -> None:
    attempts = {"n": 0}

    def invalid(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(400, json={"error": {"code": -32602, "message": "Invalid params"}})

    client = make_client(invalid)
    with pytest.raises(DeribitError) as info:
        client.book_summary("BTC", "option")
    assert info.value.code == -32602
    assert attempts["n"] == 1


def test_rate_limiter_spaces_calls() -> None:
    now = {"t": 100.0}
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    limiter = RateLimiter(5.0, clock=lambda: now["t"], sleep=sleep)
    for _ in range(3):
        limiter.wait()
    assert slept == pytest.approx([0.2, 0.2])


def test_client_context_manager_closes(make_client: Factory) -> None:
    with make_client() as client:
        assert client.delivery_prices("btc_usd", 0, 1)["records_total"] == 2

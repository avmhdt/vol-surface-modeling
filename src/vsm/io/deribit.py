"""Deribit public REST client. Public endpoints only; no key, no account."""

import time
from collections.abc import Callable
from typing import Any

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

TOO_MANY_REQUESTS = 10028


class DeribitError(RuntimeError):
    """Deribit returned a JSON-RPC error that retrying will not fix."""

    def __init__(self, method: str, code: int | None, message: str) -> None:
        super().__init__(f"{method}: {code} {message}")
        self.code = code


class _Transient(Exception):
    """Failure worth retrying: rate limit, 5xx or a dropped connection."""


class RateLimiter:
    """Blocking limiter that keeps calls at least ``1 / max_rps`` seconds apart."""

    def __init__(
        self,
        max_rps: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 1.0 / max_rps
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0

    def wait(self) -> None:
        """Block until the next call is allowed."""
        now = self._clock()
        if now < self._next:
            self._sleep(self._next - now)
            now = self._next
        self._next = now + self._interval


class DeribitClient:
    """Thin client for the public methods this project uses.

    Timestamps are venue milliseconds; premiums and vols stay in venue units.
    """

    def __init__(
        self,
        base_url: str,
        *,
        max_rps: float = 5.0,
        retries: int = 5,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/", timeout=timeout_s, transport=transport
        )
        self._limiter = RateLimiter(max_rps, sleep=sleep)
        self._attempts = retries + 1
        self._sleep = sleep
        self.retries_used = 0

    def close(self) -> None:
        """Release the HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "DeribitClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def call(self, method: str, **params: Any) -> Any:
        """Call one public method and return its ``result``, retrying transient failures."""
        if not method.startswith("public/"):
            raise ValueError(f"only public methods are allowed: {method}")
        query = {k: _encode(v) for k, v in params.items() if v is not None}
        retrying = Retrying(
            stop=stop_after_attempt(self._attempts),
            wait=wait_exponential_jitter(initial=0.5, max=30.0),
            retry=retry_if_exception_type(_Transient),
            before_sleep=self._count_retry,
            sleep=self._sleep,
            reraise=True,
        )
        try:
            return retrying(self._once, method, query)
        except _Transient as exc:
            raise DeribitError(method, None, str(exc)) from exc

    def _count_retry(self, state: RetryCallState) -> None:
        self.retries_used += 1

    def _once(self, method: str, query: dict[str, str]) -> Any:
        self._limiter.wait()
        try:
            response = self._http.get(method, params=query)
        except httpx.TransportError as exc:
            raise _Transient(repr(exc)) from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise _Transient(f"HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise _Transient(f"invalid JSON, HTTP {response.status_code}") from exc
        error = body.get("error")
        if error:
            if error.get("code") == TOO_MANY_REQUESTS:
                raise _Transient(str(error))
            raise DeribitError(method, error.get("code"), str(error.get("message")))
        return body["result"]

    def book_summary(self, currency: str, kind: str) -> list[dict[str, Any]]:
        """``get_book_summary_by_currency``: one row per live instrument."""
        result: list[dict[str, Any]] = self.call(
            "public/get_book_summary_by_currency", currency=currency, kind=kind
        )
        return result

    def instruments(
        self, currency: str, kind: str, *, expired: bool = False
    ) -> list[dict[str, Any]]:
        """``get_instruments``: active, or recently expired with ``expired=True``."""
        result: list[dict[str, Any]] = self.call(
            "public/get_instruments", currency=currency, kind=kind, expired=expired
        )
        return result

    def index_price(self, index_name: str) -> float:
        """``get_index_price``: spot index in USD."""
        return float(self.call("public/get_index_price", index_name=index_name)["index_price"])

    def ticker(self, instrument_name: str) -> dict[str, Any]:
        """``ticker``: top of book with sizes, IVs and greeks for one instrument."""
        result: dict[str, Any] = self.call("public/ticker", instrument_name=instrument_name)
        return result

    def volatility_index(
        self, currency: str, start_ms: int, end_ms: int, resolution: str
    ) -> dict[str, Any]:
        """``get_volatility_index_data``: DVOL candles, newest page first."""
        result: dict[str, Any] = self.call(
            "public/get_volatility_index_data",
            currency=currency,
            start_timestamp=start_ms,
            end_timestamp=end_ms,
            resolution=resolution,
        )
        return result

    def chart_data(
        self, instrument_name: str, start_ms: int, end_ms: int, resolution: str
    ) -> dict[str, Any]:
        """``get_tradingview_chart_data``: OHLC bars as parallel arrays."""
        result: dict[str, Any] = self.call(
            "public/get_tradingview_chart_data",
            instrument_name=instrument_name,
            start_timestamp=start_ms,
            end_timestamp=end_ms,
            resolution=resolution,
        )
        return result

    def funding_history(
        self, instrument_name: str, start_ms: int, end_ms: int
    ) -> list[dict[str, Any]]:
        """``get_funding_rate_history``: hourly funding records."""
        result: list[dict[str, Any]] = self.call(
            "public/get_funding_rate_history",
            instrument_name=instrument_name,
            start_timestamp=start_ms,
            end_timestamp=end_ms,
        )
        return result

    def delivery_prices(self, index_name: str, offset: int, count: int) -> dict[str, Any]:
        """``get_delivery_prices``: daily settlement values, newest first."""
        result: dict[str, Any] = self.call(
            "public/get_delivery_prices", index_name=index_name, offset=offset, count=count
        )
        return result


def _encode(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)

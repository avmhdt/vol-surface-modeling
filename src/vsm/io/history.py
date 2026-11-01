"""Free Deribit history: DVOL, perpetual OHLC, funding and delivery prices.

Backfill and daily append are the same code: each series resumes from its latest stored
row unless a start date is given, and writes are keyed upserts, so re-runs change nothing.
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from vsm.conventions import date_to_ns, datetime_to_ns, ms_to_ns, ns_to_ms
from vsm.io import storage
from vsm.io.deribit import DeribitClient

HISTORY_START = date(2018, 8, 1)  # just before BTC-PERPETUAL was listed
HOUR_MS = 3_600_000
PERP_CHUNK_MS = 4_800 * HOUR_MS  # the endpoint returns at most about 5,000 bars
WINDOW_CHUNK_MS = 720 * HOUR_MS  # DVOL and funding return under 1,000 rows per call
DELIVERY_PAGE = 1_000
BAR_RESOLUTION_S = 3_600

log = structlog.get_logger("vsm.history")


def _chunks(start_ms: int, end_ms: int, step_ms: int) -> Iterator[tuple[int, int]]:
    left = start_ms
    while left < end_ms:
        right = min(left + step_ms, end_ms)
        yield left, right
        left = right


def _resume_ms(directory: Path, column: str, start: date | None) -> int:
    """Where a series starts: the given date, else its latest stored row, else the epoch."""
    if start is not None:
        return ns_to_ms(date_to_ns(start))
    files = sorted(directory.glob("part-*.parquet"))
    if files:
        latest = pl.read_parquet(files[-1], columns=[column])[column].max()
        if isinstance(latest, int):
            return ns_to_ms(latest)
    return ns_to_ms(date_to_ns(HISTORY_START))


def _write_by_year(frame: pl.DataFrame, directory: Path, key: str) -> None:
    years = frame.with_columns(pl.col(key).cast(pl.Datetime("ns", "UTC")).dt.year().alias("_year"))
    for (year,), part in years.partition_by("_year", as_dict=True).items():
        storage.upsert_parquet(part.drop("_year"), directory / f"part-{year}.parquet", [key])


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def _is_complete(ts_ms: int, end_ms: int) -> bool:
    return ts_ms + HOUR_MS <= end_ms


def ingest_perp(
    client: DeribitClient, data_dir: Path, symbol: str, start: date | None, end_ms: int
) -> int:
    """Hourly perpetual OHLC; bars still open at ``end_ms`` are left out."""
    directory = storage.raw_dir(data_dir) / "perp_ohlc" / f"symbol={symbol}"
    rows: list[dict[str, Any]] = []
    for left, right in _chunks(_resume_ms(directory, "ts", start), end_ms, PERP_CHUNK_MS):
        bars = client.chart_data(symbol, left, right, "60")
        if bars.get("status") != "ok":
            continue
        fields = zip(
            bars["ticks"],
            bars["open"],
            bars["high"],
            bars["low"],
            bars["close"],
            bars["volume"],
            strict=True,
        )
        rows += [
            {
                "ts": ms_to_ns(ts),
                "venue": storage.VENUE,
                "symbol": symbol,
                "resolution_s": BAR_RESOLUTION_S,
                "open": float(o),
                "high": float(h),
                "low": float(lo),
                "close": float(c),
                "volume": float(v),
            }
            for ts, o, h, lo, c, v in fields
            if _is_complete(ts, end_ms)
        ]
    if rows:
        _write_by_year(pl.DataFrame(rows, schema=storage.PERP_OHLC_SCHEMA), directory, "ts")
    return len(rows)


def ingest_dvol(
    client: DeribitClient, data_dir: Path, underlying: str, start: date | None, end_ms: int
) -> int:
    """Hourly DVOL candles in percent; candles still open at ``end_ms`` are left out."""
    directory = storage.raw_dir(data_dir) / "dvol" / f"underlying={underlying}"
    rows: list[dict[str, Any]] = []
    for left, right in _chunks(_resume_ms(directory, "ts", start), end_ms, WINDOW_CHUNK_MS):
        for ts, o, h, lo, c in client.volatility_index(underlying, left, right, "3600")["data"]:
            if _is_complete(ts, end_ms):
                rows.append(
                    {
                        "ts": ms_to_ns(ts),
                        "underlying": underlying,
                        "open": float(o),
                        "high": float(h),
                        "low": float(lo),
                        "close": float(c),
                    }
                )
    if rows:
        _write_by_year(pl.DataFrame(rows, schema=storage.DVOL_SCHEMA), directory, "ts")
    return len(rows)


def ingest_funding(
    client: DeribitClient, data_dir: Path, symbol: str, start: date | None, end_ms: int
) -> int:
    """Hourly perpetual funding records as delivered; early records have null fields."""
    directory = storage.raw_dir(data_dir) / "funding" / f"symbol={symbol}"
    rows: list[dict[str, Any]] = []
    for left, right in _chunks(_resume_ms(directory, "ts", start), end_ms, WINDOW_CHUNK_MS):
        rows += [
            {
                "ts": ms_to_ns(r["timestamp"]),
                "symbol": symbol,
                "interest_1h": _float(r.get("interest_1h")),
                "interest_8h": _float(r.get("interest_8h")),
                "index_px": _float(r.get("index_price")),
                "prev_index_px": _float(r.get("prev_index_price")),
            }
            for r in client.funding_history(symbol, left, right)
        ]
    if rows:
        frame = pl.DataFrame(rows, schema=storage.FUNDING_SCHEMA).unique("ts", keep="last")
        _write_by_year(frame, directory, "ts")
    return len(rows)


def ingest_delivery(client: DeribitClient, data_dir: Path, index_name: str) -> int:
    """All daily delivery prices of an index; the endpoint pages newest first."""
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = client.delivery_prices(index_name, offset, DELIVERY_PAGE)
        batch = page["data"]
        rows += [
            {
                "date": date.fromisoformat(r["date"]),
                "index_name": index_name,
                "delivery_price": float(r["delivery_price"]),
            }
            for r in batch
        ]
        offset += len(batch)
        if not batch or offset >= page["records_total"]:
            break
    frame = pl.DataFrame(rows, schema=storage.DELIVERY_SCHEMA)
    path = storage.raw_dir(data_dir) / "delivery" / f"index_name={index_name}" / "delivery.parquet"
    storage.upsert_parquet(frame, path, ["date"])
    return frame.height


def ingest_history(
    client: DeribitClient,
    data_dir: Path,
    *,
    what: list[str],
    currencies: list[str],
    start: date | None,
    end: date | None,
    now: datetime | None = None,
) -> dict[str, int]:
    """Run the requested series for each currency; returns rows fetched per series."""
    latest = now or datetime.now(UTC)
    if end is not None:
        limit = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)
        latest = min(latest, limit)
    end_ms = ns_to_ms(datetime_to_ns(latest))
    unknown = set(what) - {"dvol", "perp", "funding", "delivery"}
    if unknown:
        raise ValueError(f"unknown series: {sorted(unknown)}")
    counts: dict[str, int] = {}
    for ccy in currencies:
        perp = f"{ccy}-PERPETUAL"
        if "dvol" in what:
            counts[f"dvol {ccy}"] = ingest_dvol(client, data_dir, ccy, start, end_ms)
        if "perp" in what:
            counts[f"perp {ccy}"] = ingest_perp(client, data_dir, perp, start, end_ms)
        if "funding" in what:
            counts[f"funding {ccy}"] = ingest_funding(client, data_dir, perp, start, end_ms)
        if "delivery" in what:
            counts[f"delivery {ccy}"] = ingest_delivery(client, data_dir, f"{ccy.lower()}_usd")
    log.info("history_ingested", **{k.replace(" ", "_"): v for k, v in counts.items()})
    return counts

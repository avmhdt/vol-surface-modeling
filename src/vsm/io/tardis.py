"""Free Tardis first-of-month ``options_chain`` files to 08:15 UTC snapshots (FR-04, DQ-15).

The files are large, up to 18 GB compressed in 2026, and only one moment of each is needed. The
stream is read as it arrives, filtered, and never written to disk, and it stops once
``local_timestamp`` passes the end of the window. That only works if the column is in order, so
the stream counts every step backwards and reports the largest.

Downloads are capped at ``tardis.max_bytes_per_second``, one file at a time. Between files
``IngestGuard`` pauses the run if free space falls below ``tardis.min_free_gb``, or if the crypto
recorder, which shares this machine and its network link, restarts, shows new resyncs, opens a new
Binance connection, or reports an ingest p99 above ``tardis.max_ingest_p99_us``.
Paid daily history is not approved (D-02), so nothing here uses an API key.
"""

import io
import json
import shutil
import time
import zlib
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import structlog

from vsm.config import TardisConfig
from vsm.conventions import datetime_to_ns, us_to_ns
from vsm.io import storage

CHUNK_BYTES = 1 << 20
BLOCK_BYTES = 32 << 20
STALENESS_S = 300  # DQ-15
UNDERLYINGS = ("BTC", "ETH")
CONNECT_RETRIES = 3
PAUSE_POLL_S = 300.0
STATUS_MAX_AGE_NS = 300_000_000_000
BINANCE_VENUE = "binance-usdm"

CSV_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "exchange": pl.String,
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "local_timestamp": pl.Int64,
    "type": pl.String,
    "strike_price": pl.Float64,
    "expiration": pl.Int64,
    "open_interest": pl.Float64,
    "last_price": pl.Float64,
    "bid_price": pl.Float64,
    "bid_amount": pl.Float64,
    "bid_iv": pl.Float64,
    "ask_price": pl.Float64,
    "ask_amount": pl.Float64,
    "ask_iv": pl.Float64,
    "mark_price": pl.Float64,
    "mark_iv": pl.Float64,
    "underlying_index": pl.String,
    "underlying_price": pl.Float64,
    "delta": pl.Float64,
    "gamma": pl.Float64,
    "vega": pl.Float64,
    "theta": pl.Float64,
    "rho": pl.Float64,
}

log = structlog.get_logger("vsm.tardis")


@dataclass
class IngestStats:
    """What one file cost and what it held."""

    date: str
    total_bytes: int = 0
    bytes_read: int = 0
    decompressed_bytes: int = 0
    rows_read: int = 0
    rows_kept: int = 0
    seconds: float = 0.0
    inversions: int = 0
    max_regression_ms: float = 0.0
    last_local_ts_us: int = 0
    reached_end: bool = False


class ByteRateLimiter:
    """Blocks so the long-run byte rate stays at or below ``rate`` bytes per second."""

    def __init__(
        self,
        rate: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._rate = rate
        self._clock = clock
        self._sleep = sleep
        self._start = clock()
        self._sent = 0

    def acquire(self, size: int) -> None:
        """Account for ``size`` bytes, sleeping until they are within the rate."""
        self._sent += size
        due = self._sent / self._rate
        elapsed = self._clock() - self._start
        if due > elapsed:
            self._sleep(due - elapsed)


class IngestPaused(RuntimeError):
    """The guard held the run back and the caller asked to stop rather than wait."""


@dataclass(frozen=True)
class CryptoHealth:
    """What the guard reads from the crypto recorder's status file."""

    run: str
    resyncs: int
    ingest_p99_us: float
    binance_connection: int


def _connection_serial(conn_ids: list[str]) -> int:
    """Highest ``<name>.g<n>``: a venue's ``n`` counts every connection its run has opened."""
    serials = [conn_id.rsplit(".g", 1)[-1] for conn_id in conn_ids]
    return max((int(serial) for serial in serials if serial.isdigit()), default=0)


class IngestGuard:
    """Why the bulk run should wait right now, if at all."""

    def __init__(self, cfg: TardisConfig) -> None:
        self._cfg = cfg
        self._status = Path(cfg.crypto_status_path).expanduser()
        self._baseline = self._read_status()

    def _read_status(self) -> CryptoHealth | None:
        """Counters from a live crypto recorder, else None."""
        if not self._status.exists():
            return None
        try:
            status = json.loads(self._status.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (
            status.get("stopped")
            or time.time_ns() - int(status.get("ts_ns", 0)) > STATUS_MAX_AGE_NS
        ):
            return None
        resyncs = 0
        worst = 0.0
        for venue in status.get("venues", {}).values():
            worst = max(worst, float(venue.get("ingest_us_p99") or 0.0))
            resyncs += sum(int(s.get("resyncs", 0)) for s in venue.get("streams", []))
        binance = status.get("venues", {}).get(BINANCE_VENUE, {})
        return CryptoHealth(
            run=str(status.get("run", "")),
            resyncs=resyncs,
            ingest_p99_us=worst,
            binance_connection=_connection_serial(binance.get("conns", [])),
        )

    def reason(self) -> str | None:
        """The condition holding the run back, or None when it may continue."""
        free_gb = shutil.disk_usage(self._cfg.free_space_path).free / 1e9
        if free_gb < self._cfg.min_free_gb:
            return f"free space {free_gb:.0f} GB is below {self._cfg.min_free_gb:.0f} GB"
        now = self._read_status()
        if now is None:
            return None
        base = self._baseline
        if base is not None:
            # A new run restarts every counter, so nothing else can be compared across one.
            if now.run != base.run:
                return f"crypto recorder restarted, run {base.run} to {now.run}"
            if now.resyncs > base.resyncs:
                return f"crypto recorder resyncs rose from {base.resyncs} to {now.resyncs}"
            if now.binance_connection > base.binance_connection:
                return (
                    "Binance reconnected, connection serial rose from "
                    f"{base.binance_connection} to {now.binance_connection}"
                )
        if now.ingest_p99_us > self._cfg.max_ingest_p99_us:
            limit = self._cfg.max_ingest_p99_us
            return f"crypto ingest p99 is {now.ingest_p99_us:.0f} us, above {limit:.0f} us"
        return None


def first_of_month_days(start: date, end: date) -> list[date]:
    """First days of months between two dates, inclusive."""
    days = []
    year, month = start.year, start.month
    if start.day > 1:
        month += 1
        year, month = (year + 1, 1) if month == 13 else (year, month)
    current = date(year, month, 1)
    while current <= end:
        days.append(current)
        month += 1
        year, month = (year + 1, 1) if month == 13 else (year, month)
        current = date(year, month, 1)
    return days


def file_url(base: str, day: date) -> str:
    """The dataset URL of one day."""
    return f"{base.rstrip('/')}/deribit/options_chain/{day:%Y/%m/%d}/OPTIONS.csv.gz"


def snapshot_path(data_dir: Path, underlying: str, day: date, label: str) -> Path:
    """One underlying's extracted snapshot."""
    return (
        data_dir
        / "interim"
        / "tardis_snapshots"
        / f"underlying={underlying}"
        / f"date={day.isoformat()}"
        / f"part-{label}.parquet"
    )


def _decompressed_blocks(chunks: Iterator[bytes], stats: IngestStats) -> Iterator[bytes]:
    """Whole CSV lines, in blocks, from a stream of gzip bytes."""
    inflate = zlib.decompressobj(31)
    pending = b""
    for chunk in chunks:
        pending += inflate.decompress(chunk)
        if len(pending) >= BLOCK_BYTES:
            cut = pending.rfind(b"\n") + 1
            stats.decompressed_bytes += cut
            yield pending[:cut]
            pending = pending[cut:]
    pending += inflate.flush()
    if pending:
        stats.decompressed_bytes += len(pending)
        yield pending


def _parse(block: bytes, *, header: bool) -> pl.DataFrame:
    return pl.read_csv(
        io.BytesIO(block),
        has_header=False,
        skip_rows=1 if header else 0,
        new_columns=list(CSV_SCHEMA),
        schema_overrides=CSV_SCHEMA,
        truncate_ragged_lines=True,
    )


def _track_order(block: pl.DataFrame, previous: int, stats: IngestStats) -> int:
    """Count steps backwards in ``local_timestamp``, inside the block and across blocks."""
    times = block["local_timestamp"].to_numpy()
    if not times.size:
        return previous
    steps = times[1:] - times[:-1]
    backwards = steps[steps < 0]
    if previous > times[0]:
        stats.inversions += 1
        stats.max_regression_ms = max(stats.max_regression_ms, (previous - times[0]) / 1e3)
    if backwards.size:
        stats.inversions += int(backwards.size)
        stats.max_regression_ms = max(stats.max_regression_ms, float(-backwards.min()) / 1e3)
    return int(times[-1])


def latest_quotes(blocks: list[pl.DataFrame]) -> pl.DataFrame:
    """The last update per instrument."""
    return (
        pl.concat(blocks)
        .sort("local_timestamp", maintain_order=True)
        .unique("symbol", keep="last", maintain_order=True)
    )


def read_snapshot_window(
    client: httpx.Client,
    url: str,
    as_of_us: int,
    end_us: int,
    rate: float,
    stats: IngestStats,
    *,
    raw_sink: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pl.DataFrame:
    """Stream a file until ``end_us``; keep each BTC or ETH option's last quote at as-of."""
    earliest = as_of_us - STALENESS_S * 1_000_000
    inverse = pl.col("symbol").str.starts_with("BTC-") | pl.col("symbol").str.starts_with("ETH-")
    limiter = ByteRateLimiter(rate, sleep=sleep)
    kept: list[pl.DataFrame] = []
    started = time.monotonic()
    with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        stats.total_bytes = int(response.headers.get("content-length", 0))
        sink = raw_sink.open("wb") if raw_sink is not None else None

        def chunks() -> Iterator[bytes]:
            for chunk in response.iter_raw(CHUNK_BYTES):
                stats.bytes_read += len(chunk)
                limiter.acquire(len(chunk))
                if sink is not None:
                    sink.write(chunk)
                yield chunk

        previous = 0
        first = True
        try:
            for block in _decompressed_blocks(chunks(), stats):
                rows = _parse(block, header=first)
                first = False
                stats.rows_read += rows.height
                previous = _track_order(rows, previous, stats)
                stats.last_local_ts_us = previous
                window = rows.filter(
                    inverse & pl.col("local_timestamp").is_between(earliest, as_of_us)
                )
                if window.height:
                    kept.append(latest_quotes([window]))
                if previous >= end_us:
                    stats.reached_end = True
                    break
        finally:
            if sink is not None:
                sink.close()
    stats.seconds = time.monotonic() - started
    empty = pl.DataFrame(schema=CSV_SCHEMA)
    return latest_quotes(kept) if kept else empty


def to_option_quotes(
    rows: pl.DataFrame, underlying: str, label: str, as_of_us: int
) -> pl.DataFrame:
    """DQ-15 selection, mapped to the ``option_quotes`` schema with ``source=tardis_free``."""
    expiry_us = pl.col("expiration")
    selected = rows.filter(
        pl.col("symbol").str.starts_with(f"{underlying}-") & (expiry_us > as_of_us)
    )
    mapped = selected.select(
        pl.lit(f"{storage.VENUE}-{underlying}-{label}").alias("snapshot_id"),
        us_to_ns(pl.col("local_timestamp")).alias("ts_recv"),
        us_to_ns(pl.col("timestamp")).alias("ts_exchange"),
        pl.lit(storage.VENUE).alias("venue"),
        pl.col("symbol"),
        pl.lit(underlying).alias("underlying"),
        us_to_ns(expiry_us).alias("expiry_ns"),
        pl.col("strike_price").alias("strike"),
        pl.when(pl.col("type") == "call").then(1).otherwise(-1).cast(pl.Int8).alias("cp"),
        pl.col("bid_price").alias("bid_px"),
        pl.col("ask_price").alias("ask_px"),
        ((pl.col("bid_price") + pl.col("ask_price")) / 2.0).alias("mid_px"),
        pl.col("mark_price").alias("mark_px"),
        pl.col("last_price").alias("last_px"),
        pl.col("mark_iv").alias("mark_iv_pct"),
        pl.col("underlying_index"),
        pl.col("underlying_price").alias("underlying_px"),
        pl.lit(0.0).alias("interest_rate"),
        pl.col("open_interest"),
        pl.lit(None, dtype=pl.Float64).alias("volume_24h"),
        pl.col("bid_amount").alias("bid_qty"),
        pl.col("ask_amount").alias("ask_qty"),
        pl.col("bid_iv").alias("bid_iv_pct"),
        pl.col("ask_iv").alias("ask_iv_pct"),
        pl.col("delta"),
        pl.col("gamma"),
        pl.col("vega"),
        pl.lit("tardis_free").alias("source"),
    )
    return mapped.select(list(storage.OPTION_QUOTES_SCHEMA))


def as_of_of(day: date, at_utc: str) -> datetime:
    """The snapshot time on one day."""
    hour, minute = (int(part) for part in at_utc.split(":"))
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


def ingest_day(
    client: httpx.Client,
    data_dir: Path,
    base_url: str,
    day: date,
    at_utc: str,
    cfg: TardisConfig,
    *,
    keep_raw: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> IngestStats | None:
    """Extract one day. Returns None if its snapshots already exist, so a run can resume."""
    as_of = as_of_of(day, at_utc)
    label = as_of.strftime("%Y%m%dT%H%MZ")
    paths = {u: snapshot_path(data_dir, u, day, label) for u in UNDERLYINGS}
    if all(path.exists() for path in paths.values()):
        return None
    hour, minute = (int(part) for part in cfg.window_utc[1].split(":"))
    end = as_of.replace(hour=hour, minute=minute)
    as_of_us = datetime_to_ns(as_of) // 1_000
    end_us = datetime_to_ns(end) // 1_000
    raw_sink = None
    if keep_raw:
        raw_sink = (
            data_dir / "raw" / "tardis" / "options_chain" / f"{day:%Y/%m/%d}" / "OPTIONS.csv.gz"
        )
        raw_sink.parent.mkdir(parents=True, exist_ok=True)
    stats = IngestStats(date=day.isoformat())
    rows = read_snapshot_window(
        client,
        file_url(base_url, day),
        as_of_us,
        end_us,
        cfg.max_bytes_per_second,
        stats,
        raw_sink=raw_sink,
        sleep=sleep,
    )
    for underlying, path in paths.items():
        quotes = to_option_quotes(rows, underlying, label, as_of_us)
        stats.rows_kept += quotes.height
        storage.write_parquet_atomic(quotes, path)
    log.info("tardis_day", **asdict(stats))
    return stats


def _record(results_dir: Path, stats: IngestStats) -> None:
    path = results_dir / "metrics" / "tardis_ingest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    if path.exists():
        entries = json.loads(path.read_text(encoding="utf-8"))
    entries[stats.date] = {k: v for k, v in asdict(stats).items() if k != "date"}
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ingest_months(
    client: httpx.Client,
    data_dir: Path,
    results_dir: Path,
    base_url: str,
    days: list[date],
    at_utc: str,
    cfg: TardisConfig,
    *,
    keep_raw: bool = False,
    stop_on_pause: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> list[IngestStats]:
    """Extract days one at a time; when the guard holds the run back, wait, or stop if asked.

    The same command resumes a stopped run, because finished days are skipped.
    """
    guard = IngestGuard(cfg)
    done = []
    for day in days:
        while (reason := guard.reason()) is not None:
            log.warning("tardis_paused", reason=reason, day=day.isoformat())
            if stop_on_pause:
                raise IngestPaused(reason)
            sleep(PAUSE_POLL_S)
        failures = 0
        while True:
            try:
                stats = ingest_day(
                    client, data_dir, base_url, day, at_utc, cfg, keep_raw=keep_raw, sleep=sleep
                )
                break
            except httpx.TransportError as exc:
                log.warning("tardis_retry", day=day.isoformat(), error=repr(exc)[:200])
                failures += 1
                if failures == CONNECT_RETRIES:
                    raise
                sleep(float(timedelta(minutes=1).total_seconds()))
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != httpx.codes.TOO_MANY_REQUESTS:
                    raise
                wait = float(exc.response.headers.get("retry-after", PAUSE_POLL_S))
                reason = f"Tardis refused the download, retry after {wait:.0f} s"
                log.warning("tardis_paused", reason=reason, day=day.isoformat())
                if stop_on_pause:
                    raise IngestPaused(reason) from exc
                sleep(wait)
        if stats is not None:
            _record(results_dir, stats)
            done.append(stats)
    return done

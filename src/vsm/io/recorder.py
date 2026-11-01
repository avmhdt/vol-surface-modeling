"""Hourly chain snapshots: one atomic file per snapshot_id, a gap log, a run log."""

import csv
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from vsm.conventions import datetime_to_ns, ms_to_ns, parse_option_symbol
from vsm.io import storage
from vsm.io.deribit import DeribitClient

ON_SCHEDULE_GRACE_S = 600
CANONICAL_TOLERANCE_MIN = 30
DISK_WARNING_FRACTION = 0.80
GAP_COLUMNS = ["slot_utc", "underlying", "dataset", "detected_at_utc", "reason"]
LABEL_FORMAT = "%Y%m%dT%H%MZ"

log = structlog.get_logger("vsm.recorder")


@dataclass(frozen=True)
class Slot:
    """Minute a snapshot is filed under; ``on_schedule`` is False for manual runs."""

    ts: datetime
    on_schedule: bool

    @property
    def label(self) -> str:
        return self.ts.strftime(LABEL_FORMAT)

    @property
    def day(self) -> str:
        return self.ts.strftime("%Y-%m-%d")


@dataclass
class DatasetOutcome:
    """Result of writing one dataset for one underlying."""

    underlying: str
    dataset: str
    snapshot_id: str
    status: str  # written | exists | failed
    rows: int = 0
    dropped: int = 0
    latency_ms: float = 0.0
    reason: str = ""


@dataclass
class RecordSummary:
    """Everything one ``vsm record`` run did."""

    slot: Slot
    outcomes: list[DatasetOutcome] = field(default_factory=list)
    ticker_rows: int = 0
    retries: int = 0

    @property
    def ok(self) -> bool:
        return all(o.status != "failed" for o in self.outcomes)


def snapshot_id(underlying: str, slot: Slot) -> str:
    """``{venue}-{underlying}-{YYYYMMDDTHHMMZ}``, the idempotency key."""
    return f"{storage.VENUE}-{underlying}-{slot.label}"


def current_slot(now: datetime, minute_utc: int) -> Slot:
    """Latest scheduled slot if ``now`` is within the grace period, else ``now``'s minute."""
    scheduled = now.replace(minute=minute_utc, second=0, microsecond=0)
    if scheduled > now:
        scheduled -= timedelta(hours=1)
    if (now - scheduled).total_seconds() <= ON_SCHEDULE_GRACE_S:
        return Slot(scheduled, True)
    return Slot(now.replace(second=0, microsecond=0), False)


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def instrument_row(raw: dict[str, Any], seen_ns: int) -> dict[str, Any]:
    """Map one ``get_instruments`` row to the ``instruments`` schema."""
    perpetual = raw.get("settlement_period") == "perpetual"
    return {
        "venue": storage.VENUE,
        "symbol": raw["instrument_name"],
        "underlying": raw["base_currency"],
        "contract_type": "perpetual" if perpetual else raw["kind"],
        "option_type": raw.get("option_type"),
        "strike": _float(raw.get("strike")),
        "expiry_ns": None if perpetual else ms_to_ns(raw["expiration_timestamp"]),
        "tick_size": float(raw["tick_size"]),
        "tick_size_steps": json.dumps(raw.get("tick_size_steps") or []),
        "lot_size": float(raw["min_trade_amount"]),
        "contract_size": float(raw["contract_size"]),
        "settlement_currency": raw["settlement_currency"],
        "is_active": bool(raw["is_active"]),
        "first_seen_ns": seen_ns,
        "last_seen_ns": seen_ns,
    }


def refresh_instruments(
    client: DeribitClient,
    data_dir: Path,
    underlying: str,
    kinds: list[str],
    seen_ns: int,
    *,
    expired: bool = False,
) -> pl.DataFrame:
    """Fetch instrument metadata and merge it into the stored table."""
    rows: list[dict[str, Any]] = []
    for kind in kinds:
        rows += [instrument_row(r, seen_ns) for r in client.instruments(underlying, kind)]
        if expired:
            rows += [
                instrument_row(r, seen_ns)
                for r in client.instruments(underlying, kind, expired=True)
            ]
    new = pl.DataFrame(rows, schema=storage.INSTRUMENTS_SCHEMA)
    return storage.upsert_parquet(
        new,
        storage.instruments_path(data_dir, underlying),
        ["symbol"],
        keep_first=("first_seen_ns",),
    )


def _stored_instruments(data_dir: Path, underlying: str) -> pl.DataFrame | None:
    path = storage.instruments_path(data_dir, underlying)
    return pl.read_parquet(path) if path.exists() else None


def _option_meta(instruments: pl.DataFrame | None) -> dict[str, tuple[int, float, int]]:
    if instruments is None:
        return {}
    rows = instruments.filter(pl.col("contract_type") == "option").select(
        "symbol", "expiry_ns", "strike", "option_type"
    )
    return {
        row[0]: (int(row[1]), float(row[2]), 1 if row[3] == "call" else -1)
        for row in rows.iter_rows()
    }


def _future_expiries(instruments: pl.DataFrame | None) -> dict[str, int | None]:
    if instruments is None:
        return {}
    rows = instruments.filter(pl.col("contract_type") != "option").select("symbol", "expiry_ns")
    return dict(rows.iter_rows())


def option_row(
    raw: dict[str, Any],
    meta: dict[str, tuple[int, float, int]],
    snapshot: str,
    ts_recv: int,
) -> dict[str, Any]:
    """Map one book-summary row to ``option_quotes``. Metadata wins; the symbol is checked."""
    symbol = raw["instrument_name"]
    parsed = parse_option_symbol(symbol)
    expiry_ns, strike, cp = meta.get(symbol, parsed)
    if (expiry_ns, strike, cp) != parsed:
        log.warning("symbol_metadata_mismatch", symbol=symbol, metadata=[expiry_ns, strike, cp])
    return {
        "snapshot_id": snapshot,
        "ts_recv": ts_recv,
        "ts_exchange": ms_to_ns(raw["creation_timestamp"]),
        "venue": storage.VENUE,
        "symbol": symbol,
        "underlying": raw["base_currency"],
        "expiry_ns": expiry_ns,
        "strike": strike,
        "cp": cp,
        "bid_px": _float(raw.get("bid_price")),
        "ask_px": _float(raw.get("ask_price")),
        "mid_px": _float(raw.get("mid_price")),
        "mark_px": _float(raw.get("mark_price")),
        "last_px": _float(raw.get("last")),
        "mark_iv_pct": _float(raw.get("mark_iv")),
        "underlying_index": raw.get("underlying_index"),
        "underlying_px": _float(raw.get("underlying_price")),
        "interest_rate": _float(raw.get("interest_rate")),
        "open_interest": _float(raw.get("open_interest")),
        "volume_24h": _float(raw.get("volume")),
        "bid_qty": None,
        "ask_qty": None,
        "bid_iv_pct": None,
        "ask_iv_pct": None,
        "delta": None,
        "gamma": None,
        "vega": None,
        "source": "recorder",
    }


def future_row(
    raw: dict[str, Any],
    expiries: dict[str, int | None],
    index_px: float,
    snapshot: str,
    ts_recv: int,
) -> dict[str, Any]:
    """Map one book-summary row to ``future_quotes``."""
    symbol = raw["instrument_name"]
    return {
        "snapshot_id": snapshot,
        "ts_recv": ts_recv,
        "ts_exchange": ms_to_ns(raw["creation_timestamp"]),
        "venue": storage.VENUE,
        "symbol": symbol,
        "underlying": raw["base_currency"],
        "expiry_ns": expiries.get(symbol),
        "bid_px": _float(raw.get("bid_price")),
        "ask_px": _float(raw.get("ask_price")),
        "mark_px": _float(raw.get("mark_price")),
        "index_px": index_px,
        "current_funding": _float(raw.get("current_funding")),
        "funding_8h": _float(raw.get("funding_8h")),
        "open_interest": _float(raw.get("open_interest")),
    }


def ticker_row(raw: dict[str, Any], snapshot: str, ts_recv: int) -> dict[str, Any]:
    """Map one ``ticker`` response to ``option_tickers``."""
    greeks = raw.get("greeks") or {}
    return {
        "snapshot_id": snapshot,
        "ts_recv": ts_recv,
        "ts_exchange": ms_to_ns(raw["timestamp"]),
        "venue": storage.VENUE,
        "symbol": raw["instrument_name"],
        "underlying": raw["instrument_name"].split("-")[0],
        "bid_qty": _float(raw.get("best_bid_amount")),
        "ask_qty": _float(raw.get("best_ask_amount")),
        "bid_iv_pct": _float(raw.get("bid_iv")),
        "ask_iv_pct": _float(raw.get("ask_iv")),
        "delta": _float(greeks.get("delta")),
        "gamma": _float(greeks.get("gamma")),
        "vega": _float(greeks.get("vega")),
    }


def _build_rows(
    raws: list[dict[str, Any]], build: Any, context: str
) -> tuple[list[dict[str, Any]], int]:
    """Apply ``build`` row by row; one malformed row is dropped and logged, never fatal."""
    rows: list[dict[str, Any]] = []
    dropped = 0
    for raw in raws:
        try:
            rows.append(build(raw))
        except (KeyError, TypeError, ValueError) as exc:
            dropped += 1
            log.warning("row_dropped", context=context, row=str(raw)[:200], error=repr(exc))
    return rows, dropped


def _now_ns() -> int:
    return datetime_to_ns(datetime.now(UTC))


def _record_quotes(
    client: DeribitClient,
    data_dir: Path,
    underlying: str,
    kind: str,
    slot: Slot,
    instruments: dict[str, Any],
) -> DatasetOutcome:
    dataset = storage.OPTION_QUOTES if kind == "option" else storage.FUTURE_QUOTES
    schema = storage.OPTION_QUOTES_SCHEMA if kind == "option" else storage.FUTURE_QUOTES_SCHEMA
    snapshot = snapshot_id(underlying, slot)
    path = storage.snapshot_path(data_dir, dataset, underlying, slot.day, snapshot)
    outcome = DatasetOutcome(underlying, dataset, snapshot, "failed")
    if path.exists():
        outcome.status = "exists"
        return outcome

    started = time.monotonic()
    try:
        raws = client.book_summary(underlying, kind)
        ts_recv = _now_ns()
        if not raws:
            raise ValueError("empty book summary")
        table = instruments.get(underlying)
        if table is None:
            table = _refresh_or_stored(client, data_dir, underlying, instruments, ts_recv)
        if kind == "option":
            meta = _option_meta(table)
            rows, dropped = _build_rows(
                raws, lambda raw: option_row(raw, meta, snapshot, ts_recv), snapshot
            )
        else:
            index_px = client.index_price(f"{underlying.lower()}_usd")
            expiries = _future_expiries(table)
            rows, dropped = _build_rows(
                raws,
                lambda raw: future_row(raw, expiries, index_px, snapshot, ts_recv),
                snapshot,
            )
        if not rows:
            raise ValueError("no usable rows")
        storage.write_parquet_atomic(pl.DataFrame(rows, schema=schema), path)
    except Exception as exc:
        outcome.reason = repr(exc)[:300]
        outcome.latency_ms = (time.monotonic() - started) * 1e3
        log.error(
            "snapshot_failed",
            snapshot_id=snapshot,
            dataset=dataset,
            error=outcome.reason,
        )
        return outcome

    outcome.status = "written"
    outcome.rows = len(rows)
    outcome.dropped = dropped
    outcome.latency_ms = (time.monotonic() - started) * 1e3
    log.info(
        "snapshot_written",
        snapshot_id=snapshot,
        dataset=dataset,
        rows=outcome.rows,
        dropped=dropped,
        latency_ms=round(outcome.latency_ms, 1),
        retries=client.retries_used,
    )
    return outcome


def _refresh_or_stored(
    client: DeribitClient,
    data_dir: Path,
    underlying: str,
    cache: dict[str, Any],
    seen_ns: int,
) -> pl.DataFrame | None:
    """Metadata for this run. A failed refresh falls back to the stored table."""
    table: pl.DataFrame | None
    try:
        table = refresh_instruments(client, data_dir, underlying, ["option", "future"], seen_ns)
    except Exception as exc:
        log.warning("instruments_refresh_failed", underlying=underlying, error=repr(exc)[:300])
        table = _stored_instruments(data_dir, underlying)
    cache[underlying] = table
    return table


def sweep_tickers(client: DeribitClient, data_dir: Path, underlying: str, slot: Slot) -> int:
    """Fetch ``ticker`` for every option in the snapshot; rows keep their own ts_recv.

    The sweep takes minutes, so it is a separate dataset written after the snapshot.
    """
    snapshot = snapshot_id(underlying, slot)
    quotes = storage.snapshot_path(data_dir, storage.OPTION_QUOTES, underlying, slot.day, snapshot)
    path = storage.snapshot_path(data_dir, storage.OPTION_TICKERS, underlying, slot.day, snapshot)
    if not quotes.exists() or path.exists():
        return 0
    rows: list[dict[str, Any]] = []
    failed = 0
    for symbol in pl.read_parquet(quotes, columns=["symbol"])["symbol"].to_list():
        try:
            raw = client.ticker(symbol)
            rows.append(ticker_row(raw, snapshot, _now_ns()))
        except Exception as exc:
            failed += 1
            log.warning("ticker_failed", symbol=symbol, error=repr(exc)[:200])
    if rows:
        frame = pl.DataFrame(rows, schema=storage.OPTION_TICKERS_SCHEMA)
        storage.write_parquet_atomic(frame, path)
    log.info("ticker_sweep_done", snapshot_id=snapshot, rows=len(rows), failed=failed)
    return len(rows)


def gaps_path(data_dir: Path) -> Path:
    """The append-only gap log."""
    return storage.raw_dir(data_dir) / "_gaps.csv"


def read_gaps(data_dir: Path) -> list[dict[str, str]]:
    """All logged gaps."""
    path = gaps_path(data_dir)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_gaps(data_dir: Path, rows: list[dict[str, str]]) -> int:
    """Append gaps not logged yet; returns how many were new."""
    known = {(g["slot_utc"], g["underlying"], g["dataset"]) for g in read_gaps(data_dir)}
    fresh = [r for r in rows if (r["slot_utc"], r["underlying"], r["dataset"]) not in known]
    if not fresh:
        return 0
    path = gaps_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GAP_COLUMNS, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerows(fresh)
    return len(fresh)


def captured_labels(data_dir: Path, dataset: str, underlying: str) -> set[str]:
    """``YYYYMMDDTHHMMZ`` of every snapshot file on disk."""
    files = storage.snapshot_files(data_dir, dataset, underlying)
    return {path.stem.rsplit("-", 1)[1] for path in files}


def label_time(label: str) -> datetime:
    """``YYYYMMDDTHHMMZ`` to an aware datetime."""
    return datetime.strptime(label, LABEL_FORMAT).replace(tzinfo=UTC)


def scheduled_slots(start: datetime, end: datetime, minute_utc: int) -> list[datetime]:
    """Scheduled hh:mm slots in ``[start, end)``."""
    slot = start.replace(minute=minute_utc, second=0, microsecond=0)
    if slot < start:
        slot += timedelta(hours=1)
    slots = []
    while slot < end:
        slots.append(slot)
        slot += timedelta(hours=1)
    return slots


def find_gaps(
    data_dir: Path, dataset: str, underlying: str, minute_utc: int, until: datetime
) -> list[str]:
    """Scheduled slots since the first capture, before ``until``, with no snapshot file."""
    captured = captured_labels(data_dir, dataset, underlying)
    on_schedule = sorted(c for c in captured if label_time(c).minute == minute_utc)
    if not on_schedule:
        return []
    slots = scheduled_slots(label_time(on_schedule[0]), until, minute_utc)
    return [s.strftime(LABEL_FORMAT) for s in slots if s.strftime(LABEL_FORMAT) not in captured]


def _log_gaps(data_dir: Path, summary: RecordSummary, minute_utc: int, now: datetime) -> int:
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [
        {
            "slot_utc": summary.slot.label,
            "underlying": o.underlying,
            "dataset": o.dataset,
            "detected_at_utc": stamp,
            "reason": o.reason,
        }
        for o in summary.outcomes
        if o.status == "failed" and summary.slot.on_schedule
    ]
    for underlying, dataset in sorted({(o.underlying, o.dataset) for o in summary.outcomes}):
        rows += [
            {
                "slot_utc": label,
                "underlying": underlying,
                "dataset": dataset,
                "detected_at_utc": stamp,
                "reason": "not_captured",
            }
            for label in find_gaps(data_dir, dataset, underlying, minute_utc, summary.slot.ts)
        ]
    return append_gaps(data_dir, rows)


def check_disk(data_dir: Path) -> float:
    """Used fraction of the filesystem holding the data; warns at 80%."""
    data_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(data_dir)
    fraction = usage.used / usage.total
    if fraction >= DISK_WARNING_FRACTION:
        log.warning("disk_warning", used_pct=round(100 * fraction, 1), path=str(data_dir))
    return fraction


def wants_ticker_sweep(mode: str, slot: Slot, canonical_time_utc: str) -> bool:
    """Whether this slot sweeps ``ticker``, per ``recorder.ticker_sweep``."""
    if mode == "hourly":
        return True
    return mode == "daily" and slot.ts.strftime("%H:%M") == canonical_time_utc


def record_snapshot(
    client: DeribitClient,
    data_dir: Path,
    *,
    currencies: list[str],
    kinds: list[str],
    minute_utc: int,
    ticker_sweep: bool,
    now: datetime | None = None,
) -> RecordSummary:
    """Take one snapshot for each currency and kind.

    A failure in one currency or kind is logged as a gap and the rest still run.
    """
    started = now or datetime.now(UTC)
    slot = current_slot(started, minute_utc)
    summary = RecordSummary(slot)
    check_disk(data_dir)
    instruments: dict[str, Any] = {}
    for underlying in currencies:
        for kind in kinds:
            outcome = _record_quotes(client, data_dir, underlying, kind, slot, instruments)
            summary.outcomes.append(outcome)
    gaps = _log_gaps(data_dir, summary, minute_utc, started)
    if ticker_sweep:
        for underlying in currencies:
            summary.ticker_rows += sweep_tickers(client, data_dir, underlying, slot)
    summary.retries = client.retries_used
    log.info(
        "run_done",
        slot=slot.label,
        on_schedule=slot.on_schedule,
        rows=sum(o.rows for o in summary.outcomes),
        failed=sum(o.status == "failed" for o in summary.outcomes),
        new_gaps=gaps,
        ticker_rows=summary.ticker_rows,
        retries=summary.retries,
    )
    return summary


def coverage(
    data_dir: Path,
    underlying: str,
    *,
    minute_utc: int,
    canonical_time_utc: str,
    since: date | None,
    now: datetime,
) -> dict[str, Any]:
    """Scheduled vs captured snapshots of one underlying (AC-W-04)."""
    captured = sorted(captured_labels(data_dir, storage.OPTION_QUOTES, underlying))
    times = [label_time(label) for label in captured]
    if not times:
        return {"underlying": underlying, "first_snapshot": None, "captured": 0}
    start = times[0]
    if since is not None:
        start = max(start, datetime(since.year, since.month, since.day, tzinfo=UTC))
    times = [t for t in times if t >= start]
    scheduled = scheduled_slots(start, now, minute_utc)
    on_schedule = {t for t in times if t.minute == minute_utc}
    up_days = {t.date() for t in times}
    scheduled_up = [s for s in scheduled if s.date() in up_days]

    hour, minute = (int(part) for part in canonical_time_utc.split(":"))
    tolerance = timedelta(minutes=CANONICAL_TOLERANCE_MIN)
    canonical_days = 0
    calendar_days = 0
    day = start.date()
    while True:
        target = datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)
        if target > now:
            break
        if target >= start:
            calendar_days += 1
            canonical_days += any(abs(t - target) <= tolerance for t in times)
        day += timedelta(days=1)

    return {
        "underlying": underlying,
        "first_snapshot": captured[0],
        "last_snapshot": captured[-1],
        "captured": len(times),
        "hourly_scheduled": len(scheduled),
        "hourly_captured": len(on_schedule & set(scheduled)),
        "hourly_scheduled_on_up_days": len(scheduled_up),
        "hourly_frac_on_up_days": (
            round(len(on_schedule & set(scheduled_up)) / len(scheduled_up), 4)
            if scheduled_up
            else None
        ),
        "canonical_days": canonical_days,
        "calendar_days": calendar_days,
        "canonical_frac": round(canonical_days / calendar_days, 4) if calendar_days else None,
    }


def consecutive_hourly(data_dir: Path, underlying: str, minute_utc: int) -> int:
    """Length of the unbroken run of scheduled snapshots ending at the latest one."""
    on_schedule = sorted(
        (
            t
            for t in map(label_time, captured_labels(data_dir, storage.OPTION_QUOTES, underlying))
            if t.minute == minute_utc
        ),
        reverse=True,
    )
    run = 0
    expected = on_schedule[0] if on_schedule else None
    for t in on_schedule:
        if t != expected:
            break
        run += 1
        expected = t - timedelta(hours=1)
    return run


def disk_use_mb(data_dir: Path) -> float:
    """Megabytes under the raw Deribit tree."""
    root = storage.raw_dir(data_dir)
    return round(sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6, 2)

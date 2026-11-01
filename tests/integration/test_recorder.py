"""Recorder against the mocked transport: atomic files, idempotency, gaps, coverage."""

import json
from collections import namedtuple
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
import pytest
import structlog

from vsm.conventions import datetime_to_ns
from vsm.io import recorder, storage
from vsm.io.deribit import DeribitClient

Factory = Callable[..., DeribitClient]
SLOT_TIME = datetime(2026, 11, 2, 9, 15, 3, tzinfo=UTC)


def record(
    client: DeribitClient, data: Path, now: datetime = SLOT_TIME, **kw: object
) -> recorder.RecordSummary:
    return recorder.record_snapshot(
        client,
        data,
        currencies=["BTC", "ETH"],
        kinds=["option", "future"],
        minute_utc=15,
        ticker_sweep=False,
        now=now,
        **kw,  # type: ignore[arg-type]
    )


def quotes(data: Path, dataset: str, underlying: str) -> pl.DataFrame:
    return pl.concat(pl.read_parquet(p) for p in storage.snapshot_files(data, dataset, underlying))


def test_snapshot_writes_one_file_per_dataset_and_currency(
    make_client: Factory, tmp_path: Path
) -> None:
    summary = record(make_client(), tmp_path)
    assert summary.ok
    assert summary.slot.label == "20261102T0915Z"
    assert summary.slot.on_schedule
    for underlying in ("BTC", "ETH"):
        files = storage.snapshot_files(tmp_path, storage.OPTION_QUOTES, underlying)
        assert [f.name for f in files] == [f"part-deribit-{underlying}-20261102T0915Z.parquet"]
        assert files[0].parent.name == "date=2026-11-02"


def test_option_row_matches_schema_and_venue_units(make_client: Factory, tmp_path: Path) -> None:
    record(make_client(), tmp_path)
    frame = quotes(tmp_path, storage.OPTION_QUOTES, "BTC")
    assert dict(frame.schema) == {
        k: pl.Int8 if v is pl.Int8 else v for k, v in storage.OPTION_QUOTES_SCHEMA.items()
    }
    row = frame.filter(pl.col("bid_px").is_not_null()).row(0, named=True)
    assert row["snapshot_id"] == "deribit-BTC-20261102T0915Z"
    assert row["source"] == "recorder"
    assert row["cp"] in (1, -1)
    assert row["mark_iv_pct"] > 1  # percent as delivered, not decimal
    assert row["ts_exchange"] % 1_000_000 == 0  # milliseconds scaled to ns
    expiry = datetime.fromtimestamp(row["expiry_ns"] / 1e9, UTC)
    assert (expiry.hour, expiry.minute) == (8, 0)


def test_null_bid_is_kept(make_client: Factory, tmp_path: Path) -> None:
    record(make_client(), tmp_path)
    frame = quotes(tmp_path, storage.OPTION_QUOTES, "BTC")
    assert frame["bid_px"].null_count() == 1


def test_future_rows_have_index_funding_and_null_perp_expiry(
    make_client: Factory, tmp_path: Path
) -> None:
    record(make_client(), tmp_path)
    frame = quotes(tmp_path, storage.FUTURE_QUOTES, "BTC")
    perp = frame.filter(pl.col("symbol") == "BTC-PERPETUAL").row(0, named=True)
    dated = frame.filter(pl.col("symbol") != "BTC-PERPETUAL").row(0, named=True)
    assert perp["expiry_ns"] is None
    assert perp["funding_8h"] is not None
    assert perp["index_px"] == pytest.approx(77830.71)
    assert dated["expiry_ns"] is not None
    assert dated["current_funding"] is None


def test_rerun_of_a_slot_is_idempotent(make_client: Factory, tmp_path: Path) -> None:
    record(make_client(), tmp_path)
    before = quotes(tmp_path, storage.OPTION_QUOTES, "BTC")
    summary = record(make_client(), tmp_path, now=SLOT_TIME + timedelta(seconds=40))
    assert {o.status for o in summary.outcomes} == {"exists"}
    assert quotes(tmp_path, storage.OPTION_QUOTES, "BTC").equals(before)


def test_manual_run_files_under_its_own_minute(make_client: Factory, tmp_path: Path) -> None:
    summary = record(make_client(), tmp_path, now=datetime(2026, 11, 2, 9, 47, 30, tzinfo=UTC))
    assert summary.slot.label == "20261102T0947Z"
    assert not summary.slot.on_schedule


def test_run_before_the_minute_belongs_to_the_previous_hour() -> None:
    slot = recorder.current_slot(datetime(2026, 11, 2, 9, 5, tzinfo=UTC), 15)
    assert slot.label == "20261102T0905Z"
    late = recorder.current_slot(datetime(2026, 11, 2, 9, 24, 59, tzinfo=UTC), 15)
    assert (late.label, late.on_schedule) == ("20261102T0915Z", True)


def test_malformed_row_is_dropped_not_fatal(
    make_client: Factory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(request: httpx.Request) -> httpx.Response | None:
        if request.url.params.get("kind") == "option" and "book_summary" in request.url.path:
            rows = json.loads(Path("tests/fixtures/deribit/book_option.json").read_text())
            rows[0].pop("creation_timestamp")
            rows[1]["instrument_name"] = "garbage"
            return httpx.Response(200, json={"result": rows})
        return None

    summary = record(make_client(broken), tmp_path)
    assert summary.ok
    outcome = summary.outcomes[0]
    assert (outcome.rows, outcome.dropped) == (2, 2)
    assert "row_dropped" in capsys.readouterr().out


def test_failed_currency_logs_gap_and_others_still_run(
    make_client: Factory, tmp_path: Path
) -> None:
    def btc_down(request: httpx.Request) -> httpx.Response | None:
        is_btc = request.url.params.get("currency") == "BTC"
        return httpx.Response(503) if is_btc and "book_summary" in request.url.path else None

    summary = record(make_client(btc_down, retries=1), tmp_path)
    assert not summary.ok
    failed = {(o.underlying, o.dataset) for o in summary.outcomes if o.status == "failed"}
    assert failed == {("BTC", storage.OPTION_QUOTES), ("BTC", storage.FUTURE_QUOTES)}
    assert storage.snapshot_files(tmp_path, storage.OPTION_QUOTES, "ETH")
    gaps = recorder.read_gaps(tmp_path)
    assert {(g["slot_utc"], g["underlying"]) for g in gaps} == {("20261102T0915Z", "BTC")}
    assert all("503" in g["reason"] or "DeribitError" in g["reason"] for g in gaps)


def test_restart_mid_write_is_logged_and_the_next_run_resumes(
    make_client: Factory, tmp_path: Path
) -> None:
    record(make_client(), tmp_path)
    lost = storage.snapshot_path(
        tmp_path, storage.OPTION_QUOTES, "BTC", "2026-11-02", "deribit-BTC-20261102T1015Z"
    )
    lost.parent.mkdir(parents=True, exist_ok=True)
    lost.with_name(f".{lost.name}.4242.tmp").write_bytes(b"partial")

    summary = record(make_client(), tmp_path, now=SLOT_TIME + timedelta(hours=2))

    assert summary.ok
    assert recorder.captured_labels(tmp_path, storage.OPTION_QUOTES, "BTC") == {
        "20261102T0915Z",
        "20261102T1115Z",
    }
    assert quotes(tmp_path, storage.OPTION_QUOTES, "BTC")["snapshot_id"].n_unique() == 2
    gaps = recorder.read_gaps(tmp_path)
    assert {g["slot_utc"] for g in gaps} == {"20261102T1015Z"}
    assert {g["reason"] for g in gaps} == {"not_captured"}


def test_missing_scheduled_slots_are_logged_once(make_client: Factory, tmp_path: Path) -> None:
    record(make_client(), tmp_path)
    later = SLOT_TIME + timedelta(hours=3)
    record(make_client(), tmp_path, now=later)
    record(make_client(), tmp_path, now=later + timedelta(hours=1))
    gaps = recorder.read_gaps(tmp_path)
    btc_options = [
        g["slot_utc"]
        for g in gaps
        if g["underlying"] == "BTC" and g["dataset"] == storage.OPTION_QUOTES
    ]
    assert btc_options == ["20261102T1015Z", "20261102T1115Z"]
    assert {g["reason"] for g in gaps} == {"not_captured"}


def test_instrument_failure_falls_back_to_symbol_and_store(
    make_client: Factory, tmp_path: Path
) -> None:
    def no_instruments(request: httpx.Request) -> httpx.Response | None:
        return httpx.Response(503) if "get_instruments" in request.url.path else None

    summary = record(make_client(no_instruments, retries=0), tmp_path)
    assert summary.ok
    assert quotes(tmp_path, storage.OPTION_QUOTES, "BTC")["expiry_ns"].null_count() == 0

    record(make_client(), tmp_path, now=SLOT_TIME + timedelta(hours=1))
    summary = record(
        make_client(no_instruments, retries=0), tmp_path, now=SLOT_TIME + timedelta(hours=2)
    )
    assert summary.ok


def test_instrument_table_tracks_first_and_last_seen(make_client: Factory, tmp_path: Path) -> None:
    first = datetime_to_ns(SLOT_TIME)
    later = first + 3_600_000_000_000
    client = make_client()
    recorder.refresh_instruments(client, tmp_path, "BTC", ["option", "future"], first)
    table = recorder.refresh_instruments(
        client, tmp_path, "BTC", ["option", "future"], later, expired=True
    )
    assert table["symbol"].n_unique() == table.height
    active = table.filter(pl.col("symbol") == "BTC-PERPETUAL").row(0, named=True)
    assert active["contract_type"] == "perpetual"
    assert active["expiry_ns"] is None
    assert (active["first_seen_ns"], active["last_seen_ns"]) == (first, later)
    delivered = table.filter(~pl.col("is_active"))
    assert delivered.height == 1
    assert delivered["first_seen_ns"][0] == later
    tick_steps = table.filter(pl.col("contract_type") == "option")["tick_size_steps"][0]
    assert isinstance(json.loads(tick_steps), list)


def test_ticker_sweep_is_a_separate_dataset(make_client: Factory, tmp_path: Path) -> None:
    client = make_client()
    summary = recorder.record_snapshot(
        client,
        tmp_path,
        currencies=["BTC"],
        kinds=["option"],
        minute_utc=15,
        ticker_sweep=True,
        now=SLOT_TIME,
    )
    assert summary.ticker_rows == 4
    tickers = quotes(tmp_path, storage.OPTION_TICKERS, "BTC")
    assert tickers["bid_qty"].null_count() == 0
    assert tickers["delta"][0] == pytest.approx(-0.80763)
    assert quotes(tmp_path, storage.OPTION_QUOTES, "BTC")["bid_qty"].null_count() == 4


def test_one_failed_ticker_does_not_lose_the_sweep(make_client: Factory, tmp_path: Path) -> None:
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response | None:
        if request.url.path.endswith("/ticker"):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(400, json={"error": {"code": 1, "message": "nope"}})
        return None

    summary = recorder.record_snapshot(
        make_client(flaky),
        tmp_path,
        currencies=["BTC"],
        kinds=["option"],
        minute_utc=15,
        ticker_sweep=True,
        now=SLOT_TIME,
    )
    assert summary.ticker_rows == 3


@pytest.mark.parametrize(
    ("mode", "time", "expected"),
    [
        ("none", "08:15", False),
        ("daily", "08:15", True),
        ("daily", "09:15", False),
        ("hourly", "09:15", True),
    ],
)
def test_ticker_sweep_schedule(mode: str, time: str, expected: bool) -> None:
    hour, minute = map(int, time.split(":"))
    slot = recorder.Slot(datetime(2026, 11, 2, hour, minute, tzinfo=UTC), True)
    assert recorder.wants_ticker_sweep(mode, slot, "08:15") is expected


def test_disk_warning_at_80_percent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(recorder.shutil, "disk_usage", lambda _: usage(100, 80, 20))
    assert recorder.check_disk(tmp_path) == pytest.approx(0.8)
    assert "disk_warning" in capsys.readouterr().out
    monkeypatch.setattr(recorder.shutil, "disk_usage", lambda _: usage(100, 79, 21))
    recorder.check_disk(tmp_path)
    assert "disk_warning" not in capsys.readouterr().out


def test_coverage_counts_scheduled_captured_and_canonical(
    make_client: Factory, tmp_path: Path
) -> None:
    start = datetime(2026, 11, 2, 7, 15, 1, tzinfo=UTC)
    for hours in (0, 1, 2, 4):  # 11:15 missing
        record(make_client(), tmp_path, now=start + timedelta(hours=hours))
    report = recorder.coverage(
        tmp_path,
        "BTC",
        minute_utc=15,
        canonical_time_utc="08:15",
        since=None,
        now=datetime(2026, 11, 2, 12, 0, tzinfo=UTC),
    )
    assert report["first_snapshot"] == "20261102T0715Z"
    assert report["last_snapshot"] == "20261102T1115Z"
    assert report["hourly_scheduled"] == 5
    assert report["hourly_captured"] == 4
    assert report["hourly_frac_on_up_days"] == 0.8
    assert (report["canonical_days"], report["calendar_days"]) == (1, 1)
    assert recorder.consecutive_hourly(tmp_path, "BTC", 15) == 1
    assert recorder.disk_use_mb(tmp_path) >= 0


def test_canonical_tolerance_accepts_a_late_manual_snapshot(
    make_client: Factory, tmp_path: Path
) -> None:
    record(make_client(), tmp_path, now=datetime(2026, 11, 2, 7, 15, tzinfo=UTC))
    record(make_client(), tmp_path, now=datetime(2026, 11, 2, 8, 40, tzinfo=UTC))
    report = recorder.coverage(
        tmp_path,
        "BTC",
        minute_utc=15,
        canonical_time_utc="08:15",
        since=None,
        now=datetime(2026, 11, 2, 12, 0, tzinfo=UTC),
    )
    assert report["canonical_days"] == 1


def test_coverage_since_and_empty_store(make_client: Factory, tmp_path: Path) -> None:
    empty = recorder.coverage(
        tmp_path,
        "BTC",
        minute_utc=15,
        canonical_time_utc="08:15",
        since=None,
        now=SLOT_TIME,
    )
    assert empty == {"underlying": "BTC", "first_snapshot": None, "captured": 0}
    record(make_client(), tmp_path, now=SLOT_TIME)
    record(make_client(), tmp_path, now=SLOT_TIME + timedelta(days=1))
    report = recorder.coverage(
        tmp_path,
        "BTC",
        minute_utc=15,
        canonical_time_utc="08:15",
        since=date(2026, 11, 3),
        now=SLOT_TIME + timedelta(days=1, hours=1),
    )
    assert report["captured"] == 1


@pytest.fixture(autouse=True)
def reset_structlog() -> None:
    structlog.reset_defaults()

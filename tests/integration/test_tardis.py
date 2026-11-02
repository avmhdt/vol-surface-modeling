"""The Tardis ingest on a synthetic gzip stream, through a mocked transport."""

import gzip
import json
from collections import namedtuple
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
from typer.testing import CliRunner

from vsm import cli
from vsm.config import AppConfig, TardisConfig, load_config
from vsm.conventions import datetime_to_ns
from vsm.io import tardis

DAY = date(2026, 10, 1)
LABEL = "20261001T0815Z"
AS_OF = datetime(2026, 10, 1, 8, 15, tzinfo=UTC)
AS_OF_US = datetime_to_ns(AS_OF) // 1_000
EXPIRY_US = datetime_to_ns(datetime(2026, 10, 30, 8, tzinfo=UTC)) // 1_000
TODAY_08_US = datetime_to_ns(datetime(2026, 10, 1, 8, tzinfo=UTC)) // 1_000
HEADER = ",".join(tardis.CSV_SCHEMA)
SECOND = 1_000_000
runner = CliRunner()


def row(symbol: str, local_us: int, *, bid: float = 0.05, expiry: int = EXPIRY_US) -> str:
    kind = "call" if symbol.endswith("C") else "put"
    fields: list[Any] = [
        "deribit", symbol, local_us - 1000, local_us, kind, 80000, expiry, 12.0, 0.05,
        bid, 3.0, 45.0, 0.06, 4.0, 47.0, 0.055, 46.0, "BTC-30OCT26", 78000.0,
        0.4, 1e-5, 30.0, -50.0, 10.0,
    ]  # fmt: skip
    return ",".join(str(field) for field in fields)


def gz(lines: list[str]) -> bytes:
    return gzip.compress(("\n".join([HEADER, *lines]) + "\n").encode(), compresslevel=1)


def body(extra_blocks: int = 0) -> bytes:
    lines = [
        row("BTC-30OCT26-80000-C", AS_OF_US - 1000 * SECOND, bid=0.01),  # too old
        row("BTC-30OCT26-80000-C", AS_OF_US - 100 * SECOND, bid=0.02),
        row("BTC-30OCT26-80000-C", AS_OF_US - 10 * SECOND, bid=0.03),  # the one to keep
        row("BTC-30OCT26-80000-C", AS_OF_US + 10 * SECOND, bid=0.99),  # after as-of
        row("BTC-30OCT26-70000-P", AS_OF_US - 200 * SECOND),
        row("ETH-30OCT26-3000-C", AS_OF_US - 20 * SECOND),
        row("BTC_USDC-30OCT26-80000-C", AS_OF_US - 5 * SECOND),  # linear, excluded
        row("SOL_USDC-30OCT26-200-C", AS_OF_US - 5 * SECOND),  # other underlying
        row("BTC-1OCT26-80000-C", AS_OF_US - 5 * SECOND, expiry=TODAY_08_US),  # expired at 08:00
    ]
    return gz(lines)


def transport_for(data: bytes) -> tuple[httpx.MockTransport, dict[str, int]]:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.url.path.endswith("/deribit/options_chain/2026/10/01/OPTIONS.csv.gz")
        return httpx.Response(
            200, stream=httpx.ByteStream(data), headers={"content-length": str(len(data))}
        )

    return httpx.MockTransport(handler), calls


@pytest.fixture
def config() -> TardisConfig:
    return TardisConfig(
        crypto_status_path="/nonexistent/status.json", free_space_path=".", min_free_gb=0.0
    )


def run_day(
    data_dir: Path, cfg: TardisConfig, data: bytes | None = None
) -> tardis.IngestStats | None:
    transport, _ = transport_for(data or body())
    with httpx.Client(transport=transport) as client:
        return tardis.ingest_day(
            client, data_dir, "https://test.invalid/v1", DAY, "08:15", cfg, sleep=lambda _: None
        )


def snapshot(data_dir: Path, underlying: str) -> pl.DataFrame:
    return pl.read_parquet(tardis.snapshot_path(data_dir, underlying, DAY, LABEL))


def test_keeps_the_last_quote_at_as_of_per_instrument(tmp_path: Path, config: TardisConfig) -> None:
    run_day(tmp_path, config)
    frame = snapshot(tmp_path, "BTC").sort("symbol")
    assert frame["symbol"].to_list() == ["BTC-30OCT26-70000-P", "BTC-30OCT26-80000-C"]
    assert frame.filter(pl.col("symbol") == "BTC-30OCT26-80000-C")["bid_px"][0] == 0.03


def test_nothing_after_as_of_or_older_than_the_staleness_limit(
    tmp_path: Path, config: TardisConfig
) -> None:
    run_day(tmp_path, config)
    frame = snapshot(tmp_path, "BTC")
    as_of_ns = AS_OF_US * 1_000
    assert frame["ts_recv"].max() <= as_of_ns
    assert frame["ts_recv"].min() >= as_of_ns - tardis.STALENESS_S * 1_000_000_000


def test_linear_other_underlyings_and_same_day_expiry_are_excluded(
    tmp_path: Path, config: TardisConfig
) -> None:
    run_day(tmp_path, config)
    symbols = snapshot(tmp_path, "BTC")["symbol"].to_list()
    assert not any("USDC" in symbol or symbol.startswith("SOL") for symbol in symbols)
    assert "BTC-1OCT26-80000-C" not in symbols
    assert snapshot(tmp_path, "ETH")["symbol"].to_list() == ["ETH-30OCT26-3000-C"]


def test_rows_map_to_the_option_quotes_schema(tmp_path: Path, config: TardisConfig) -> None:
    from vsm.io import storage

    run_day(tmp_path, config)
    frame = snapshot(tmp_path, "BTC")
    assert dict(frame.schema) == storage.OPTION_QUOTES_SCHEMA
    quote = frame.filter(pl.col("symbol") == "BTC-30OCT26-80000-C").row(0, named=True)
    assert quote["snapshot_id"] == f"deribit-BTC-{LABEL}"
    assert quote["source"] == "tardis_free"
    assert quote["cp"] == 1
    assert quote["expiry_ns"] == EXPIRY_US * 1_000
    assert quote["bid_iv_pct"] == 45.0  # percent as delivered
    assert quote["bid_qty"] == 3.0
    assert quote["mid_px"] == pytest.approx(0.045)


def test_stream_stops_at_the_end_of_the_window(tmp_path: Path, config: TardisConfig) -> None:
    filler = row("BTC-30OCT26-90000-C", AS_OF_US - 50 * SECOND)
    later = row("BTC-30OCT26-90000-C", AS_OF_US + 400 * SECOND)
    tail = row("BTC-30OCT26-90000-C", AS_OF_US + 9000 * SECOND)
    data = gz([filler] * 400_000 + [later] * 400_000 + [tail] * 3_000_000)
    stats = run_day(tmp_path, config, data)
    assert stats is not None
    assert stats.reached_end
    assert stats.rows_read < 3_800_000
    assert stats.bytes_read <= len(data)


def test_order_is_measured(tmp_path: Path, config: TardisConfig) -> None:
    stats = run_day(tmp_path, config)
    assert stats is not None
    assert stats.inversions == 1  # the deliberately older row after the +10 s one
    assert stats.max_regression_ms == pytest.approx(210_000.0)
    assert stats.total_bytes == stats.bytes_read


def test_a_day_with_no_rows_in_the_window_writes_empty_snapshots(
    tmp_path: Path, config: TardisConfig
) -> None:
    run_day(tmp_path, config, gz([row("BTC-30OCT26-80000-C", AS_OF_US + 9000 * SECOND)]))
    assert snapshot(tmp_path, "BTC").is_empty()


def test_existing_day_is_skipped(tmp_path: Path, config: TardisConfig) -> None:
    assert run_day(tmp_path, config) is not None
    assert run_day(tmp_path, config) is None


def test_keep_raw_writes_what_was_read(tmp_path: Path, config: TardisConfig) -> None:
    data = body()
    transport, _ = transport_for(data)
    with httpx.Client(transport=transport) as client:
        tardis.ingest_day(
            client, tmp_path, "https://test.invalid/v1", DAY, "08:15", config, keep_raw=True
        )
    raw = tmp_path / "raw" / "tardis" / "options_chain" / "2026" / "10" / "01" / "OPTIONS.csv.gz"
    assert raw.read_bytes() == data


def test_rate_limiter_holds_the_long_run_rate() -> None:
    now = {"t": 0.0}
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    limiter = tardis.ByteRateLimiter(10_000_000.0, clock=lambda: now["t"], sleep=sleep)
    for _ in range(100):
        limiter.acquire(1_000_000)
    assert now["t"] == pytest.approx(10.0)
    assert len(slept) == 100


def test_rate_limiter_does_not_sleep_when_already_slow() -> None:
    now = {"t": 0.0}
    slept: list[float] = []
    limiter = tardis.ByteRateLimiter(10_000_000.0, clock=lambda: now["t"], sleep=slept.append)
    now["t"] = 5.0
    limiter.acquire(1_000_000)
    assert slept == []


def test_cap_defaults_to_the_owner_limit() -> None:
    assert load_config().tardis.max_bytes_per_second == 10_000_000.0
    assert load_config().tardis.min_free_gb == 100.0


def test_first_of_month_days() -> None:
    assert tardis.first_of_month_days(date(2026, 11, 2), date(2027, 2, 1)) == [
        date(2026, 12, 1),
        date(2027, 1, 1),
        date(2027, 2, 1),
    ]
    assert len(tardis.first_of_month_days(date(2020, 3, 1), date(2026, 10, 1))) == 80


def status_file(
    path: Path,
    resyncs: int,
    p99: float,
    *,
    age_s: float = 0.0,
    run: str = "run-1",
    conns: tuple[str, ...] = ("market.g2", "public.g0", "ticker.g1"),
    other_conns: tuple[str, ...] = ("ws.g0",),
) -> str:
    import time

    status = {
        "ts_ns": time.time_ns() - int(age_s * 1e9),
        "run": run,
        "stopped": False,
        "venues": {
            tardis.BINANCE_VENUE: {
                "ingest_us_p99": p99,
                "streams": [{"resyncs": resyncs}, {"resyncs": 0}],
                "conns": list(conns),
            },
            "coinbase": {"ingest_us_p99": 100.0, "streams": [], "conns": list(other_conns)},
        },
    }
    path.write_text(json.dumps(status), encoding="utf-8")
    return str(path)


def guard_for(tmp_path: Path, **overrides: Any) -> tuple[tardis.IngestGuard, Path]:
    path = tmp_path / "status.json"
    status_file(path, 3, 100.0)
    settings = {"free_space_path": ".", "min_free_gb": 0.0, **overrides}
    cfg = TardisConfig(crypto_status_path=str(path), **settings)
    return tardis.IngestGuard(cfg), path


def test_guard_clear_when_nothing_changed(tmp_path: Path) -> None:
    guard, _ = guard_for(tmp_path)
    assert guard.reason() is None


def test_guard_pauses_on_a_new_resync(tmp_path: Path) -> None:
    guard, path = guard_for(tmp_path)
    status_file(path, 4, 100.0)
    assert "resyncs" in str(guard.reason())


def test_guard_pauses_when_ingest_p99_climbs(tmp_path: Path) -> None:
    guard, path = guard_for(tmp_path)
    status_file(path, 3, 1001.0)
    assert "p99" in str(guard.reason())


def test_guard_pauses_on_a_binance_reconnect(tmp_path: Path) -> None:
    guard, path = guard_for(tmp_path)
    status_file(path, 3, 100.0, conns=("market.g3", "public.g0", "ticker.g1"))
    assert "Binance reconnected" in str(guard.reason())


def test_guard_remembers_a_reconnect_seen_only_after_the_socket_is_back(tmp_path: Path) -> None:
    # A connection that is down right now just drops off the list, so the serial can dip.
    guard, path = guard_for(tmp_path)
    status_file(path, 3, 100.0, conns=("public.g0", "ticker.g1"))
    assert guard.reason() is None
    status_file(path, 3, 100.0, conns=("market.g3", "public.g0", "ticker.g1"))
    assert "Binance reconnected" in str(guard.reason())


def test_guard_ignores_reconnects_on_other_venues(tmp_path: Path) -> None:
    guard, path = guard_for(tmp_path)
    status_file(path, 3, 100.0, other_conns=("ws.g7",))
    assert guard.reason() is None


def test_guard_pauses_when_the_recorder_restarts(tmp_path: Path) -> None:
    # Every counter starts again in a new run, so a quiet comparison would mean nothing.
    guard, path = guard_for(tmp_path)
    status_file(path, 0, 100.0, run="run-2", conns=("market.g0",))
    assert "restarted" in str(guard.reason())


def test_guard_ignores_normal_variation_in_p99(tmp_path: Path) -> None:
    # The recorder's p99 moves between 170 and 560 us minute to minute with no load at all.
    guard, path = guard_for(tmp_path)
    status_file(path, 3, 558.0)
    assert guard.reason() is None


def test_guard_pauses_on_low_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard, _ = guard_for(tmp_path, min_free_gb=100.0)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(tardis.shutil, "disk_usage", lambda _: usage(0, 0, 99_000_000_000))
    assert "free space" in str(guard.reason())


def test_guard_ignores_a_stopped_or_stale_recorder(tmp_path: Path) -> None:
    guard, path = guard_for(tmp_path)
    status_file(path, 9, 900.0, age_s=3600.0)
    assert guard.reason() is None
    path.write_text("not json", encoding="utf-8")
    assert guard.reason() is None


def test_run_waits_until_the_guard_clears(
    tmp_path: Path, config: TardisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    reasons = iter(["crypto ingest p99 rose", None])
    monkeypatch.setattr(tardis.IngestGuard, "reason", lambda self: next(reasons))
    slept: list[float] = []
    transport, _ = transport_for(body())
    with httpx.Client(transport=transport) as client:
        done = tardis.ingest_months(
            client,
            tmp_path,
            tmp_path / "results",
            "https://test.invalid/v1",
            [DAY],
            "08:15",
            config,
            sleep=slept.append,
        )
    assert len(done) == 1
    assert tardis.PAUSE_POLL_S in slept
    saved = json.loads((tmp_path / "results" / "metrics" / "tardis_ingest.json").read_text())
    assert saved["2026-10-01"]["rows_kept"] == 3


def test_run_stops_when_asked_and_resumes_from_finished_days(
    tmp_path: Path, config: TardisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    reasons = iter([None, "Binance reconnected"])
    monkeypatch.setattr(tardis.IngestGuard, "reason", lambda self: next(reasons))
    slept: list[float] = []
    transport, _ = transport_for(body())
    results = tmp_path / "results"
    days = [DAY, date(2026, 11, 1)]
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(tardis.IngestPaused, match="Binance reconnected"),
    ):
        tardis.ingest_months(
            client,
            tmp_path,
            results,
            "https://test.invalid/v1",
            days,
            "08:15",
            config,
            stop_on_pause=True,
            sleep=slept.append,
        )
    assert tardis.PAUSE_POLL_S not in slept
    saved = json.loads((results / "metrics" / "tardis_ingest.json").read_text())
    assert list(saved) == ["2026-10-01"]


def test_dropped_connection_retries_then_gives_up(tmp_path: Path, config: TardisConfig) -> None:
    calls = {"n": 0}

    def failing(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, stream=httpx.ByteStream(body()))

    with httpx.Client(transport=httpx.MockTransport(failing)) as client:
        done = tardis.ingest_months(
            client,
            tmp_path,
            tmp_path / "results",
            "https://test.invalid/v1",
            [DAY],
            "08:15",
            config,
            sleep=lambda _: None,
        )
    assert calls["n"] == 2
    assert len(done) == 1

    def always(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(always)) as client,
        pytest.raises(httpx.ConnectError),
    ):
        tardis.ingest_months(
            client,
            tmp_path / "other",
            tmp_path / "results",
            "https://test.invalid/v1",
            [DAY],
            "08:15",
            config,
            sleep=lambda _: None,
        )


def cli_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[], httpx.Client]:
    config: AppConfig = load_config()
    config.paths.results_dir = tmp_path / "results"
    config.tardis.crypto_status_path = "/nonexistent/status.json"
    config.tardis.free_space_path = "."
    config.tardis.min_free_gb = 0.0
    monkeypatch.setenv("VSM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    transport, _ = transport_for(body())

    def client() -> httpx.Client:
        return httpx.Client(transport=transport)

    monkeypatch.setattr(cli, "_tardis_client", client)
    return client


def test_cli_ingests_a_month(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cli_setup(monkeypatch, tmp_path)
    span = ["--start", "2026-10-01", "--end", "2026-10-01"]
    result = runner.invoke(cli.app, ["ingest", "tardis", "--mode", "first-of-month", *span])
    assert result.exit_code == 0, result.output
    assert "kept 3 rows" in result.output
    assert "1 of 1 days fetched" in result.output


def test_cli_refuses_paid_ranges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cli_setup(monkeypatch, tmp_path)
    span = ["--start", "2026-10-01", "--end", "2026-10-02"]
    result = runner.invoke(cli.app, ["ingest", "tardis", "--mode", "range", *span])
    assert result.exit_code == 2
    assert "D-02" in result.output


def test_cli_ingests_named_days(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cli_setup(monkeypatch, tmp_path)
    args = ["ingest", "tardis", "--mode", "first-of-month", "--days", "2026-10-01"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "1 of 1 days fetched" in result.output


def test_cli_refuses_a_named_day_that_is_not_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli_setup(monkeypatch, tmp_path)
    args = ["ingest", "tardis", "--mode", "first-of-month", "--days", "2026-10-01,2026-10-02"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 2
    assert "D-02" in result.output


def test_cli_needs_days_or_a_full_span(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cli_setup(monkeypatch, tmp_path)
    args = ["ingest", "tardis", "--mode", "first-of-month", "--start", "2026-10-01"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 2
    assert "--days" in result.output


def test_cli_stops_and_says_why(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cli_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(tardis.IngestGuard, "reason", lambda self: "Binance reconnected")
    args = ["ingest", "tardis", "--mode", "first-of-month", "--days", "2026-10-01"]
    result = runner.invoke(cli.app, [*args, "--stop-on-pause"])
    assert result.exit_code == 3
    assert "Binance reconnected" in result.output

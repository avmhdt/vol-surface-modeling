"""History ingest against the mocked transport: paging, resume, idempotency."""

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from vsm.conventions import date_to_ns
from vsm.io import history, storage
from vsm.io.deribit import DeribitClient

Factory = Callable[..., DeribitClient]
NOW = datetime(2026, 11, 2, 9, 30, tzinfo=UTC)
START = date(2026, 10, 1)


def run(
    client: DeribitClient, data: Path, *what: str, start: date | None = START
) -> dict[str, int]:
    return history.ingest_history(
        client, data, what=list(what), currencies=["BTC"], start=start, end=None, now=NOW
    )


def read(data: Path, *parts: str) -> pl.DataFrame:
    root = storage.raw_dir(data).joinpath(*parts)
    return pl.concat(pl.read_parquet(p) for p in sorted(root.glob("*.parquet")))


def test_perp_chunks_cover_the_range_and_match_schema(make_client: Factory, tmp_path: Path) -> None:
    seen: list[int] = []

    def spy(request: httpx.Request) -> None:
        seen.append(int(request.url.params["start_timestamp"]))

    run(make_client(spy), tmp_path, "perp", start=date(2026, 1, 1))
    frame = read(tmp_path, "perp_ohlc", "symbol=BTC-PERPETUAL")
    assert len(seen) == 2  # just over 4,800 hours from January to November
    assert frame.height == 2
    assert dict(frame.schema) == storage.PERP_OHLC_SCHEMA
    assert frame["resolution_s"].unique().to_list() == [3600]
    assert frame["ts"].min() == date_to_ns(date(2026, 1, 1))


def test_open_bar_is_left_out(make_client: Factory, tmp_path: Path) -> None:
    def open_bar(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": {
                    "status": "ok",
                    "ticks": [int(NOW.replace(minute=0).timestamp() * 1000)],
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "volume": [1.0],
                }
            },
        )

    counts = run(make_client(open_bar), tmp_path, "perp", start=date(2026, 11, 2))
    assert counts["perp BTC"] == 0


def test_no_data_status_is_skipped(make_client: Factory, tmp_path: Path) -> None:
    def no_data(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": {"status": "no_data"}})

    assert run(make_client(no_data), tmp_path, "perp")["perp BTC"] == 0


def test_dvol_funding_delivery_rows(make_client: Factory, tmp_path: Path) -> None:
    counts = run(make_client(), tmp_path, "dvol", "funding", "delivery")
    assert counts["dvol BTC"] == counts["funding BTC"] > 0
    assert counts["delivery BTC"] == 2
    assert dict(read(tmp_path, "dvol", "underlying=BTC").schema) == storage.DVOL_SCHEMA
    assert dict(read(tmp_path, "funding", "symbol=BTC-PERPETUAL").schema) == storage.FUNDING_SCHEMA
    delivery = read(tmp_path, "delivery", "index_name=btc_usd")
    assert dict(delivery.schema) == storage.DELIVERY_SCHEMA
    assert delivery["date"].to_list() == [date(2026, 10, 31), date(2026, 11, 1)]


def test_funding_keeps_records_with_null_fields(make_client: Factory, tmp_path: Path) -> None:
    def early(request: httpx.Request) -> httpx.Response | None:
        if "get_funding_rate_history" not in request.url.path:
            return None
        record = {
            "timestamp": int(request.url.params["start_timestamp"]),
            "index_price": 100.0,
            "prev_index_price": None,
            "interest_8h": 1e-5,
            "interest_1h": None,
        }
        return httpx.Response(200, json={"result": [record]})

    assert run(make_client(early), tmp_path, "funding", start=date(2026, 10, 30))["funding BTC"] > 0
    frame = read(tmp_path, "funding", "symbol=BTC-PERPETUAL")
    assert frame["prev_index_px"].null_count() == frame.height


def test_rerun_adds_no_rows_and_append_resumes(make_client: Factory, tmp_path: Path) -> None:
    run(make_client(), tmp_path, "dvol")
    first = read(tmp_path, "dvol", "underlying=BTC")
    run(make_client(), tmp_path, "dvol")
    assert read(tmp_path, "dvol", "underlying=BTC").equals(first)

    seen: list[int] = []

    def spy(request: httpx.Request) -> None:
        seen.append(int(request.url.params["start_timestamp"]))

    run(make_client(spy), tmp_path, "dvol", start=None)
    assert seen[0] == first["ts"].max() // 1_000_000


def test_empty_store_resumes_from_the_epoch(tmp_path: Path) -> None:
    assert history._resume_ms(tmp_path, "ts", None) == date_to_ns(history.HISTORY_START) // 10**6


def test_end_date_caps_the_range(make_client: Factory, tmp_path: Path) -> None:
    history.ingest_history(
        make_client(),
        tmp_path,
        what=["perp"],
        currencies=["BTC"],
        start=date(2026, 10, 1),
        end=date(2026, 10, 2),
        now=NOW,
    )
    frame = read(tmp_path, "perp_ohlc", "symbol=BTC-PERPETUAL")
    assert frame["ts"].max() <= date_to_ns(date(2026, 10, 3))


def test_unknown_series_is_refused(make_client: Factory, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown series"):
        run(make_client(), tmp_path, "bogus")


def test_rows_split_into_yearly_files(make_client: Factory, tmp_path: Path) -> None:
    run(make_client(), tmp_path, "dvol", start=date(2025, 12, 15))
    root = storage.raw_dir(tmp_path) / "dvol" / "underlying=BTC"
    assert sorted(p.name for p in root.iterdir()) == ["part-2025.parquet", "part-2026.parquet"]

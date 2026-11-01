"""AC-W-04: data access works and the recorder accumulates history."""

import json
from pathlib import Path

import pytest

from vsm.io import recorder
from vsm.settings import Settings

M1_DATE = "2026-11-01"
EXPECTED_ENDPOINTS = {
    "get_book_summary_by_currency",
    "get_instruments",
    "get_index_price",
    "ticker",
    "get_volatility_index_data",
    "get_tradingview_chart_data",
    "get_funding_rate_history",
    "get_delivery_prices",
}


def test_smoke_covers_every_endpoint_in_the_spec(make_client: object) -> None:
    from vsm.io.smoke import run_smoke

    assert callable(make_client)
    results = run_smoke(make_client(), ["BTC"])
    assert {r.endpoint.split()[0] for r in results} == EXPECTED_ENDPOINTS
    assert all(r.ok for r in results)


@pytest.mark.data
def test_recorder_coverage_on_the_real_store() -> None:
    data_dir = Settings().data_dir
    report = json.loads(Path("results/metrics/recorder_coverage.json").read_text(encoding="utf-8"))
    assert recorder.storage.snapshot_files(data_dir, "option_quotes", "BTC")
    for underlying in ("BTC", "ETH"):
        coverage = report["underlyings"][underlying]
        assert coverage["first_snapshot"] >= M1_DATE.replace("-", "")
        if coverage["calendar_days"] >= 7:
            assert coverage["canonical_frac"] >= 0.90
            assert coverage["hourly_frac_on_up_days"] >= 0.95

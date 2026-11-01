"""CLI commands end to end on the mocked transport."""

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from vsm import cli
from vsm.config import AppConfig
from vsm.io import recorder, storage
from vsm.io.deribit import DeribitClient
from vsm.settings import Settings

Factory = Callable[..., DeribitClient]
runner = CliRunner()


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_client: Factory) -> None:
    monkeypatch.setenv("VSM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VSM_RECORDER_CURRENCIES", "BTC")

    def fake_client(settings: Settings, config: AppConfig) -> DeribitClient:
        return make_client()

    monkeypatch.setattr(cli, "_client", fake_client)


def test_smoke_passes_on_expected_fields() -> None:
    result = runner.invoke(cli.app, ["smoke", "--currency", "BTC"])
    assert result.exit_code == 0, result.output
    assert "11/11 checks passed" in result.output


def test_smoke_fails_when_a_field_disappears(
    monkeypatch: pytest.MonkeyPatch, make_client: Factory
) -> None:
    def renamed(request: httpx.Request) -> httpx.Response | None:
        if "get_book_summary" in request.url.path and request.url.params["kind"] == "future":
            return httpx.Response(200, json={"result": [{"instrument_name": "BTC-PERPETUAL"}]})
        return None

    monkeypatch.setattr(cli, "_client", lambda *_: make_client(renamed))
    result = runner.invoke(cli.app, ["smoke", "--currency", "BTC"])
    assert result.exit_code == 1
    assert "FAIL  get_book_summary_by_currency BTC future" in result.output


def test_record_writes_snapshot_and_run_log(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["record", "--no-ticker-sweep"])
    assert result.exit_code == 0, result.output
    data = tmp_path / "data"
    assert storage.snapshot_files(data, storage.OPTION_QUOTES, "BTC")
    logs = list((data / "logs").glob("recorder-*.jsonl"))
    events = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]
    written = next(e for e in events if e["event"] == "snapshot_written")
    assert {"snapshot_id", "rows", "latency_ms", "retries"} <= written.keys()


def test_record_exits_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch, make_client: Factory
) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    monkeypatch.setattr(cli, "_client", lambda *_: make_client(down, retries=0))
    result = runner.invoke(cli.app, ["record", "--no-ticker-sweep"])
    assert result.exit_code == 1


def test_status_reports_and_writes_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner.invoke(cli.app, ["record", "--no-ticker-sweep"])
    original = cli.load_config

    def relocated(path: Path | None = None) -> AppConfig:
        config = original(path)
        config.paths.results_dir = tmp_path / "results"
        return config

    monkeypatch.setattr(cli, "load_config", relocated)
    result = runner.invoke(cli.app, ["status", "--since", "2026-01-01"])
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "results" / "metrics" / "recorder_coverage.json").read_text())
    assert report["underlyings"]["BTC"]["captured"] == 1
    assert report["gaps_logged"] == 0


def test_ingest_instruments_and_history(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["ingest", "instruments", "--currency", "BTC", "--expired"])
    assert result.exit_code == 0, result.output
    assert "BTC: 7 instruments stored" in result.output
    result = runner.invoke(
        cli.app,
        [
            "ingest",
            "deribit-history",
            "--what",
            "delivery,dvol",
            "--currency",
            "BTC",
            "--start",
            "2026-10-01",
            "--end",
            "2026-10-02",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "delivery BTC: 2 rows" in result.output
    assert recorder.gaps_path(tmp_path / "data").exists() is False

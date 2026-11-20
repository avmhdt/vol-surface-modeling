"""``vsm reproduce`` end to end on a one-day synthetic store."""

from pathlib import Path

import pytest
from test_build import DAY, write_snapshot
from typer.testing import CliRunner

from vsm import cli
from vsm.config import AppConfig, load_config
from vsm.reproduce import reproduce

cli_runner = CliRunner()


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    loaded = load_config()
    loaded.paths.results_dir = tmp_path / "results"
    loaded.paths.reports_dir = tmp_path / "reports"
    loaded.underlyings = ["BTC"]
    loaded.mc.paths = 1_000
    loaded.mc.steps_per_day = 1
    loaded.straddle.bootstrap.n = 100
    loaded.reproduce.first_day = DAY.isoformat()
    loaded.reproduce.dvol_start = DAY.isoformat()
    loaded.reproduce.recorder_start = DAY.isoformat()
    loaded.reproduce.barrier_dates = [DAY.isoformat()]
    loaded.reproduce.showcase_date = DAY.isoformat()
    loaded.stability.lam = None
    return loaded


def test_reproduce_runs_every_step(tmp_path: Path, config: AppConfig) -> None:
    data = tmp_path / "data"
    write_snapshot(data)
    summary = reproduce(data, config, ["BTC"], DAY)
    assert summary["BTC"]["days_built"] == 1
    assert summary["BTC"]["heston_days"] == 1
    results = config.results_dir()
    for name in (
        "metrics/localvol_BTC.json",
        "tables/barrier_lv_vs_heston_BTC.csv",
        "metrics/vrp_surfaces_BTC.json",
        "metrics/straddle_tardis_monthly_entry_iv_BTC.json",
        "metrics/sticky_BTC.json",
    ):
        assert (results / name).exists(), name
    assert (config.reports_dir() / "viewer" / "surface_BTC.html").exists()
    assert "smile_and_density_BTC" in summary["figures"]["drawn"]


def test_reproduce_command_skips_ingest(
    tmp_path: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setenv("VSM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    monkeypatch.setattr(cli, "reproduce", lambda *args: calls.append(args[2:]) or {"ok": 1})
    monkeypatch.setattr(cli, "ingest_deribit_history", lambda: calls.append(("ingest",)))
    args = ["reproduce", "--skip-ingest", "--underlying", "BTC", "--end", DAY.isoformat()]
    result = cli_runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert calls == [(["BTC"], DAY)]

"""The 3D viewer and the report figures on a synthetic stored day."""

from pathlib import Path

import pytest
from test_build import DAY, write_snapshot
from typer.testing import CliRunner

from vsm import cli, pipeline
from vsm.config import AppConfig, load_config
from vsm.localvol.runner import run_localvol_day
from vsm.viz import report, viewer

cli_runner = CliRunner()


def make_config(root: Path) -> AppConfig:
    loaded = load_config()
    loaded.paths.results_dir = root / "results"
    loaded.paths.reports_dir = root / "reports"
    loaded.mc.paths = 1_000
    loaded.mc.steps_per_day = 1
    loaded.underlyings = ["BTC"]
    return loaded


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return make_config(tmp_path)


@pytest.fixture(scope="module")
def stored(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("viewer")
    data = root / "data"
    write_snapshot(data)
    assert pipeline.run_day(data, make_config(root), "BTC", DAY) is not None
    return data


def test_viewer_is_one_static_file_under_the_limit(stored: Path, config: AppConfig) -> None:
    path = viewer.build_viewer(stored, config, "BTC", "2026-11-01", "2026-11-30", ["2026-11-02"])
    assert path is not None
    html = path.read_text(encoding="utf-8")
    assert path.stat().st_size < viewer.MAX_BYTES
    assert "Quote mids" in html
    assert "Deribit" in html
    assert viewer.build_viewer(stored, config, "BTC", "2020-01-01", "2020-01-02") is None


def test_viewer_refuses_too_many_quote_dates(stored: Path, config: AppConfig) -> None:
    many = [f"2026-11-0{i}" for i in range(1, 7)]
    with pytest.raises(ValueError, match="D-07"):
        viewer.build_viewer(stored, config, "BTC", "2026-11-01", "2026-11-30", many)


def test_viewer_thins_a_long_history() -> None:
    assert viewer._thin(list(range(10)), 4) == [0, 3, 6, 9]  # type: ignore[arg-type]
    assert viewer._thin(list(range(3)), 4) == [0, 1, 2]  # type: ignore[arg-type]


def test_report_figures_draw_what_their_inputs_allow(stored: Path, config: AppConfig) -> None:
    first = report.draw_all(stored, config, DAY)
    assert "smile_and_density_BTC" in first["drawn"]
    assert "local_vol_BTC" in first["skipped"]
    run_localvol_day(stored, config, "BTC", DAY)
    second = report.draw_all(stored, config, DAY)
    assert "local_vol_BTC" in second["drawn"]
    figures = config.reports_dir() / "figures"
    for name in ("smile_fits_BTC.png", "g_density_BTC.png", "localvol_BTC.png"):
        assert (figures / name).exists()


def test_viewer_command(stored: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VSM_DATA_DIR", str(stored))
    monkeypatch.setattr(cli, "load_config", lambda path=None: config)
    result = cli_runner.invoke(cli.app, ["viewer", "--underlying", "BTC", "--quotes-dates", ""])
    assert result.exit_code == 0, result.output
    assert "MB" in result.output
    missing = ["viewer", "--underlying", "BTC", "--start", "2020-01-01", "--end", "2020-01-02"]
    assert cli_runner.invoke(cli.app, missing).exit_code == 1

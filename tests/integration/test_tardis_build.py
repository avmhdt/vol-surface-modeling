"""``vsm build`` on a Tardis-derived snapshot: same pipeline, no futures file."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from fixtures.make_fixtures import AS_OF_NS, build_chain
from vsm import pipeline
from vsm.clean import clean_snapshot
from vsm.config import AppConfig, load_config
from vsm.conventions import NS_PER_S
from vsm.io import storage, tardis

DAY = date(2026, 11, 2)
LABEL = "20261102T0815Z"


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    loaded = load_config()
    loaded.paths.results_dir = tmp_path / "results"
    loaded.paths.reports_dir = tmp_path / "reports"
    return loaded


def chain_frame() -> pl.DataFrame:
    chain = build_chain()
    missing = {
        name: kind
        for name, kind in storage.OPTION_QUOTES_SCHEMA.items()
        if name not in chain.columns
    }
    frame = chain.with_columns(
        [pl.lit(None, dtype=kind).alias(name) for name, kind in missing.items()]
    ).with_columns(pl.lit("tardis_free").alias("source"))
    return frame.select(list(storage.OPTION_QUOTES_SCHEMA))


def write_snapshot(data: Path, frame: pl.DataFrame) -> None:
    storage.write_parquet_atomic(frame, tardis.snapshot_path(data, "BTC", DAY, LABEL))


def test_tardis_snapshot_is_found_when_the_recorder_has_none(tmp_path: Path) -> None:
    write_snapshot(tmp_path, chain_frame())
    assert pipeline.select_snapshot(tmp_path, "BTC", DAY, "08:15") == pipeline.Snapshot(
        LABEL, False, "tardis"
    )


def test_build_runs_through_the_gate_without_futures(tmp_path: Path, config: AppConfig) -> None:
    write_snapshot(tmp_path, chain_frame())
    report = pipeline.run_day(tmp_path, config, "BTC", DAY)
    assert report is not None
    assert report["source"] == "tardis"
    assert report["surface"]["passed"]
    assert report["surface"]["exported"]
    assert report["forward_methods"]["parity_wls"] >= 6
    assert (tmp_path / "exports" / "BTC" / LABEL / "surface.json").exists()


def test_as_of_is_the_nominal_time_and_spot_falls_back(tmp_path: Path, config: AppConfig) -> None:
    write_snapshot(tmp_path, chain_frame())
    pipeline.run_day(tmp_path, config, "BTC", DAY)
    forwards = pl.read_parquet(
        tmp_path
        / "interim"
        / "forwards"
        / "underlying=BTC"
        / "date=2026-11-02"
        / f"part-{LABEL}.parquet"
    )
    assert forwards["as_of_ns"][0] == AS_OF_NS
    assert forwards["spot_index"][0] == pytest.approx(
        chain_frame().sort("expiry_ns")["underlying_px"][0]
    )


def test_spot_comes_from_the_last_closed_perpetual_bar(tmp_path: Path, config: AppConfig) -> None:
    hour = 3_600_000_000_000
    bars = pl.DataFrame(
        {
            "ts": [AS_OF_NS - 2 * hour, AS_OF_NS - hour + 1, AS_OF_NS - hour // 2],
            "close": [1.0, 2.0, 3.0],
        }
    )
    path = storage.raw_dir(tmp_path) / "perp_ohlc" / "symbol=BTC-PERPETUAL" / "part-2026.parquet"
    storage.write_parquet_atomic(bars, path)
    # The second and third bars had not closed at as-of, so only the first can be used.
    assert pipeline._spot_before(tmp_path, "BTC", AS_OF_NS, chain_frame()) == 1.0


def test_spot_falls_back_when_no_bar_has_closed(tmp_path: Path) -> None:
    bars = pl.DataFrame({"ts": [AS_OF_NS], "close": [1.0]})
    path = storage.raw_dir(tmp_path) / "perp_ohlc" / "symbol=BTC-PERPETUAL" / "part-2026.parquet"
    storage.write_parquet_atomic(bars, path)
    assert pipeline._spot_before(tmp_path, "BTC", AS_OF_NS, chain_frame()) > 1.0


def test_empty_snapshot_means_no_surface(tmp_path: Path, config: AppConfig) -> None:
    write_snapshot(tmp_path, chain_frame().head(0))
    assert pipeline.run_day(tmp_path, config, "BTC", DAY) is None


def test_staleness_override_is_applied(config: AppConfig) -> None:
    frame = chain_frame().with_columns(pl.lit(AS_OF_NS - 200 * NS_PER_S).alias("ts_recv"))
    ticks = pl.DataFrame({"symbol": ["none"], "tick_size": [0.0001]})
    futures = pl.DataFrame(schema=storage.FUTURE_QUOTES_SCHEMA)
    stale = clean_snapshot(frame, futures, 78_000.0, AS_OF_NS, ticks, config)
    assert set(stale.quotes["drop_reason"].to_list()) == {"stale"}
    fresh = clean_snapshot(frame, futures, 78_000.0, AS_OF_NS, ticks, config, staleness_s=300)
    assert fresh.report["kept"] > 0

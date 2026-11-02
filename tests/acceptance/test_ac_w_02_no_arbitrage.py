"""AC-W-02: zero static-arbitrage violations in fitted surfaces, checked for every day."""

import csv
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fixtures.make_fixtures import AS_OF_NS, SPOT, build_chain, build_futures, tick_table
from vsm.clean import clean_snapshot
from vsm.config import load_config
from vsm.export.bundle import surface_document
from vsm.surface.arbitrage import check_surface
from vsm.surface.fit import build_slices
from vsm.surface.store import with_report
from vsm.surface.types import Surface


@pytest.fixture(scope="module")
def surface() -> Surface:
    config = load_config()
    chain = build_chain()
    result = clean_snapshot(chain, build_futures(), SPOT, AS_OF_NS, tick_table(chain), config)
    built = build_slices(result.quotes, result.forwards, config)
    assert built is not None
    ssvi, slices, dropped, sources = built
    raw = Surface(
        "BTC",
        "deribit",
        AS_OF_NS,
        "deribit-BTC-20261102T0815Z",
        SPOT,
        ssvi,
        slices,
        dropped,
        sources,
    )
    return with_report(raw, check_surface(raw, config.arbitrage))


def test_synthetic_surface_has_no_violations(surface: Surface) -> None:
    cfg = load_config().arbitrage
    report = surface.report
    assert report is not None
    assert report.passed
    assert report.min_g >= -cfg.tol_g
    assert report.min_calendar_dw >= -cfg.tol_calendar
    assert report.density_mass_max_err <= cfg.tol_density_mass
    assert all(check.price_ok for check in report.slices)


def test_no_expiry_is_lost(surface: Surface) -> None:
    assert len(surface.slices) == 7  # the 1-day expiry is below min_days_to_expiry
    assert surface.dropped_expiries == ()
    assert sum(surface.fallback_counts.values()) == 7


def test_failing_surface_is_never_written(surface: Surface) -> None:
    report = surface.report
    assert report is not None
    failed = with_report(surface, type(report)(**{**report.__dict__, "passed": False}))
    with pytest.raises(ValueError, match="did not pass"):
        surface_document(failed, load_config().arbitrage)


def test_unchecked_surface_is_never_written(surface: Surface) -> None:
    unchecked = Surface(
        surface.underlying,
        surface.venue,
        surface.as_of_ns,
        surface.snapshot_id,
        surface.spot_index,
        surface.ssvi,
        surface.slices,
    )
    with pytest.raises(ValueError, match="did not pass"):
        surface_document(unchecked, load_config().arbitrage)


@pytest.mark.data
@pytest.mark.parametrize("underlying", ["BTC", "ETH"])
def test_every_recorded_day_is_clean(underlying: str) -> None:
    path = Path(f"results/arbcheck/summary_{underlying}.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        assert row["passed"] == "true"
        assert int(row["failed_slices_after_fallback"]) == 0
        assert float(row["min_g"]) >= -1e-12
        assert float(row["min_calendar_dw"]) >= -1e-12
        assert float(row["density_mass_max_err"]) <= 1e-3


def test_slices_rise_in_total_variance(surface: Surface) -> None:
    k = np.linspace(-0.3, 0.3, 61)
    stacked = pl.DataFrame({str(i): s.svi.w(k) for i, s in enumerate(surface.slices)}).to_numpy()
    assert np.all(np.diff(stacked, axis=1) >= -1e-12)

"""``vsm reproduce``: every committed result, table and figure from local data (FR-26).

Nothing here downloads. The CLI runs the ingest steps first unless ``--skip-ingest`` is given.
The steps are the same ones the individual commands run, in dependency order: build every
stored day, calibrate the models, then the local vol, VRP, straddle, sticky and stability
studies, then the viewer and the report figures. Seeds, fixed grids and sorted outputs make
two runs byte-identical under ``results/``, except ``experiment_log.csv`` and ``bench/``
(AC-W-08). Timings come from ``results/bench`` and are not re-measured here.
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

from vsm.config import AppConfig, config_hash
from vsm.explog import append_run
from vsm.localvol.runner import run_barrier_day, run_localvol_day
from vsm.models.runner import calibrate_heston_day, calibrate_sabr_day, refresh_comparison
from vsm.pipeline import refresh_outputs, run_day
from vsm.studies.sticky import run_sticky
from vsm.studies.straddle import HEDGES, run_straddles
from vsm.studies.vrp import run_vrp
from vsm.surface.regularized import run_stability
from vsm.viz import report
from vsm.viz.viewer import build_viewer

log = structlog.get_logger("vsm.reproduce")


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def reproduce_underlying(
    data_dir: Path, config: AppConfig, underlying: str, last: date
) -> dict[str, Any]:
    """Every per-underlying step, in order; returns counts for the log."""
    cfg = config.reproduce
    first = date.fromisoformat(cfg.first_day)
    days = _days(first, last)
    built = [day for day in days if run_day(data_dir, config, underlying, day) is not None]
    refresh_outputs(data_dir, config, underlying)
    heston = [d for d in built if calibrate_heston_day(data_dir, config, underlying, d)]
    sabr = [d for d in built if calibrate_sabr_day(data_dir, config, underlying, d)]
    refresh_comparison(data_dir, config, underlying)
    for text in cfg.barrier_dates:
        day = date.fromisoformat(text)
        run_localvol_day(data_dir, config, underlying, day)
        run_barrier_day(data_dir, config, underlying, day)
    run_vrp(data_dir, config, underlying, first, last, "surfaces")
    run_vrp(data_dir, config, underlying, date.fromisoformat(cfg.dvol_start), last, "dvol")
    for hedge in HEDGES:
        run_straddles(data_dir, config, underlying, "tardis_monthly", hedge)
    run_straddles(data_dir, config, underlying, "recorder_daily", "entry_iv")
    run_sticky(data_dir, config, underlying, cfg.recorder_start, last.isoformat())
    if config.stability.lam is not None:
        run_stability(data_dir, config, underlying)
    build_viewer(data_dir, config, underlying, first.isoformat(), last.isoformat())
    return {"days_built": len(built), "heston_days": len(heston), "sabr_days": len(sabr)}


def reproduce(
    data_dir: Path, config: AppConfig, underlyings: list[str], last: date | None = None
) -> dict[str, Any]:
    """Run everything for each underlying, then the cross-underlying figures."""
    end = last or datetime.now(UTC).date()
    summary: dict[str, Any] = {
        underlying: reproduce_underlying(data_dir, config, underlying, end)
        for underlying in underlyings
    }
    figures = report.draw_all(data_dir, config, date.fromisoformat(config.reproduce.showcase_date))
    summary["figures"] = figures
    append_run(
        config.results_dir(),
        "reproduce",
        config_hash(config),
        start=config.reproduce.first_day,
        end=end.isoformat(),
        params={"underlyings": underlyings},
        metrics={k: v for k, v in summary.items() if k != "figures"},
    )
    log.info("reproduced", **{k: v for k, v in summary.items() if k != "figures"})
    return summary

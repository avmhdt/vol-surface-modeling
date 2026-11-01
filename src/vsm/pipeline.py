"""Daily build: pick the snapshot for an as-of date and run the cleaning steps on it."""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

from vsm.clean import CleanResult, clean_snapshot
from vsm.config import AppConfig
from vsm.io import recorder, storage

CANONICAL_TOLERANCE = timedelta(minutes=recorder.CANONICAL_TOLERANCE_MIN)

log = structlog.get_logger("vsm.pipeline")


@dataclass(frozen=True)
class Snapshot:
    """The snapshot used for one as-of date; ``flagged`` means it was off the exact time."""

    label: str
    flagged: bool


def select_snapshot(data_dir: Path, underlying: str, day: date, time_utc: str) -> Snapshot | None:
    """The snapshot at ``time_utc``, else the nearest within 30 minutes, else None (DQ-13)."""
    hour, minute = (int(part) for part in time_utc.split(":"))
    target = datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)
    captured = recorder.captured_labels(data_dir, storage.OPTION_QUOTES, underlying)
    exact = target.strftime(recorder.LABEL_FORMAT)
    if exact in captured:
        return Snapshot(exact, False)
    near = [
        (abs(when - target), label)
        for label in captured
        if abs((when := recorder.label_time(label)) - target) <= CANONICAL_TOLERANCE
    ]
    return Snapshot(min(near)[1], True) if near else None


def _read_snapshot(
    data_dir: Path, dataset: str, underlying: str, label: str
) -> pl.DataFrame | None:
    day = recorder.label_time(label).strftime("%Y-%m-%d")
    snapshot = f"{storage.VENUE}-{underlying}-{label}"
    path = storage.snapshot_path(data_dir, dataset, underlying, day, snapshot)
    return pl.read_parquet(path) if path.exists() else None


def _tick_sizes(data_dir: Path, underlying: str) -> pl.DataFrame:
    path = storage.instruments_path(data_dir, underlying)
    if not path.exists():
        return pl.DataFrame(schema={"symbol": pl.String, "tick_size": pl.Float64})
    return pl.read_parquet(path, columns=["symbol", "tick_size"])


def forwards_frame(result: CleanResult, underlying: str, spot_index: float) -> pl.DataFrame:
    """``forwards`` table of one snapshot."""
    as_of_ns = int(result.quotes["as_of_ns"][0])
    rows = [
        {
            **asdict(estimate),
            "as_of_ns": as_of_ns,
            "underlying": underlying,
            "spot_index": spot_index,
        }
        for estimate in result.forwards
    ]
    return pl.DataFrame(rows, schema=storage.FORWARDS_SCHEMA).select(list(storage.FORWARDS_SCHEMA))


def run_day(
    data_dir: Path,
    config: AppConfig,
    underlying: str,
    day: date,
    *,
    at_utc: str | None = None,
) -> dict[str, Any] | None:
    """Clean one as-of date. Returns its DQ report, or None if no snapshot exists (DQ-13)."""
    time_utc = at_utc or config.recorder.canonical_time_utc
    snapshot = select_snapshot(data_dir, underlying, day, time_utc)
    if snapshot is None:
        log.warning("no_snapshot", underlying=underlying, day=day.isoformat(), time_utc=time_utc)
        return None
    options = _read_snapshot(data_dir, storage.OPTION_QUOTES, underlying, snapshot.label)
    futures = _read_snapshot(data_dir, storage.FUTURE_QUOTES, underlying, snapshot.label)
    if options is None or futures is None:
        log.warning("incomplete_snapshot", underlying=underlying, label=snapshot.label)
        return None

    as_of_ns = max(int(options["ts_recv"].max()), int(futures["ts_recv"].max()))  # type: ignore[arg-type]
    spot_index = float(futures["index_px"][0])
    result = clean_snapshot(
        options, futures, spot_index, as_of_ns, _tick_sizes(data_dir, underlying), config
    )

    stamp = day.isoformat()
    part = f"part-{snapshot.label}.parquet"
    interim = data_dir / "interim"
    storage.write_parquet_atomic(
        result.quotes,
        interim / "clean_quotes" / f"underlying={underlying}" / f"date={stamp}" / part,
    )
    storage.write_parquet_atomic(
        forwards_frame(result, underlying, spot_index),
        interim / "forwards" / f"underlying={underlying}" / f"date={stamp}" / part,
    )
    report = {
        "underlying": underlying,
        "date": stamp,
        "snapshot": snapshot.label,
        "snapshot_flagged": snapshot.flagged,
        **result.report,
    }
    out = config.results_dir() / "dq" / f"{underlying}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def forward_gap_metrics(data_dir: Path, config: AppConfig, underlying: str) -> dict[str, Any]:
    """AC-W-05: |ln(F_parity / F_future)| over all built days, written to results/metrics."""
    root = data_dir / "interim" / "forwards" / f"underlying={underlying}"
    files = sorted(root.glob("date=*/part-*.parquet"))
    metrics: dict[str, Any] = {"underlying": underlying, "days": 0, "n": 0}
    if files:
        frame = pl.concat([pl.read_parquet(path) for path in files])
        gaps = frame["gap_bps"].drop_nulls().abs().to_numpy()
        metrics["days"] = len(files)
        metrics["n"] = int(gaps.size)
        if gaps.size:
            metrics["median_abs_bps"] = float(np.median(gaps))
            metrics["p95_abs_bps"] = float(np.percentile(gaps, 95))
            metrics["max_abs_bps"] = float(gaps.max())
    out = config.results_dir() / "metrics" / f"forward_gap_{underlying}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics

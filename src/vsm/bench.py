"""Benchmarks: IV throughput and calibration times, written to results/bench (NFR-01)."""

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from vsm.config import AppConfig
from vsm.iv.black import black76_price
from vsm.iv.solver import OK, implied_vol
from vsm.models.runner import measure_times
from vsm.pipeline import read_params

IV_QUOTES = 2_000_000
QUICK_QUOTES = 2_000
IV_TARGET = 1_000_000
HESTON_TARGET_SECONDS = 1.0
HESTON_MIN_DAYS = 20  # AC-H-04


def _write(directory: Path, name: str, content: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(content, indent=2, sort_keys=True) + "\n"
    (directory / name).write_text(text, encoding="utf-8")


def cpu_model() -> str:
    """CPU name from /proc/cpuinfo where it exists."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def hardware() -> dict[str, Any]:
    """The machine the numbers were measured on."""
    info: dict[str, Any] = {
        "cpu": cpu_model(),
        "logical_cores": os.cpu_count(),
        "platform": platform.platform(),
    }
    if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
        info["ram_gb"] = round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9, 1)
    return info


def iv_throughput(quotes: int = IV_QUOTES) -> dict[str, Any]:
    """Vols per second on random OTM quotes, JIT warm."""
    rng = np.random.default_rng(20261102)
    strike = np.exp(rng.uniform(-1.0, 1.0, quotes))
    vol = np.exp(rng.uniform(np.log(0.05), np.log(2.0), quotes))
    is_call = strike >= 1.0
    price = black76_price(1.0, strike, 1.0, vol, is_call)
    implied_vol(price[:1000], 1.0, strike[:1000], 1.0, is_call[:1000])
    started = time.perf_counter()
    _, status = implied_vol(price, 1.0, strike, 1.0, is_call)
    elapsed = time.perf_counter() - started
    return {
        "vols_per_second": round(quotes / elapsed),
        "quotes": quotes,
        "failed": int((status != OK).sum()),
        "target": IV_TARGET,
    }


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def heston_report(results_dir: Path, per_underlying: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """AC-H-04: median calibration time over at least 20 days, per underlying."""
    report: dict[str, Any] = {
        "target_seconds": HESTON_TARGET_SECONDS,
        "min_days": HESTON_MIN_DAYS,
        "jacobian": "analytic",
        "underlyings": {},
    }
    for underlying, times in per_underlying.items():
        seconds = times["heston_seconds"]
        median = _median(seconds)
        details = times["details"]
        report["underlyings"][underlying] = {
            "days": len(seconds),
            "median_seconds": median,
            "max_seconds": max(seconds) if seconds else None,
            "median_evaluations": _median([d["evaluations"] for d in details]),
            "median_rmse_iv_bp": _median([d["rmse_iv_bp"] for d in details]),
            "enough_days": len(seconds) >= HESTON_MIN_DAYS,
            "under_target": median is not None and median <= HESTON_TARGET_SECONDS,
            "details": details,
        }
    return report


def run_bench(data_dir: Path, config: AppConfig, *, quick: bool = False) -> dict[str, Any]:
    """Write every benchmark file and return a summary."""
    out = config.results_dir() / "bench"
    throughput = iv_throughput(QUICK_QUOTES if quick else IV_QUOTES)
    _write(out, "iv_throughput.json", throughput)
    _write(out, "hardware.json", hardware())
    summary: dict[str, Any] = {"iv": throughput}
    if quick:
        return summary
    per_underlying: dict[str, dict[str, Any]] = {}
    for underlying in config.underlyings:
        params = read_params(data_dir, underlying)
        if params.is_empty():
            continue
        days = sorted(
            {
                _day
                for _day in params.select(
                    pl.col("snapshot_id").str.extract(r"-(\d{8})T").str.to_date("%Y%m%d")
                )
                .to_series()
                .to_list()
            }
        )
        times = measure_times(data_dir, config, underlying, days)
        per_underlying[underlying] = times
        _write(
            out,
            f"calibration_times_{underlying}.json",
            {
                "SVI": _median(times["surface_seconds"]),
                "SSVI": _median(times["surface_seconds"]),
                "Heston": _median(times["heston_seconds"]),
                "SABR": _median(times["sabr_seconds"]),
            },
        )
    report = heston_report(out, per_underlying)
    _write(out, "heston_calib.json", report)
    summary["heston"] = {
        name: {k: v for k, v in entry.items() if k != "details"}
        for name, entry in report["underlyings"].items()
    }
    return summary

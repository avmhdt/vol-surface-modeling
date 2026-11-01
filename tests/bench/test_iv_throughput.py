"""NFR-01: IV solver throughput on this machine, written to results/bench."""

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pytest

from vsm.iv.black import black76_price
from vsm.iv.solver import OK, implied_vol

TARGET_VOLS_PER_SECOND = 1_000_000
N = 2_000_000
RESULTS = Path("results/bench")


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


@pytest.mark.bench
def test_iv_throughput() -> None:
    rng = np.random.default_rng(20261102)
    strike = np.exp(rng.uniform(-1.0, 1.0, N))
    vol = np.exp(rng.uniform(np.log(0.05), np.log(2.0), N))
    is_call = strike >= 1.0
    price = black76_price(1.0, strike, 1.0, vol, is_call)
    implied_vol(price[:1000], 1.0, strike[:1000], 1.0, is_call[:1000])  # JIT warm-up

    started = time.perf_counter()
    _, status = implied_vol(price, 1.0, strike, 1.0, is_call)
    elapsed = time.perf_counter() - started

    rate = N / elapsed
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "iv_throughput.json").write_text(
        json.dumps(
            {
                "vols_per_second": round(rate),
                "quotes": N,
                "failed": int((status != OK).sum()),
                "target": TARGET_VOLS_PER_SECOND,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (RESULTS / "hardware.json").write_text(
        json.dumps(
            {
                "cpu": cpu_model(),
                "logical_cores": os.cpu_count(),
                "platform": platform.platform(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert np.all(status == OK)
    assert rate >= TARGET_VOLS_PER_SECOND

"""Local vol grids and barrier tables for stored surfaces (FR-19, FR-20, AC-H-03).

Only surfaces that pass the gate are used, and the gate is re-run here rather than trusted from
the stored rows.
"""

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

from vsm.config import AppConfig, config_hash
from vsm.conventions import F64
from vsm.explog import append_run
from vsm.io import storage
from vsm.iv.black import black76_price
from vsm.localvol.barrier import BarrierSpec, black_barrier_price
from vsm.localvol.dupire import LOCAL_VOL_CAP, LocalVolGrid, flat_grid, local_vol_grid
from vsm.localvol.mc import pair_mean, price_barrier_mc, simulate
from vsm.models.heston_cf import HestonParams
from vsm.pipeline import stored_rows
from vsm.surface.arbitrage import check_surface
from vsm.surface.store import heston_path, load_surface
from vsm.surface.types import Surface

DAYS_PER_YEAR = 365.0
Z_LIMIT = 3.0  # AC-H-03
MC_K_POINTS = 1501  # 0.002 in k over the export range
BARRIER_COLUMNS = ("date", "product")

log = structlog.get_logger("vsm.localvol")


def gated_surface(data_dir: Path, config: AppConfig, underlying: str, day: date) -> Surface | None:
    """The stored surface of a day, or None if there is none or it fails the gate."""
    rows = stored_rows(data_dir, config, underlying, day)
    if rows is None:
        return None
    surface = load_surface(rows)
    if not check_surface(surface, config.arbitrage).passed:
        log.warning("surface_failed_gate", snapshot=surface.snapshot_id)
        return None
    return surface


def forward_at(surface: Surface, t: float) -> float:
    """Forward at time ``t``, linear in log between expiries and flat outside them."""
    times = [item.t for item in surface.slices]
    logs = [math.log(item.forward) for item in surface.slices]
    return float(np.exp(np.interp(t, times, logs)))


def _k_grid(config: AppConfig) -> F64:
    grid = config.localvol.k_grid
    return np.linspace(grid.min, grid.max, grid.n)


def _dense_grid(surface: Surface, config: AppConfig, days: float) -> LocalVolGrid:
    """Local vol at each Monte Carlo step's midpoint out to ``days``, on a fine k grid.

    Near expiry local vol is sharply curved in k around the money, and linear interpolation on
    the export grid overstates it there.
    """
    steps = round(days * config.mc.steps_per_day)
    dt = 1.0 / (DAYS_PER_YEAR * config.mc.steps_per_day)
    middle = np.asarray((np.arange(steps) + 0.5) * dt, dtype=np.float64)
    grid = config.localvol.k_grid
    k = np.linspace(grid.min, grid.max, MC_K_POINTS)
    return local_vol_grid(surface.ssvi, k, middle)


def repricing_check(surface: Surface, config: AppConfig) -> dict[str, Any]:
    """Local vol Monte Carlo against the SSVI vanillas it came from, in standard errors."""
    tenors = config.localvol.reprice_tenor_days
    grid = _dense_grid(surface, config, tenors[-1])
    paths = simulate(grid, tenors, config.mc.paths, config.mc.steps_per_day, config.mc.seed)
    rows = []
    for index, days in enumerate(tenors):
        t = days / DAYS_PER_YEAR
        terminal = np.exp(paths.ends[index])
        for strike in config.localvol.reprice_strike_pct:
            is_call = strike >= 1.0
            sign = 1.0 if is_call else -1.0
            w = float(surface.ssvi.w(np.array(math.log(strike)), np.array(t)))
            target = float(black76_price(1.0, strike, t, math.sqrt(w / t), is_call))
            price, error = pair_mean(np.maximum(sign * (terminal - strike), 0.0))
            rows.append(
                {
                    "tenor_days": days,
                    "strike_pct": strike,
                    "ssvi_price": target,
                    "mc_price": price,
                    "se": error,
                    "z": (price - target) / error,
                }
            )
    worst = max(abs(row["z"]) for row in rows)
    return {"worst_abs_z": worst, "passed": worst <= Z_LIMIT, "points": rows}


def _merge_metric(config: AppConfig, name: str, day: date, entry: dict[str, Any]) -> None:
    path = config.results_dir() / "metrics" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    content: dict[str, Any] = {}
    if path.exists():
        content = json.loads(path.read_text(encoding="utf-8"))
    content[day.isoformat()] = entry
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_localvol_day(
    data_dir: Path, config: AppConfig, underlying: str, day: date
) -> dict[str, Any] | None:
    """Store the configured grid of one day and check that it reprices its source."""
    surface = gated_surface(data_dir, config, underlying, day)
    if surface is None:
        return None
    times = np.array(config.localvol.t_grid_days) / DAYS_PER_YEAR
    grid = local_vol_grid(surface.ssvi, _k_grid(config), times)
    frame = pl.DataFrame(
        {
            "t": np.repeat(grid.t, grid.k.size),
            "k": np.tile(grid.k, grid.t.size),
            "local_vol": grid.vol.ravel(),
        }
    )
    root = data_dir / "processed" / "localvol" / f"underlying={underlying}"
    storage.write_parquet_atomic(frame, root / f"date={day.isoformat()}" / "grid.parquet")
    repricing = repricing_check(surface, config)
    entry = {
        "snapshot_id": surface.snapshot_id,
        "finite": bool(np.all(np.isfinite(grid.vol))),
        "positive": bool(np.all(grid.vol > 0.0)),
        "min_vol": float(grid.vol.min()),
        "max_vol": float(grid.vol.max()),
        "capped_share": float(np.mean(grid.vol >= LOCAL_VOL_CAP)),
        "repricing": repricing,
    }
    _merge_metric(config, f"localvol_{underlying}", day, entry)
    append_run(
        config.results_dir(),
        "localvol",
        config_hash(config),
        underlying=underlying,
        start=day.isoformat(),
        end=day.isoformat(),
        metrics={k: v for k, v in entry.items() if k != "repricing"}
        | {"worst_abs_z": repricing["worst_abs_z"]},
    )
    return entry


def _heston_fit(data_dir: Path, surface: Surface) -> tuple[HestonParams, float] | None:
    path = heston_path(data_dir, surface.underlying)
    if not path.exists():
        return None
    rows = pl.read_parquet(path).filter(
        (pl.col("snapshot_id") == surface.snapshot_id) & (pl.col("variant") == "all")
    )
    if rows.is_empty():
        return None
    row = rows.row(0, named=True)
    params = HestonParams(row["v0"], row["kappa"], row["theta"], row["xi"], row["rho"])
    return params, float(row["rmse_iv_bp"])


def run_barrier_day(
    data_dir: Path,
    config: AppConfig,
    underlying: str,
    day: date,
    *,
    paths: int | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]] | None:
    """Price each configured product under local vol, Heston and flat Black for one day."""
    surface = gated_surface(data_dir, config, underlying, day)
    if surface is None:
        return None
    fit = _heston_fit(data_dir, surface)
    if fit is None:
        log.warning("no_heston_fit", snapshot=surface.snapshot_id)
        return None
    params, rmse = fit
    n_paths = config.mc.paths if paths is None else paths
    base_seed = config.mc.seed if seed is None else seed
    days = float(config.barrier.tenor_days)
    t = days / DAYS_PER_YEAR
    forward = forward_at(surface, t)
    atm_vol = math.sqrt(float(surface.ssvi.theta(np.array(t))) / t)
    grid = _dense_grid(surface, config, days)
    flat = flat_grid(atm_vol)
    steps = config.mc.steps_per_day
    rows: list[dict[str, Any]] = []
    for item in config.barrier.products:
        spec = BarrierSpec(item.type, item.strike_pct, item.barrier_pct, days)
        local_price, local_error = price_barrier_mc(grid, spec, forward, n_paths, steps, base_seed)
        heston_price, heston_error = price_barrier_mc(
            params, spec, forward, n_paths, steps, base_seed
        )
        black_price, black_error = price_barrier_mc(flat, spec, forward, n_paths, steps, base_seed)
        rows.append(
            {
                "date": day.isoformat(),
                "product": item.type,
                "snapshot_id": surface.snapshot_id,
                "strike": forward * item.strike_pct,
                "barrier": forward * item.barrier_pct,
                "tenor_days": days,
                "forward": forward,
                "atm_vol": atm_vol,
                "local_vol_price": local_price,
                "local_vol_se": local_error,
                "heston_price": heston_price,
                "heston_se": heston_error,
                "black_price": black_price,
                "black_se": black_error,
                "black_closed_form": black_barrier_price(spec, forward, atm_vol),
                "heston_rmse_iv_bp": rmse,
                "paths": n_paths,
                "steps_per_day": steps,
                "seed": base_seed,
            }
        )
    path = config.results_dir() / "tables" / f"barrier_lv_vs_heston_{underlying}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pl.DataFrame(rows)
    if path.exists():
        kept = pl.read_csv(path, schema_overrides=table.schema).join(
            table.select(BARRIER_COLUMNS), on=list(BARRIER_COLUMNS), how="anti"
        )
        table = pl.concat([kept, table])
    table.sort(*BARRIER_COLUMNS).write_csv(path)
    append_run(
        config.results_dir(),
        "barrier",
        config_hash(config),
        underlying=underlying,
        start=day.isoformat(),
        end=day.isoformat(),
        params={"paths": n_paths, "seed": base_seed},
        metrics={row["product"]: row["local_vol_price"] for row in rows},
    )
    return rows

"""Write ``vsm.surface.v1``. Only a surface that passed the arbitrage gate can be written."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import vsm
from vsm.config import ArbitrageConfig
from vsm.conventions import ns_to_datetime
from vsm.surface.types import Surface

SCHEMA_NAME = "vsm.surface.v1"
PRODUCER = "vol-surface-modeling"
ISO_SECONDS = "%Y-%m-%dT%H:%M:%SZ"


def _iso(ns: int) -> str:
    return ns_to_datetime(ns).strftime(ISO_SECONDS)


def _number(value: float) -> float | None:
    return None if math.isnan(value) else value


def surface_document(
    surface: Surface,
    cfg: ArbitrageConfig,
    *,
    coin_rate: str = "zero",
    git_sha: str = "unknown",
    config_hash: str = "",
    created_at: datetime | None = None,
    heston: dict[str, Any] | None = None,
    sabr: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The JSON document of a gated surface. Refuses a surface that failed the gate."""
    report = surface.report
    if report is None or not report.passed:
        raise ValueError(f"{surface.snapshot_id} did not pass the arbitrage gate")
    ssvi = surface.ssvi
    return {
        "schema": SCHEMA_NAME,
        "underlying": surface.underlying,
        "venue": surface.venue,
        "as_of": _iso(surface.as_of_ns),
        "as_of_ns": surface.as_of_ns,
        "spot": {"index_name": f"{surface.underlying.lower()}_usd", "value": surface.spot_index},
        "conventions": {
            "option_style": "european",
            "settlement": "inverse",
            "premium_currency": surface.underlying,
            "pricing": "undiscounted Black-76 on the expiry forward",
            "year_fraction": "ACT/365 seconds",
            "expiry_time_utc": "08:00:00",
            "moneyness": "k = ln(K/F)",
            "total_variance": "w = sigma^2 * t",
            "coin_rate": coin_rate,
        },
        "canonical": "ssvi",
        "ssvi": {
            "phi": "power_law",
            "rho": ssvi.rho,
            "eta": ssvi.eta,
            "gamma": ssvi.gamma,
            "lambda": None,
            "theta_interpolation": "linear_in_t",
            "t_nodes": list(ssvi.t_nodes),
            "theta_nodes": list(ssvi.theta_nodes),
        },
        "expiries": [
            {
                "expiry": _iso(item.expiry_ns),
                "expiry_ns": item.expiry_ns,
                "t": item.t,
                "forward": item.forward,
                "forward_method": item.forward_method,
                "forward_se": _number(item.forward_se),
                "discount_factor": 1.0 if coin_rate == "zero" else item.df,
                "basis_annualized": item.basis_ann,
                "dividend_yield": None,
                "svi_raw": {
                    "a": item.svi.a,
                    "b": item.svi.b,
                    "rho": item.svi.rho,
                    "m": item.svi.m,
                    "sigma": item.svi.sigma,
                },
                "slice_source": item.source,
                "k_range_quoted": [item.k_min, item.k_max],
                "fit": {
                    "n_quotes": item.n_quotes,
                    "n_liquid": item.n_liquid,
                    "rmse_iv_bp": item.rmse_iv_bp,
                    "within_spread_frac": _number(item.within_spread_frac),
                },
            }
            for item in surface.slices
        ],
        "arbitrage": {
            "passed": report.passed,
            "min_g": report.min_g,
            "min_calendar_dw": report.min_calendar_dw,
            "density_mass_max_err": report.density_mass_max_err,
            "grid": {"z_min": cfg.grid_z.min, "z_max": cfg.grid_z.max, "n": cfg.grid_z.n},
        },
        "heston": heston,
        "sabr": sabr or [],
        "local_vol": None,
        "provenance": {
            "vsm_version": vsm.__version__,
            "git_sha": git_sha,
            "config_hash": config_hash,
            "snapshot_id": surface.snapshot_id,
            "created_at": (created_at or datetime.now(UTC)).strftime(ISO_SECONDS),
            "producer": PRODUCER,
        },
    }


def bundle_path(surface: Surface, out_dir: str | Path) -> Path:
    """``<out_dir>/<underlying>/<YYYYMMDDTHHMMZ>/surface.json``."""
    label = surface.snapshot_id.rsplit("-", 1)[1]
    return Path(out_dir) / surface.underlying / label / "surface.json"


def write_surface_bundle(
    surface: Surface,
    out_dir: str,
    *,
    cfg: ArbitrageConfig | None = None,
    coin_rate: str = "zero",
    git_sha: str = "unknown",
    config_hash: str = "",
    heston: dict[str, Any] | None = None,
    sabr: list[dict[str, Any]] | None = None,
) -> str:
    """Write the JSON file of a gated surface and return its path."""
    document = surface_document(
        surface,
        cfg or ArbitrageConfig(),
        coin_rate=coin_rate,
        git_sha=git_sha,
        config_hash=config_hash,
        heston=heston,
        sabr=sabr,
    )
    path = bundle_path(surface, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return str(path)

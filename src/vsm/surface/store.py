"""The ``surface_params`` table: one row per slice per as-of, and back to a Surface."""

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from vsm.conventions import ns_to_datetime, year_fraction
from vsm.io import storage
from vsm.surface.ssvi import SSVI
from vsm.surface.svi import RawSVI
from vsm.surface.types import ArbReport, Surface, SurfaceSlice

DROPPED = "dropped"


def params_path(data_dir: Path, underlying: str) -> Path:
    """Internal history, gate failures included."""
    return (
        data_dir / "processed" / "surfaces" / f"underlying={underlying}" / "surface_params.parquet"
    )


def export_params_path(data_dir: Path, underlying: str) -> Path:
    """Consumer history: gated surfaces only."""
    return data_dir / "exports" / underlying / "surface_params.parquet"


def heston_path(data_dir: Path, underlying: str) -> Path:
    """Heston fits, one row per snapshot and variant."""
    root = data_dir / "processed" / "model_params" / f"underlying={underlying}"
    return root / "heston_params.parquet"


def sabr_path(data_dir: Path, underlying: str) -> Path:
    """SABR fits, one row per snapshot, expiry and beta."""
    root = data_dir / "processed" / "model_params" / f"underlying={underlying}"
    return root / "sabr_params.parquet"


def model_blocks(
    data_dir: Path, underlying: str, snapshot_id: str, beta: float
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """The ``heston`` and ``sabr`` blocks of an export, if those fits exist."""
    heston: dict[str, Any] | None = None
    path = heston_path(data_dir, underlying)
    if path.exists():
        rows = pl.read_parquet(path).filter(
            (pl.col("snapshot_id") == snapshot_id) & (pl.col("variant") == "all")
        )
        if rows.height:
            row = rows.row(0, named=True)
            heston = {
                name: row[name]
                for name in ("v0", "kappa", "theta", "xi", "rho", "rmse_iv_bp", "converged")
            }
    sabr: list[dict[str, Any]] = []
    path = sabr_path(data_dir, underlying)
    if path.exists():
        rows = pl.read_parquet(path).filter(
            (pl.col("snapshot_id") == snapshot_id) & (pl.col("beta") == beta)
        )
        sabr = [
            {
                "expiry": ns_to_datetime(row["expiry_ns"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "alpha": row["alpha"],
                "beta": row["beta"],
                "rho": row["rho"],
                "nu": row["nu"],
            }
            for row in rows.sort("expiry_ns").iter_rows(named=True)
        ]
    return heston, sabr


def surface_rows(surface: Surface, git_sha: str, config_hash: str) -> pl.DataFrame:
    """One row per fitted slice, plus a null-parameter row per dropped slice.

    SSVI nodes include dropped expiries, so the dropped rows are what lets a surface be
    rebuilt exactly from this table.
    """
    report = surface.report
    assert report is not None, "gate the surface before storing it"
    ssvi = surface.ssvi
    common = {
        "as_of_ns": surface.as_of_ns,
        "underlying": surface.underlying,
        "snapshot_id": surface.snapshot_id,
        "spot_index": surface.spot_index,
        "ssvi_rho": ssvi.rho,
        "ssvi_eta": ssvi.eta,
        "ssvi_gamma": ssvi.gamma,
        "git_sha": git_sha,
        "config_hash": config_hash,
    }
    rows = []
    for item, check in zip(surface.slices, report.slices, strict=True):
        jw = item.svi.to_jw(item.t)
        rows.append(
            {
                **common,
                "expiry_ns": item.expiry_ns,
                "t": item.t,
                "forward": item.forward,
                "forward_se": item.forward_se,
                "forward_method": item.forward_method,
                "df": item.df,
                "basis_ann": item.basis_ann,
                "svi_a": item.svi.a,
                "svi_b": item.svi.b,
                "svi_rho": item.svi.rho,
                "svi_m": item.svi.m,
                "svi_sigma": item.svi.sigma,
                "jw_v": jw.v,
                "jw_psi": jw.psi,
                "jw_p": jw.p,
                "jw_c": jw.c,
                "jw_vtilde": jw.vtilde,
                "ssvi_theta": float(ssvi.theta(np.array(item.t))),
                "slice_source": item.source,
                "k_min": item.k_min,
                "k_max": item.k_max,
                "rmse_iv_bp": item.rmse_iv_bp,
                "within_spread_frac": item.within_spread_frac,
                "ssvi_within_spread_frac": item.ssvi_within_spread_frac,
                "n_quotes": item.n_quotes,
                "n_liquid": item.n_liquid,
                "min_g": check.min_g,
                "min_calendar_dw": check.min_calendar_dw,
                "density_mass": check.density_mass_err + 1.0,
                "arb_free": check.passed and report.passed,
            }
        )
    for expiry_ns in surface.dropped_expiries:
        t = float(year_fraction(np.array([expiry_ns]), surface.as_of_ns)[0])
        rows.append(
            {
                **common,
                "expiry_ns": expiry_ns,
                "t": t,
                "ssvi_theta": float(ssvi.theta(np.array(t))),
                "slice_source": DROPPED,
                "arb_free": False,
            }
        )
    schema = storage.SURFACE_PARAMS_SCHEMA
    return pl.DataFrame(rows, schema=schema).select(list(schema)).sort("expiry_ns")


def _optional(value: float | None) -> float:
    return math.nan if value is None else float(value)


def load_surface(frame: pl.DataFrame) -> Surface:
    """Rebuild one as-of's Surface, without its gate report, from its rows."""
    rows = frame.sort("expiry_ns").to_dicts()
    first = rows[0]
    ssvi = SSVI(
        rho=first["ssvi_rho"],
        eta=first["ssvi_eta"],
        gamma=first["ssvi_gamma"],
        t_nodes=tuple(row["t"] for row in rows),
        theta_nodes=tuple(row["ssvi_theta"] for row in rows),
    )
    slices = tuple(
        SurfaceSlice(
            expiry_ns=row["expiry_ns"],
            t=row["t"],
            forward=row["forward"],
            forward_se=_optional(row["forward_se"]),
            forward_method=row["forward_method"],
            df=row["df"],
            basis_ann=row["basis_ann"],
            svi=RawSVI(row["svi_a"], row["svi_b"], row["svi_rho"], row["svi_m"], row["svi_sigma"]),
            source=row["slice_source"],
            k_min=row["k_min"],
            k_max=row["k_max"],
            n_quotes=row["n_quotes"],
            n_liquid=row["n_liquid"],
            rmse_iv_bp=row["rmse_iv_bp"],
            within_spread_frac=_optional(row["within_spread_frac"]),
            ssvi_within_spread_frac=_optional(row["ssvi_within_spread_frac"]),
        )
        for row in rows
        if row["slice_source"] != DROPPED
    )
    return Surface(
        underlying=first["underlying"],
        venue=storage.VENUE,
        as_of_ns=first["as_of_ns"],
        snapshot_id=first["snapshot_id"],
        spot_index=first["spot_index"],
        ssvi=ssvi,
        slices=slices,
        dropped_expiries=tuple(row["expiry_ns"] for row in rows if row["slice_source"] == DROPPED),
    )


def store_surface(data_dir: Path, surface: Surface, git_sha: str, config_hash: str) -> None:
    """Upsert a gated surface; only passing surfaces reach the consumer history."""
    rows = surface_rows(surface, git_sha, config_hash)
    keys = ["as_of_ns", "expiry_ns"]
    storage.upsert_parquet(rows, params_path(data_dir, surface.underlying), keys)
    report = surface.report
    if report is not None and report.passed:
        exported = rows.filter(pl.col("slice_source") != DROPPED)
        storage.upsert_parquet(exported, export_params_path(data_dir, surface.underlying), keys)


def with_report(surface: Surface, report: ArbReport) -> Surface:
    """The surface with its gate report attached."""
    return replace(surface, report=report)

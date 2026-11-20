"""The report figures that come from stored results rather than from one pipeline step (§14).

Each figure reads files the studies already wrote, so it can be redrawn without refitting.
A figure whose inputs are missing is skipped and named in the return value.
"""

import math
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from vsm.config import AppConfig
from vsm.localvol.runner import gated_surface
from vsm.pipeline import clean_quotes_of
from vsm.surface.density import density_k
from vsm.surface.svi import slice_g
from vsm.viz import figures

SMILE_PANELS = 9
K_RANGE = 0.8


def _out(config: AppConfig, name: str) -> Path:
    return config.reports_dir() / "figures" / name


def smile_and_density(data_dir: Path, config: AppConfig, underlying: str, day: date) -> bool:
    """Smile fits per expiry, and g(k) and density per slice, for one gated day."""
    surface = gated_surface(data_dir, config, underlying, day)
    if surface is None:
        return False
    label = surface.snapshot_id.rsplit("-", 1)[1]
    quotes = clean_quotes_of(data_dir, underlying, label)
    if quotes is None:
        return False
    kept = quotes.filter(pl.col("drop_reason").is_null() & pl.col("is_otm"))
    panels = []
    for item in surface.slices[:SMILE_PANELS]:
        rows = kept.filter(pl.col("expiry_ns") == item.expiry_ns).sort("k")
        if rows.is_empty():
            continue
        grid = np.linspace(item.k_min - 0.05, item.k_max + 0.05, 200)
        model = np.sqrt(item.svi.w(grid) / item.t)
        panels.append(
            (
                f"{item.t * 365.0:.0f} days",
                rows["k"].to_numpy(),
                rows["bid_iv"].to_numpy(),
                rows["ask_iv"].to_numpy(),
                grid,
                model,
            )
        )
    stamp = day.isoformat()
    figures.smile_fits(
        panels,
        f"{underlying} {stamp}: smiles by expiry",
        _out(config, f"smile_fits_{underlying}.png"),
    )
    k = np.linspace(-K_RANGE, K_RANGE, 401)
    curves = [
        (f"{item.t * 365.0:.0f} d", slice_g(item.svi, k), density_k(k, item.svi))
        for item in surface.slices
    ]
    figures.g_and_density(
        k,
        curves,
        f"{underlying} {stamp}: arbitrage diagnostics by slice",
        _out(config, f"g_density_{underlying}.png"),
    )
    return True


def local_vol_map(data_dir: Path, config: AppConfig, underlying: str, day: date) -> bool:
    """Heatmap of the stored local vol grid of one day."""
    path = (
        data_dir
        / "processed"
        / "localvol"
        / f"underlying={underlying}"
        / f"date={day.isoformat()}"
        / "grid.parquet"
    )
    if not path.exists():
        return False
    grid = pl.read_parquet(path).sort("t", "k")
    k = np.unique(grid["k"].to_numpy())
    t = np.unique(grid["t"].to_numpy())
    values = grid["local_vol"].to_numpy().reshape(t.size, k.size)
    figures.heatmap(
        k,
        t * 365.0,
        values,
        f"{underlying} {day.isoformat()}: Dupire local vol from SSVI",
        _out(config, f"localvol_{underlying}.png"),
    )
    return True


def barrier_chart(config: AppConfig, underlying: str) -> bool:
    """Knock-out prices by model and date, from the barrier table."""
    path = config.results_dir() / "tables" / f"barrier_lv_vs_heston_{underlying}.csv"
    if not path.exists():
        return False
    table = pl.read_csv(path).sort("date", "product")
    labels = [
        f"{row['date']}\n{row['product'].replace('_call', '').replace('_', '-')}"
        for row in table.iter_rows(named=True)
    ]
    prices = {
        "Local vol": (table["local_vol_price"].to_numpy(), table["local_vol_se"].to_numpy()),
        "Heston": (table["heston_price"].to_numpy(), table["heston_se"].to_numpy()),
        "Flat Black": (table["black_price"].to_numpy(), table["black_se"].to_numpy()),
    }
    figures.barrier_bars(
        labels,
        prices,
        f"{underlying}: 30-day knock-out calls at the forward",
        _out(config, f"barrier_{underlying}.png"),
    )
    return True


def _years(days: list[str]) -> np.ndarray:
    out = []
    for value in days:
        d = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
        out.append(d.year + (d.timetuple().tm_yday - 1) / 365.0)
    return np.array(out)


def vrp_chart(config: AppConfig, underlying: str) -> bool:
    """DVOL squared against realized variance over time."""
    path = config.results_dir() / "tables" / f"vrp_dvol_{underlying}.csv"
    if not path.exists():
        return False
    table = pl.read_csv(path).sort("date")
    if table.is_empty():
        return False
    figures.vrp_series(
        _years(table["date"].to_list()),
        table["implied_var"].to_numpy(),
        table["realized_var"].to_numpy(),
        f"{underlying}: 30-day implied against realized variance",
        _out(config, f"vrp_{underlying}.png"),
    )
    return True


def pnl_decomposition(config: AppConfig, underlying: str) -> bool:
    """Realized hedged P&L, funding left out, against the theoretical gamma P&L."""
    path = config.results_dir() / "tables" / f"straddles_tardis_monthly_entry_iv_{underlying}.csv"
    if not path.exists():
        return False
    try:
        table = pl.read_csv(path)
    except pl.exceptions.NoDataError:  # a source with no trades writes an empty table
        return False
    if table.height < 3:
        return False
    theory = (table["gamma_pnl"] / table["forward"]).to_numpy()
    realized = ((table["gross_pnl"] - table["funding_pnl"]) / table["forward"]).to_numpy()
    r2 = float(np.corrcoef(theory, realized)[0, 1] ** 2)
    figures.pnl_scatter(
        theory,
        realized,
        r2 if math.isfinite(r2) else math.nan,
        f"{underlying}: monthly short straddles",
        _out(config, f"pnl_decomposition_{underlying}.png"),
    )
    return True


def draw_all(data_dir: Path, config: AppConfig, showcase: date) -> dict[str, list[str]]:
    """Every figure here for every underlying; returns what was drawn and what was skipped."""
    drawn: list[str] = []
    skipped: list[str] = []
    for underlying in config.underlyings:
        for name, ok in (
            ("smile_and_density", smile_and_density(data_dir, config, underlying, showcase)),
            ("local_vol", local_vol_map(data_dir, config, underlying, showcase)),
            ("barrier", barrier_chart(config, underlying)),
            ("vrp", vrp_chart(config, underlying)),
            ("pnl_decomposition", pnl_decomposition(config, underlying)),
        ):
            (drawn if ok else skipped).append(f"{name}_{underlying}")
    return {"drawn": drawn, "skipped": skipped}

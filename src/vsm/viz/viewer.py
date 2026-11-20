"""Interactive 3D surface viewer: one static HTML per underlying (FR-24, AC-H-05, D-07).

Implied vol of the exported SSVI surface over log-moneyness and days to expiry, with a slider
across the gated history. Hover shows the strike, the days and the IV. On at most five
showcase dates, the quote mids the surface was fitted to are drawn as markers. Plotly's
JavaScript is inlined, so the file opens offline.
"""

import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

from vsm.config import AppConfig
from vsm.conventions import F64
from vsm.pipeline import clean_quotes_of
from vsm.surface.store import export_params_path, load_surface
from vsm.surface.types import Surface

K_GRID = np.linspace(-0.6, 0.6, 41)
T_DAYS = np.array([7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365], dtype=np.float64)
MAX_QUOTE_DATES = 5  # D-07
MAX_BYTES = 10_000_000  # AC-H-05
ATTRIBUTION = "Data: Deribit public market data. Surfaces fitted by this project."


def _day(surface: Surface) -> str:
    return datetime.fromtimestamp(surface.as_of_ns / 1e9, UTC).strftime("%Y-%m-%d")


def history(data_dir: Path, underlying: str, start: str, end: str) -> list[Surface]:
    """Exported (gated) surfaces in the range, one per day, the canonical one where there is one."""
    path = export_params_path(data_dir, underlying)
    if not path.exists():
        return []
    params = pl.read_parquet(path)
    chosen: dict[str, Surface] = {}
    for (_,), frame in sorted(params.partition_by("snapshot_id", as_dict=True).items()):
        surface = load_surface(frame)
        day = _day(surface)
        if start <= day <= end and (day not in chosen or "T0815Z" in surface.snapshot_id):
            chosen[day] = surface
    return [chosen[day] for day in sorted(chosen)]


def _thin(surfaces: list[Surface], limit: int) -> list[Surface]:
    if len(surfaces) <= limit:
        return surfaces
    keep = np.unique(np.linspace(0, len(surfaces) - 1, limit).round().astype(int))
    return [surfaces[i] for i in keep]


def _grid(surface: Surface) -> tuple[F64, F64, F64]:
    """(IV in %, strike, days) on the K_GRID x T_DAYS grid."""
    t = T_DAYS / 365.0
    k, tt = np.meshgrid(K_GRID, t)
    iv = 100.0 * np.sqrt(surface.ssvi.w(k, tt) / tt)
    times = [item.t for item in surface.slices]
    logs = [math.log(item.forward) for item in surface.slices]
    forward = np.exp(np.interp(tt, times, logs))
    return iv, forward * np.exp(k), np.broadcast_to(T_DAYS[:, None], k.shape)


def _surface_trace(surface: Surface) -> go.Surface:
    iv, strike, days = _grid(surface)
    return go.Surface(
        x=K_GRID,
        y=T_DAYS,
        z=np.round(iv, 2),
        customdata=np.stack([np.round(strike, 0), days], axis=-1),
        colorscale="Blues",
        cmin=20,
        cmax=150,
        colorbar={"title": "IV %"},
        hovertemplate="K %{customdata[0]:,.0f}<br>T %{customdata[1]:.0f} d<br>"
        "IV %{z:.1f}%<extra></extra>",
        name="Fitted SSVI",
    )


def _quote_trace(surface: Surface, quotes: pl.DataFrame) -> go.Scatter3d:
    rows = quotes.filter(
        pl.col("drop_reason").is_null() & pl.col("is_otm") & (pl.col("k").abs() <= 0.6)
    ).filter(pl.col("t") * 365.0 <= 365.0)
    return go.Scatter3d(
        x=rows["k"].to_numpy(),
        y=rows["t"].to_numpy() * 365.0,
        z=np.round(100.0 * rows["mid_iv"].to_numpy(), 2),
        customdata=rows["strike"].to_numpy(),
        mode="markers",
        marker={"size": 2.5, "color": "#eb6834"},
        hovertemplate="Quote mid<br>K %{customdata:,.0f}<br>T %{y:.0f} d<br>"
        "IV %{z:.1f}%<extra></extra>",
        name="Quote mids",
    )


def build_viewer(
    data_dir: Path,
    config: AppConfig,
    underlying: str,
    start: str,
    end: str,
    quote_dates: list[str] | None = None,
) -> Path | None:
    """Write ``reports/viewer/surface_<underlying>.html``; None if there is nothing to show."""
    surfaces = _thin(history(data_dir, underlying, start, end), config.viewer.max_dates)
    if not surfaces:
        return None
    showcase = set(quote_dates if quote_dates is not None else config.viewer.quote_dates)
    if len(showcase) > MAX_QUOTE_DATES:
        raise ValueError(f"at most {MAX_QUOTE_DATES} quote dates (D-07)")
    frames = []
    for surface in surfaces:
        day = _day(surface)
        data = [_surface_trace(surface)]
        if day in showcase:
            label = surface.snapshot_id.rsplit("-", 1)[1]
            quotes = clean_quotes_of(data_dir, underlying, label)
            if quotes is not None:
                data.append(_quote_trace(surface, quotes))
        frames.append(go.Frame(data=data, name=day))
    figure = go.Figure(data=frames[-1].data, frames=frames)
    steps = [
        {
            "args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": f.name,
            "method": "animate",
        }
        for f in frames
    ]
    figure.update_layout(
        title=f"{underlying} implied volatility surface",
        scene={
            "xaxis_title": "ln(K / F)",
            "yaxis_title": "Days to expiry",
            "zaxis_title": "IV %",
        },
        sliders=[{"active": len(frames) - 1, "steps": steps, "currentvalue": {"prefix": "As of "}}],
        annotations=[
            {
                "text": ATTRIBUTION,
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": -0.08,
                "font": {"size": 10, "color": "#52514e"},
            }
        ],
        margin={"l": 0, "r": 0, "t": 40, "b": 60},
    )
    path = config.reports_dir() / "viewer" / f"surface_{underlying}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True, full_html=True, auto_play=False)
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"{path.name} is {size} bytes, over AC-H-05's {MAX_BYTES}")
    return path

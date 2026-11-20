"""Matplotlib report figures, written as static PNGs at 150 dpi."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from vsm.conventions import F64
from vsm.surface.density import density_k
from vsm.surface.spline_baseline import SplineSlice
from vsm.surface.svi import RawSVI

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
DPI = 150


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _spline_density(spline: SplineSlice, k: F64) -> F64:
    """Density of ln(K/F) from the spline's own g(k)."""
    w = spline.w(k)
    root = np.sqrt(w)
    d2 = -k / root - 0.5 * root
    density: F64 = spline.g(k) / (np.sqrt(2.0 * np.pi) * root) * np.exp(-0.5 * d2 * d2)
    return density


def spline_vs_svi_density(spline: SplineSlice, svi: RawSVI, title: str, path: Path) -> None:
    """The risk-neutral density each interpolation implies on one slice."""
    k = spline.grid()
    spline_p = _spline_density(spline, k)
    svi_p = density_k(k, svi)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.plot(k, svi_p, color=BLUE, linewidth=2.0, label="SVI")
    ax.plot(k, spline_p, color=ORANGE, linewidth=2.0, label="Cubic spline in IV")
    ax.set_xlabel("Log-moneyness k = ln(K/F)")
    ax.set_ylabel("Density of k")
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, labelcolor=MUTED, fontsize=8)
    _save(fig, path)


def model_fit_comparison(
    models: list[str], rmse_bp: list[float], inside: list[float], title: str, path: Path
) -> None:
    """IV RMSE by model, with the share of quotes inside the spread beside each bar."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    positions = np.arange(len(models))[::-1]
    ax.barh(positions, rmse_bp, height=0.55, color=BLUE)
    ax.set_yticks(positions, models)
    limit = max(rmse_bp) * 1.32
    ax.set_xlim(0.0, limit)
    for position, value, share in zip(positions, rmse_bp, inside, strict=True):
        ax.text(
            value + limit * 0.012,
            position,
            f"{value:.0f} bp, {100.0 * share:.0f}% inside spread",
            va="center",
            color=INK,
            fontsize=8,
        )
    ax.set_xlabel("IV RMSE, vol bp")
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    _save(fig, path)


def smile_fits(panels: list[tuple[str, F64, F64, F64, F64, F64]], title: str, path: Path) -> None:
    """One panel per expiry: quoted bid and ask IV against the fitted SVI slice.

    Each panel is (label, k, bid_iv, ask_iv, k_grid, model_iv), with vols as fractions.
    """
    columns = 3
    rows = max(1, math.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(9.6, 2.6 * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (label, k, bid, ask, grid, model) in zip(axes.ravel(), panels, strict=False):
        _style(ax)
        ax.vlines(k, 100.0 * bid, 100.0 * ask, color=MUTED, linewidth=1.2)
        ax.plot(grid, 100.0 * model, color=BLUE, linewidth=2.0)
        ax.set_title(label, loc="left", color=INK, fontsize=9)
    for ax in axes.ravel()[len(panels) :]:
        ax.set_visible(False)
    fig.supxlabel("k = ln(K/F). Grey bars are bid to ask, blue is the SVI fit.", color=MUTED)
    fig.supylabel("Implied vol, %", color=MUTED)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def g_and_density(k: F64, curves: list[tuple[str, F64, F64]], title: str, path: Path) -> None:
    """Durrleman's g and the implied density of k, one line per slice."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    shades = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, max(len(curves), 1)))
    for (label, g, density), color in zip(curves, shades, strict=False):
        top.plot(k, g, color=color, linewidth=1.5, label=label)
        bottom.plot(k, density, color=color, linewidth=1.5)
    for ax in (top, bottom):
        _style(ax)
        ax.axhline(0.0, color=MUTED, linewidth=0.8)
    top.set_ylabel("g(k): no butterfly arbitrage where > 0")
    bottom.set_ylabel("Density of k")
    bottom.set_xlabel("Log-moneyness k = ln(K/F)")
    top.legend(frameon=False, fontsize=7, labelcolor=MUTED, ncol=2)
    top.set_title(title, loc="left", color=INK, fontsize=11)
    _save(fig, path)


def heatmap(k: F64, days: F64, values: F64, title: str, path: Path) -> None:
    """A field over (k, days), such as local vol, in percent."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(False)
    mesh = ax.pcolormesh(k, days, 100.0 * values, cmap="Blues", shading="nearest")
    bar = fig.colorbar(mesh, ax=ax)
    bar.set_label("Local vol, %", color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("Log-moneyness k = ln(K/F)")
    ax.set_ylabel("Days")
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    _save(fig, path)


def barrier_bars(
    labels: list[str], prices: dict[str, tuple[F64, F64]], title: str, path: Path
) -> None:
    """Knock-out prices by model, with 2 standard error whiskers."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    width = 0.8 / max(len(prices), 1)
    colors = (BLUE, ORANGE, MUTED)
    positions = np.arange(len(labels))
    for index, (name, (values, errors)) in enumerate(prices.items()):
        ax.bar(
            positions + index * width,
            values,
            width=width * 0.92,
            yerr=2.0 * errors,
            color=colors[index % len(colors)],
            label=name,
            error_kw={"ecolor": INK, "linewidth": 0.8},
        )
    ax.set_xticks(positions + width * (len(prices) - 1) / 2, labels, fontsize=8)
    ax.set_ylabel("Price, USD per coin")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    _save(fig, path)


def vrp_series(years: F64, implied: F64, realized: F64, title: str, path: Path) -> None:
    """Implied (DVOL squared) and realized 30-day variance by observation date."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot(years, implied, color=BLUE, linewidth=1.2, label="DVOL squared, 30-day implied")
    ax.plot(years, realized, color=ORANGE, linewidth=1.2, label="Realized over the next 30 days")
    ax.set_xlabel("Observation date")
    ax.set_ylabel("Annualized variance")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    _save(fig, path)


def pnl_scatter(theory: F64, realized: F64, r2: float, title: str, path: Path) -> None:
    """Each straddle's hedged P&L against its theoretical gamma P&L, as % of the forward."""
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.axhline(0.0, color=GRID, linewidth=0.8)
    ax.axvline(0.0, color=GRID, linewidth=0.8)
    ax.scatter(100.0 * theory, 100.0 * realized, s=22, color=BLUE, edgecolor=SURFACE)
    span = 100.0 * float(np.max(np.abs(np.concatenate([theory, realized])))) * 1.05
    ax.plot([-span, span], [-span, span], color=MUTED, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Theoretical gamma P&L, % of forward")
    ax.set_ylabel("Hedged P&L before costs, % of forward")
    ax.set_title(f"{title}, R^2 {r2:.2f}", loc="left", color=INK, fontsize=11)
    _save(fig, path)


def param_history(
    plain_ns: F64,
    plain: dict[str, F64],
    regularized_ns: F64,
    regularized: dict[str, F64],
    title: str,
    path: Path,
) -> None:
    """SSVI rho, eta and 30-day theta by day, unregularized and regularized."""
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    days_plain = (plain_ns - plain_ns[0]) / 86_400e9
    days_reg = (regularized_ns - plain_ns[0]) / 86_400e9
    for ax, name in zip(axes, ("rho", "eta", "theta_30d"), strict=True):
        _style(ax)
        ax.plot(days_plain, plain[name], color=BLUE, linewidth=2, marker="o", markersize=3)
        ax.plot(days_reg, regularized[name], color=ORANGE, linewidth=2, marker="o", markersize=3)
        ax.set_ylabel(name, color=MUTED)
    axes[0].plot([], [], color=BLUE, linewidth=2, label="Unregularized")
    axes[0].plot([], [], color=ORANGE, linewidth=2, label="Regularized")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED)
    axes[-1].set_xlabel("Days since the first recorder day")
    axes[0].set_title(title, loc="left", color=INK, fontsize=11)
    _save(fig, path)


def sticky_regression(
    returns: F64, fixed_strike: F64, fixed_moneyness: F64, title: str, path: Path
) -> None:
    """Daily IV changes against spot log returns, at a fixed strike and a fixed moneyness."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.axhline(0.0, color=GRID, linewidth=0.8)
    ax.axvline(0.0, color=GRID, linewidth=0.8)
    for values, color, label in (
        (fixed_strike, BLUE, "Fixed strike"),
        (fixed_moneyness, ORANGE, "Fixed moneyness (sticky delta)"),
    ):
        ax.scatter(100.0 * returns, 100.0 * values, s=18, color=color, edgecolor=SURFACE)
        if returns.size >= 2 and np.ptp(returns) > 0.0:
            slope, intercept = np.polyfit(returns, values, 1)
            line = np.linspace(returns.min(), returns.max(), 2)
            ax.plot(100.0 * line, 100.0 * (intercept + slope * line), color=color, linewidth=2)
        ax.plot([], [], color=color, linewidth=2, label=label)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    ax.set_xlabel("Daily log return of the 30-day forward, %")
    ax.set_ylabel("Change in 30-day IV, vol points")
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    _save(fig, path)


def straddle_pnl(entry_years: F64, gross: F64, net: F64, title: str, path: Path) -> None:
    """Cumulative gross and net P&L of the straddle study, as a fraction of the forward."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.axhline(0.0, color=GRID, linewidth=0.8)
    ax.plot(entry_years, 100.0 * np.cumsum(gross), color=BLUE, linewidth=2, label="Gross")
    ax.plot(entry_years, 100.0 * np.cumsum(net), color=ORANGE, linewidth=2, label="Net of costs")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    ax.set_xlabel("Entry date")
    ax.set_ylabel("Cumulative P&L, % of forward")
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    _save(fig, path)


def rnd_skew(risk_reversal: F64, skewness: F64, t_days: F64, title: str, path: Path) -> None:
    """Density skewness against the 25-delta risk reversal, one point per slice."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.axhline(0.0, color=GRID, linewidth=0.8)
    ax.axvline(0.0, color=GRID, linewidth=0.8)
    ax.scatter(
        100.0 * risk_reversal,
        skewness,
        s=np.clip(t_days, 12.0, 120.0),
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=1.0,
    )
    ax.set_xlabel("25-delta risk reversal, vol points. Larger markers are longer expiries.")
    ax.set_ylabel("Skewness of ln(S_T/F)")
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    _save(fig, path)

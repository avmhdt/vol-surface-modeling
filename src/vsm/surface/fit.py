"""Fit one day's surface: SSVI first, then each SVI slice through the fallback chain.

SSVI comes first because it anchors the wings. A slice is only identified inside its quoted
range, and without an anchor a short slice can extrapolate to a wing so steep that no later
slice can stay above it. Light pseudo-points from the SSVI slice, outside the quoted range,
hold the wings where SSVI puts them, and SSVI is calendar-free by construction.

Each slice then goes constrained refit -> SSVI slice -> drop (FR-13), shortest expiry first,
with the previous accepted slice as the calendar floor.
"""

from collections import Counter

import numpy as np
import polars as pl
import structlog

from vsm.config import AppConfig
from vsm.conventions import F64
from vsm.forwards import ForwardEstimate
from vsm.surface.arbitrage import check_slice
from vsm.surface.ssvi import SSVI, SliceData, fit_ssvi
from vsm.surface.svi import RawSVI, SliceFit, fit_raw_svi
from vsm.surface.types import SurfaceSlice

ANCHOR_Z = np.array([3.0, 4.0, 5.0, 6.0, 8.0])
FIRST_PASS_GRID = 61
REFIT_GRID = 301
CONSTRAINT_STRETCH = 1.25

log = structlog.get_logger("vsm.surface")


def _fit_weights(rows: pl.DataFrame, config: AppConfig) -> F64:
    """Least-squares weights on total variance, so residuals read as vol errors."""
    iv = rows["target_iv"].to_numpy()
    t = rows["t"].to_numpy()
    if config.svi.weights == "inverse_spread":
        per_vol = rows["weight"].to_numpy()
    elif config.svi.weights == "vega":
        per_vol = rows["vega_fwd"].to_numpy()
    else:
        per_vol = np.ones(rows.height)
    weights: F64 = (per_vol / (2.0 * iv * t)) ** 2
    return weights


def slice_data(quotes: pl.DataFrame, config: AppConfig) -> list[tuple[pl.DataFrame, SliceData]]:
    """Kept OTM quotes of each expiry, shortest first."""
    kept = quotes.filter(pl.col("drop_reason").is_null() & pl.col("is_otm"))
    groups = []
    for _, rows in kept.sort("expiry_ns", "strike").group_by("expiry_ns", maintain_order=True):
        iv = rows["target_iv"].to_numpy()
        t = rows["t"].to_numpy()
        data = SliceData(
            t=float(t[0]),
            k=rows["k"].to_numpy(),
            w=iv * iv * t,
            weight=_fit_weights(rows, config),
        )
        groups.append((rows, data))
    return groups


def _with_wing_anchor(
    data: SliceData, anchor: RawSVI, theta: float, weight: float
) -> tuple[F64, F64, F64]:
    """Quotes plus SSVI pseudo-points outside the quoted range."""
    candidates = np.concatenate([-ANCHOR_Z, ANCHOR_Z]) * np.sqrt(theta)
    outside = candidates[(candidates < data.k.min()) | (candidates > data.k.max())]
    return (
        np.concatenate([data.k, outside]),
        np.concatenate([data.w, anchor.w(outside)]),
        np.concatenate([data.weight, np.full(outside.size, weight * float(np.mean(data.weight)))]),
    )


def _best_fit(
    data: SliceData,
    anchor: RawSVI,
    theta: float,
    lower: RawSVI | None,
    lower_quoted: F64 | None,
    grid_points: int,
    config: AppConfig,
) -> SliceFit | None:
    """Best converged fit from the SSVI start and the quasi-explicit start.

    The constraint grid covers every region the gate checks, the lower slice's included, and
    runs past its edge: the gate scales z by the fitted theta, which is not known yet here.
    """
    k, w, weight = _with_wing_anchor(data, anchor, theta, config.svi.wing_anchor_weight)
    grid_z = config.arbitrage.grid_z
    z = np.linspace(grid_z.min, grid_z.max, grid_points) * CONSTRAINT_STRETCH
    parts = [
        z * np.sqrt(theta),
        data.k,
        np.linspace(float(data.k.min()), float(data.k.max()), grid_points),
    ]
    if lower is not None and lower_quoted is not None:
        parts += [
            z * np.sqrt(float(lower.w(np.array(0.0)))),
            np.linspace(float(lower_quoted.min()), float(lower_quoted.max()), grid_points),
        ]
    grid = np.unique(np.concatenate(parts))
    fits = [
        fit_raw_svi(k, w, weight, init=start, lower_slice=lower, grid=grid)
        for start in (anchor, None)
    ]
    converged = [fit for fit in fits if fit.converged]
    return min(converged, key=lambda fit: fit.rmse_w) if converged else None


def fit_slice_chain(
    data: SliceData,
    ssvi: SSVI,
    lower: RawSVI | None,
    lower_quoted: F64 | None,
    config: AppConfig,
) -> tuple[RawSVI, str] | None:
    """Run the fallback chain; returns (slice, source), or None if the slice is dropped."""
    cfg = config.arbitrage
    theta = float(ssvi.theta(np.array(data.t)))
    anchor = ssvi.slice_at(data.t)
    quoted = np.array([float(data.k.min()), float(data.k.max())])

    def passes(candidate: RawSVI) -> bool:
        return check_slice(candidate, quoted, cfg, lower=lower, lower_quoted=lower_quoted).passed

    fit = _best_fit(data, anchor, theta, lower, lower_quoted, FIRST_PASS_GRID, config)
    if fit is not None and passes(fit.svi):
        return fit.svi, "svi"
    if "constrained_refit" in cfg.fallback:
        fit = _best_fit(data, anchor, theta, lower, lower_quoted, REFIT_GRID, config)
        if fit is not None and passes(fit.svi):
            return fit.svi, "svi_constrained"
    if "ssvi_slice" in cfg.fallback and passes(anchor):
        return anchor, "ssvi_fallback"
    return None


def _spread_stats(rows: pl.DataFrame, model_iv: F64) -> tuple[int, float]:
    liquid = rows["is_liquid"].to_numpy()
    inside = (model_iv >= rows["bid_iv"].to_numpy()) & (model_iv <= rows["ask_iv"].to_numpy())
    count = int(liquid.sum())
    return count, float(inside[liquid].mean()) if count else float("nan")


def build_slices(
    quotes: pl.DataFrame, forwards: list[ForwardEstimate], config: AppConfig
) -> tuple[SSVI, tuple[SurfaceSlice, ...], tuple[int, ...], dict[str, int]] | None:
    """Fit SSVI and every slice. None when there is nothing to fit or SSVI does not converge."""
    groups = slice_data(quotes, config)
    if not groups:
        return None
    fitted = fit_ssvi([data for _, data in groups])
    if not fitted.converged:
        log.warning("ssvi_not_converged", message=fitted.message)
        return None
    ssvi = fitted.ssvi
    by_expiry = {f.expiry_ns: f for f in forwards}

    slices: list[SurfaceSlice] = []
    dropped: list[int] = []
    sources: Counter[str] = Counter()
    lower: RawSVI | None = None
    lower_quoted: F64 | None = None
    for rows, data in groups:
        expiry_ns = int(rows["expiry_ns"][0])
        result = fit_slice_chain(data, ssvi, lower, lower_quoted, config)
        if result is None:
            dropped.append(expiry_ns)
            sources["dropped"] += 1
            log.warning("slice_dropped", expiry_ns=expiry_ns)
            continue
        svi, source = result
        sources[source] += 1
        forward = by_expiry[expiry_ns]
        model_iv = np.sqrt(svi.w(data.k) / data.t)
        ssvi_iv = np.sqrt(ssvi.w(data.k, np.full(data.k.size, data.t)) / data.t)
        n_liquid, within = _spread_stats(rows, model_iv)
        _, ssvi_within = _spread_stats(rows, ssvi_iv)
        error = model_iv - rows["target_iv"].to_numpy()
        slices.append(
            SurfaceSlice(
                expiry_ns=expiry_ns,
                t=data.t,
                forward=forward.forward,
                forward_se=forward.forward_se,
                forward_method=forward.method,
                df=forward.df,
                basis_ann=forward.basis_ann,
                svi=svi,
                source=source,
                k_min=float(data.k.min()),
                k_max=float(data.k.max()),
                n_quotes=rows.height,
                n_liquid=n_liquid,
                rmse_iv_bp=1e4 * float(np.sqrt(np.mean(error * error))),
                within_spread_frac=within,
                ssvi_within_spread_frac=ssvi_within,
            )
        )
        lower, lower_quoted = svi, np.array([float(data.k.min()), float(data.k.max())])
    return ssvi, tuple(slices), tuple(dropped), dict(sources)

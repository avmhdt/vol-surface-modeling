"""Dupire local vol from SSVI in total-variance form (FR-19).

    sigma_loc^2(k, T) = dw/dT / g(k, T)

    g = 1 - (k / w) dw/dk + 1/4 (-1/4 - 1/w + k^2 / w^2) (dw/dk)^2 + 1/2 d2w/dk2

The k-derivatives are analytic. g is Durrleman's function, so it is positive on a surface that
passes the gate. theta(t) is piecewise linear, so dw/dT is analytic too, and local vol jumps at
the expiry nodes.
"""

from dataclasses import dataclass

import numpy as np

from vsm.conventions import F64
from vsm.surface.ssvi import SSVI, phi_power

LOCAL_VOL_CAP = 10.0  # 1000%: only reached in the far wings of very short maturities


@dataclass(frozen=True)
class LocalVolGrid:
    """Local vol on a uniform k grid. ``vol`` has shape (t.size, k.size); t is in years."""

    k: F64
    t: F64
    vol: F64


def _theta_slope(ssvi: SSVI, t: F64) -> F64:
    """d theta / dt, with the right-hand slope at a node."""
    nodes = np.array(ssvi.t_nodes)
    thetas = np.array(ssvi.theta_nodes)
    inner = np.diff(thetas) / np.diff(nodes) if nodes.size > 1 else np.empty(0)
    slopes = np.concatenate([[thetas[0] / nodes[0]], inner, [thetas[-1] / nodes[-1]]])
    return slopes[np.searchsorted(nodes, t, side="right")]


def dupire_local_vol(ssvi: SSVI, k: F64, t: F64) -> F64:
    """Local vol at log-moneyness ``k`` and time ``t`` in years, with broadcasting."""
    log_moneyness, times = np.broadcast_arrays(
        np.asarray(k, dtype=np.float64), np.asarray(t, dtype=np.float64)
    )
    rho = ssvi.rho
    theta = ssvi.theta(times)
    phi = phi_power(theta, ssvi.eta, ssvi.gamma)
    dphi = phi * (-ssvi.gamma / theta + (ssvi.gamma - 1.0) / (1.0 + theta))

    x = phi * log_moneyness
    root = np.sqrt((x + rho) ** 2 + 1.0 - rho * rho)
    shape = 1.0 + rho * x + root
    skew = rho + (x + rho) / root
    w = 0.5 * theta * shape
    dw = 0.5 * theta * phi * skew
    d2w = 0.5 * theta * phi * phi * (1.0 - rho * rho) / root**3

    dw_dtheta = 0.5 * shape + 0.5 * theta * skew * log_moneyness * dphi
    dw_dt = dw_dtheta * _theta_slope(ssvi, times)
    ratio = log_moneyness / w
    g = 1.0 - ratio * dw + 0.25 * (-0.25 - 1.0 / w + ratio * ratio) * dw * dw + 0.5 * d2w
    local_vol: F64 = np.sqrt(dw_dt / g)
    return local_vol


def local_vol_grid(ssvi: SSVI, k: F64, t: F64) -> LocalVolGrid:
    """Local vol on the grid ``t`` x ``k``, capped at ``LOCAL_VOL_CAP``."""
    times = np.asarray(t, dtype=np.float64)
    moneyness = np.asarray(k, dtype=np.float64)
    vol = dupire_local_vol(ssvi, moneyness[np.newaxis, :], times[:, np.newaxis])
    return LocalVolGrid(k=moneyness, t=times, vol=np.minimum(vol, LOCAL_VOL_CAP))


def flat_grid(vol: float) -> LocalVolGrid:
    """A constant local vol, for flat Black paths through the same simulator."""
    k = np.array([-1.0, 0.0, 1.0])
    t = np.array([0.0, 1.0])
    return LocalVolGrid(k=k, t=t, vol=np.full((t.size, k.size), vol))

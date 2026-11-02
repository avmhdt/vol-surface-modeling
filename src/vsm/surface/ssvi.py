"""Global SSVI surface (Gatheral and Jacquier 2014, FR-12).

    w(k, t) = theta_t / 2 * (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))

with power-law phi(theta) = eta theta^-gamma (1 + theta)^(gamma - 1) and theta_t non-decreasing.

Theorem 4.2 gives butterfly freedom if theta phi (1 + |rho|) < 4 and theta phi^2 (1 + |rho|) <= 4.
Consumers extrapolate theta to 0 and to infinity, so these must hold for every theta > 0, not
only at the nodes. With gamma <= 1/2 the second quantity vanishes at both ends, so a grid of
theta catches its maximum. Theorem 4.1's calendar condition holds by construction: for gamma in
(0, 1), d(theta phi)/dtheta = phi (1 - gamma) / (1 + theta), which is positive and below
phi (1 + sqrt(1 - rho^2)) / rho^2.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from vsm.conventions import F64
from vsm.surface.svi import RawSVI

RHO_MAX = 0.999
GAMMA_MIN = 1e-3
GAMMA_MAX = 0.5
ETA_MIN = 1e-4
ETA_MAX = 20.0
THETA_MIN = 1e-8
THETA_GRID = np.logspace(-6.0, 2.0, 49)
CONDITION_MARGIN = 1e-6
SLSQP_MAX_ITER = 300


def phi_power(theta: F64 | float, eta: float, gamma: float) -> F64:
    """Power-law phi(theta)."""
    theta_array = np.asarray(theta, dtype=np.float64)
    return eta * theta_array**-gamma * (1.0 + theta_array) ** (gamma - 1.0)


def _ssvi_w(k: F64, theta: F64, phi: F64, rho: float) -> F64:
    x = phi * k
    return 0.5 * theta * (1.0 + rho * x + np.sqrt((x + rho) ** 2 + 1.0 - rho * rho))


@dataclass(frozen=True)
class SSVI:
    """SSVI with ATM total variance piecewise linear in t between nodes (t in years)."""

    rho: float
    eta: float
    gamma: float
    t_nodes: tuple[float, ...]
    theta_nodes: tuple[float, ...]

    def theta(self, t: F64) -> F64:
        """ATM total variance: linear between nodes, constant ATM vol outside them."""
        times = np.asarray(t, dtype=np.float64)
        nodes = np.array(self.t_nodes)
        thetas = np.array(self.theta_nodes)
        inside = np.interp(times, nodes, thetas)
        below = thetas[0] * times / nodes[0]
        above = thetas[-1] * times / nodes[-1]
        theta: F64 = np.where(times < nodes[0], below, np.where(times > nodes[-1], above, inside))
        return theta

    def phi(self, t: F64) -> F64:
        """phi(theta(t))."""
        return phi_power(self.theta(t), self.eta, self.gamma)

    def w(self, k: F64, t: F64) -> F64:
        """Total variance at log-moneyness ``k`` and time ``t``."""
        theta = self.theta(t)
        return _ssvi_w(np.asarray(k, dtype=np.float64), theta, self.phi(t), self.rho)

    def slice_at(self, t: float) -> RawSVI:
        """The raw SVI parameters of the slice at ``t``; an SSVI slice is exactly raw SVI."""
        theta = float(self.theta(np.array(t)))
        phi = float(self.phi(np.array(t)))
        root = math.sqrt(1.0 - self.rho * self.rho)
        return RawSVI(
            a=0.5 * theta * (1.0 - self.rho * self.rho),
            b=0.5 * theta * phi,
            rho=self.rho,
            m=-self.rho / phi,
            sigma=root / phi,
        )


@dataclass(frozen=True)
class SliceData:
    """Quotes of one expiry for fitting: log-moneyness, total variance, weights."""

    t: float
    k: F64
    w: F64
    weight: F64


@dataclass(frozen=True)
class SSVIFit:
    """Fitted surface. ``converged`` is False when the optimizer did not report success."""

    ssvi: SSVI
    converged: bool
    rmse_w: float
    message: str


def butterfly_margins(rho: float, eta: float, gamma: float) -> tuple[float, float]:
    """(4 - sup theta phi (1 + |rho|), 4 - sup theta phi^2 (1 + |rho|)) over the theta grid."""
    phi = phi_power(THETA_GRID, eta, gamma)
    scale = 1.0 + abs(rho)
    return (
        4.0 - float(np.max(THETA_GRID * phi)) * scale,
        4.0 - float(np.max(THETA_GRID * phi * phi)) * scale,
    )


def _atm_variance(data: SliceData) -> float:
    order = np.argsort(data.k)
    return float(np.interp(0.0, data.k[order], data.w[order]))


def fit_ssvi(slices: list[SliceData], *, prior: SSVI | None = None, lam: float = 0.0) -> SSVIFit:
    """Fit (rho, eta, gamma) and one theta per slice, jointly and under the SSVI conditions.

    With ``prior`` and ``lam > 0`` a penalty lam * sum(((p - p_prev) / s_p)^2) pulls the shape
    parameters toward the previous day, with s_p = 1 for rho, eta and gamma (FR-14).
    """
    ordered = sorted(slices, key=lambda s: s.t)
    times = np.array([s.t for s in ordered])
    theta0 = np.maximum.accumulate([max(_atm_variance(s), THETA_MIN) for s in ordered])
    increments0 = np.diff(theta0, prepend=0.0)
    increments0[1:] = np.maximum(increments0[1:], THETA_MIN)

    k = np.concatenate([s.k for s in ordered])
    target = np.concatenate([s.w for s in ordered])
    weight = np.concatenate([s.weight / np.sum(s.weight) for s in ordered]) / len(ordered)
    slice_of = np.concatenate([np.full(s.k.size, i) for i, s in enumerate(ordered)])
    scale = float(np.mean(target))

    def unpack(x: F64) -> tuple[float, float, float, F64]:
        return float(x[0]), float(x[1]), float(x[2]), np.cumsum(x[3:]) * scale

    def objective(x: F64) -> float:
        rho, eta, gamma, theta = unpack(x)
        theta_q = theta[slice_of]
        model = _ssvi_w(k, theta_q, phi_power(theta_q, eta, gamma), rho)
        residual = (model - target) / scale
        cost = float(np.sum(weight * residual * residual))
        if prior is not None and lam > 0.0:
            cost += lam * (
                (rho - prior.rho) ** 2 + (eta - prior.eta) ** 2 + (gamma - prior.gamma) ** 2
            )
        return cost

    def conditions(x: F64) -> F64:
        rho, eta, gamma, _ = unpack(x)
        first, second = butterfly_margins(rho, eta, gamma)
        return np.array([first - CONDITION_MARGIN, second - CONDITION_MARGIN])

    start_eta = prior.eta if prior is not None else 1.0
    start_rho = prior.rho if prior is not None else -0.1
    start_gamma = prior.gamma if prior is not None else 0.4
    x0 = np.concatenate([[start_rho, start_eta, start_gamma], increments0 / scale])
    bounds = [
        (-RHO_MAX, RHO_MAX),
        (ETA_MIN, ETA_MAX),
        (GAMMA_MIN, GAMMA_MAX),
        *[(THETA_MIN / scale, None)] * len(ordered),
    ]
    solved = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": conditions}],
        options={"maxiter": SLSQP_MAX_ITER, "ftol": 1e-12},
    )
    rho, eta, gamma, theta = unpack(solved.x)
    ssvi = SSVI(
        rho=rho,
        eta=eta,
        gamma=gamma,
        t_nodes=tuple(float(t) for t in times),
        theta_nodes=tuple(float(v) for v in theta),
    )
    theta_q = theta[slice_of]
    residual = _ssvi_w(k, theta_q, phi_power(theta_q, eta, gamma), rho) - target
    rmse = math.sqrt(float(np.sum(weight * residual * residual)))
    return SSVIFit(
        ssvi=ssvi, converged=bool(solved.success), rmse_w=rmse, message=str(solved.message)
    )

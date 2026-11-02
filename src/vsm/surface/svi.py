"""Raw SVI per expiry, with butterfly and calendar constraints (FR-11).

    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))

Initialization is quasi-explicit (Zeliade 2009): for fixed (m, sigma) the model is linear in
(a, d, c), so only two parameters need a search. SLSQP then fits all five under the
constraints, with g(k) >= 0 on a grid and, if a lower slice is given, w above it on a grid.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from vsm.conventions import F64

B_MAX = 2.0  # Lee's bound b (1 + |rho|) <= 2 implies b <= 2
RHO_MAX = 0.999
SIGMA_MIN = 1e-4
SIGMA_MAX = 5.0
CONSTRAINT_MARGIN = 1e-7  # solvers stop slightly outside a constraint; the gate has no slack
SLSQP_MAX_ITER = 200
SLSQP_FTOL = 1e-12


@dataclass(frozen=True)
class SVIJW:
    """SVI jump-wing parameters: ATM variance, skew, wing slopes and minimum variance."""

    v: float
    psi: float
    p: float
    c: float
    vtilde: float


@dataclass(frozen=True)
class RawSVI:
    """Raw SVI in total variance at log-moneyness k = ln(K/F)."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def _root(self, k: F64) -> F64:
        shifted = np.asarray(k, dtype=np.float64) - self.m
        return np.sqrt(shifted * shifted + self.sigma * self.sigma)

    def w(self, k: F64) -> F64:
        """Total implied variance."""
        shifted = np.asarray(k, dtype=np.float64) - self.m
        return self.a + self.b * (self.rho * shifted + self._root(k))

    def dw(self, k: F64) -> F64:
        """dw/dk."""
        shifted = np.asarray(k, dtype=np.float64) - self.m
        return self.b * (self.rho + shifted / self._root(k))

    def d2w(self, k: F64) -> F64:
        """d2w/dk2."""
        return self.b * self.sigma * self.sigma / self._root(k) ** 3

    def to_jw(self, t: float) -> SVIJW:
        """Jump-wing form at time to expiry ``t`` in years (Gatheral and Jacquier 2014)."""
        root = math.sqrt(self.m * self.m + self.sigma * self.sigma)
        w_atm = self.a + self.b * (root - self.rho * self.m)
        scale = self.b / math.sqrt(w_atm)
        return SVIJW(
            v=w_atm / t,
            psi=0.5 * scale * (self.rho - self.m / root),
            p=scale * (1.0 - self.rho),
            c=scale * (1.0 + self.rho),
            vtilde=(self.a + self.b * self.sigma * math.sqrt(1.0 - self.rho * self.rho)) / t,
        )


@dataclass(frozen=True)
class SliceFit:
    """A fitted slice. ``converged`` is False when the optimizer did not report success."""

    svi: RawSVI
    converged: bool
    rmse_w: float
    message: str


def durrleman_g(k: F64, w: F64, dw: F64, d2w: F64) -> F64:
    """Durrleman's g(k): the density in log-strike is g(k) times a positive factor."""
    first = 1.0 - k * dw / (2.0 * w)
    return first * first - 0.25 * dw * dw * (1.0 / w + 0.25) + 0.5 * d2w


def slice_g(svi: RawSVI, k: F64) -> F64:
    """Durrleman's g for one slice."""
    return durrleman_g(k, svi.w(k), svi.dw(k), svi.d2w(k))


def _quasi_explicit_inner(
    k: F64, target: F64, weight: F64, m: float, sigma: float
) -> tuple[float, float, float, float]:
    """Best (a, d, c) for fixed (m, sigma), clipped to the valid domain, and its cost."""
    y = (k - m) / sigma
    z = np.sqrt(y * y + 1.0)
    design = np.column_stack([np.ones_like(y), y, z])
    root = np.sqrt(weight)
    coef, *_ = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)
    a, d, c = float(coef[0]), float(coef[1]), float(coef[2])
    c = float(np.clip(c, 1e-10, 4.0 * sigma))
    bound = min(c, 4.0 * sigma - c)
    d = float(np.clip(d, -bound, bound))
    residual = a + d * y + c * z - target
    return a, d, c, float(np.sum(weight * residual * residual))


def quasi_explicit(k: F64, target: F64, weight: F64) -> RawSVI:
    """Quasi-explicit start: search (m, sigma), solve (a, d, c) in closed form."""
    span = float(np.max(k) - np.min(k))
    best: tuple[float, float, float] | None = None
    best_cost = math.inf
    for m0 in np.linspace(float(np.min(k)), float(np.max(k)), 7):
        for sigma0 in (0.05, 0.15, 0.4, 1.0):
            sigma_try = sigma0 * max(span, 0.05)
            cost = _quasi_explicit_inner(k, target, weight, float(m0), sigma_try)[3]
            if cost < best_cost:
                best_cost = cost
                best = (float(m0), sigma_try, cost)
    assert best is not None

    def objective(x: F64) -> float:
        return _quasi_explicit_inner(k, target, weight, float(x[0]), math.exp(float(x[1])))[3]

    start = np.array([best[0], math.log(best[1])])
    solved = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-12},
    )
    m = float(solved.x[0])
    sigma = float(np.clip(math.exp(float(solved.x[1])), SIGMA_MIN, SIGMA_MAX))
    a, d, c, _ = _quasi_explicit_inner(k, target, weight, m, sigma)
    b = min(c / sigma, B_MAX / (1.0 + min(abs(d / c), RHO_MAX)))
    rho = float(np.clip(d / c, -RHO_MAX, RHO_MAX))
    return RawSVI(a=a, b=max(b, 1e-8), rho=rho, m=m, sigma=sigma)


def default_grid(k: F64) -> F64:
    """Constraint grid: the quoted range, stretched well into the wings."""
    low, high = float(np.min(k)), float(np.max(k))
    span = max(high - low, 0.2)
    return np.linspace(low - span, high + span, 41)


def fit_raw_svi(
    k: F64,
    w_target: F64,
    weights: F64,
    *,
    init: RawSVI | None = None,
    enforce_butterfly: bool = True,
    lower_slice: RawSVI | None = None,
    grid: F64 | None = None,
) -> SliceFit:
    """Weighted least squares in total variance under the no-arbitrage constraints.

    ``weights`` multiply squared total-variance errors. ``grid`` holds the log-moneyness points
    where g and the calendar constraint apply.
    """
    k = np.asarray(k, dtype=np.float64)
    target = np.asarray(w_target, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    weight = weight / np.sum(weight)
    scale = float(np.mean(target))
    start = init if init is not None else quasi_explicit(k, target, weight)
    points = default_grid(k) if grid is None else np.asarray(grid, dtype=np.float64)

    def unpack(x: F64) -> RawSVI:
        return RawSVI(
            a=float(x[0]) * scale,
            b=float(x[1]),
            rho=float(x[2]),
            m=float(x[3]),
            sigma=float(x[4]),
        )

    def objective(x: F64) -> float:
        residual = (unpack(x).w(k) - target) / scale
        return float(np.sum(weight * residual * residual))

    def floor(x: F64) -> F64:
        svi = unpack(x)
        minimum = svi.a + svi.b * svi.sigma * math.sqrt(1.0 - svi.rho * svi.rho)
        return np.array([minimum / scale - 1e-6])

    def lee(x: F64) -> F64:
        return np.array([B_MAX - x[1] * (1.0 + x[2]), B_MAX - x[1] * (1.0 - x[2])])

    def butterfly(x: F64) -> F64:
        return slice_g(unpack(x), points) - CONSTRAINT_MARGIN

    constraints = [{"type": "ineq", "fun": floor}, {"type": "ineq", "fun": lee}]
    if enforce_butterfly:
        constraints.append({"type": "ineq", "fun": butterfly})
    if lower_slice is not None:
        lower_w = lower_slice.w(points)

        def calendar(x: F64) -> F64:
            return (unpack(x).w(points) - lower_w) / scale - CONSTRAINT_MARGIN

        constraints.append({"type": "ineq", "fun": calendar})

    x0 = np.array([start.a / scale, start.b, start.rho, start.m, start.sigma])
    span = float(np.max(np.abs(k))) + 1.0
    bounds = [
        (None, None),
        (1e-8, B_MAX),
        (-RHO_MAX, RHO_MAX),
        (-2.0 * span, 2.0 * span),
        (SIGMA_MIN, SIGMA_MAX),
    ]
    solved = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": SLSQP_MAX_ITER, "ftol": SLSQP_FTOL},
    )
    svi = unpack(solved.x)
    rmse = math.sqrt(float(np.sum(weight * (svi.w(k) - target) ** 2)))
    return SliceFit(
        svi=svi, converged=bool(solved.success), rmse_w=rmse, message=str(solved.message)
    )

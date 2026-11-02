"""Heston characteristic function of ln(S_T/F), with its analytic gradient.

The form is Albrecher et al.'s (2007) "little Heston trap", which stays on the principal branch
of the complex logarithm:

    phi(u) = exp(C + D v0)
    beta = kappa - xi rho i u,  d = sqrt(beta^2 + xi^2 (u^2 + i u)),  g = (beta - d) / (beta + d)
    C = kappa theta / xi^2 * ((beta - d) t - 2 ln((1 - g e^(-d t)) / (1 - g)))
    D = (beta - d) / xi^2 * (1 - e^(-d t)) / (1 - g e^(-d t))

Two rewrites keep precision when xi is small and kappa large. There beta - d subtracts nearly
equal numbers and C multiplies the result by a large kappa theta / xi^2, so beta - d is computed
as -xi^2 (u^2 + i u) / (beta + d), and the logarithm as log1p of a small number.

The gradient is the chain rule through (beta, d, g, e^(-d t)), as in Cui et al. (2017).
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

C128 = NDArray[np.complex128]


@dataclass(frozen=True)
class HestonParams:
    """Initial variance, mean-reversion speed, long-run variance, vol of vol, correlation."""

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def as_array(self) -> NDArray[np.float64]:
        """Parameters in gradient order."""
        return np.array([self.v0, self.kappa, self.theta, self.xi, self.rho])


def heston_cf(u: ArrayLike, t: float, p: HestonParams) -> C128:
    """E[exp(i u ln(S_T/F))]; ``u`` may be complex, as Carr-Madan needs."""
    w = np.asarray(u, dtype=np.complex128)
    xi2 = p.xi * p.xi
    moment = w * w + 1j * w
    beta = p.kappa - p.xi * p.rho * 1j * w
    d = np.sqrt(beta * beta + xi2 * moment)
    plus = beta + d
    minus = -xi2 * moment / plus
    g = minus / plus
    exp_dt = np.exp(-d * t)
    log_term = np.log1p(g * (1.0 - exp_dt) / (1.0 - g))
    c = p.kappa * p.theta / xi2 * (minus * t - 2.0 * log_term)
    big_d = minus / xi2 * (1.0 - exp_dt) / (1.0 - g * exp_dt)
    result: C128 = np.exp(c + big_d * p.v0)
    return result


def heston_cf_and_grad(u: ArrayLike, t: float, p: HestonParams) -> tuple[C128, C128]:
    """phi(u) and its gradient, shape (5, u.size), in the order of ``HestonParams``."""
    w = np.asarray(u, dtype=np.complex128)
    xi2 = p.xi * p.xi
    moment = w * w + 1j * w
    beta = p.kappa - p.xi * p.rho * 1j * w
    d = np.sqrt(beta * beta + xi2 * moment)
    plus = beta + d
    minus = -xi2 * moment / plus
    g = minus / plus
    exp_dt = np.exp(-d * t)
    denominator = 1.0 - g * exp_dt
    log_term = np.log1p(g * (1.0 - exp_dt) / (1.0 - g))
    bracket = minus * t - 2.0 * log_term
    ratio = (1.0 - exp_dt) / denominator
    level = p.kappa * p.theta / xi2
    big_c = level * bracket
    big_d = minus / xi2 * ratio
    phi = np.exp(big_c + big_d * p.v0)

    def partials(dbeta: C128, dd: C128, dnumerator: C128) -> tuple[C128, C128]:
        """d(bracket) and d(big_d) at fixed prefactors, for one parameter.

        ``dnumerator`` is the derivative of xi^2 (u^2 + i u).
        """
        dplus = dbeta + dd
        dminus = (-dnumerator - minus * dplus) / plus
        dg = (dminus - g * dplus) / plus
        dexp = -t * dd * exp_dt
        ddenominator = -dg * exp_dt - g * dexp
        dlog = ddenominator / denominator + dg / (1.0 - g)
        dbracket = dminus * t - 2.0 * dlog
        dratio = (-dexp * denominator - (1.0 - exp_dt) * ddenominator) / (denominator * denominator)
        return dbracket, (dminus * ratio + minus * dratio) / xi2

    ones = np.ones_like(w)
    zeros = np.zeros_like(w)
    dbracket_kappa, dbig_d_kappa = partials(ones, beta / d, zeros)

    dbeta_rho = -p.xi * 1j * w
    dbracket_rho, dbig_d_rho = partials(dbeta_rho, beta * dbeta_rho / d, zeros)

    dbeta_xi = -p.rho * 1j * w
    dd_xi = (beta * dbeta_xi + p.xi * moment) / d
    dbracket_xi, dbig_d_xi = partials(dbeta_xi, dd_xi, 2.0 * p.xi * moment)

    grad = np.empty((5, w.size), dtype=np.complex128)
    grad[0] = phi * big_d
    grad[1] = phi * (p.theta / xi2 * bracket + level * dbracket_kappa + dbig_d_kappa * p.v0)
    grad[2] = phi * (p.kappa / xi2 * bracket)
    grad[3] = phi * (
        -2.0 * big_c / p.xi + level * dbracket_xi + (dbig_d_xi - 2.0 * big_d / p.xi) * p.v0
    )
    grad[4] = phi * (level * dbracket_rho + dbig_d_rho * p.v0)
    return phi, grad


def cumulants(t: float, p: HestonParams) -> tuple[float, float]:
    """First two cumulants of ln(S_T/F): the mean in closed form, the variance from phi.

    The second is read from the curvature of ln phi at 0. It only sets the COS interval, so
    a small error is harmless, and this form holds for any parameters.
    """
    decay = math.exp(-p.kappa * t)
    c1 = (1.0 - decay) * (p.theta - p.v0) / (2.0 * p.kappa) - 0.5 * p.theta * t
    h = 0.05 / math.sqrt(max(p.v0, p.theta) * t)
    log_phi = np.log(heston_cf(np.array([h + 0j]), t, p))
    return c1, float(-2.0 * log_phi.real[0] / (h * h))

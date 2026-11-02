"""COS pricing of Heston (Fang and Oosterlee 2008).

The density of z = ln(S_T/F) is expanded on a fixed interval [a, b], and each strike integrates
its own payoff against that series. The usual form shifts the interval by ln(F/K), which fails
for short expiries and wide strikes, where the strike can sit outside the interval altogether.

Each OTM option is priced so its payoff stays bounded and only its own tail matters:

- A put is int (e^k - e^z)^+ dQ per unit of F, bounded by e^k.
- A call is int (1 - e^(k - z))^+ dS under the stock measure, bounded by 1. The usual call
  coefficients contain e^b, which turns rounding in the series into large errors, and a call
  from put parity inherits the put's left-tail truncation error, which swamps a small call.

The stock measure has characteristic function phi(u - i). ITM prices come from parity.
"""

import math

import numpy as np
from numpy.typing import NDArray

from vsm.conventions import F64
from vsm.models.heston_cf import C128, HestonParams, cumulants, heston_cf, heston_cf_and_grad

DEFAULT_TERMS = 400
DEFAULT_L = 20.0
N_PARAMS = 5
Bools = NDArray[np.bool_]


def _bounds(t: float, p: HestonParams, truncation: float) -> tuple[float, float, float]:
    """(a, b) under Q, and the shift of the interval under the stock measure."""
    c1, c2 = cumulants(t, p)
    half = truncation * math.sqrt(abs(c2))
    return c1 - half, c1 + half, abs(c2)


def _frequencies(n_terms: int, a: float, b: float) -> F64:
    u: F64 = np.arange(n_terms, dtype=np.float64) * math.pi / (b - a)
    return u


def _put_coefficients(k: F64, a: float, b: float, n_terms: int) -> F64:
    """int_a^k (e^k - e^z) cos(u (z - a)) dz, shape (strikes, terms)."""
    u = _frequencies(n_terms, a, b)
    high = np.clip(k, a, b)[:, None]
    exp_high = np.exp(high)
    cos_high = np.cos(u * (high - a))
    sin_high = np.sin(u * (high - a))
    chi = (cos_high * exp_high - math.exp(a) + u * sin_high * exp_high) / (1.0 + u * u)
    psi = np.empty_like(chi)
    psi[:, 0] = (high - a)[:, 0]
    psi[:, 1:] = sin_high[:, 1:] / u[1:]
    coefficients: F64 = np.exp(k)[:, None] * psi - chi
    return coefficients


def _call_coefficients(k: F64, a: float, b: float, n_terms: int) -> F64:
    """int_k^b (1 - e^(k - z)) cos(u (z - a)) dz, shape (strikes, terms)."""
    u = _frequencies(n_terms, a, b)
    low = np.clip(k, a, b)[:, None]
    exp_minus_low = np.exp(-low)
    cos_low = np.cos(u * (low - a))
    sin_low = np.sin(u * (low - a))
    chi = (
        math.exp(-b) * (u * np.sin(u * (b - a)) - np.cos(u * (b - a)))
        - exp_minus_low * (u * sin_low - cos_low)
    ) / (1.0 + u * u)
    psi = np.empty_like(chi)
    psi[:, 0] = b - low[:, 0]
    psi[:, 1:] = -sin_low[:, 1:] / u[1:]
    coefficients: F64 = psi - np.exp(k)[:, None] * chi
    return coefficients


def _series_weights(values: C128, a: float, b: float, u: F64) -> F64:
    """Density coefficients 2/(b-a) Re[phi e^(-i u a)], with the first term halved."""
    weights: F64 = 2.0 / (b - a) * np.real(values * np.exp(-1j * u * a))
    weights[..., 0] *= 0.5
    return weights


def _broadcast(forward: F64, strike: F64, t: F64, is_call: Bools) -> tuple[F64, F64, F64, Bools]:
    f, k, times, calls = np.broadcast_arrays(
        np.asarray(forward, dtype=np.float64),
        np.asarray(strike, dtype=np.float64),
        np.asarray(t, dtype=np.float64),
        np.asarray(is_call, dtype=np.bool_),
    )
    return np.ravel(f), np.ravel(k), np.ravel(times), np.ravel(calls)


def _stock_cf(u: F64, t: float, p: HestonParams) -> C128:
    """phi(u - i). phi(-i) = 1 by the martingale property, and is set rather than computed."""
    values = np.ones(u.size, dtype=np.complex128)
    values[1:] = heston_cf(u[1:] - 1j, t, p)
    return values


def _stock_cf_and_grad(u: F64, t: float, p: HestonParams) -> tuple[C128, C128]:
    values = np.ones(u.size, dtype=np.complex128)
    grad = np.zeros((N_PARAMS, u.size), dtype=np.complex128)
    values[1:], grad[:, 1:] = heston_cf_and_grad(u[1:] - 1j, t, p)
    return values, grad


def _otm_units(k: F64, t: float, p: HestonParams, n_terms: int, L: float) -> F64:
    """OTM prices per unit of F for one expiry: puts below the forward, calls at or above."""
    a, b, shift = _bounds(t, p, L)
    u = _frequencies(n_terms, a, b)
    units = np.empty(k.size)
    puts = k < 0.0
    if puts.any():
        weights = _series_weights(heston_cf(u, t, p), a, b, u)
        units[puts] = _put_coefficients(k[puts], a, b, n_terms) @ weights
    if not puts.all():
        a_s, b_s = a + shift, b + shift
        weights = _series_weights(_stock_cf(u, t, p), a_s, b_s, u)
        units[~puts] = _call_coefficients(k[~puts], a_s, b_s, n_terms) @ weights
    return units


def _otm_units_and_grad(
    k: F64, t: float, p: HestonParams, n_terms: int, L: float
) -> tuple[F64, F64]:
    """``_otm_units`` with gradients, shape (strikes, 5). The interval is held fixed."""
    a, b, shift = _bounds(t, p, L)
    u = _frequencies(n_terms, a, b)
    units = np.empty(k.size)
    grad = np.empty((k.size, N_PARAMS))
    puts = k < 0.0
    if puts.any():
        phi, dphi = heston_cf_and_grad(u, t, p)
        coefficients = _put_coefficients(k[puts], a, b, n_terms)
        units[puts] = coefficients @ _series_weights(phi, a, b, u)
        grad[puts] = coefficients @ _series_weights(dphi, a, b, u).T
    if not puts.all():
        a_s, b_s = a + shift, b + shift
        phi, dphi = _stock_cf_and_grad(u, t, p)
        coefficients = _call_coefficients(k[~puts], a_s, b_s, n_terms)
        units[~puts] = coefficients @ _series_weights(phi, a_s, b_s, u)
        grad[~puts] = coefficients @ _series_weights(dphi, a_s, b_s, u).T
    return units, grad


def _requested(otm: F64, f: F64, k: F64, calls: Bools) -> F64:
    """Parity from the OTM price to the requested side."""
    otm_is_call = k >= f
    prices: F64 = np.where(calls == otm_is_call, otm, otm + np.where(calls, f - k, k - f))
    return prices


def heston_price_cos(
    p: HestonParams,
    forward: F64,
    strike: F64,
    t: F64,
    is_call: Bools,
    n_terms: int = DEFAULT_TERMS,
    L: float = DEFAULT_L,
) -> F64:
    """Undiscounted Heston price, in the units of ``forward``; ``t`` in years."""
    shape = np.broadcast(forward, strike, t, is_call).shape
    f, k, times, calls = _broadcast(forward, strike, t, is_call)
    log_moneyness = np.log(k / f)
    otm = np.empty(f.size)
    for expiry in np.unique(times):
        mask = times == expiry
        otm[mask] = f[mask] * _otm_units(log_moneyness[mask], float(expiry), p, n_terms, L)
    return _requested(otm, f, k, calls).reshape(shape)


def heston_price_and_grad_cos(
    p: HestonParams,
    forward: F64,
    strike: F64,
    t: F64,
    is_call: Bools,
    n_terms: int = DEFAULT_TERMS,
    L: float = DEFAULT_L,
) -> tuple[F64, F64]:
    """Prices and analytic gradients, shape (n, 5), in the order of ``HestonParams``.

    Parity is parameter-free, so a call and a put share a gradient. The interval [a, b] is
    held fixed in the gradient: moving it changes prices by less than the truncation error.
    """
    f, k, times, calls = _broadcast(forward, strike, t, is_call)
    log_moneyness = np.log(k / f)
    otm = np.empty(f.size)
    grad = np.empty((f.size, N_PARAMS))
    for expiry in np.unique(times):
        mask = times == expiry
        units, unit_grad = _otm_units_and_grad(log_moneyness[mask], float(expiry), p, n_terms, L)
        otm[mask] = f[mask] * units
        grad[mask] = f[mask, None] * unit_grad
    return _requested(otm, f, k, calls), grad

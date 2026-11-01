"""Black-76 in normalized form, accurate deep in the wings.

Prices are built from the out-of-the-money time value. With a = |ln(F/K)| and total vol
s = sigma * sqrt(T), the normalized OTM price is

    b(a, s) = exp(-a/2) Phi(h + t) - exp(a/2) Phi(h - t),    h = -a/s, t = s/2.

Both terms carry the factor exp(-(h^2 + t^2)/2), so in the tail b is computed from the scaled
complementary error function, which keeps full relative precision where the plain formula
would subtract two numbers that have underflowed to noise. ``price = sqrt(F K) * b``.
"""

import math

import numpy as np
from numba import prange
from numpy.typing import NDArray

from vsm._jit import njit

F64 = NDArray[np.float64]

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)
ONE_OVER_SQRT_PI = 1.0 / math.sqrt(math.pi)
PARALLEL_MIN_SIZE = 512  # below this, waking the thread pool costs more than it saves

# Rational Chebyshev coefficients of Cody (1969), Math. Comp. 23, as in netlib CALERF.
_A = (
    3.16112374387056560e00,
    1.13864154151050156e02,
    3.77485237685302021e02,
    3.20937758913846947e03,
    1.85777706184603330e-1,
)
_B = (
    2.36012909523441209e01,
    2.44024637934444173e02,
    1.28261652607737228e03,
    2.84423683343917062e03,
)
_C = (
    5.64188496988670089e-1,
    8.88314979438837594e00,
    6.61191906371416295e01,
    2.98635138197400131e02,
    8.81952221241769090e02,
    1.71204761263407058e03,
    2.05107837782607147e03,
    1.23033935479799725e03,
    2.15311535474403846e-8,
)
_D = (
    1.57449261107098347e01,
    1.17693950891312499e02,
    5.37181101862009858e02,
    1.62138957456669019e03,
    3.29079923573345963e03,
    4.36261909014324716e03,
    3.43936767414372164e03,
    1.23033935480374942e03,
)
_P = (
    3.05326634961232344e-1,
    3.60344899949804439e-1,
    1.25781726111229246e-1,
    1.60837851487422766e-2,
    6.58749161529837803e-4,
    1.63153871373020978e-2,
)
_Q = (
    2.56852019228982242e00,
    1.87295284992346725e00,
    5.27905102951428412e-1,
    6.05183413124413191e-2,
    2.33520497626869185e-3,
)


@njit()
def erfcx(y: float) -> float:
    """Scaled complementary error function exp(y^2) erfc(y) for y >= 0."""
    if y <= 0.46875:
        ysq = y * y
        xnum = _A[4] * ysq
        xden = ysq
        for i in range(3):
            xnum = (xnum + _A[i]) * ysq
            xden = (xden + _B[i]) * ysq
        return float(math.exp(ysq) * (1.0 - y * (xnum + _A[3]) / (xden + _B[3])))
    if y <= 4.0:
        xnum = _C[8] * y
        xden = y
        for i in range(7):
            xnum = (xnum + _C[i]) * y
            xden = (xden + _D[i]) * y
        return float((xnum + _C[7]) / (xden + _D[7]))
    ysq = 1.0 / (y * y)
    xnum = _P[5] * ysq
    xden = ysq
    for i in range(4):
        xnum = (xnum + _P[i]) * ysq
        xden = (xden + _Q[i]) * ysq
    return float((ONE_OVER_SQRT_PI - ysq * (xnum + _P[4]) / (xden + _Q[4])) / y)


@njit()
def norm_cdf(z: float) -> float:
    """Standard normal CDF with full relative precision in the lower tail."""
    y = abs(z) / SQRT_2
    lower = 0.5 * math.exp(-y * y) * erfcx(y)
    return lower if z < 0.0 else 1.0 - lower


@njit()
def norm_otm_price(a: float, s: float) -> float:
    """Normalized OTM Black price b(a, s) for a = |ln(F/K)| >= 0 and total vol s > 0."""
    h = -a / s
    t = 0.5 * s
    if h + t < 0.0:
        scale = 0.5 * math.exp(-0.5 * (h * h + t * t))
        return float(scale * (erfcx(-(h + t) / SQRT_2) - erfcx(-(h - t) / SQRT_2)))
    half_a = 0.5 * a
    return float(math.exp(-half_a) * norm_cdf(h + t) - math.exp(half_a) * norm_cdf(h - t))


@njit()
def _price_one(forward: float, strike: float, total_vol: float, is_call: bool) -> float:
    intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
    if total_vol == 0.0:
        return intrinsic
    a = abs(math.log(forward / strike))
    return intrinsic + math.sqrt(forward * strike) * norm_otm_price(a, total_vol)


@njit()
def _checked_price(
    forward: float, strike: float, t: float, vol: float, is_call: bool, discount: float
) -> float:
    if not (forward > 0.0 and strike > 0.0 and t >= 0.0 and vol >= 0.0):
        return math.nan
    return discount * _price_one(forward, strike, vol * math.sqrt(t), is_call)


@njit()
def _vega_one(forward: float, strike: float, t: float, vol: float) -> float:
    total = vol * math.sqrt(t)
    if total == 0.0:
        return 0.0
    if not (forward > 0.0 and strike > 0.0 and total > 0.0):
        return math.nan
    h = math.log(forward / strike) / total
    q = 0.5 * total
    weight = math.exp(-0.5 * (h * h + q * q)) / SQRT_2PI
    return math.sqrt(forward * strike) * weight * math.sqrt(t)


@njit()
def _price_serial(
    forward: F64, strike: F64, t: F64, vol: F64, is_call: NDArray[np.bool_], discount: F64
) -> F64:
    out = np.empty(forward.size)
    for i in range(forward.size):
        out[i] = _checked_price(forward[i], strike[i], t[i], vol[i], is_call[i], discount[i])
    return out


@njit(parallel=True)
def _price_parallel(
    forward: F64, strike: F64, t: F64, vol: F64, is_call: NDArray[np.bool_], discount: F64
) -> F64:
    out = np.empty(forward.size)
    for i in prange(forward.size):
        out[i] = _checked_price(forward[i], strike[i], t[i], vol[i], is_call[i], discount[i])
    return out


@njit()
def _vega_serial(forward: F64, strike: F64, t: F64, vol: F64) -> F64:
    out = np.empty(forward.size)
    for i in range(forward.size):
        out[i] = _vega_one(forward[i], strike[i], t[i], vol[i])
    return out


@njit(parallel=True)
def _vega_parallel(forward: F64, strike: F64, t: F64, vol: F64) -> F64:
    out = np.empty(forward.size)
    for i in prange(forward.size):
        out[i] = _vega_one(forward[i], strike[i], t[i], vol[i])
    return out


def black76_price(
    forward: F64,
    strike: F64,
    t: F64,
    vol: F64,
    is_call: NDArray[np.bool_],
    discount: F64 | float = 1.0,
) -> F64:
    """Black-76 price in the units of ``forward``; ``t`` in years, ``vol`` decimal.

    Invalid inputs (non-positive forward or strike, negative vol or time) give NaN.
    """
    arrays = np.broadcast_arrays(
        np.asarray(forward, dtype=np.float64),
        np.asarray(strike, dtype=np.float64),
        np.asarray(t, dtype=np.float64),
        np.asarray(vol, dtype=np.float64),
        np.asarray(is_call, dtype=np.bool_),
        np.asarray(discount, dtype=np.float64),
    )
    shape = arrays[0].shape
    flat = [np.ravel(a).copy() for a in arrays]
    kernel = _price_parallel if flat[0].size >= PARALLEL_MIN_SIZE else _price_serial
    return kernel(*flat).reshape(shape)


def black76_vega(forward: F64, strike: F64, t: F64, vol: F64) -> F64:
    """Undiscounted dPrice/dVol, in the units of ``forward`` per 1.0 of vol."""
    arrays = np.broadcast_arrays(
        np.asarray(forward, dtype=np.float64),
        np.asarray(strike, dtype=np.float64),
        np.asarray(t, dtype=np.float64),
        np.asarray(vol, dtype=np.float64),
    )
    shape = arrays[0].shape
    flat = [np.ravel(a).copy() for a in arrays]
    kernel = _vega_parallel if flat[0].size >= PARALLEL_MIN_SIZE else _vega_serial
    return kernel(*flat).reshape(shape)

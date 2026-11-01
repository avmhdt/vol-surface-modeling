"""Implied vol: safeguarded Householder(3) on the normalized price, with a Brent fallback.

The solve runs on the OTM time value, so an ITM quote is first reduced by its intrinsic value.
A closed-form start puts the iterate close to the root, Householder(3) converges in a few
steps, and below 1e-3 of the upper bound it works on the log of the price, which is nearly
linear in the wing. Every step is held inside a bracket; if the iteration stalls, Brent's
method finishes from that bracket.
"""

import math

import numpy as np
from numba import prange
from numpy.typing import NDArray

from vsm._jit import njit
from vsm.iv.black import F64, PARALLEL_MIN_SIZE, SQRT_2PI, norm_otm_price

OK = 0
BELOW_INTRINSIC = 1
ABOVE_UPPER_BOUND = 2
NO_CONVERGENCE = 3

MAX_TOTAL_VOL = 1.0e3
BRENT_MAX_ITER = 200
LOG_OBJECTIVE_BELOW = 1.0e-3
STEP_TOL = 4.0 * np.finfo(np.float64).eps
TAIL_BELOW = 1.0e-5
FINAL_STEP = 1.0e-6  # Householder(3) has order 4, so the step after this lands at machine eps
ROUNDING_ULPS = 4.0 * np.finfo(np.float64).eps


@njit()
def _initial_guess(beta: float, a: float) -> float:
    """Closed-form start for total vol, split at the inflection point s_c = sqrt(2a).

    Above it the Polya approximation of the normal CDF is close. Below it log(b) is close to
    quadratic in 1/s, so a parabola matched at s_c works, and far out the exponent of the
    tail asymptotic b ~ exp(-(h^2 + t^2)/2) * 2t / (sqrt(2 pi) (h^2 - t^2)) is sharper.
    """
    upper = math.exp(-0.5 * a)
    ratio = beta / upper
    s_c = math.sqrt(2.0 * a)
    b_c = norm_otm_price(a, s_c) if a > 0.0 else 0.0
    if beta >= b_c:
        polya = 2.0 * math.sqrt(-0.5 * math.pi * math.log1p(-min(ratio * ratio, 1.0 - 1e-16)))
        return math.sqrt(polya * polya + 2.0 * a * (1.0 - ratio))
    log_beta = math.log(beta)
    if ratio > TAIL_BELOW:
        slope = math.exp(-0.5 * a) / SQRT_2PI
        curvature = 0.5 * s_c**3 * slope / b_c
        return 1.0 / math.sqrt(1.0 / (s_c * s_c) + (math.log(b_c) - log_beta) / curvature)
    s = s_c
    for _ in range(3):
        h2 = (a / s) ** 2
        t2 = 0.25 * s * s
        prefactor = s / (SQRT_2PI * max(abs(h2 - t2), 1e-12))
        exponent = max(math.log(prefactor) - log_beta, 0.5 * a + 1e-9)
        s = a / math.sqrt(exponent + math.sqrt(exponent * exponent - 0.25 * a * a))
    return s


@njit()
def _brent(beta: float, a: float, lo: float, hi: float) -> tuple[float, int]:
    """Brent's root finder on log(b) - log(beta) inside a bracket."""
    log_beta = math.log(beta)
    fa = math.log(max(norm_otm_price(a, lo), 1e-300)) - log_beta
    fb = math.log(max(norm_otm_price(a, hi), 1e-300)) - log_beta
    x0, x1 = lo, hi
    if fa == 0.0:
        return x0, OK
    if fb == 0.0:
        return x1, OK
    if fa * fb > 0.0:
        return math.nan, NO_CONVERGENCE
    c, fc = x0, fa
    d = e = x1 - x0
    for _ in range(BRENT_MAX_ITER):
        if abs(fc) < abs(fb):
            x0, x1, c = x1, c, x1
            fa, fb, fc = fb, fc, fb
        tol = STEP_TOL * abs(x1)
        mid = 0.5 * (c - x1)
        if abs(mid) <= tol or fb == 0.0:
            return x1, OK
        if abs(e) >= tol and abs(fa) > abs(fb):
            ratio = fb / fa
            if x0 == c:
                p = 2.0 * mid * ratio
                q = 1.0 - ratio
            else:
                g = fa / fc
                r = fb / fc
                p = ratio * (2.0 * mid * g * (g - r) - (x1 - x0) * (r - 1.0))
                q = (g - 1.0) * (r - 1.0) * (ratio - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * mid * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = mid
                e = d
        else:
            d = mid
            e = d
        x0, fa = x1, fb
        x1 += d if abs(d) > tol else math.copysign(tol, mid)
        fb = math.log(max(norm_otm_price(a, x1), 1e-300)) - log_beta
        if fb * fc > 0.0:
            c, fc = x0, fa
            d = e = x1 - x0
    return math.nan, NO_CONVERGENCE


@njit()
def solve_normalized(beta: float, a: float, max_iter: int) -> tuple[float, int]:
    """Total vol s with b(a, s) = beta, for 0 < beta < exp(-a/2). Returns (s, iterations)."""
    upper = math.exp(-0.5 * a)
    use_log = beta < LOG_OBJECTIVE_BELOW * upper
    log_beta = math.log(beta)
    lo = 0.0
    hi = MAX_TOTAL_VOL
    s = _initial_guess(beta, a)
    if not (0.0 < s < hi):
        s = 1.0 if math.isnan(s) else max(min(s, 0.5 * hi), 1e-8)

    for iteration in range(max_iter):
        b = norm_otm_price(a, s)
        if b < beta:
            lo = s
        else:
            hi = s
        a2_s3 = a * a / (s * s * s)
        gamma = a2_s3 - 0.25 * s
        delta = gamma * gamma - 3.0 * a2_s3 / s - 0.25
        exponent = -0.5 * ((a / s) ** 2 + 0.25 * s * s)
        slope = math.exp(exponent) / SQRT_2PI
        if b <= 0.0 or slope <= 0.0:
            step = math.nan
        else:
            if use_log:
                r = slope / b
                nu = (log_beta - math.log(b)) / r
                gamma_f = gamma - r
                delta_f = delta - 3.0 * r * gamma + 2.0 * r * r
            else:
                nu = (beta - b) / slope
                gamma_f = gamma
                delta_f = delta
            denominator = 1.0 + nu * (gamma_f + delta_f * nu / 6.0)
            step = nu * (1.0 + 0.5 * gamma_f * nu) / denominator
        if abs(step) <= FINAL_STEP * s:
            return s + step, iteration + 1
        next_s = s + step
        if lo < next_s < hi:
            s = next_s
        elif lo <= 0.0:
            s = 0.5 * hi
        elif hi >= MAX_TOTAL_VOL:
            s = 2.0 * lo
        else:
            s = math.sqrt(lo * hi)

    if lo <= 0.0:
        lo = min(s, 1.0)
        while norm_otm_price(a, lo) > beta and lo > 1e-300:
            lo *= 0.5
    solved, status = _brent(beta, a, lo, hi)
    return solved, (max_iter + 1 if status == OK else -1)


@njit()
def _iv_one(
    price: float,
    forward: float,
    strike: float,
    t: float,
    is_call: bool,
    discount: float,
    max_iter: int,
) -> tuple[float, int, int]:
    """(vol, status, iterations) for one quote."""
    p = price / discount
    if math.isnan(p) or not (forward > 0.0 and strike > 0.0 and t > 0.0):
        return math.nan, NO_CONVERGENCE, 0
    intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
    upper = forward if is_call else strike
    if p >= upper:
        return math.nan, ABOVE_UPPER_BOUND, 0
    time_value = p - intrinsic
    if time_value < -ROUNDING_ULPS * max(intrinsic, 1e-300):
        return math.nan, BELOW_INTRINSIC, 0
    if time_value <= 0.0:
        return 0.0, OK, 0
    a = abs(math.log(forward / strike))
    beta = time_value / math.sqrt(forward * strike)
    if beta >= math.exp(-0.5 * a):
        return math.nan, ABOVE_UPPER_BOUND, 0
    s, used = solve_normalized(beta, a, max_iter)
    if used < 0 or math.isnan(s):
        return math.nan, NO_CONVERGENCE, max(used, 0)
    return s / math.sqrt(t), OK, used


@njit()
def _iv_serial(
    price: F64,
    forward: F64,
    strike: F64,
    t: F64,
    is_call: NDArray[np.bool_],
    discount: F64,
    max_iter: int,
) -> tuple[F64, NDArray[np.int8], NDArray[np.int32]]:
    n = price.size
    vol = np.empty(n)
    status = np.zeros(n, dtype=np.int8)
    iterations = np.zeros(n, dtype=np.int32)
    for i in range(n):
        vol[i], status[i], iterations[i] = _iv_one(
            price[i], forward[i], strike[i], t[i], is_call[i], discount[i], max_iter
        )
    return vol, status, iterations


@njit(parallel=True)
def _iv_parallel(
    price: F64,
    forward: F64,
    strike: F64,
    t: F64,
    is_call: NDArray[np.bool_],
    discount: F64,
    max_iter: int,
) -> tuple[F64, NDArray[np.int8], NDArray[np.int32]]:
    n = price.size
    vol = np.empty(n)
    status = np.zeros(n, dtype=np.int8)
    iterations = np.zeros(n, dtype=np.int32)
    for i in prange(n):
        vol[i], status[i], iterations[i] = _iv_one(
            price[i], forward[i], strike[i], t[i], is_call[i], discount[i], max_iter
        )
    return vol, status, iterations


def implied_vol(
    price: F64,
    forward: F64,
    strike: F64,
    t: F64,
    is_call: NDArray[np.bool_],
    discount: F64 | float = 1.0,
    *,
    max_iter: int = 50,
) -> tuple[F64, NDArray[np.int8]]:
    """Black-76 implied vol (decimal) and a status per quote.

    Status 0 = ok, 1 = below intrinsic, 2 = above upper bound, 3 = no convergence or invalid
    input. Every status other than 0 returns NaN. A price exactly at intrinsic is the
    zero-vol price, so it returns 0.0 with status 0.
    """
    vol, status, _ = implied_vol_detail(price, forward, strike, t, is_call, discount, max_iter)
    return vol, status


def implied_vol_detail(
    price: F64,
    forward: F64,
    strike: F64,
    t: F64,
    is_call: NDArray[np.bool_],
    discount: F64 | float = 1.0,
    max_iter: int = 50,
) -> tuple[F64, NDArray[np.int8], NDArray[np.int32]]:
    """``implied_vol`` plus iterations per quote; ``max_iter + 1`` marks a Brent finish."""
    arrays = np.broadcast_arrays(
        np.asarray(price, dtype=np.float64),
        np.asarray(forward, dtype=np.float64),
        np.asarray(strike, dtype=np.float64),
        np.asarray(t, dtype=np.float64),
        np.asarray(is_call, dtype=np.bool_),
        np.asarray(discount, dtype=np.float64),
    )
    shape = arrays[0].shape
    flat = [np.ravel(a).copy() for a in arrays]
    kernel = _iv_parallel if flat[0].size >= PARALLEL_MIN_SIZE else _iv_serial
    vol, status, iterations = kernel(flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], max_iter)
    return vol.reshape(shape), status.reshape(shape), iterations.reshape(shape)

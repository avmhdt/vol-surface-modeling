"""Monte Carlo of the T-forward under local vol and Heston (FR-20).

Paths are in x = ln(F_t / F_0), a martingale in levels, since every price here is undiscounted
in the forward measure. Local vol uses log-Euler and Heston uses Andersen's (2008) QE scheme with
his martingale correction. Antithetic pairs share normals with opposite signs.

A barrier is handled by the Brownian-bridge correction: each step multiplies the path's weight by
the probability that the bridge between its ends did not touch the barrier, so a discrete grid
does not miss crossings between steps.

Normals come per block from ``SeedSequence.spawn`` and the kernels are serial, so results do not
depend on the machine's thread count (NFR-02).
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vsm._jit import njit
from vsm.conventions import F64
from vsm.iv.black import norm_cdf
from vsm.localvol.barrier import BarrierSpec
from vsm.localvol.dupire import LocalVolGrid
from vsm.models.heston_cf import HestonParams

BLOCK_PAIRS = 2048
DAYS_PER_YEAR = 365.0
PSI_CRITICAL = 1.5
I64 = NDArray[np.int64]


@dataclass(frozen=True)
class Simulation:
    """``ends`` has shape (horizons, paths); ``weights`` holds each path's survival weight."""

    ends: F64
    weights: F64


@njit()
def _stay(x0: float, x1: float, down: float, up: float, variance: float) -> float:
    """Probability that a bridge from x0 to x1 stays strictly between the barriers."""
    if x1 <= down or x1 >= up:
        return 0.0
    if variance <= 0.0:
        return 1.0
    below = 1.0 - math.exp(-2.0 * (x0 - down) * (x1 - down) / variance)
    above = 1.0 - math.exp(-2.0 * (up - x0) * (up - x1) / variance)
    return below * above


@njit()
def _vol_at(table: F64, step: int, x: float, k0: float, dk: float, last: int) -> float:
    position = (x - k0) / dk
    if position <= 0.0:
        return float(table[step, 0])
    if position >= last:
        return float(table[step, last])
    left = int(position)
    weight = position - left
    return float((1.0 - weight) * table[step, left] + weight * table[step, left + 1])


@njit()
def _local_vol_block(
    normals: F64,
    table: F64,
    k0: float,
    dk: float,
    dt: float,
    down: float,
    up: float,
    marks: I64,
    ends: F64,
    weights: F64,
) -> None:
    """Fill ``ends`` and ``weights`` for 2 * normals.shape[0] paths."""
    pairs, steps = normals.shape
    last = table.shape[1] - 1
    root = math.sqrt(dt)
    for path in range(2 * pairs):
        row = path // 2
        sign = 1.0 if path % 2 == 0 else -1.0
        x = 0.0
        weight = 1.0
        mark = 0
        for step in range(steps):
            vol = _vol_at(table, step, x, k0, dk, last)
            after = x - 0.5 * vol * vol * dt + vol * root * sign * normals[row, step]
            weight *= _stay(x, after, down, up, vol * vol * dt)
            x = after
            if mark < marks.size and step + 1 == marks[mark]:
                ends[mark, path] = x
                mark += 1
        weights[path] = weight


@njit()
def _qe_step(v: float, z: float, ex: float, c1: float, c2: float, theta: float) -> float:
    """One QE step of the variance, with the same normal driving both branches."""
    mean = theta + (v - theta) * ex
    psi = (v * c1 + c2) / (mean * mean)
    if psi <= PSI_CRITICAL:
        inverse = 2.0 / psi
        b2 = inverse - 1.0 + math.sqrt(inverse) * math.sqrt(inverse - 1.0)
        return float(mean / (1.0 + b2) * (math.sqrt(b2) + z) ** 2)
    p = (psi - 1.0) / (psi + 1.0)
    u = float(norm_cdf(z))
    if u <= p:
        return 0.0
    return float(math.log((1.0 - p) / (1.0 - u)) * mean * (1.0 + psi) / 2.0)


@njit()
def _log_mgf(v: float, a: float, ex: float, c1: float, c2: float, theta: float) -> float:
    """ln E[exp(a V') | V] under the QE scheme, or nan where Andersen's bound fails."""
    mean = theta + (v - theta) * ex
    psi = (v * c1 + c2) / (mean * mean)
    if psi <= PSI_CRITICAL:
        inverse = 2.0 / psi
        b2 = inverse - 1.0 + math.sqrt(inverse) * math.sqrt(inverse - 1.0)
        scale = mean / (1.0 + b2)
        room = 1.0 - 2.0 * a * scale
        if room <= 0.0:
            return math.nan
        return float(a * b2 * scale / room - 0.5 * math.log(room))
    p = (psi - 1.0) / (psi + 1.0)
    beta = (1.0 - p) / mean
    if a >= beta:
        return math.nan
    return float(math.log(p + beta * (1.0 - p) / (beta - a)))


@njit()
def _heston_block(
    normals_v: F64,
    normals_x: F64,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    dt: float,
    down: float,
    up: float,
    marks: I64,
    ends: F64,
    weights: F64,
) -> None:
    """Fill ``ends`` and ``weights``, with Andersen's central weights gamma1 = gamma2 = 1/2."""
    pairs, steps = normals_v.shape
    ex = math.exp(-kappa * dt)
    c1 = xi * xi * ex * (1.0 - ex) / kappa
    c2 = theta * xi * xi * (1.0 - ex) ** 2 / (2.0 * kappa)
    drift = kappa * rho / xi - 0.5
    k0 = -rho * kappa * theta * dt / xi
    k1 = 0.5 * dt * drift - rho / xi
    k2 = 0.5 * dt * drift + rho / xi
    k3 = 0.5 * dt * (1.0 - rho * rho)
    tilt = k2 + 0.5 * k3
    for path in range(2 * pairs):
        row = path // 2
        sign = 1.0 if path % 2 == 0 else -1.0
        x = 0.0
        v = v0
        weight = 1.0
        mark = 0
        for step in range(steps):
            after_v = _qe_step(v, sign * normals_v[row, step], ex, c1, c2, theta)
            log_m = _log_mgf(v, tilt, ex, c1, c2, theta)
            shift = k0 if math.isnan(log_m) else -log_m - (k1 + 0.5 * k3) * v
            scale = math.sqrt(k3 * (v + after_v))
            after = x + shift + k1 * v + k2 * after_v + scale * sign * normals_x[row, step]
            weight *= _stay(x, after, down, up, 0.5 * (v + after_v) * dt)
            x = after
            v = after_v
            if mark < marks.size and step + 1 == marks[mark]:
                ends[mark, path] = x
                mark += 1
        weights[path] = weight


def _step_table(grid: LocalVolGrid, steps: int, dt: float) -> F64:
    """Local vol at each step's midpoint, one row per step, flat outside the grid's times."""
    middle = (np.arange(steps) + 0.5) * dt
    if grid.t.size == steps and np.allclose(grid.t, middle):
        return grid.vol
    return np.column_stack(
        [np.interp(middle, grid.t, grid.vol[:, column]) for column in range(grid.k.size)]
    )


def _block_sizes(pairs: int) -> list[int]:
    whole, part = divmod(pairs, BLOCK_PAIRS)
    return [BLOCK_PAIRS] * whole + ([part] if part else [])


def simulate(
    model: LocalVolGrid | HestonParams,
    horizons_days: list[float],
    paths: int,
    steps_per_day: int,
    seed: int,
    *,
    down: float = -math.inf,
    up: float = math.inf,
) -> Simulation:
    """Simulate ln(F_t / F_0) to each horizon. ``paths`` rounds down to an even number.

    ``down`` and ``up`` are barriers in the same log units.
    """
    marks = np.array([round(days * steps_per_day) for days in horizons_days], dtype=np.int64)
    if np.any(np.diff(marks) <= 0):
        raise ValueError("horizons must be increasing")
    steps = int(marks[-1])
    dt = 1.0 / (DAYS_PER_YEAR * steps_per_day)
    pairs = paths // 2
    ends = np.empty((marks.size, 2 * pairs))
    weights = np.empty(2 * pairs)
    sizes = _block_sizes(pairs)
    seeds = np.random.SeedSequence(seed).spawn(len(sizes))
    table = _step_table(model, steps, dt) if isinstance(model, LocalVolGrid) else None
    start = 0
    for size, block_seed in zip(sizes, seeds, strict=True):
        rng = np.random.Generator(np.random.PCG64(block_seed))
        stop = start + 2 * size
        block_ends = ends[:, start:stop]
        block_weights = weights[start:stop]
        if isinstance(model, LocalVolGrid):
            assert table is not None
            k0 = float(model.k[0])
            dk = float(model.k[1] - model.k[0])
            normals = rng.standard_normal((size, steps))
            _local_vol_block(normals, table, k0, dk, dt, down, up, marks, block_ends, block_weights)
        else:
            normals_v = rng.standard_normal((size, steps))
            normals_x = rng.standard_normal((size, steps))
            _heston_block(
                normals_v,
                normals_x,
                model.v0,
                model.kappa,
                model.theta,
                model.xi,
                model.rho,
                dt,
                down,
                up,
                marks,
                block_ends,
                block_weights,
            )
        start = stop
    return Simulation(ends=ends, weights=weights)


def pair_mean(values: F64) -> tuple[float, float]:
    """(mean, standard error) of a sample of antithetic pairs."""
    pairs = values.reshape(-1, 2).mean(axis=1)
    return float(pairs.mean()), float(pairs.std(ddof=1) / math.sqrt(pairs.size))


def price_barrier_mc(
    model: LocalVolGrid | HestonParams,
    product: BarrierSpec,
    forward: float,
    paths: int,
    steps_per_day: int,
    seed: int,
) -> tuple[float, float]:
    """(price, standard error) of a knock-out call, undiscounted, in the forward's units."""
    level = math.log(product.barrier_pct)
    down = level if product.is_down else -math.inf
    up = math.inf if product.is_down else level
    result = simulate(model, [product.tenor_days], paths, steps_per_day, seed, down=down, up=up)
    prices = forward * np.exp(result.ends[0])
    payoff = np.maximum(prices - forward * product.strike_pct, 0.0) * result.weights
    return pair_mean(payoff)

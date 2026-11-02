"""Read ``vsm.surface.v1`` and evaluate it: the reference for consumers.

The consumer rules live in ``docs/export-format.md``. This module is their implementation.
"""

import json
import math
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

from vsm.conventions import F64
from vsm.settings import REPO_ROOT
from vsm.surface.svi import RawSVI

SCHEMA_PATH = REPO_ROOT / "schemas" / "vsm.surface.v1.schema.json"


def load_schema() -> dict[str, Any]:
    """The committed JSON Schema."""
    schema: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema


def validate_document(document: dict[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` or ``ValueError`` if the document is invalid."""
    jsonschema.Draft202012Validator(load_schema()).validate(document)
    ssvi = document["ssvi"]
    if ssvi is not None:
        t_nodes = np.array(ssvi["t_nodes"])
        theta_nodes = np.array(ssvi["theta_nodes"])
        if t_nodes.size != theta_nodes.size or np.any(np.diff(t_nodes) <= 0.0):
            raise ValueError("ssvi.t_nodes must be strictly increasing and match theta_nodes")
        if np.any(np.diff(theta_nodes) < 0.0):
            raise ValueError("ssvi.theta_nodes must be non-decreasing")
        if ssvi["phi"] == "heston_like" and ssvi.get("lambda") is None:
            raise ValueError("ssvi.lambda is required when phi is heston_like")
    elif document["canonical"] != "svi":
        raise ValueError("canonical must be svi when ssvi is null")
    if not document["arbitrage"]["passed"]:
        raise ValueError("only surfaces that passed the arbitrage gate may be exported")


class SurfaceBundle:
    """A validated surface file with the consumer rules applied."""

    def __init__(self, document: dict[str, Any]) -> None:
        validate_document(document)
        self.document = document
        self._ssvi: dict[str, Any] | None = document["ssvi"]
        self._expiries = sorted(document["expiries"], key=lambda e: e["t"])
        self._times = np.array([e["t"] for e in self._expiries])
        self._log_forwards = np.log([e["forward"] for e in self._expiries])
        self._slices = [RawSVI(**e["svi_raw"]) for e in self._expiries]
        self._spot = float(document["spot"]["value"])

    def _theta(self, t: F64) -> F64:
        ssvi = self._ssvi
        assert ssvi is not None
        nodes = np.array(ssvi["t_nodes"])
        thetas = np.array(ssvi["theta_nodes"])
        inside = np.interp(t, nodes, thetas)
        below = thetas[0] * t / nodes[0]
        above = thetas[-1] * t / nodes[-1]
        theta: F64 = np.where(t < nodes[0], below, np.where(t > nodes[-1], above, inside))
        return theta

    def _phi(self, theta: F64) -> F64:
        ssvi = self._ssvi
        assert ssvi is not None
        if ssvi["phi"] == "heston_like":
            scaled = float(ssvi["lambda"]) * theta
            heston_like: F64 = (1.0 - (1.0 - np.exp(-scaled)) / scaled) / scaled
            return heston_like
        gamma = float(ssvi["gamma"])
        power_law: F64 = float(ssvi["eta"]) * theta**-gamma * (1.0 + theta) ** (gamma - 1.0)
        return power_law

    def _ssvi_variance(self, k: F64, t: F64) -> F64:
        ssvi = self._ssvi
        assert ssvi is not None
        rho = float(ssvi["rho"])
        theta = self._theta(t)
        x = self._phi(theta) * k
        variance: F64 = 0.5 * theta * (1.0 + rho * x + np.sqrt((x + rho) ** 2 + 1.0 - rho * rho))
        return variance

    def _slice_variance(self, k: F64, t: F64) -> F64:
        """Linear in w at fixed k between expiries, constant ATM vol outside them."""
        columns = np.stack([svi.w(k) for svi in self._slices])
        times = self._times
        first: F64 = columns[0] * t / times[0]
        if times.size == 1:
            return first
        position = np.asarray(
            np.clip(np.searchsorted(times, t, side="right") - 1, 0, times.size - 2)
        )
        left = times[position]
        weight = (t - left) / (times[position + 1] - left)
        low = np.take_along_axis(columns, position[None, ...], axis=0)[0]
        high = np.take_along_axis(columns, position[None, ...] + 1, axis=0)[0]
        inside = low + weight * (high - low)
        last = columns[-1] * t / times[-1]
        variance: F64 = np.where(t < times[0], first, np.where(t > times[-1], last, inside))
        return variance

    def total_variance(self, k: F64, t: F64, source: str = "ssvi") -> F64:
        """w(k, t). ``ssvi`` is the arbitrage-free continuum; ``svi`` interpolates the slices."""
        moneyness, times = np.broadcast_arrays(
            np.asarray(k, dtype=np.float64), np.asarray(t, dtype=np.float64)
        )
        if source == "ssvi" and self._ssvi is not None:
            return self._ssvi_variance(moneyness, times)
        if source in ("ssvi", "svi"):
            return self._slice_variance(moneyness, times)
        raise ValueError(f"unknown source: {source}")

    def forward(self, t: F64) -> F64:
        """F(t): ln F linear between expiries, flat annualized basis beyond the ends."""
        times = np.asarray(t, dtype=np.float64)
        inside = np.interp(times, self._times, self._log_forwards)
        log_spot = math.log(self._spot)
        below = log_spot + (self._log_forwards[0] - log_spot) * times / self._times[0]
        above = log_spot + (self._log_forwards[-1] - log_spot) * times / self._times[-1]
        return np.exp(
            np.where(
                times < self._times[0], below, np.where(times > self._times[-1], above, inside)
            )
        )

    def implied_vol(self, strike: F64, t: F64, source: str = "ssvi") -> F64:
        """Implied vol at a strike in USD and time ``t`` in years."""
        times = np.asarray(t, dtype=np.float64)
        k = np.log(np.asarray(strike, dtype=np.float64) / self.forward(times))
        return np.sqrt(self.total_variance(k, times, source) / times)


def read_surface_bundle(path: str) -> SurfaceBundle:
    """Read and validate a ``surface.json``."""
    return SurfaceBundle(json.loads(Path(path).read_text(encoding="utf-8")))

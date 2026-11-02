"""The synthetic surface must not drift. Changing the frozen file takes its own commit.

The comparison is on total variance rather than on parameters, because raw SVI parameters can
trade off against each other across platforms while the slice itself stays put.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from fixtures.make_fixtures import AS_OF_NS, SPOT, build_chain, build_futures, tick_table
from vsm.clean import clean_snapshot
from vsm.config import load_config
from vsm.surface.fit import build_slices
from vsm.surface.ssvi import SSVI
from vsm.surface.svi import RawSVI

FROZEN = Path(__file__).parents[1] / "fixtures" / "golden" / "surface_params.json"
K = np.linspace(-0.6, 0.6, 25)
SVI_TOL = 2e-3  # relative, on total variance inside the quoted range
SSVI_TOL = 1e-3


@pytest.fixture(scope="module")
def built() -> tuple[SSVI, tuple[object, ...]]:
    config = load_config()
    chain = build_chain()
    result = clean_snapshot(chain, build_futures(), SPOT, AS_OF_NS, tick_table(chain), config)
    surface = build_slices(result.quotes, result.forwards, config)
    assert surface is not None
    return surface[0], surface[1]


def test_ssvi_matches_the_frozen_surface(built: tuple[SSVI, tuple[object, ...]]) -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["ssvi"]
    ssvi = built[0]
    assert ssvi.rho == pytest.approx(frozen["rho"], abs=SSVI_TOL)
    assert ssvi.eta == pytest.approx(frozen["eta"], rel=SSVI_TOL)
    assert ssvi.gamma == pytest.approx(frozen["gamma"], abs=SSVI_TOL)
    assert list(ssvi.theta_nodes) == pytest.approx(frozen["theta_nodes"], rel=SSVI_TOL)


def test_slices_match_the_frozen_surface(built: tuple[SSVI, tuple[object, ...]]) -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["slices"]
    slices = built[1]
    assert len(slices) == len(frozen)
    for item, expected in zip(slices, frozen, strict=True):
        assert item.source == expected["source"]  # type: ignore[attr-defined]
        reference = RawSVI(
            expected["a"], expected["b"], expected["rho"], expected["m"], expected["sigma"]
        )
        root = np.sqrt(float(reference.w(np.array(0.0))))
        k = K * root * 2.0
        assert item.svi.w(k) == pytest.approx(reference.w(k), rel=SVI_TOL)  # type: ignore[attr-defined]

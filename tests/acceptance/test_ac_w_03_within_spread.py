"""AC-W-03: fit error inside the bid-ask spread for over 90% of liquid quotes."""

import json
from pathlib import Path

import pytest

from fixtures.make_fixtures import AS_OF_NS, SPOT, build_chain, build_futures, tick_table
from vsm.clean import clean_snapshot
from vsm.config import load_config
from vsm.surface.fit import build_slices


def test_synthetic_fit_is_inside_the_spread() -> None:
    config = load_config()
    chain = build_chain()
    result = clean_snapshot(chain, build_futures(), SPOT, AS_OF_NS, tick_table(chain), config)
    built = build_slices(result.quotes, result.forwards, config)
    assert built is not None
    _, slices, _, _ = built
    liquid = sum(item.n_liquid for item in slices)
    inside = sum(item.n_liquid * item.within_spread_frac for item in slices) / liquid
    assert inside > 0.90


@pytest.mark.data
@pytest.mark.parametrize("underlying", ["BTC", "ETH"])
def test_real_fit_is_inside_the_spread(underlying: str) -> None:
    metrics = json.loads(
        Path(f"results/metrics/fit_within_spread_{underlying}.json").read_text(encoding="utf-8")
    )
    assert metrics["n_liquid"] > 0
    assert metrics["pooled_svi"] > 0.90
    assert metrics["median_day_svi"] >= 0.90

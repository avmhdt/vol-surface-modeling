"""Regenerate ``tests/fixtures/golden/surface_params.json``.

Run ``uv run python tests/fixtures/make_golden.py``. A change to this file needs its own commit,
because it means the fitted surface changed on purpose.
"""

import json
from pathlib import Path

from make_fixtures import AS_OF_NS, SPOT, build_chain, build_futures, tick_table

from vsm.clean import clean_snapshot
from vsm.config import load_config
from vsm.surface.fit import build_slices

OUT = Path(__file__).parent / "golden" / "surface_params.json"


def main() -> None:
    config = load_config()
    chain = build_chain()
    result = clean_snapshot(chain, build_futures(), SPOT, AS_OF_NS, tick_table(chain), config)
    built = build_slices(result.quotes, result.forwards, config)
    assert built is not None
    ssvi, slices, _, _ = built
    frozen = {
        "ssvi": {
            "rho": ssvi.rho,
            "eta": ssvi.eta,
            "gamma": ssvi.gamma,
            "t_nodes": list(ssvi.t_nodes),
            "theta_nodes": list(ssvi.theta_nodes),
        },
        "slices": [
            {
                "t": item.t,
                "source": item.source,
                "a": item.svi.a,
                "b": item.svi.b,
                "rho": item.svi.rho,
                "m": item.svi.m,
                "sigma": item.svi.sigma,
            }
            for item in slices
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}, {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()

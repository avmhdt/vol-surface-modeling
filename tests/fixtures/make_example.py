"""Regenerate ``docs/examples/surface_example.json`` from the synthetic chain.

Run ``uv run python tests/fixtures/make_example.py``.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from make_fixtures import AS_OF_NS, SPOT, build_chain, build_futures, tick_table

from vsm.clean import clean_snapshot
from vsm.config import load_config
from vsm.export.bundle import surface_document
from vsm.surface.arbitrage import check_surface
from vsm.surface.fit import build_slices
from vsm.surface.store import with_report
from vsm.surface.types import Surface

OUT = Path(__file__).parents[2] / "docs" / "examples" / "surface_example.json"


def main() -> None:
    config = load_config()
    chain = build_chain()
    result = clean_snapshot(chain, build_futures(), SPOT, AS_OF_NS, tick_table(chain), config)
    built = build_slices(result.quotes, result.forwards, config)
    assert built is not None
    ssvi, slices, dropped, sources = built
    raw = Surface(
        "BTC",
        "deribit",
        AS_OF_NS,
        "deribit-BTC-20261102T0815Z",
        SPOT,
        ssvi,
        slices,
        dropped,
        sources,
    )
    document = surface_document(
        with_report(raw, check_surface(raw, config.arbitrage)),
        config.arbitrage,
        git_sha="synthetic",
        config_hash="synthetic",
        created_at=datetime(2026, 11, 2, 8, 21, tzinfo=UTC),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}, {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()

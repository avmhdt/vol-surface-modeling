"""Nightly: endpoint names and fields still match the schemas."""

import pytest

from vsm.io.deribit import DeribitClient
from vsm.io.smoke import run_smoke
from vsm.settings import Settings


@pytest.mark.live
def test_public_endpoints_match_schemas() -> None:
    with DeribitClient(Settings().deribit_base_url) as client:
        results = run_smoke(client, ["BTC", "ETH"])
    assert [r for r in results if not r.ok] == []

"""AC-W-09: the export is valid and round-trips."""

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest

from fixtures.make_fixtures import AS_OF_NS, SPOT, build_chain, build_futures, tick_table
from vsm.clean import clean_snapshot
from vsm.config import load_config
from vsm.export.bundle import write_surface_bundle
from vsm.export.reader import SurfaceBundle, read_surface_bundle, validate_document
from vsm.settings import Settings
from vsm.surface.arbitrage import check_surface
from vsm.surface.fit import build_slices
from vsm.surface.ssvi import SSVI
from vsm.surface.store import with_report
from vsm.surface.types import Surface

EXAMPLE = Path("docs/examples/surface_example.json")


@pytest.fixture(scope="module")
def surface() -> Surface:
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
    return with_report(raw, check_surface(raw, config.arbitrage))


@pytest.fixture
def path(surface: Surface, tmp_path: Path) -> str:
    return write_surface_bundle(surface, str(tmp_path), git_sha="abc1234", config_hash="sha256:x")


def document_of(path: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return document


def test_path_follows_the_contract(path: str) -> None:
    assert Path(path).as_posix().endswith("BTC/20261102T0815Z/surface.json")


def test_written_file_validates(path: str) -> None:
    validate_document(document_of(path))


def test_example_validates() -> None:
    validate_document(json.loads(EXAMPLE.read_text(encoding="utf-8")))


def test_reader_reproduces_the_stored_surface_to_1e_12(surface: Surface, path: str) -> None:
    bundle = read_surface_bundle(path)
    k = np.linspace(-1.0, 1.0, 50)
    times = np.linspace(0.01, 2.0, 20)
    grid_k, grid_t = np.meshgrid(k, times, indexing="ij")
    stored = surface.ssvi.w(grid_k, grid_t)
    assert np.abs(bundle.total_variance(grid_k, grid_t) - stored).max() <= 1e-12


def test_svi_source_matches_slices_at_listed_expiries(surface: Surface, path: str) -> None:
    bundle = read_surface_bundle(path)
    k = np.linspace(-0.5, 0.5, 11)
    for item in surface.slices:
        read = bundle.total_variance(k, np.full(k.size, item.t), source="svi")
        assert np.abs(read - item.svi.w(k)).max() <= 1e-12


def test_svi_source_is_linear_in_w_between_expiries(surface: Surface, path: str) -> None:
    bundle = read_surface_bundle(path)
    first, second = surface.slices[1], surface.slices[2]
    k = np.linspace(-0.3, 0.3, 7)
    t = 0.5 * (first.t + second.t)
    expected = 0.5 * (first.svi.w(k) + second.svi.w(k))
    assert bundle.total_variance(k, np.full(k.size, t), "svi") == pytest.approx(expected)


def test_svi_source_extrapolates_at_constant_atm_vol(surface: Surface, path: str) -> None:
    bundle = read_surface_bundle(path)
    last = surface.slices[-1]
    k = np.array([0.0, 0.2])
    t = 2.0 * last.t
    assert bundle.total_variance(k, np.full(2, t), "svi") == pytest.approx(2.0 * last.svi.w(k))
    first = surface.slices[0]
    t = 0.5 * first.t
    assert bundle.total_variance(k, np.full(2, t), "svi") == pytest.approx(0.5 * first.svi.w(k))


def test_forward_interpolates_log_forward_and_extrapolates_basis(
    surface: Surface, path: str
) -> None:
    bundle = read_surface_bundle(path)
    first, second, last = surface.slices[0], surface.slices[1], surface.slices[-1]
    assert float(bundle.forward(np.array(first.t))) == pytest.approx(first.forward)
    middle = float(bundle.forward(np.array(0.5 * (first.t + second.t))))
    assert middle == pytest.approx(np.sqrt(first.forward * second.forward))
    far = float(bundle.forward(np.array(2.0 * last.t)))
    assert far == pytest.approx(SPOT * (last.forward / SPOT) ** 2)


def test_implied_vol_uses_moneyness_at_t(surface: Surface, path: str) -> None:
    bundle = read_surface_bundle(path)
    item = surface.slices[3]
    atm = float(bundle.implied_vol(np.array(item.forward), np.array(item.t)))
    theta = float(surface.ssvi.theta(np.array(item.t)))
    assert atm == pytest.approx(np.sqrt(theta / item.t), rel=1e-12)


def test_unknown_source_is_refused(path: str) -> None:
    with pytest.raises(ValueError, match="unknown source"):
        read_surface_bundle(path).total_variance(np.zeros(1), np.ones(1), "heston")


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda d: d.update(schema="vsm.surface.v2"), jsonschema.ValidationError),
        (lambda d: d["arbitrage"].update(passed=False), jsonschema.ValidationError),
        (lambda d: d.pop("expiries"), jsonschema.ValidationError),
        (lambda d: d["ssvi"]["t_nodes"].reverse(), ValueError),
        (lambda d: d["ssvi"]["theta_nodes"].reverse(), ValueError),
        (lambda d: d["ssvi"].update(phi="heston_like"), ValueError),
        (lambda d: d.update(ssvi=None), ValueError),
    ],
)
def test_broken_documents_are_refused(path: str, change: Any, error: type[Exception]) -> None:
    document = document_of(path)
    change(document)
    with pytest.raises(error):
        validate_document(document)


def test_additive_fields_are_allowed(path: str) -> None:
    document = document_of(path)
    document["a_new_optional_field"] = {"anything": 1}
    validate_document(document)


def test_svi_canonical_without_ssvi_uses_the_slices(path: str) -> None:
    document = document_of(path)
    document["canonical"] = "svi"
    document["ssvi"] = None
    bundle = SurfaceBundle(document)
    k = np.array([0.0])
    t = np.array(document["expiries"][2]["t"])
    assert bundle.total_variance(k, t) == pytest.approx(bundle.total_variance(k, t, "svi"))


def test_single_expiry_extrapolates_from_that_slice(path: str) -> None:
    document = document_of(path)
    document["expiries"] = document["expiries"][:1]
    bundle = SurfaceBundle(document)
    t = document["expiries"][0]["t"]
    k = np.array([0.1])
    assert bundle.total_variance(k, np.array([2 * t]), "svi") == pytest.approx(
        2 * bundle.total_variance(k, np.array([t]), "svi")
    )


def test_heston_like_phi_is_supported_for_other_writers(path: str) -> None:
    document = copy.deepcopy(document_of(path))
    document["ssvi"].update(phi="heston_like", **{"lambda": 2.0})
    bundle = SurfaceBundle(document)
    theta = np.array(document["ssvi"]["theta_nodes"][2])
    t = np.array(document["ssvi"]["t_nodes"][2])
    assert float(bundle.total_variance(np.array(0.0), t)) == pytest.approx(float(theta))


def test_bundle_uses_the_applied_discount_factor(surface: Surface, tmp_path: Path) -> None:
    written = write_surface_bundle(surface, str(tmp_path), coin_rate="implied")
    document = document_of(written)
    assert document["conventions"]["coin_rate"] == "implied"
    assert document["expiries"][0]["discount_factor"] == surface.slices[0].df


@pytest.mark.data
def test_every_exported_file_validates() -> None:
    files = sorted((Settings().data_dir / "exports").glob("*/*/surface.json"))
    assert files
    for file in files:
        read_surface_bundle(str(file))


def test_ssvi_type_is_reachable() -> None:
    assert SSVI(0.0, 1.0, 0.4, (1.0,), (0.1,)).theta(np.array(1.0)) == pytest.approx(0.1)

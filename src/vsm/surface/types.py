"""The fitted surface of one underlying at one as-of time."""

from dataclasses import dataclass, field

from vsm.surface.ssvi import SSVI
from vsm.surface.svi import RawSVI


@dataclass(frozen=True)
class SliceCheck:
    """Gate results of one slice. Calendar compares it with the slice before it."""

    min_g: float
    density_mass_err: float
    density_mean_err: float
    price_ok: bool
    min_calendar_dw: float | None
    passed: bool


@dataclass(frozen=True)
class ArbReport:
    """Gate results of a whole surface."""

    passed: bool
    min_g: float
    min_calendar_dw: float
    density_mass_max_err: float
    density_mean_max_err: float
    slices: tuple[SliceCheck, ...]
    ssvi_slices: tuple[SliceCheck, ...]


@dataclass(frozen=True)
class SurfaceSlice:
    """One expiry: forward, fitted raw SVI, where it came from and how well it fits."""

    expiry_ns: int
    t: float
    forward: float
    forward_se: float
    forward_method: str
    df: float
    basis_ann: float
    svi: RawSVI
    source: str  # svi | svi_constrained | ssvi_fallback
    k_min: float
    k_max: float
    n_quotes: int
    n_liquid: int
    rmse_iv_bp: float
    within_spread_frac: float
    ssvi_within_spread_frac: float


@dataclass(frozen=True)
class Surface:
    """A surface before or after the gate; ``report`` is set once it has been checked."""

    underlying: str
    venue: str
    as_of_ns: int
    snapshot_id: str
    spot_index: float
    ssvi: SSVI
    slices: tuple[SurfaceSlice, ...]
    dropped_expiries: tuple[int, ...] = ()
    fallback_counts: dict[str, int] = field(default_factory=dict)
    report: ArbReport | None = None

"""YAML experiment config: typed models plus a stable config hash."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from vsm.settings import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class PathsConfig(BaseModel):
    """Output directories, relative to the repo root unless absolute."""

    model_config = ConfigDict(extra="forbid")

    results_dir: Path = Path("./results")
    reports_dir: Path = Path("./reports")


class RecorderConfig(BaseModel):
    """Snapshot recorder schedule and limits."""

    model_config = ConfigDict(extra="forbid")

    currencies: list[str] = ["BTC", "ETH"]
    kinds: list[Literal["option", "future"]] = ["option", "future"]
    minute_utc: int = Field(default=15, ge=0, le=59)
    canonical_time_utc: str = "08:15"
    ticker_sweep: Literal["none", "daily", "hourly"] = "daily"
    max_requests_per_second: float = Field(default=5.0, gt=0)
    retries: int = Field(default=5, ge=0, le=5)


class ConventionsConfig(BaseModel):
    """Day count and the coin-rate convention (D-04)."""

    model_config = ConfigDict(extra="forbid")

    year_seconds: int = 31_536_000
    coin_rate: Literal["zero", "implied"] = "zero"


class CleaningConfig(BaseModel):
    """Quote filters, DQ-01 to DQ-07."""

    model_config = ConfigDict(extra="forbid")

    staleness_s: int = 120
    min_days_to_expiry: float = 2.0
    min_bid_ticks: float = 1.0
    max_iv_spread: float = 0.10
    max_rel_spread: float = 1.0
    min_strikes_per_slice: int = 5


class ForwardsConfig(BaseModel):
    """Put-call parity regression and its fallbacks, FR-08 and DQ-09."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["parity_wls"] = "parity_wls"
    k_window: float = 0.20
    min_pairs: int = 3
    huber_c: float = 1.345
    max_futures_gap_bps: float = 25.0


class IvConfig(BaseModel):
    """IV solver and the fit target."""

    model_config = ConfigDict(extra="forbid")

    solver: Literal["householder_brent", "lbr"] = "householder_brent"
    max_iter: int = 50
    target: Literal["mid", "microprice"] = "mid"


class LiquidConfig(BaseModel):
    """Liquid-quote definition, DQ-08. Frozen before any AC metric is computed."""

    model_config = ConfigDict(extra="forbid")

    min_abs_delta: float = 0.05
    max_abs_delta: float = 0.95
    max_iv_spread: float = 0.10
    min_days_to_expiry: float = 2.0


class SviConfig(BaseModel):
    """Raw SVI slice fit."""

    model_config = ConfigDict(extra="forbid")

    init: Literal["quasi_explicit", "ssvi_slice"] = "quasi_explicit"
    weights: Literal["inverse_spread", "vega", "uniform"] = "inverse_spread"
    lee_bound: bool = True
    wing_anchor_weight: float = Field(default=1.0e-3, ge=0.0)


class SsviConfig(BaseModel):
    """Global SSVI fit. Only the power-law phi is built."""

    model_config = ConfigDict(extra="forbid")

    phi: Literal["power_law"] = "power_law"


class GridZ(BaseModel):
    """Check grid in z = k / sqrt(theta)."""

    model_config = ConfigDict(extra="forbid")

    min: float = -6.0
    max: float = 6.0
    n: int = 1201


class ArbitrageConfig(BaseModel):
    """Static-arbitrage gate. Tolerances here are never relaxed to pass a day (rule 13)."""

    model_config = ConfigDict(extra="forbid")

    grid_z: GridZ = GridZ()
    tol_g: float = 1.0e-12
    tol_calendar: float = 1.0e-12
    tol_density_mass: float = 1.0e-3
    tol_density_mean: float = 1.0e-4
    tol_price: float = 1.0e-9
    fallback: list[Literal["constrained_refit", "ssvi_slice", "drop"]] = [
        "constrained_refit",
        "ssvi_slice",
        "drop",
    ]


class HestonConfig(BaseModel):
    """COS pricing and calibration. COS is the only pricer that has a gradient."""

    model_config = ConfigDict(extra="forbid")

    pricer: Literal["cos"] = "cos"
    cos_terms: int = 400
    cos_truncation_L: float = 20.0
    optimizer: Literal["lm", "trf"] = "lm"
    init: dict[str, float] = {"v0": 0.36, "kappa": 2.0, "theta": 0.36, "xi": 1.0, "rho": -0.2}
    bounds: dict[str, tuple[float, float]] = {
        "v0": (1.0e-4, 4.0),
        "kappa": (1.0e-3, 50.0),
        "theta": (1.0e-4, 4.0),
        "xi": (1.0e-3, 10.0),
        "rho": (-0.999, 0.999),
    }


class SabrConfig(BaseModel):
    """SABR per expiry; beta is fixed (D-05)."""

    model_config = ConfigDict(extra="forbid")

    beta: float = 1.0
    obloj_correction: bool = False


class LocalVolKGrid(BaseModel):
    """Uniform log-moneyness grid."""

    model_config = ConfigDict(extra="forbid")

    min: float = -1.5
    max: float = 1.5
    n: int = 121


class LocalVolConfig(BaseModel):
    """Dupire grid (FR-19) and the vanillas it must reprice (AC-H-03)."""

    model_config = ConfigDict(extra="forbid")

    k_grid: LocalVolKGrid = LocalVolKGrid()
    t_grid_days: list[float] = [2, 7, 14, 30, 60, 90, 180, 365]
    reprice_tenor_days: list[float] = [14, 30, 60]
    reprice_strike_pct: list[float] = [0.8, 0.9, 1.0, 1.1, 1.2]


class McConfig(BaseModel):
    """Monte Carlo size and seed (FR-20, NFR-02)."""

    model_config = ConfigDict(extra="forbid")

    paths: int = 200_000
    steps_per_day: int = 24
    antithetic: bool = True
    seed: int = 20261030


class BarrierProduct(BaseModel):
    """One knock-out call, as fractions of the forward."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["down_and_out_call", "up_and_out_call"]
    strike_pct: float
    barrier_pct: float


class BarrierConfig(BaseModel):
    """Products priced under local vol, Heston and flat Black (FR-20)."""

    model_config = ConfigDict(extra="forbid")

    tenor_days: int = 30
    products: list[BarrierProduct] = [
        BarrierProduct(type="down_and_out_call", strike_pct=1.0, barrier_pct=0.85),
        BarrierProduct(type="up_and_out_call", strike_pct=1.0, barrier_pct=1.2),
    ]


class VrpConfig(BaseModel):
    """Implied against realized variance (FR-21)."""

    model_config = ConfigDict(extra="forbid")

    tenor_days: int = 30
    rv_bar_s: int = 3600
    hac_lags: Literal["auto"] = "auto"


class StraddleHedge(BaseModel):
    """How the straddle's delta is hedged (D-06)."""

    model_config = ConfigDict(extra="forbid")

    instrument: Literal["perpetual"] = "perpetual"
    rebalance: Literal["daily"] = "daily"
    delta: Literal["entry_iv", "current_iv", "min_variance"] = "entry_iv"


class StraddleCosts(BaseModel):
    """Fills at the touch, fees from the venue's table, and hedge slippage."""

    model_config = ConfigDict(extra="forbid")

    fee_table: str = "configs/fees/deribit.yaml"
    option_fill: Literal["bid_ask", "mid"] = "bid_ask"
    hedge_slippage_bps: float = 2.0


class Bootstrap(BaseModel):
    """Stationary bootstrap of a mean."""

    model_config = ConfigDict(extra="forbid")

    n: int = 10_000
    method: Literal["stationary"] = "stationary"
    mean_block: float = 3.0
    seed: int = 7


class StraddleConfig(BaseModel):
    """The straddle design (FR-22), frozen before the AC-H-02 metrics are computed."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["tardis_monthly", "recorder_daily", "tardis_daily"] = "tardis_monthly"
    entry_time_utc: str = "08:15"
    target_days: float = 30.0
    min_days: float = 20.0
    max_days: float = 45.0
    side: Literal["short", "long"] = "short"
    hedge: StraddleHedge = StraddleHedge()
    costs: StraddleCosts = StraddleCosts()
    bootstrap: Bootstrap = Bootstrap()


class TardisConfig(BaseModel):
    """Free first-of-month ingest. The rate cap and guards are the owner's conditions."""

    model_config = ConfigDict(extra="forbid")

    keep_raw: bool = False
    window_utc: tuple[str, str] = ("07:45", "08:20")
    max_bytes_per_second: float = Field(default=10_000_000.0, gt=0)
    min_free_gb: float = 100.0
    free_space_path: str = "/mnt/d"
    crypto_status_path: str = "~/data/crypto-lob-alpha/status.json"
    max_ingest_p99_us: float = 1000.0


class ExportConfig(BaseModel):
    """Surface export contract."""

    model_config = ConfigDict(extra="forbid")

    schema_name: str = Field(default="vsm.surface.v1", alias="schema")
    canonical: Literal["ssvi"] = "ssvi"


class AppConfig(BaseModel):
    """Top-level config. Sections no milestone has typed yet pass through unchanged."""

    model_config = ConfigDict(extra="allow")

    version: int = 1
    underlyings: list[str] = ["BTC", "ETH"]
    venue: str = "deribit"
    paths: PathsConfig = PathsConfig()
    recorder: RecorderConfig = RecorderConfig()
    conventions: ConventionsConfig = ConventionsConfig()
    cleaning: CleaningConfig = CleaningConfig()
    forwards: ForwardsConfig = ForwardsConfig()
    iv: IvConfig = IvConfig()
    liquid: LiquidConfig = LiquidConfig()
    svi: SviConfig = SviConfig()
    ssvi: SsviConfig = SsviConfig()
    arbitrage: ArbitrageConfig = ArbitrageConfig()
    heston: HestonConfig = HestonConfig()
    sabr: SabrConfig = SabrConfig()
    localvol: LocalVolConfig = LocalVolConfig()
    mc: McConfig = McConfig()
    barrier: BarrierConfig = BarrierConfig()
    vrp: VrpConfig = VrpConfig()
    straddle: StraddleConfig = StraddleConfig()
    tardis: TardisConfig = TardisConfig()
    export: ExportConfig = ExportConfig()

    def results_dir(self) -> Path:
        """Absolute results directory."""
        return _resolve(self.paths.results_dir)

    def reports_dir(self) -> Path:
        """Absolute reports directory."""
        return _resolve(self.paths.reports_dir)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate a YAML config (``configs/default.yaml`` by default)."""
    raw: dict[str, Any] = yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)


def config_hash(config: AppConfig) -> str:
    """``sha256:<hex>`` of the canonical JSON form, independent of YAML formatting."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

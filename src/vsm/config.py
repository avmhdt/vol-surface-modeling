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

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


class AppConfig(BaseModel):
    """Top-level config. Sections no milestone has typed yet pass through unchanged."""

    model_config = ConfigDict(extra="allow")

    version: int = 1
    underlyings: list[str] = ["BTC", "ETH"]
    venue: str = "deribit"
    paths: PathsConfig = PathsConfig()
    recorder: RecorderConfig = RecorderConfig()

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

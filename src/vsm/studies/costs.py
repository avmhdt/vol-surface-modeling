"""Deribit fees by date, read from ``configs/fees/deribit.yaml`` (spec rule 18).

Option fees are a share of the underlying, capped at a share of the option's premium, both in
coin. Perpetual fees are a share of the notional.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from vsm.settings import REPO_ROOT


@dataclass(frozen=True)
class FeePeriod:
    """One fee schedule and the date it took effect."""

    effective: date
    option_taker: float
    option_cap: float
    delivery_rate: float
    delivery_cap: float
    perpetual_taker: dict[str, float]


@dataclass(frozen=True)
class FeeTable:
    """Fee schedules in date order."""

    source: str
    periods: tuple[FeePeriod, ...]

    def on(self, day: date) -> FeePeriod:
        """The schedule in effect on ``day``."""
        chosen = [period for period in self.periods if period.effective <= day]
        if not chosen:
            raise ValueError(f"no fee schedule covers {day}")
        return chosen[-1]


def load_fees(path: str | Path) -> FeeTable:
    """Read the fee table; a relative path is from the repository root."""
    location = Path(path)
    if not location.is_absolute():
        location = REPO_ROOT / location
    content: dict[str, Any] = yaml.safe_load(location.read_text(encoding="utf-8"))
    periods = []
    for item in content["periods"]:
        periods.append(
            FeePeriod(
                effective=item["effective"],
                option_taker=float(item["option"]["taker"]),
                option_cap=float(item["option"]["cap_of_premium"]),
                delivery_rate=float(item["option_delivery"]["rate"]),
                delivery_cap=float(item["option_delivery"]["cap_of_premium"]),
                perpetual_taker={
                    coin: float(rates["taker"]) for coin, rates in item["perpetual"].items()
                },
            )
        )
    periods.sort(key=lambda period: period.effective)
    return FeeTable(source=str(content["source"]), periods=tuple(periods))


def option_fee_coin(premium_coin: float, rate: float, cap: float) -> float:
    """Fee on one option, in coin: ``rate`` of the underlying, at most ``cap`` of the premium."""
    return min(rate, cap * premium_coin)

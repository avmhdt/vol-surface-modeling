"""Synthetic Deribit-style chains from a known SSVI surface.

Run ``uv run python tests/fixtures/make_fixtures.py`` to regenerate the committed CSVs. They are
synthetic, seeded, and carry the raw ``option_quotes`` and ``future_quotes`` columns.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from vsm.conventions import datetime_to_ns, inverse_to_forward_usd, year_fraction
from vsm.iv.black import black76_price
from vsm.iv.solver import implied_vol

SEED = 20261102
AS_OF = datetime(2026, 11, 2, 8, 15, tzinfo=UTC)
AS_OF_NS = datetime_to_ns(AS_OF)
SPOT = 78_000.0
BASIS = 0.06
SSVI = {"rho": -0.25, "eta": 1.2, "gamma": 0.45}
ATM_VOL = {1: 0.60, 3: 0.56, 7: 0.54, 14: 0.53, 30: 0.52, 60: 0.53, 90: 0.54, 180: 0.56}
LISTED_FUTURES = (7, 30, 90, 180)
TICK = 0.0001
OUT_DIR = Path(__file__).parent


def ssvi_vol(k: np.ndarray, t: float, atm_vol: float) -> np.ndarray:
    """SSVI implied vol at log-moneyness ``k``."""
    theta = atm_vol * atm_vol * t
    phi = SSVI["eta"] * theta ** -SSVI["gamma"] * (1 + theta) ** (SSVI["gamma"] - 1)
    rho = SSVI["rho"]
    w = 0.5 * theta * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho * rho))
    return np.sqrt(w / t)


def expiry_of(days: int) -> datetime:
    """08:00 UTC, ``days`` calendar days after the as-of date."""
    return (AS_OF + timedelta(days=days)).replace(hour=8, minute=0)


def forward_of(days: int) -> float:
    return SPOT * float(np.exp(BASIS * days / 365.0))


def build_chain(*, noise: bool = True, seed: int = SEED) -> pl.DataFrame:
    """Raw option rows. ``noise=False`` gives exact mids and untouched parity."""
    rng = np.random.default_rng(seed)
    rows = []
    for days, atm_vol in ATM_VOL.items():
        expiry = expiry_of(days)
        expiry_ns = datetime_to_ns(expiry)
        t = float(year_fraction(np.array([expiry_ns]), AS_OF_NS)[0])
        forward = forward_of(days)
        width = 2.5 * atm_vol * np.sqrt(t)
        strikes = np.unique(np.round(forward * np.exp(np.linspace(-width, width, 21)), -2))
        code = f"{expiry.day}{expiry.strftime('%b').upper()}{expiry.strftime('%y')}"
        for strike in strikes:
            k = np.log(strike / forward)
            vol = float(ssvi_vol(np.array([k]), t, atm_vol)[0])
            if noise:
                vol += float(rng.normal(0.0, 0.0008))
            half_iv = 0.004 + 0.012 * min(abs(k) / width, 1.0)
            for cp, is_call in ((1, True), (-1, False)):
                mid = black76_price(forward, strike, t, vol, is_call) / forward
                up = black76_price(forward, strike, t, vol + half_iv, is_call) / forward
                down = (
                    black76_price(forward, strike, t, max(vol - half_iv, 1e-4), is_call) / forward
                )
                half = max(float(up - mid), float(mid - down), TICK)
                bid, ask = float(mid) - half, float(mid) + half
                if noise:
                    bid = np.floor(bid / TICK) * TICK
                    ask = np.ceil(ask / TICK) * TICK
                rows.append(
                    {
                        "snapshot_id": "deribit-BTC-20261102T0815Z",
                        "ts_recv": AS_OF_NS - 1_000_000_000,
                        "ts_exchange": AS_OF_NS - 1_500_000_000,
                        "venue": "deribit",
                        "symbol": f"BTC-{code}-{int(strike)}-{'C' if is_call else 'P'}",
                        "underlying": "BTC",
                        "expiry_ns": expiry_ns,
                        "strike": float(strike),
                        "cp": cp,
                        "bid_px": bid if bid > 0.0 else None,
                        "ask_px": ask,
                        "mid_px": float(mid),
                        "mark_px": float(mid),
                        "last_px": None,
                        "mark_iv_pct": 100.0 * vol,
                        "underlying_index": f"BTC-{code}",
                        "underlying_px": forward,
                        "interest_rate": 0.0,
                        "open_interest": 1.0,
                        "volume_24h": 1.0,
                        "source": "recorder",
                    }
                )
    return pl.DataFrame(rows).with_columns(pl.col("cp").cast(pl.Int8))


def build_futures() -> pl.DataFrame:
    """Dated futures at the true forward plus a perpetual at spot."""
    rows = [
        {
            "snapshot_id": "deribit-BTC-20261102T0815Z",
            "ts_recv": AS_OF_NS - 1_000_000_000,
            "ts_exchange": AS_OF_NS - 1_500_000_000,
            "venue": "deribit",
            "symbol": "BTC-PERPETUAL",
            "underlying": "BTC",
            "expiry_ns": None,
            "bid_px": SPOT - 0.5,
            "ask_px": SPOT + 0.5,
            "mark_px": SPOT,
            "index_px": SPOT,
            "current_funding": 0.0,
            "funding_8h": 1e-5,
            "open_interest": 1e8,
        }
    ]
    for days in LISTED_FUTURES:
        forward = forward_of(days)
        rows.append(
            {
                **rows[0],
                "symbol": f"BTC-FUT-{days}",
                "expiry_ns": datetime_to_ns(expiry_of(days)),
                "bid_px": forward - 0.5,
                "ask_px": forward + 0.5,
                "mark_px": forward,
                "current_funding": None,
                "funding_8h": None,
            }
        )
    return pl.DataFrame(rows, schema_overrides={"expiry_ns": pl.Int64})


def tick_table(chain: pl.DataFrame) -> pl.DataFrame:
    return chain.select("symbol").with_columns(pl.lit(TICK).alias("tick_size"))


def main() -> None:
    chain = build_chain()
    chain.write_csv(OUT_DIR / "synthetic_chain.csv")
    build_futures().write_csv(OUT_DIR / "synthetic_futures.csv")
    vols, status = implied_vol(
        inverse_to_forward_usd(chain["mid_px"].to_numpy(), chain["underlying_px"].to_numpy()),
        chain["underlying_px"].to_numpy(),
        chain["strike"].to_numpy(),
        year_fraction(chain["expiry_ns"].to_numpy(), AS_OF_NS),
        chain["cp"].to_numpy() == 1,
    )
    print(f"{chain.height} rows, IV range {np.nanmin(vols):.3f}-{np.nanmax(vols):.3f}", set(status))


if __name__ == "__main__":
    main()

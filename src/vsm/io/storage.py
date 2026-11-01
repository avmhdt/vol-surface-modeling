"""Partitioned Parquet layout, dataset schemas and atomic writes."""

import os
from pathlib import Path

import polars as pl

OPTION_QUOTES = "option_quotes"
FUTURE_QUOTES = "future_quotes"
OPTION_TICKERS = "option_tickers"
INSTRUMENTS = "instruments"

VENUE = "deribit"

OPTION_QUOTES_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "snapshot_id": pl.String,
    "ts_recv": pl.Int64,
    "ts_exchange": pl.Int64,
    "venue": pl.String,
    "symbol": pl.String,
    "underlying": pl.String,
    "expiry_ns": pl.Int64,
    "strike": pl.Float64,
    "cp": pl.Int8,
    "bid_px": pl.Float64,
    "ask_px": pl.Float64,
    "mid_px": pl.Float64,
    "mark_px": pl.Float64,
    "last_px": pl.Float64,
    "mark_iv_pct": pl.Float64,
    "underlying_index": pl.String,
    "underlying_px": pl.Float64,
    "interest_rate": pl.Float64,
    "open_interest": pl.Float64,
    "volume_24h": pl.Float64,
    "bid_qty": pl.Float64,
    "ask_qty": pl.Float64,
    "bid_iv_pct": pl.Float64,
    "ask_iv_pct": pl.Float64,
    "delta": pl.Float64,
    "gamma": pl.Float64,
    "vega": pl.Float64,
    "source": pl.String,
}

FUTURE_QUOTES_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "snapshot_id": pl.String,
    "ts_recv": pl.Int64,
    "ts_exchange": pl.Int64,
    "venue": pl.String,
    "symbol": pl.String,
    "underlying": pl.String,
    "expiry_ns": pl.Int64,
    "bid_px": pl.Float64,
    "ask_px": pl.Float64,
    "mark_px": pl.Float64,
    "index_px": pl.Float64,
    "current_funding": pl.Float64,
    "funding_8h": pl.Float64,
    "open_interest": pl.Float64,
}

OPTION_TICKERS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "snapshot_id": pl.String,
    "ts_recv": pl.Int64,
    "ts_exchange": pl.Int64,
    "venue": pl.String,
    "symbol": pl.String,
    "underlying": pl.String,
    "bid_qty": pl.Float64,
    "ask_qty": pl.Float64,
    "bid_iv_pct": pl.Float64,
    "ask_iv_pct": pl.Float64,
    "delta": pl.Float64,
    "gamma": pl.Float64,
    "vega": pl.Float64,
}

INSTRUMENTS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "venue": pl.String,
    "symbol": pl.String,
    "underlying": pl.String,
    "contract_type": pl.String,
    "option_type": pl.String,
    "strike": pl.Float64,
    "expiry_ns": pl.Int64,
    "tick_size": pl.Float64,
    "tick_size_steps": pl.String,
    "lot_size": pl.Float64,
    "contract_size": pl.Float64,
    "settlement_currency": pl.String,
    "is_active": pl.Boolean,
    "first_seen_ns": pl.Int64,
    "last_seen_ns": pl.Int64,
}

PERP_OHLC_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "ts": pl.Int64,
    "venue": pl.String,
    "symbol": pl.String,
    "resolution_s": pl.Int32,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

DVOL_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "ts": pl.Int64,
    "underlying": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
}

FUNDING_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "ts": pl.Int64,
    "symbol": pl.String,
    "interest_1h": pl.Float64,
    "interest_8h": pl.Float64,
    "index_px": pl.Float64,
    "prev_index_px": pl.Float64,
}

DELIVERY_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "date": pl.Date,
    "index_name": pl.String,
    "delivery_price": pl.Float64,
}


def raw_dir(data_dir: Path) -> Path:
    """Root of the irreplaceable Deribit history."""
    return data_dir / "raw" / VENUE


def snapshot_path(
    data_dir: Path, dataset: str, underlying: str, day: str, snapshot_id: str
) -> Path:
    """Path of the one file a snapshot writes in a quote dataset."""
    return (
        raw_dir(data_dir)
        / dataset
        / f"venue={VENUE}"
        / f"underlying={underlying}"
        / f"date={day}"
        / f"part-{snapshot_id}.parquet"
    )


def snapshot_files(data_dir: Path, dataset: str, underlying: str) -> list[Path]:
    """All snapshot files of one dataset and underlying, oldest first."""
    root = raw_dir(data_dir) / dataset / f"venue={VENUE}" / f"underlying={underlying}"
    return sorted(root.glob("date=*/part-*.parquet"))


def instruments_path(data_dir: Path, underlying: str) -> Path:
    """Instrument metadata table of one underlying."""
    return raw_dir(data_dir) / INSTRUMENTS / f"underlying={underlying}" / "instruments.parquet"


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write zstd Parquet through a temp file and rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        df.write_parquet(tmp, compression="zstd")
        with tmp.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def upsert_parquet(
    new: pl.DataFrame,
    path: Path,
    keys: list[str],
    *,
    keep_first: tuple[str, ...] = (),
) -> pl.DataFrame:
    """Merge ``new`` into the file at ``path`` on ``keys``; new rows win.

    Columns in ``keep_first`` keep their earliest value across old and new rows.
    """
    merged = new
    if path.exists():
        old = pl.read_parquet(path)
        merged = pl.concat([old, new], how="vertical_relaxed")
    earliest = [pl.col(c).min().over(keys).alias(c) for c in keep_first]
    merged = (
        merged.with_columns(earliest).unique(subset=keys, keep="last", maintain_order=True)
    ).sort(keys)
    write_parquet_atomic(merged, path)
    return merged

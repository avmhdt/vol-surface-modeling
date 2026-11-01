# Data

Draft from M1. Nothing in this file's tree is committed: raw data lives under `$DATA_ROOT` (default `~/data/vol-surface-modeling`) on the WSL ext4 filesystem, never on `/mnt/*`. The settings loader refuses a `/mnt` data directory.

## Sources

| Source | Endpoint or file | Use | Auth |
|---|---|---|---|
| Deribit REST | `public/get_book_summary_by_currency` | Hourly option and future quotes | None |
| Deribit REST | `public/get_instruments` | Strike, expiry, tick and contract size, with `expired=true` for recent expiries | None |
| Deribit REST | `public/get_index_price` | Spot index | None |
| Deribit REST | `public/ticker` | Sizes, bid and ask IV, greeks in the daily sweep | None |
| Deribit REST | `public/get_volatility_index_data` | DVOL | None |
| Deribit REST | `public/get_tradingview_chart_data` | Perpetual OHLC | None |
| Deribit REST | `public/get_funding_rate_history` | Perpetual funding | None |
| Deribit REST | `public/get_delivery_prices` | Settlement values | None |
| Tardis.dev | First-of-month `options_chain` | Monthly history, from M5 | None for these files |

Only public endpoints are used. `DeribitClient` refuses any other method. Check Deribit's API terms before reusing the data. Raw data is never committed; committed figures show derived surfaces with attribution.

## Recorder

- Cron runs `vsm record` hourly at hh:15 UTC from the deploy clone (`scripts/recorder.cron`, `scripts/deploy_recorder.sh`).
- First scheduled snapshot: `20261101T2115Z`. No earlier snapshots exist, and gaps cannot be backfilled for free.
- The 08:15 UTC snapshot is canonical. It also sweeps `public/ticker` into `option_tickers`.
- One file per `snapshot_id`, written through a temp file and an atomic rename. A re-run of a slot writes nothing.
- A missed slot is appended to `raw/deribit/_gaps.csv` with its reason, either the error or `not_captured`. Gaps are never filled.
- Run log: `logs/recorder-YYYY-MM-DD.jsonl` with `snapshot_id`, rows, latency and retries. Cron output: `logs/cron.log`.
- Each run warns with a `disk_warning` line once the filesystem is 80% used.
- `vsm status` writes coverage to `results/metrics/recorder_coverage.json`.

## Measured size

One snapshot of both currencies, about 1,770 rows, takes about 148 kB of zstd Parquet, measured on the first run on 2026-11-01. That is roughly 1.3 GB per year, inside the spec's 1 to 3 GB estimate. Ticker sweeps add to this and will be measured after the first 08:15 run.

## History backfill

Run on 2026-11-01. Coverage by series, latest rows at that date:

| Series | BTC starts | ETH starts | Rows BTC | Rows ETH |
|---|---|---|---|---|
| Perpetual hourly OHLC | 2018-08-14 | 2019-03-14 | 72,035 | 66,944 |
| DVOL hourly | 2021-03-24 | 2021-03-24 | 49,173 | 49,173 |
| Funding hourly | 2019-04-30 | 2019-04-30 | 65,818 | 65,818 |
| Delivery prices daily | 2016-03-18 | 2019-03-29 | 2,656 | 2,496 |

The whole history takes under 10 MB. A daily cron line appends new rows.

Endpoint behaviour found while backfilling:

- `get_tradingview_chart_data` returns at most about 5,000 bars, the most recent in the range. The ingest pages in 4,800-hour chunks.
- `get_funding_rate_history` returns at most 744 records, the most recent in the range. The ingest pages in 720-hour chunks.
- Early funding records have null fields, so all four numeric columns are nullable.
- `get_instruments` with `expired=true` returns only recent expiries (62 BTC options on 2026-11-01), so older instruments cannot be recovered. Strike, expiry and type for old snapshots are safe because the recorder stores them on each row.

## Datasets

```text
raw/deribit/option_quotes/venue=deribit/underlying=BTC/date=YYYY-MM-DD/part-<snapshot_id>.parquet
raw/deribit/future_quotes/venue=deribit/underlying=BTC/date=YYYY-MM-DD/part-<snapshot_id>.parquet
raw/deribit/option_tickers/venue=deribit/underlying=BTC/date=YYYY-MM-DD/part-<snapshot_id>.parquet
raw/deribit/instruments/underlying=BTC/instruments.parquet
raw/deribit/perp_ohlc/symbol=BTC-PERPETUAL/part-YYYY.parquet
raw/deribit/dvol/underlying=BTC/part-YYYY.parquet
raw/deribit/funding/symbol=BTC-PERPETUAL/part-YYYY.parquet
raw/deribit/delivery/index_name=btc_usd/delivery.parquet
raw/deribit/_gaps.csv
```

Column-level schemas are the `*_SCHEMA` dicts in `src/vsm/io/storage.py`. `raw/deribit/` is the irreplaceable history: it is kept indefinitely, and the owner backs it up weekly.

## Licensing

Deribit raw data and Tardis files are never committed or redistributed. `tests/fixtures/deribit/` holds a handful of trimmed responses for the mocked transport, as the spec allows. The repo guard (`scripts/check_repo_guard.py`) fails CI on any tracked path under `data/`, any `.csv.gz`, and any fixture over 500 kB.

# Decisions log

Owner and agent decisions for this repo. Owner entries are copied from the portfolio's `DECISIONS.md` (recorded 2026-10-31), which overrides `SPEC.html`, which overrides the README. Thresholds change only with owner approval recorded here.

## Owner decisions

| Item | Decision |
|---|---|
| Merging | Agents push feature branches and open PRs. PRs merge automatically once CI is green. Branch protection and auto-merge are on for this repo. |
| GitHub | Active account `avmhdt`, admin on the repo, Actions enabled. |
| Country (P-07) | Cayman Islands, `OWNER_COUNTRY=KY`. Public Deribit data needs no account. |
| WRDS (P-11) | No access. OptionMetrics path stays off. |
| Downloads | Tardis free samples approved. Raw and row-level data stay out of git; aggregates only. |
| Money | USD 0 for this project. Nothing here is on the approved spend list. |
| Machine | Toolchain and always-on setup done by the owner. Storage stays on `D:`; free space is checked at each milestone. |
| Spec defaults | Every other default in spec section 3 stands. No project 4 default was changed in `DECISIONS.md`. |

## Spec decisions as they stand

| ID | Decision | Outcome |
|---|---|---|
| D-01 | Recorder host and cadence | Owner's PC under WSL2, cron, hourly at hh:15 UTC, canonical snapshot 08:15 UTC. Gaps logged, never filled. |
| D-02 | Tardis paid budget | Not approved. Free first-of-month files plus the recorder. |
| D-03 | SPX path | None. Crypto path only, no SPX source. FR-06 and AC-H-09 do not apply. |
| D-04 | Coin rate | Zero, implied DF still reported daily. |
| D-05 | SABR beta | 1.0, with a 0.5 sensitivity table. |
| D-06 | Straddle hedge | Perpetual with funding; dated future as robustness. |
| D-07 | 3D viewer contents | Fitted surfaces plus quote mids for at most 5 showcase dates, Deribit attributed. |
| D-08 | Export contract | `vsm.surface.v1` accepted as specified. |
| D-09 | Extra `.gitignore` entries | Added: `.numba_cache/`, `.env.local`. Coverage and benchmark outputs also added. |
| D-10 | IV solver | Safeguarded Householder plus Brent primary. |
| D-11 | Licence | MIT, code only. |

## Agent decisions

### M1

- **Recorder start.** First scheduled snapshot: `20261101T2115Z` (2026-11-01 21:15 UTC) for both underlyings, BTC 956 and ETH 786 options. No earlier history exists in this store.
- **`scripts/schedule_recorder.ps1` is not created.** The spec has it register an "At startup" task that starts WSL. The owner's `setup/always-on.ps1` already registers `Start WSL at boot`, so a second task would only duplicate it. The recorder needs just its cron lines inside WSL, installed by `scripts/deploy_recorder.sh`.
- **Run-only deploy clone.** Cron runs from a separate clone at `~/deploy/vol-surface-modeling`, with its own `.venv`, pinned to a detached commit. Development stays in the repo on `/mnt/d`. Switching branches there can never change what the next hh:15 run executes, which is how rule 11 ("do not break it") is met. `deploy_recorder.sh` refuses to run within minutes 12 to 20 of the hour, so a deploy lands between scheduled runs.
- **Commits come from Windows git.** WSL's global git config has a different email from this repo's history.
- **Time zone.** WSL's clock is `America/Sao_Paulo`. Cron uses local time, which is harmless for the minute-only hourly line. The daily history line is a nominal 09:40 local (12:40 UTC) and nothing depends on its exact hour. Slots are always computed in UTC in Python.
- **Off-schedule runs.** A run within 10 minutes after hh:15 files under that slot. Any other run, such as a manual test, files under its own minute and does not count as a scheduled snapshot, so `snapshot_id` never claims a time the data was not taken at.
- **Ticker sweep is a separate dataset, `option_tickers`.** The sweep walks about 1,750 instruments at 5 requests/s, so it takes about 6 minutes, and each row has its own `ts_recv`. Merging those into `option_quotes` would put data received minutes after the as-of time into the snapshot and break DQ-01. So `option_quotes` keeps `bid_qty` and related fields null for recorder rows, the snapshot is written before the sweep starts, and M2 joins tickers on (`symbol`, `snapshot_id`) only where `ts_recv` allows it.
- **Instrument metadata refreshes on every run.** That adds `get_instruments` calls per currency and kind, within its documented 1 request/s sustained limit. If the call fails, the stored table is used; if there is none, strike, expiry and type come from the symbol. A snapshot is never lost for want of metadata.
- **Malformed symbols are dropped.** Rows whose symbol cannot be parsed are dropped and logged per NFR-09. Metadata and symbol disagreements log `symbol_metadata_mismatch` and keep the metadata (DQ-11).
- **`lot_size`** holds the API's `min_trade_amount`, matching the spec's "min trade amount" semantic. The API's own `lot_size` field is a different number.
- **Experiment log scope.** `vsm record`, `smoke`, `status` and `ingest` do not append to `results/experiment_log.csv`. They are operations, not trials, and hourly writes would dirty the deploy clone. Rule 6 lists builds, calibrations, studies and config runs, and those append from M2 onward. The recorder has its own JSONL run log.
- **`vsm ingest deribit-history --start` is optional.** Each series resumes from its latest stored row, or from 2018-08-01 on an empty store, so backfill and daily append are one command.
- **History layout.** One Parquet file per year per series, keyed upserts, in `raw/deribit/{perp_ohlc,dvol,funding}/<key>/part-YYYY.parquet` and `raw/deribit/delivery/index_name=<index>/delivery.parquet`. Bars still open at fetch time are left out.
- **`vsm.conventions` started in M1** with timestamp and symbol helpers, because rule 12 routes all conversions through it. Price and vol conversions arrive in M2.
- **Endpoint limits read from the docs, not memory (rule 18).** `get_instruments`: 10,000 credits per call, 1 request/s sustained, burst 50. Public requests are limited per IP. Source: `docs.deribit.com/articles/rate-limits`, retrieved 2026-11-01. The recorder stays at the spec's 5 requests/s. No endpoint was renamed from the spec; `vsm smoke` passes 22 of 22 checks.
- **Open risk: WSL keep-alive.** `Start WSL at boot` runs `wsl.exe --exec /bin/true`, which exits at once. `.wslconfig` sets only `vmIdleTimeout`, which covers the VM and not the distro; Microsoft's `wsl-config` page documents `[general] instanceIdleTimeout`, default 15,000 ms, with -1 disabling auto shutdown. On 2026-11-01 the distro was up with many `wsl.exe` clients attached, so nothing yet shows cron keeps running with no client attached. The owner is deciding the fix, a shared setting applied through `setup/always-on.ps1`. This repo does not touch the boot task or `.wslconfig`, and no agent restarts WSL.
- **One planned WSL restart is expected** to apply that setting. The recorder needs nothing else to resume. `cron` is enabled under systemd, and the deploy clone, its venv, Python and the data are all on ext4. The `flock` lock is released when the process dies. The next hh:15 run logs every slot missed across the restart as `not_captured` in `_gaps.csv`, including a slot cut off mid-write, whose leftover temp file readers ignore. A gap restarts the count of consecutive snapshots for the M1 exit.
- **`configs/fees/deribit.yaml` waits for M5**, the first milestone that reads fees, so the table is copied from Deribit's fee page when it is used.

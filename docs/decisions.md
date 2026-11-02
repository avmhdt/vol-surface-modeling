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
- **Ticker sweep is a separate dataset, `option_tickers`.** The sweep walks about 1,750 instruments at 5 requests/s, so it takes about 6 minutes, and each row has its own `ts_recv`. Merging those into `option_quotes` would put data received minutes after the as-of time into the snapshot and break DQ-01. So `option_quotes` keeps `bid_qty` and related fields null for recorder rows, and the snapshot is written before the sweep starts.
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

### M2

- **AC-W-01 amended by the owner, 2026-11-01.** The owner was asked whether AC-W-01 should require the ITM vol round-trip only where the price can resolve it, and answered "Yes, amend it". The vol round-trip of 1e-10 where vega/F >= 1e-8 now applies to every OTM point, and to ITM points where 2 ulp(p) / vega <= 1e-10. The 1e-10 threshold itself is unchanged. The reason is that the literal criterion cannot be met in double precision. On the 121 x 60 grid, 127 of the 3,077 ITM points with vega/F >= 1e-8 miss the 1e-10 vol round-trip (50 calls, 77 puts; worst 4.3e-8). An ITM price is intrinsic plus a much smaller time value, so the time value is known only to about one ulp of the whole price, and where vega is tiny that alone moves vol by more than 1e-10: half an ulp exceeds 1e-10 of vol at 170 of those points. It is not the solver. Solving the same points from their OTM price gives 0 misses and a worst error of 1.8e-15. Everything else passes: the price round-trip at every point (worst 3.6e-13), and the vol round-trip at every OTM point. The pipeline fits OTM quotes only (DQ-06), so no fitted surface is affected. `tests/acceptance/test_ac_w_01_iv_roundtrip.py` asserts the amended criterion, bounds the remaining ITM misses by the rounding of the input, and keeps the original wording as a strict xfail so the shortfall stays visible.
- **A price exactly at intrinsic returns vol 0.0 with status 0.** FR-09 returns NaN "outside (intrinsic, upper bound)", and intrinsic is the exact zero-vol price, so the inverse is defined. Deep ITM grid prices can round to intrinsic, and AC-W-01 needs those to round-trip. A price more than a few ulps below intrinsic still returns status 1.
- **Solver start.** FR-09 asks for a start from a rational approximation. The start here is closed-form instead: Polya's normal CDF above the inflection point s_c = sqrt(2a), a parabola in 1/s matched at s_c below it, and the tail asymptotic far out. It needs at most 4 iterations over 1,000,000 random points, a mean of 2.7, with no Brent finishes, so a rational table would add nothing. Brent stays as the safeguard and has a test that forces it.
- **`erfcx` is implemented in the Numba kernel** from Cody's (1969) coefficients, because Numba has no scaled error function. It agrees with SciPy to 9e-16.
- **Small arrays run serially.** Below 512 quotes the kernels skip `prange`, because waking the thread pool costs more than it saves.
- **`as_of_ns` of a recorder snapshot is its latest `ts_recv`,** across options and futures, not the nominal hh:15. Cron starts at hh:15:00 and rows arrive seconds later, so a nominal as-of would mark every row as coming from the future under DQ-01. The snapshot label keeps the nominal minute. On the first real run the as-of was 9 s after the slot for BTC and 18 s for ETH.
- **Parity regression.** F comes from the two-parameter fit, F = -alpha/beta, and `df` reports alpha. Under `coin_rate: zero` the price conversion uses DF = 1 as D-04 says, so `df` is reported but not applied.
- **Short expiries skip the parity regression,** since DQ-05 does not fit them. Their forward comes from the future, or else from `underlying_price`.
- **Extra `drop_reason` values:** `no_ask` (ask missing), `no_forward`, and `thin_slice` (DQ-07). The spec's list is given as examples.
- **`iv_fail` on ITM rows.** A deep ITM quote can have its bid below intrinsic while its mid is valid, and it is tagged `iv_fail`. On the first real snapshots all 175 such rows were ITM and none OTM. ITM rows are never fitted (DQ-06), so nothing is lost.
- **DQ-17 runs on OTM quotes with at least 2 days to expiry,** because ITM marks and near-expiry marks are too coarse to compare in vol terms.
- **Tick rule.** DQ-03 uses `tick_size`, not `tick_size_steps`, since the rule looks at the lowest prices, where the base tick applies.
- **Microprice stays off on recorder data.** The ticker sweep arrives after the as-of time, so DQ-01 excludes it, and target IV stays at mid. The microprice path works where sizes exist on the row, as in Tardis data.
- **`vsm build --at HH:MM` is added** to build a snapshot other than the canonical one. The first canonical 08:15 snapshot is 2026-11-02, so the first real-data run used 22:15 on 2026-11-01.

### M3

- **Fit order: SSVI first, then SVI slices anchored to it.** A slice is only identified inside its quoted range. Unanchored, an 11-day BTC slice extrapolated to rho near 0.93, close to Lee's bound, and the calendar constraint then pushed every longer slice off its quotes: only 28% of liquid quotes stayed inside the spread. So each slice gets light pseudo-points from its SSVI slice at z = 3, 4, 5, 6 and 8, outside the quoted range only.
- **`svi.wing_anchor_weight` = 1e-3, chosen from three trials.** On one snapshot (2026-11-01 22:15 UTC), the share of liquid quotes inside the spread was BTC 96.5%, ETH 98.0% at 1e-3; 90.4%, 90.2% at 1e-2; and 83.7%, 81.1% at 1e-1. All three had positive g and calendar margins. The weight is frozen at 1e-3, and any change is a new logged trial (rule 14).
- **`gamma` is bounded by 1/2.** Consumers extrapolate theta to zero, so the conditions must hold for every theta, and theta phi^2 diverges as theta goes to 0 when gamma is above 1/2. Real days sit on this bound.
- **The calendar condition of SSVI needs no constraint.** For gamma in (0, 1) it holds by construction, and `tests/unit/test_ssvi.py` checks that.
- **`phi: heston_like` is not built.** FR-12 lists it as an option, and power law is the default. The config rejects it, but the reader handles it, because other writers of the format may use it.
- **Calendar check grid.** FR-13 does not say which theta scales z for a pair of slices. The gate uses both slices' grids and both quoted ranges. The fit's constraint grid covers the same regions and runs 25% past them, because the gate scales z by the fitted theta, which is unknown during the fit. Without that, wings crossed just past the edge and slices dropped.
- **The gate covers the whole quoted range,** at the same density as the z grid. FR-13 says "plus the quoted range", and a short slice can be quoted well past z = 6. An early version added only the two end points of the range, and a test with Axel Vogt's slice showed g negative inside the quotes while the gate reported it clean. Real days were not affected: their min g was about 0.12.
- **A finite grid is still a finite grid.** g can dip between grid points by an amount that shrinks with the grid spacing. On the Vogt repair it was 3.6e-7 at the gate's density. A fit that only just satisfies the constraint could pass with such a dip; real days have margins near 0.12, far from that case.
- **Added gate settings.** `tol_density_mean` holds FR-13's 1e-4 of F, which the spec's YAML omitted. `tol_price` is 1e-9 in units of F, because the spec sets no tolerance for monotone and convex prices and rounding in the wings needs one. These add checks; nothing was relaxed. Density mass and mean integrate over 30 sqrt(theta) of log-moneyness, since 6 sqrt(theta) cuts off real tail mass.
- **`arbitrage.min_calendar_dw` is 0.0 for a one-slice surface,** since JSON has no infinity and the field is a required number.
- **`discount_factor` in the export is the factor applied:** 1 under `coin_rate: zero`.
- **Null in the export.** `forward_se` and `fit.within_spread_frac` may be null, when a fallback forward has no standard error or a slice has no liquid quotes. Both were bare numbers in the spec's example, and the schema allows null before the format freezes.
- **`surface_params` has additive columns:** `snapshot_id`, `spot_index`, `forward_se`, `forward_method`, `df`, `basis_ann`, `k_min`, `k_max` and `ssvi_within_spread_frac`. Dropped slices keep a row with null parameters, because they are SSVI nodes and `vsm check` must rebuild the surface exactly. The consumer history leaves those rows out.
- **`summary_<underlying>.csv` reports "before fallback" as fallback counts:** slices that failed the first pass are the ones counted under `fallback_constrained`, `fallback_ssvi` and `dropped`. "After" is `failed_slices_after_fallback`.
- **The gate runs inside `fit`.** `check` stays a valid step name, and `vsm check` re-runs the gate on stored surfaces without refitting.
- **Density moments include the anchored wings.** Skewness and kurtosis in `rnd_skew_<underlying>.csv` depend on the tails, which come from SSVI beyond the quotes, so read them as model values, not market ones.
- **SSVI fits much worse than the slices:** 62.9% (BTC) and 48.0% (ETH) of liquid quotes inside the spread, against 96.5% and 97.6%. AC-W-03 is on the slices, but gpu-pricing-risk-engine reads the SSVI continuum by default, so `docs/export-format.md` states the gap.
- **`spline_vs_svi_density.png` covers BTC only,** as the spec has a single file. The spline baseline runs for both underlyings.

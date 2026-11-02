# Export format `vsm.surface.v1`

One JSON file per underlying and as-of time, at `data/exports/<underlying>/<YYYYMMDDTHHMMZ>/surface.json`, UTF-8, validated by `schemas/vsm.surface.v1.schema.json` (JSON Schema draft 2020-12). `docs/examples/surface_example.json` is a synthetic example.

The file is written only for surfaces that passed the arbitrage gate. `vsm.export.reader` is the reference implementation of the rules below.

## Units

- Prices are USD per coin.
- Times are years, ACT/365 in seconds.
- Vols are decimal.
- Timestamps are ISO-8601 UTC, each with a matching int64 nanosecond field.
- `as_of` is the latest receive time in the snapshot, seconds after the nominal hh:15. The directory name and `provenance.snapshot_id` carry the nominal minute.

## Consumer rules

1. **Canonical continuum.** w(k, t) is the SSVI formula with θ(t) linear between nodes. For t < t₁, θ(t) = θ₁·t/t₁. For t > tₙ, θ(t) = θₙ·t/tₙ, which is constant ATM vol. This continuum is arbitrage-free under the enforced SSVI conditions.
2. **Forward at arbitrary t.** ln F is linear in t between expiry forwards, with a flat annualized basis beyond the ends: F(t) = S·exp(b·t), using the nearest end's b.
3. **Per-expiry SVI slices** are the best fit at listed expiries. Between expiries they may be interpolated linearly in w at fixed k, which is calendar-safe because the slices pass the calendar check, but butterfly safety between slices is not guaranteed. Monte Carlo and local-vol consumers should use the SSVI continuum.
4. **Local vol.** Bilinear in (k, t) on the grid, flat outside it. k is relative to the forward at that t. `local_vol` is null until M5.
5. **History.** `data/exports/<underlying>/surface_params.parquet` has one row per slice per as-of, with the columns in `SURFACE_PARAMS_SCHEMA` in `src/vsm/io/storage.py`.
6. **Optional fields**, additive under NFR-11:
   - `ssvi.lambda` is required only when `phi` is `heston_like`. This project writes `power_law`, and the reader handles both.
   - `provenance.producer` defaults to `vol-surface-modeling`. Other writers of this format set their own name.
   - `canonical = "svi"` is allowed only in files without an SSVI fit. Then `ssvi` is null and consumers build the continuum with rule 3. This project always writes `canonical = "ssvi"`.

## Fit quality differs between the two blocks

SSVI is the arbitrage-free continuum, but it is a much stiffer model than a raw SVI slice per expiry. On the first real snapshots the SVI slices were inside the bid-ask spread for 96.5% (BTC) and 97.6% (ETH) of liquid quotes, and SSVI for 62.9% and 48.0%. Consumers pricing listed vanillas should prefer the slices. Consumers that need a consistent continuum should use SSVI and expect that gap.

## Nullable fields

- `expiries[].forward_se` is null when the forward came from a fallback, since only the parity regression has a standard error.
- `expiries[].fit.within_spread_frac` is null when a slice has no liquid quotes.
- `heston` and `local_vol` are null, and `sabr` is empty, until M4 and M5 fill them. The fields are additive, so this needs no new version.

## `discount_factor`

This is the factor applied in the price conversion. Under `coin_rate: zero` it is 1. The regression's estimate is kept in the `forwards` table either way.

## Stability

`vsm.surface.v1` freezes when M3 ends. Additive optional fields are allowed. Any breaking change needs `vsm.surface.v2` and owner approval.

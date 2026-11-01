# Conventions

Normative for every module. All conversions live in `vsm.conventions`, and raw tables keep venue units (DQ-10).

## Instruments

- Deribit BTC and ETH options are European, cash-settled and **inverse**. The premium is quoted and paid in the coin, and the payoff (S_T − K)⁺/S_T coins is worth exactly (S_T − K)⁺ USD at expiry.
- Linear USDC options are excluded. `get_book_summary_by_currency` with `currency=BTC` or `ETH` returns only the inverse chain; linear options sit under `currency=USDC`.

## Time

- All timestamps are int64 nanoseconds UTC. Deribit milliseconds are multiplied by 10⁶ in `ms_to_ns`.
- Expiry is 08:00 UTC on the expiry date. It comes from `expiration_timestamp`, and the symbol is only a check (DQ-11).
- Time to expiry: T = (expiry_ns − as_of_ns) / (365 × 86,400 × 10⁹) years, ACT/365 in seconds.
- Perpetuals report a far-future `expiration_timestamp`. They are stored with a null `expiry_ns`.
- A recorder snapshot's as-of time is its latest `ts_recv`, seconds after the nominal hh:15. That keeps every row at or before the as-of time (DQ-01).

## Pricing

- All fitting and models use undiscounted Black-76 in USD per coin on the forward F of each expiry.
- Conversion: `c_fwd_usd = premium_coin × F / DF_coin`, with DF_coin = 1 under the default zero coin-rate convention (D-04).

## Forward

- Per expiry from put–call parity on coin premiums: `C_coin − P_coin = DF_coin × (1 − K/F)`, a line in K with intercept α = DF_coin and slope β = −DF_coin/F, so F = −α/β.
- Cross-checked against the dated future of the same expiry and Deribit's `underlying_price`.
- The regression reports α as `df`. Under `coin_rate: zero`, prices convert with DF = 1 and `df` is only reported.
- Short expiries are not fitted, so they take the future or `underlying_price` instead of the regression.
- Annualized basis b = ln(F/S)/T against the spot index is reported, never used as an input.

## Moneyness and variance

- k = ln(K/F), total implied variance w = σ²T.
- Deribit IVs are percent. Decimal vols appear only after conversion.

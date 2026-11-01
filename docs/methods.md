# Methods

Draft from M2. Each milestone adds its own section.

## Cleaning

Cleaning runs in two passes around the forward. The first applies the rules that need no forward: staleness, short expiries, zero and crossed quotes. Surviving quotes, in or out of the money, go into the parity regression. The second pass converts prices to forward-space USD, solves bid, mid and ask IVs, and applies the wide-market, bound and slice-adequacy rules. A dropped row stays in `clean_quotes` with a `drop_reason`.

## Forwards

For each expiry, call and put mids at the same strike give C − P = α + βK, with α = DF_coin and β = −DF_coin/F. The fit uses pairs with |k| ≤ 0.20, weights 1/spread², and Huber reweighting with a MAD scale, so one stale quote cannot move the line. F = −α/β, and its standard error comes from the delta method.

## IV solver

With a = |ln(F/K)| and total vol s = σ√T, the solver inverts the normalized OTM price

b(a, s) = exp(−a/2) Φ(h + t) − exp(a/2) Φ(h − t), with h = −a/s and t = s/2.

Both terms share the factor exp(−(h² + t²)/2), so in the tail b is computed from the scaled complementary error function. The plain formula subtracts two numbers that have already lost their digits there.

1. **Reduce to OTM.** An ITM quote loses its intrinsic value first.
2. **Start.** b has an inflection at s_c = √(2a). Above it the start is a Polya estimate, below it a parabola in 1/s matched at s_c, and far out the tail asymptotic.
3. **Iterate.** Householder(3) has order 4, so once a step falls below 1e-6 the next iterate is at machine precision, and the solve stops without another price evaluation. Below 1e-3 of the upper bound it works on log b, which is close to quadratic in 1/s.
4. **Safeguard.** Every step stays inside a bracket, and Brent finishes if iterations run out.

### Limit on ITM vols

An ITM price is intrinsic plus time value, so the time value is known only to about one ulp of the whole price. Where vega is tiny, that ulp moves vol by more than the 1e-10 target, and no solver can recover it. From the OTM price at the same point the vol round-trips to 1.8e-15. This is why the pipeline fits OTM quotes, and why the owner amended AC-W-01 to require the ITM vol round-trip only where the input price can resolve it (`docs/decisions.md`).

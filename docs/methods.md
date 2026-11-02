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

## Surface fit

SSVI is fitted first: (ρ, η, γ) and one θ per expiry, jointly, with θ non-decreasing. The butterfly conditions θφ(1 + |ρ|) < 4 and θφ²(1 + |ρ|) ≤ 4 must hold for every θ, not only at the nodes, because consumers extrapolate. That is why γ ≤ 1/2: above it θφ² diverges as θ → 0.

Each raw SVI slice is then fitted, shortest expiry first, by SLSQP under g(k) ≥ 0, Lee's bound, non-negative variance, and the previous slice as a calendar floor. Quotes only identify a slice inside their range, so light SSVI pseudo-points hold the wings outside it. Without them, one short slice can extrapolate to a wing so steep that every later slice is forced off its quotes.

## Arbitrage gate

Every slice is checked on z = k/√θ ∈ [−6, 6] plus its quoted range: Durrleman's g(k), total variance rising between slices, a density with mass 1 and mean F, and call prices non-increasing and convex in strike. A failing slice goes constrained refit → SSVI slice → drop. A surface that fails is recorded and never exported.

## Spline baseline

A natural cubic spline of mid IV in k goes through the same checks. It fits every quote exactly, noise included, and that is the problem: density depends on the second derivative of the smile, so small differences between neighbouring quotes become large swings in the density, and it goes negative. A negative density is a butterfly arbitrage. See `reports/figures/spline_vs_svi_density.png`.

## Density

The risk-neutral density of k is g(k)/√(2πw) · exp(−d₂²/2), so g(k) ≥ 0 is the condition for a non-negative density. It is also why g is the denominator of Dupire local vol. Skew in the smile tilts this density: a negative 25-delta risk reversal goes with negative skewness of ln(S_T/F). Moments use the anchored wings, so they are model values beyond the quotes.

### Limit on ITM vols

An ITM price is intrinsic plus time value, so the time value is known only to about one ulp of the whole price. Where vega is tiny, that ulp moves vol by more than the 1e-10 target, and no solver can recover it. From the OTM price at the same point the vol round-trips to 1.8e-15. This is why the pipeline fits OTM quotes, and why the owner amended AC-W-01 to require the ITM vol round-trip only where the input price can resolve it (`docs/decisions.md`).

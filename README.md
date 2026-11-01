# Options volatility surface modeling

> Turn noisy option quotes into an arbitrage-free surface, calibrate models to it, and study what implied vol pays.

**Status:** in progress. Hourly Deribit recorder live since 2026-11-01 21:15 UTC.

| | |
|---|---|
| Track | Quant Researcher |
| Estimated effort | 5–7 weeks |
| Stack | Python (NumPy, SciPy), JAX optional |
| Data cost | Free → Paid |
| Strongest for | Options market makers, vol desks |

## Requirements

- [x] Quote cleaning: drop zero bids and wide markets; use mid or bid/ask-weighted implied vol. (FR-07, FR-10)
- [x] Implied forward and dividend or borrow from put–call parity for each expiry. (FR-08)
- [x] A robust implied-vol solver (Jäckel’s ‘Let’s Be Rational’ or a safeguarded Newton/Brent). (FR-09)
- [ ] Raw SVI per expiry plus a global SSVI fit (Gatheral and Jacquier 2014), with butterfly and calendar no-arbitrage enforced.
- [ ] Arbitrage diagnostics: Durrleman’s g(k) ≥ 0, non-decreasing total variance, positive Breeden–Litzenberger density.
- [ ] Model calibration: Heston via COS or Carr–Madan FFT, and SABR (Hagan) per expiry; compare fit error and stability.
- [ ] A variance risk premium study: implied vs realized variance, and delta-hedged option P&L.

## Data

Raw and licensed data live in `data/`, which is git-ignored. Only download scripts and small synthetic fixtures are committed.

| Dataset | Cost | What it gives you |
|---|---|---|
| [Deribit API](https://docs.deribit.com) | Free | Full BTC and ETH option chains with IVs. Deeper history is on Tardis.dev. |
| [Cboe DataShop](https://datashop.cboe.com) | Paid | End-of-day and intraday SPX and VIX options. Free sample files. |
| [OptionMetrics IvyDB (via WRDS)](https://wrds-www.wharton.upenn.edu) | Academic | The standard academic US options dataset. |
| [ThetaData](https://www.thetadata.net) | Paid | Affordable historical US option quotes and trades. |
| [FRED](https://fred.stlouisfed.org) | Free | Treasury yields and SOFR for the risk-free curve. |
| [Yahoo option chains](https://github.com/ranaroussi/yfinance) | Free | Current snapshots only. Record them daily if you have nothing else. |

## Success criteria

### Works (correct and reproducible)

- [x] The IV solver round-trips price → vol → price to 1e-10 across moneyness, deep wings included. (AC-W-01, as amended by the owner for ITM points)
- [ ] Zero static-arbitrage violations in fitted surfaces, checked automatically for every day.
- [ ] Fit error falls inside the bid–ask spread for over 90% of liquid quotes.

### Impresses hirers

- [ ] A daily surface history with stable parameters (no jumping SVI parameters from one day to the next).
- [ ] A delta-hedged straddle study that recovers the variance risk premium after costs.
- [ ] Dupire local vol extracted from the surface and used to price a barrier option, compared with Heston.
- [ ] Fast calibration (Heston in under a second with analytic gradients) and an interactive 3D surface viewer.

## Questions this project must answer

- Why not just interpolate IV with a cubic spline?
- What does skew tell you about the risk-neutral density?
- Sticky strike or sticky delta: which did you assume, and why?

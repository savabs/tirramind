---
title: "Research: Options Surface Lab (voltorch v0.2 + live Deribit surface page)"
tags:
  - doc/research
  - topic/quant
  - status/current
---

# Research: Options Surface Lab (voltorch v0.2 + live Deribit surface page)

Date: 2026-09-06. Source: `ultraresearch_venture_scan_2026-09-06.md` §2.1 (top-ranked
survivor, both skeptics passed). Owner constraint added 2026-09-06: **no ideas whose
value depends on accumulating data** — this one is engine-only; every number on the
page is computed from a live chain in seconds.

## What exists
- voltorch (github.com/savabs/voltorch, 39 tests): BlackScholes + IV solver,
  Barone-Adesi–Whaley, Fourier-COS (Heston/Bates/Merton/VG), single-slice
  `SVIParameterization`, `SABRModel`, `ImpliedVolatilitySurface`, rough Bergomi.
  Not on PyPI. 0 stars.
- Missing: SSVI (Gatheral–Jacquier 2014) — one surface across expiries that is
  butterfly- and calendar-arbitrage-free *by construction*; an arbitrage
  checker for someone else's marks; a chain fitter with a fit report.
- Compute: Apple M5 / 17 GB, torch MPS. No CUDA anywhere we own. Hetzner box
  (188.245.203.137) idle, CPU only.

## Data (free, keyless)
Deribit public REST: `get_instruments` (964 BTC / ~700 ETH options),
`get_book_summary_by_currency` (mark_iv, bid, ask, underlying_price per row,
one call), `ticker` (bid_iv / ask_iv / greeks per instrument). Options are
inverse (coin-settled), strikes in USD, `interest_rate` 0 → Black-76 on the
per-expiry forward. Public rate limit ~20 req/s; we use ≤5.

## Competitors (from the scan)
OptionStrat $40–100/mo (1 employee, ~400k visits/mo, footer: calculations
"are estimates"); Option Omega from $25; Laevitas $50 (17 staff, ~7k
visits/mo). None publishes arbitrage checks or fit error. Vola Dynamics does,
institution-only, sales-led. Libraries: QuantLib (SVI no SSVI), py_vollib.

## The claim the page makes, checkable by anyone
1. Deribit's own mark IVs contain N butterfly / calendar violations per day
   (listed by strike/expiry, with the bid–ask tolerance used).
2. Our SSVI surface has zero, fits inside bid–ask for X% of quotes, RMSE Y
   vol points, refit every 30 minutes, history kept.
3. Autograd greeks agree with closed-form Black-76 to < 1e-6 (table).

## Maths (fixed before code)
SSVI total variance: w(k,θ) = θ/2 · (1 + ρ φ(θ) k + √((φ(θ)k + ρ)² + 1 − ρ²)),
φ(θ) = η / (θ^γ (1+θ)^(1−γ)), 0 < γ ≤ 1. No butterfly iff
θφ(θ)(1+|ρ|) < 4 and θφ(θ)²(1+|ρ|) ≤ 4 (Gatheral–Jacquier Thm 4.2). No calendar
iff θ_T nondecreasing in T. Parameterise: θ_T = cumsum(softplus(·)) over sorted
expiries; ρ = tanh(·); η, γ via sigmoid to the admissible box; enforce the two
inequalities by projecting η after each step. Loss: squared IV error weighted
by 1/(ask_iv − bid_iv)² clipped, over quotes with both sides. Optimiser: Adam
then L-BFGS. Durrleman g(k) ≥ 0 sampled on a grid as a *test* of the fitted
surface, never as the constraint. Venue-mark checks: calendar — total variance
at equal log-moneyness nonincreasing in T (interpolate marks in k per slice);
butterfly — Black-76 call prices from mark IV, discrete convexity
C(K₋) − 2C(K) + C(K₊) ≥ −tol with tol from bid–ask; vertical — 0 ≤
−∂C/∂K ≤ e^{-rT}. Report violations only where the violation exceeds the
bid–ask width, so illiquid noise is not counted as arbitrage.

## Open questions (answer while building)
- Whether Deribit's `mark_iv` on far wings is a model mark or last-trade —
  check `bid_iv=0` rows (no bid); exclude one-sided quotes from the fit.
- ETH chain size and whether 30-minute Actions cadence fits the 2,000
  free minutes (≈1 min/run × 1,440 runs). Fallback: Hetzner cron.

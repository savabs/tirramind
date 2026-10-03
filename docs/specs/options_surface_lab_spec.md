---
title: "Spec: Options Surface Lab — 30-day engine-first probe"
tags:
  - doc/spec
  - topic/architecture
  - status/current
---

# Spec: Options Surface Lab — 30-day engine-first probe

Research: `docs/research/options_surface_lab.md` · Task: `tasks/active/options_surface_lab.md`
Repo: `savabs/voltorch` (library, PyPI `voltorch`) + `docs/` page on GitHub Pages.

## Goal (30 days, $0)
1. voltorch 0.2 on PyPI: SSVI surface (arb-free by construction), arbitrage
   checker for any chain, Deribit chain loader, `fit_chain()` returning a report.
2. A public page refit every 30 min: BTC + ETH smiles (marks vs fit vs bid–ask),
   venue arbitrage violations per expiry, fit quality, greeks error table,
   7-day history.
3. One write-up: "Deribit's marks contain N arbitrages a day; here is the
   surface that has none." Posted.

## Success / kill (fixed now) — day 30 = 2026-10-06
Any one: ≥300 stars on voltorch · ≥500 PyPI installs/wk · one stranger runs
`fit_chain` on their own data (issue/PR/email) · ≥10 inbound asking for the
equity version. None → stop; the page keeps refitting for free.

## Non-goals
Equity chains (BYO key) · strategy builder / backtester UI · payments ·
CUDA kernels (no CUDA hardware owned; batched torch on CPU/MPS is the engine,
CUDA is a `device=` change later) · American exercise · anything OptionStrat
already ships (probability cones, net greeks).

## Steps

### A — kernel (days 1–6)
**A1 `voltorch/ssvi.py`.** `SSVI(nn.Module)` with expiries tensor, params
(θ raw, ρ raw, η raw, γ raw), `total_variance(k, T)`, `implied_vol(k, T)`,
constraint projection, `fit(k, T, iv, w, steps)`.
*Verify:* synthetic surface from known (θ_T, ρ, η, γ) is recovered to 1e-3;
Durrleman g(k) ≥ 0 on a 200×n grid; θ_T monotone; runs on `mps` and `cpu`.

**A2 `voltorch/arbitrage.py`.** `durrleman_g(k, w, dw, d2w)`,
`calendar_violations(slices, tol)`, `butterfly_violations(strikes, calls, tol)`,
`vertical_violations(...)`; all vectorised; return rows (expiry, strike,
magnitude, tolerance).
*Verify:* hand-built arbitrageable chain (a bump) is flagged; a Black-76 flat
surface is clean; a violation smaller than tol is not reported.

**A3 `voltorch/deribit.py`.** `fetch_chain(currency)` → DataFrame (expiry,
T, strike, forward, mark_iv, bid_iv, ask_iv, bid, ask, open_interest). No key,
≤5 req/s, one-sided quotes flagged not dropped. Layer-1 only: no fitting here.
*Verify:* live call returns ≥500 BTC rows, T > 0, forward > 0, one-sided
share printed.

**A4 `voltorch/chain.py`.** `fit_chain(df, device) -> FitReport`: SSVI fit on
two-sided quotes, RMSE (vol pts), inside-bid-ask share, venue violations
(A2 on marks), our violations (A2 on fitted surface; must be 0), greeks table
(autograd vs closed-form Black-76 delta/gamma/vega/theta max abs err),
timings. `to_json()`.
*Verify:* on a live BTC chain: our violations = 0; greeks err < 1e-6; fit
< 3 s on CPU.

### B — page (days 6–10)
**B1 `scripts/live_page.py`.** Runs A3+A4 for BTC and ETH, appends a row to
`docs/history.jsonl`, renders `docs/index.html` (matplotlib → inline SVG;
no JS libraries): per-expiry smiles with bid–ask band, marks, fit; violations
table with strike/expiry/magnitude; fit table; greeks table; history table
(last 7 days: violations/day, RMSE).
*Verify:* page opens, contains today's UTC timestamp, n_quotes > 500 per
currency, our-violations = 0 shown explicitly.

**B2 GitHub Actions** `surface.yml`: cron `*/30 * * * *`, `concurrency`,
`contents: write`, runs B1, gate: exit 1 unless both currencies produced a
report with ≥300 two-sided quotes; commits `docs/`. Enable Pages.
*Verify:* two consecutive scheduled commits; forced empty chain → red.

### C — release + write-up (days 10–30)
**C1 voltorch 0.2.0 on PyPI** (twine via `~/.pypirc`, as established).
README: 30-second example: `fit_chain(fetch_chain("BTC"))`.
**C2 Write-up** with a dated table of violations per venue per day for the
first 7 days, three worked examples (strike, expiry, the exact arbitrage
trade and its size in bid–ask units), the greeks error table, and the page
link. Post: HN, r/options, r/quant, Deribit's Telegram if open.
**C3 Day-30 checkpoint** with the metrics vs the bar.

## Golden tests (before C1)
synthetic SSVI recovery · Durrleman ≥ 0 · calendar monotone · bump flagged ·
flat clean · sub-tolerance not flagged · live chain: our violations 0 ·
greeks err < 1e-6 · page contains timestamp · empty chain → non-zero exit.

## Cost
$0. Deribit public API, GitHub Actions (~1,440 min/mo of 2,000), Pages, PyPI.

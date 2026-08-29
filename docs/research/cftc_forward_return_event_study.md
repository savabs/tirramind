---
title: "Research: No detectable forward edge in the CFTC anomaly surface"
tags:
  - doc/research
  - topic/quant
  - topic/product
  - status/active
date: 2026-08-29
---

# CFTC Anomaly Forward-Return Event Study — Results

**Question:** Do the `|z| >= 2.0` anomalies the Brief tier ($29/mo) ships on CFTC
futures-positioning fields precede any real forward price move, or are they
statistically-valid noise?

**Answer: no edge survives multiple-testing correction. 0 of 51 testable
hypotheses (9 fields x 3 horizons x 2 z-thresholds) are significant at
alpha=0.05 after Benjamini-Hochberg. The best uncorrected p-value (0.002,
`mm_net_pct_oi` |z|>=2 at 20 trading days) becomes p_adj=0.102 and does not
survive. Do not sell this as a predictive claim.**

Script: `scripts/cftc_event_study.py` (read-only, `.venv/bin/python
scripts/cftc_event_study.py`).

---

## 1. The panel

All **19/19** `cftc_tracks` entity links (the only entity-graph edges that
join a CFTC contract code to a tradeable ticker) survive the join test
(>=20 CFTC weekly points, >=40 instrument daily closes, overlapping date
ranges). That number is real but misleading on its own — see attrition below.

| code | ticker | cftc pts | inst pts |
|---|---|---|---|
| 002602 | ZC=F | 160 | 753 |
| 023651 | NG=F | 160 | 791 |
| 005602 | ZS=F | 160 | 753 |
| 080732 | SB=F | 160 | 755 |
| 001602 | ZW=F | 88 | 753 |
| 088691 | GC=F | 159 | 754 |
| 111659 | RB=F | 30 | 755 |
| 057642 | LE=F | 111 | 753 |
| 073732 | CC=F | 109 | 754 |
| 054642 | HE=F | 159 | 753 |
| 083731 | KC=F | 159 | 755 |
| 033661 | CT=F | 159 | 755 |
| 06765T | BZ=F | 159 | 791 |
| 085692 | HG=F | 159 | 755 |
| 06765A | CL=F | 159 | 790 |
| 084691 | SI=F | 159 | 754 |
| 076651 | PL=F | 159 | 754 |
| 040701 | OJ=F | 159 | 755 |
| 075651 | PA=F | 159 | 753 |

**Real attrition:** of the 21,294 (contract, field, week) combinations where
the CFTC series has enough history to compute a causal z-score (>=20 points),
**7,029 (33.0%) had to be dropped** because no instrument close existed
within 7 days of the CFTC data's publication date. Root cause: CFTC
`futures_positioning` history starts **2022-01-04**, but `instrument_daily`
coverage only starts **2023-04-19** for every one of these 19 tickers — a
~15-month gap where CFTC anomalies exist but there is no tradeable price
series to score them against. There is a second, smaller attrition source at
the tail: `instrument_daily` stops updating (2026-04-17 to 2026-06-09
depending on ticker) while CFTC COT continues to 2026-08-18, cutting a
further ~10-18 weeks per contract that can never get a forward-return label.
**One in three otherwise-eligible anomalies in this product's own history
cannot be graded at all.**

## 2. Leakage audit (ran, not assumed)

1. **Z-score causality** — `zscore_causal()` in the event-study script is a
   direct copy of `_zscore_anomaly` in `scripts/live_intelligence_digest.py`:
   expanding window, `hist = x[:-1]`, min 20 points. Confirmed by code reuse,
   not by inspection alone.
2. **Publication lag (the leak this task explicitly warned about)** — CFTC
   `observed_at` is the Tuesday *as-of* position date (verified: every
   distinct `futures_positioning` timestamp in the DB falls on a Tuesday).
   The report is not public until the following Friday ~3:30pm ET
   (`agent/tools/cftc.py` docstring). The script adds
   `PUB_LAG_SECS = 3 days` and only allows entry at the first instrument
   close **on or after** that publication timestamp (7-day tolerance, else
   dropped — this is where most of the 33% attrition above comes from).
   Sample verified entries (code, ticker, obs date, publication date, actual
   entry-close date used):
   ```
   002602 ZC=F  obs=2023-04-11  pub=2023-04-14  entry_close=2023-04-19  (8.0d after obs)
   ```
   Every entry used is dated strictly after the data existed publicly — the
   naive version of this study (entering on `observed_at` itself) would have
   traded 3+ days before the report was released.
3. **No survivorship / no look-ahead in the join** — all 19 `cftc_tracks`
   links are used unconditionally; none were selected or dropped based on
   how well they performed. Attrition is 100% a function of data coverage.
4. **Evaluation window vs. feature horizon (F-05 check)** — the CFTC series
   itself is weekly and short (30-160 points per contract); the 20-trading-day
   forward horizon is well inside the available instrument history (727-1146
   points), so F-05 does not apply here — the instrument side has ample
   length. The binding constraint is the CFTC-side sample size, not window
   length.

## 3. Event study results

Baseline (the correct null) is **not zero** — it's the unconditional forward
log-return over the same horizon, pooled across all 19 tickers, for every
week where a z-score was computable at all (population n ≈ 1578-1581 per
horizon), regardless of whether it crossed the |z| threshold. This bakes in
the ambient 2023-2026 commodity drift (mostly positive — gold, silver,
copper all trended up over the sample) so that an "edge" number reflects
excess over that drift, not the drift itself.

Top rows by uncorrected p-value (full 54-row table — 51 testable after
dropping 3 fields with n<3 — is in the script output):

| field | \|z\|>= | horizon(d) | n_events | n_pop | mean_event% | mean_base% | edge% | hit_ev | hit_base | p | p_BH | sig_BH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mm_net_pct_oi | 2.0 | 20 | 123 | 1578 | 2.584 | 0.406 | 2.178 | 65.0% | 50.4% | 0.002 | 0.102 | **No** |
| swap_net | 3.0 | 20 | 9 | 1578 | 6.630 | 0.406 | 6.225 | 88.9% | 50.4% | 0.010 | 0.240 | No |
| mm_net | 2.0 | 20 | 190 | 1578 | 1.809 | 0.406 | 1.403 | 57.4% | 50.4% | 0.016 | 0.240 | No |
| conc_top4_short | 2.0 | 20 | 174 | 1578 | 1.965 | 0.406 | 1.559 | 58.1% | 50.4% | 0.019 | 0.240 | No |
| open_interest | 3.0 | 1 | 41 | 1581 | 0.680 | 0.002 | 0.678 | 70.7% | 51.4% | 0.024 | 0.240 | No |
| swap_net | 2.0 | 20 | 148 | 1578 | 2.024 | 0.406 | 1.619 | 58.8% | 50.4% | 0.029 | 0.240 | No |
| oi_change | 2.0 | 20 | 136 | 1578 | 2.010 | 0.406 | 1.604 | 56.6% | 50.4% | 0.033 | 0.240 | No |

**Multiple testing:** 9 fields x 3 horizons x 2 thresholds = 54 hypotheses,
51 testable (n>=3). Benjamini-Hochberg at alpha=0.05: **0 survive.** The
smallest adjusted p-value is 0.102.

21 of 54 rows have n_events < 30 — most of the `swap_net`, `open_interest`,
`conc_top4_*` cells at |z|>=3 are single digits (n=9, n=2) and are not
interpretable regardless of p-value.

**Second problem beyond multiple testing — pseudo-replication.** Even the
"best" cell (`mm_net_pct_oi`, |z|>=2, 20d, n=123) is not 123 independent
observations. Checking calendar-week clustering: those 123 events (251
before the instrument-availability filter) fall on only **116 distinct
weeks**, and the busiest weeks have **5 of the 5 eligible contracts firing
simultaneously** — i.e. a chunk of the "n" is the same macro/margin-driven
positioning shift hitting correlated commodities (metals, energies) in the
same week, not 123 independent trials. This is the same failure family as
F-10 (portfolio concentration illusion / correlated cluster inflating an
apparent effect) — it means the *effective* n is meaningfully smaller than
the *counted* n, so even the uncorrected p=0.002 overstates confidence
before BH is even applied.

## 4. Verdict

- **Is there any detectable forward relationship?** No field survives
  multiple-testing correction. `mm_net_pct_oi` and `swap_net` at 20-day
  horizons show the largest uncorrected edges, but both fail BH, both have
  small or clustered samples, and the 20-day horizon is exactly where
  calendar-week clustering (not independent skill) inflates apparent n.
- **Is it strong enough to justify a claim on the pricing page?** No.
  Nothing here should be described as "this Brief tier's anomalies precede
  price moves" — that claim is not supported by the only surface in the
  product that has a price join at all.
- **What would it take to detect a real effect, if one exists?** The binding
  constraint is CFTC-side sample size: 19 linked contracts x weekly data
  gives at most ~150-160 points per contract, and a third of those can't even
  be graded against a price series in this DB. To get a well-powered test
  (assuming the true effect is on the order of the noisy point estimates
  above, ~1-2% edge with ~10-15% weekly return std), you'd need roughly
  5-10x today's event count per cell — i.e., several more years of parallel
  CFTC + instrument history, or CFTC coverage extended to more than 19
  contracts (there are ~40 CFTC entities with >=20 points total; only 19 have
  a price-instrument link at all — extending `cftc_tracks` coverage is the
  cheapest lever, not waiting for more calendar time). Until then this is an
  underpowered test, and "underpowered" is not the same claim as "confirmed
  positive" — it's simply unproven, which is the honest thing to report.

## 5. What this does NOT test

This study only covers the CFTC surface, because it is the only one with an
entity-graph price join (`cftc_tracks`, 19 edges). `instrument_universe`
(realized_vol_20d, volume), `defi_flows` (tvl_usd), and `sovereign_debt`
(yield_pct) anomalies have no equivalent forward-return join available in
the current graph — that is a `schema-sentinel` / data-collection gap, not
something this study can adjudicate. Their z-score anomalies are exactly as
unvalidated today as CFTC's were before this study.

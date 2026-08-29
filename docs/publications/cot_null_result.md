---
title: "No detectable forward-return edge in CFTC Commitments-of-Traders positioning anomalies"
authors: "TirraMind"
date: 2026-08-29
tags:
  - doc/publication
  - topic/quant
status: public
code: scripts/cftc_event_study.py
commit: 85d86b0
---

# No detectable forward-return edge in CFTC Commitments-of-Traders positioning anomalies

**A null result, with the power calculation that tells you how little it proves.**

---

## Summary

We ran a forward-return event study on extreme weekly positioning readings in the
CFTC Commitments of Traders (COT) disaggregated report, across 19 futures
contracts joined to tradeable price series, 2023-04 to 2026-06.

**Result: 0 of 51 testable hypotheses survive Benjamini-Hochberg correction at
alpha = 0.05.** The best uncorrected p-value is 0.002; its BH-adjusted value is
0.102. Under a week-clustered resample that respects the calendar correlation
between contracts, the best uncorrected p rises to 0.010 and the best adjusted p
to 0.294. Nothing is significant under either.

**What this does not show.** Our specification pools *two-sided* |z| on
*level extremity* of *managed-money* positions at a *20-day* horizon. The
published literature that finds an effect finds it in *signed changes* in
*commercial* positions at a *one-week* horizon. Those are different hypotheses,
and ours cancels a signed effect by construction. Our test also has roughly
**10% power** against a plausibly-sized true effect. The honest claim is:

> *No detectable effect, at approximately 10% power, on a specification the
> literature does not endorse.*

It is **not** "COT positioning does not predict returns." Overclaiming a null is
the same error as overclaiming an edge, and this one is underpowered enough that
the null was close to guaranteed before we started. Section 7 is the part that
matters most; read it before you cite this.

Everything below is reproducible from a single script against free, keyless,
public data. The reproduction command is in Section 9.

---

## 1. Question

TirraMind shipped a paid product that flagged `|z| >= 2.0` anomalies on CFTC
futures-positioning fields. The question we needed answered before continuing to
sell it was the plain one:

> Does an extreme weekly reading in a CFTC positioning field precede an abnormal
> forward return in the underlying futures contract, relative to what that
> contract was doing anyway?

The answer determined whether the product had a defensible claim. It did not, and
the product was discontinued. This document is the study, not the postmortem.

---

## 2. The four design choices that make this checkable

Most of the value in a COT event study is in the parts that are easy to get
wrong in the direction of a false positive. We name ours up front so a reviewer
can attack them directly.

### 2.1 Publication lag is honoured (this is the big one)

Every COT observation carries a **Tuesday "as-of" date**. The report is not
public until **Friday, approximately 3:30 PM ET** — three days later.

Measuring forward returns from the as-of date grants roughly three days of
lookahead. It is, in our view, the single most likely way to manufacture a false
positive in this literature, because commodity futures autocorrelate at that
horizon and the as-of date is the timestamp the data ships with.

The study computes a publication timestamp `pub_ts = observed_at + 3 days` and
takes the entry price at the **first instrument close at or after `pub_ts`**,
with a 7-day tolerance (beyond which the event is dropped, not silently shifted).
A verbatim sample from the script's own leakage audit:

```
002602 ZC=F  obs=2023-04-11  pub=2023-04-14  entry_close=2023-04-19  (gap from obs: 8.0d)
```

Every entry used is strictly after the data existed publicly. Section 7.4
quantifies the one place where this guard is imperfect.

### 2.2 The null is the unconditional return, not zero

The baseline for each (field, horizon) cell is the mean forward log-return over
**every** week in which a z-score was computable at all — 1,578 to 1,581
observations per horizon, pooled across all 19 tickers — regardless of whether
`|z|` crossed any threshold.

This matters because 2023-2026 was a positive-drift period for this basket
(gold, silver and copper all trended up). Against a zero null, the drift alone
would print an "edge" of about +0.41% at 20 days. Our reported edge is excess
over the drift.

### 2.3 Benjamini-Hochberg across all 51 hypotheses

We tested 9 fields x 3 horizons x 2 thresholds = 54 cells, of which 51 have
`n_events >= 3` and are testable. All 51 p-values go into one BH family. We do
not report a "best" cell as if it were the only test we ran, because it was not.
See Harvey, Liu and Zhu (2016) for why this is not optional in this field.

The correction is arithmetic anyone can check:

```
smallest p = 0.002, rank 1 of m = 51  ->  0.002 x 51 / 1 = 0.102
```

### 2.4 Pseudo-replication is measured, not waved at

`n_events` is not `n_independent_observations`. When five metals contracts all
print `|z| >= 2` on the same Tuesday because of one macro or margin event, that
is closer to one observation than to five.

For the best cell (`mm_net_pct_oi`, `|z| >= 2`, 20 days), the 123 events fall on
only **67 distinct as-of weeks** — mean 1.84 contracts per firing week, with four
weeks at the maximum of 5. Section 6 re-runs every cell resampling *weeks*
instead of events, which is the correct unit.

---

## 3. Data

### 3.1 Sources — free, keyless, public domain

| Layer | Source | Access |
|---|---|---|
| Positioning | CFTC Disaggregated Futures-Only COT, weekly flat file `https://www.cftc.gov/dea/newcot/f_disagg.txt` and yearly history ZIPs `https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip` | Public domain, no key, no token |
| Prices | Daily closes for the 19 linked front-month futures tickers | Free public endpoints |

No exchange redistribution licence is involved. No credential of any kind is
required to obtain the positioning data, and none appears anywhere in the code.

### 3.2 Panel

19 contract-to-ticker links, all of which pass the join test (>= 20 CFTC weekly
points, >= 40 instrument daily closes, overlapping date ranges). All 19 are used
unconditionally — none was selected or dropped on the basis of how it performed.

| CFTC code | Ticker | CFTC weekly points | Instrument daily closes |
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

Coverage: CFTC as-of dates for the linked contracts run **2022-01-04 to
2026-08-18** (160 distinct weeks). Daily closes run **2023-04-18 to 2026-06-09**.

### 3.3 Fields tested

Nine fields, derived from the standard disaggregated columns so that anyone can
rebuild them from the raw CFTC file:

| Field | Definition from raw COT columns |
|---|---|
| `mm_net` | `M_Money_Positions_Long_All` - `M_Money_Positions_Short_All` |
| `pm_net` | `Prod_Merc_Positions_Long_All` - `Prod_Merc_Positions_Short_All` |
| `swap_net` | `Swap_Positions_Long_All` - `Swap__Positions_Short_All` |
| `mm_net_pct_oi` | `mm_net` / `Open_Interest_All` x 100, rounded to 2 decimal places |
| `mm_weekly_flow` | `Change_in_M_Money_Long_All` - `Change_in_M_Money_Short_All` |
| `open_interest` | `Open_Interest_All` |
| `oi_change` | `Change_in_Open_Interest_All` |
| `conc_top4_long` | `Conc_Net_LE_4_TDR_Long_All` |
| `conc_top4_short` | `Conc_Net_LE_4_TDR_Short_All` |

### 3.4 Attrition — one in three eligible events cannot be graded

Of the **21,294** (contract, field, week) combinations where the CFTC series has
enough history to compute a causal z-score, **7,029 (33.0%)** are dropped because
no instrument close exists within 7 days of the publication timestamp.

Two causes, both pure data coverage, neither dependent on outcomes:

1. **Head gap.** CFTC positioning history for these contracts starts 2022-01-04;
   daily closes start 2023-04-18. Roughly 15 months of positioning anomalies have
   no tradeable price series to score against.
2. **Tail gap.** Daily closes stop between 2026-04-17 and 2026-06-09 depending on
   ticker, while COT continues to 2026-08-18 — a further 10 to 18 weeks per
   contract with no forward label.

A further 36 (1-day), 63 (5-day) and 63 (20-day) events are dropped for lacking a
close far enough ahead. State this plainly: **one in three otherwise-eligible
anomalies in the product's own history could not be graded at all.**

---

## 4. Method

1. **Deduplicate.** Observations are deduplicated on `(entity_id, observed_at)`,
   keeping the highest `rowid`. Duplicate ingestion was a live failure mode in
   this codebase and inflates apparent event counts if left alone.
2. **Causal z-score.** For each (contract, field), an expanding-window z-score
   using only prior history:
   `z_t = (x_t - mean(x_{1..t-1})) / std(x_{1..t-1})`, requiring at least 20
   prior points and a non-degenerate history standard deviation. This is a direct
   reuse of the production scoring function, not a reimplementation, so the study
   grades exactly what the product shipped.
3. **Entry.** `pub_ts = observed_at + 3 days`; entry at the first close at or
   after `pub_ts`, within a 7-day tolerance.
4. **Forward return.** `log(close[i + h] / close[i])` for h in {1, 5, 20} trading
   days, where `i` is the entry index.
5. **Event set.** All entries with `|z| >= threshold`, threshold in {2.0, 3.0}.
6. **Population (null).** All entries with a computable z, unconditionally.
7. **Test.** Percentile bootstrap on the event mean (B = 2,000), two-sided
   against the population mean:
   `p = 2 x min(P(mean* <= base), P(mean* >= base))`. Confidence intervals use a
   circular block bootstrap (block length T^(1/3)); see Lahiri (1999).
8. **Correction.** Benjamini-Hochberg (1995) FDR at alpha = 0.05 across all 51
   testable cells.

Environment: Python 3.12.14, numpy 2.5.2, scipy 1.18.1, statsmodels 0.14.6.
Repository commit `85d86b0`; `scripts/cftc_event_study.py`,
`agent/quant/scoring.py` and `agent/tools/cftc.py` are unmodified at that commit.

---

## 5. Results

All 54 cells, sorted by uncorrected p-value. `mean_ev` and `mean_base` are mean
forward log-returns in percent; `edge` is their difference; `hit` rates are the
fraction of positive returns.

| field | \|z\|>= | h(d) | n_ev | n_pop | mean_ev% | mean_base% | edge% | hit_ev | hit_base | p | p_BH | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mm_net_pct_oi | 2.0 | 20 | 123 | 1578 | 2.584 | 0.406 | 2.178 | 65.04% | 50.44% | 0.002 | 0.102 | No |
| swap_net | 3.0 | 20 | 9 | 1578 | 6.630 | 0.406 | 6.225 | 88.89% | 50.44% | 0.010 | 0.240 | No |
| mm_net | 2.0 | 20 | 190 | 1578 | 1.809 | 0.406 | 1.403 | 57.37% | 50.44% | 0.016 | 0.240 | No |
| conc_top4_short | 2.0 | 20 | 174 | 1578 | 1.965 | 0.406 | 1.559 | 58.05% | 50.44% | 0.019 | 0.240 | No |
| open_interest | 3.0 | 1 | 41 | 1581 | 0.680 | 0.002 | 0.678 | 70.73% | 51.36% | 0.024 | 0.240 | No |
| swap_net | 2.0 | 20 | 148 | 1578 | 2.024 | 0.406 | 1.619 | 58.78% | 50.44% | 0.029 | 0.240 | No |
| oi_change | 2.0 | 20 | 136 | 1578 | 2.010 | 0.406 | 1.604 | 56.62% | 50.44% | 0.033 | 0.240 | No |
| oi_change | 3.0 | 1 | 30 | 1581 | 0.935 | 0.002 | 0.933 | 66.67% | 51.36% | 0.040 | 0.244 | No |
| swap_net | 2.0 | 1 | 148 | 1581 | 0.347 | 0.002 | 0.344 | 53.38% | 51.36% | 0.043 | 0.244 | No |
| conc_top4_short | 3.0 | 1 | 19 | 1581 | 0.674 | 0.002 | 0.672 | 73.68% | 51.36% | 0.075 | 0.366 | No |
| open_interest | 3.0 | 20 | 39 | 1578 | 2.027 | 0.406 | 1.622 | 61.54% | 50.44% | 0.079 | 0.366 | No |
| swap_net | 3.0 | 1 | 9 | 1581 | 1.353 | 0.002 | 1.351 | 55.56% | 51.36% | 0.094 | 0.388 | No |
| swap_net | 3.0 | 5 | 9 | 1578 | 2.611 | 0.094 | 2.517 | 66.67% | 51.33% | 0.099 | 0.388 | No |
| conc_top4_long | 2.0 | 5 | 135 | 1578 | 0.645 | 0.094 | 0.551 | 58.52% | 51.33% | 0.171 | 0.582 | No |
| pm_net | 2.0 | 20 | 142 | 1578 | 1.755 | 0.406 | 1.350 | 57.04% | 50.44% | 0.179 | 0.582 | No |
| conc_top4_short | 3.0 | 5 | 19 | 1578 | 1.321 | 0.094 | 1.227 | 63.16% | 51.33% | 0.187 | 0.582 | No |
| mm_net | 2.0 | 5 | 190 | 1578 | 0.466 | 0.094 | 0.372 | 54.74% | 51.33% | 0.204 | 0.582 | No |
| conc_top4_short | 3.0 | 20 | 19 | 1578 | -1.900 | 0.406 | -2.306 | 42.11% | 50.44% | 0.214 | 0.582 | No |
| mm_net | 3.0 | 1 | 24 | 1581 | 0.358 | 0.002 | 0.355 | 58.33% | 51.36% | 0.217 | 0.582 | No |
| conc_top4_long | 2.0 | 1 | 136 | 1581 | 0.231 | 0.002 | 0.229 | 59.56% | 51.36% | 0.234 | 0.597 | No |
| swap_net | 2.0 | 5 | 148 | 1578 | 0.534 | 0.094 | 0.440 | 52.03% | 51.33% | 0.254 | 0.617 | No |
| conc_top4_short | 2.0 | 1 | 174 | 1581 | 0.202 | 0.002 | 0.200 | 58.05% | 51.36% | 0.312 | 0.659 | No |
| open_interest | 3.0 | 5 | 39 | 1578 | 0.671 | 0.094 | 0.577 | 66.67% | 51.33% | 0.318 | 0.659 | No |
| mm_weekly_flow | 3.0 | 5 | 27 | 1578 | -0.914 | 0.094 | -1.008 | 51.85% | 51.33% | 0.332 | 0.659 | No |
| oi_change | 3.0 | 20 | 30 | 1578 | 2.109 | 0.406 | 1.703 | 60.00% | 50.44% | 0.341 | 0.659 | No |
| conc_top4_long | 2.0 | 20 | 135 | 1578 | -0.182 | 0.406 | -0.588 | 48.15% | 50.44% | 0.344 | 0.659 | No |
| oi_change | 2.0 | 1 | 138 | 1581 | 0.205 | 0.002 | 0.202 | 58.70% | 51.36% | 0.349 | 0.659 | No |
| mm_net | 2.0 | 1 | 190 | 1581 | 0.145 | 0.002 | 0.143 | 53.16% | 51.36% | 0.374 | 0.661 | No |
| open_interest | 2.0 | 1 | 218 | 1581 | 0.179 | 0.002 | 0.176 | 60.09% | 51.36% | 0.376 | 0.661 | No |
| mm_weekly_flow | 3.0 | 20 | 27 | 1578 | -1.021 | 0.406 | -1.426 | 44.44% | 50.44% | 0.452 | 0.765 | No |
| conc_top4_long | 3.0 | 20 | 20 | 1578 | -0.319 | 0.406 | -0.725 | 40.00% | 50.44% | 0.472 | 0.765 | No |
| mm_net_pct_oi | 3.0 | 20 | 25 | 1578 | 1.348 | 0.406 | 0.942 | 52.00% | 50.44% | 0.480 | 0.765 | No |
| conc_top4_long | 3.0 | 5 | 20 | 1578 | -0.444 | 0.094 | -0.538 | 55.00% | 51.33% | 0.509 | 0.787 | No |
| open_interest | 2.0 | 20 | 216 | 1578 | 0.787 | 0.406 | 0.381 | 52.31% | 50.44% | 0.561 | 0.798 | No |
| mm_net_pct_oi | 3.0 | 1 | 25 | 1581 | 0.225 | 0.002 | 0.223 | 60.00% | 51.36% | 0.573 | 0.798 | No |
| mm_net_pct_oi | 3.0 | 5 | 25 | 1578 | 0.638 | 0.094 | 0.544 | 48.00% | 51.33% | 0.584 | 0.798 | No |
| oi_change | 2.0 | 5 | 136 | 1578 | 0.315 | 0.094 | 0.221 | 57.35% | 51.33% | 0.597 | 0.798 | No |
| mm_net | 3.0 | 5 | 24 | 1578 | 0.506 | 0.094 | 0.412 | 54.17% | 51.33% | 0.606 | 0.798 | No |
| mm_weekly_flow | 2.0 | 20 | 114 | 1578 | -0.085 | 0.406 | -0.491 | 50.00% | 50.44% | 0.610 | 0.798 | No |
| oi_change | 3.0 | 5 | 30 | 1578 | 0.534 | 0.094 | 0.440 | 60.00% | 51.33% | 0.631 | 0.805 | No |
| mm_weekly_flow | 2.0 | 1 | 114 | 1581 | -0.107 | 0.002 | -0.110 | 50.88% | 51.36% | 0.678 | 0.819 | No |
| mm_net_pct_oi | 2.0 | 5 | 123 | 1578 | 0.229 | 0.094 | 0.135 | 55.28% | 51.33% | 0.685 | 0.819 | No |
| mm_weekly_flow | 2.0 | 5 | 114 | 1578 | -0.091 | 0.094 | -0.185 | 56.14% | 51.33% | 0.699 | 0.819 | No |
| mm_net_pct_oi | 2.0 | 1 | 123 | 1581 | 0.064 | 0.002 | 0.062 | 52.03% | 51.36% | 0.707 | 0.819 | No |
| conc_top4_long | 3.0 | 1 | 21 | 1581 | 0.148 | 0.002 | 0.145 | 52.38% | 51.36% | 0.760 | 0.861 | No |
| pm_net | 2.0 | 1 | 143 | 1581 | 0.058 | 0.002 | 0.055 | 52.45% | 51.36% | 0.812 | 0.899 | No |
| pm_net | 2.0 | 5 | 142 | 1578 | 0.171 | 0.094 | 0.077 | 58.45% | 51.33% | 0.854 | 0.899 | No |
| conc_top4_short | 2.0 | 5 | 174 | 1578 | 0.142 | 0.094 | 0.048 | 53.45% | 51.33% | 0.868 | 0.899 | No |
| mm_weekly_flow | 3.0 | 1 | 27 | 1581 | 0.119 | 0.002 | 0.117 | 51.85% | 51.36% | 0.871 | 0.899 | No |
| mm_net | 3.0 | 20 | 24 | 1578 | 0.224 | 0.406 | -0.182 | 45.83% | 50.44% | 0.888 | 0.899 | No |
| open_interest | 2.0 | 5 | 216 | 1578 | 0.031 | 0.094 | -0.063 | 54.63% | 51.33% | 0.899 | 0.899 | No |
| pm_net | 3.0 | 1 | 2 | 1581 | -0.125 | 0.002 | -0.128 | 50.00% | 51.36% | — | — | not testable |
| pm_net | 3.0 | 5 | 2 | 1578 | 9.834 | 0.094 | 9.740 | 100.00% | 51.33% | — | — | not testable |
| pm_net | 3.0 | 20 | 2 | 1578 | -20.634 | 0.406 | -21.039 | 50.00% | 50.44% | — | — | not testable |

**0 of 51 tested cells survive BH at alpha = 0.05. Smallest adjusted p = 0.102.**

Two things to notice before reading anything into the top rows:

- **21 of 54 cells have `n_events < 30`.** The `swap_net` `|z| >= 3` row with a
  6.2% "edge" has **n = 9**. The `pm_net` `|z| >= 3` rows have **n = 2** and are
  reported only so that the table is complete; they are not testable and the
  +9.83% and -20.63% figures are single-observation noise.
- **The largest apparent edges all sit at the 20-day horizon**, which is exactly
  where calendar clustering across correlated contracts does the most damage to
  the effective sample size. Section 6 takes that seriously.

---

## 6. Robustness: resampling weeks instead of events

The bootstrap in Section 4 resamples *events* i.i.d., which assumes the 123
events in the best cell are 123 independent draws. They are not: they fall on 67
distinct as-of weeks, and on four of those weeks five contracts fire together.

We therefore re-ran every cell with a **cluster bootstrap over as-of weeks**
(resample the 67 weeks with replacement, take all events in each drawn week,
B = 10,000), then re-applied BH across the same 51 cells.

| field | \|z\|>= | h(d) | n_ev | n_weeks | edge% | p (week-clustered) | p_BH |
|---|---|---|---|---|---|---|---|
| swap_net | 3.0 | 20 | 9 | 9 | 6.225 | 0.0098 | 0.294 |
| mm_net_pct_oi | 2.0 | 20 | 123 | 67 | 2.178 | 0.0142 | 0.294 |
| oi_change | 3.0 | 1 | 30 | 24 | 0.933 | 0.0196 | 0.294 |
| conc_top4_short | 2.0 | 20 | 174 | 75 | 1.559 | 0.0244 | 0.294 |
| open_interest | 3.0 | 1 | 41 | 32 | 0.678 | 0.0322 | 0.294 |
| mm_net | 2.0 | 20 | 190 | 89 | 1.403 | 0.0346 | 0.294 |
| pm_net | 2.0 | 20 | 142 | 78 | 1.350 | 0.0474 | 0.345 |
| oi_change | 2.0 | 20 | 136 | 69 | 1.604 | 0.0548 | 0.349 |
| swap_net | 2.0 | 20 | 148 | 73 | 1.619 | 0.0678 | 0.352 |
| swap_net | 2.0 | 1 | 148 | 73 | 0.344 | 0.0752 | 0.352 |

**Still 0 of 51 survive.** The best uncorrected p moves from 0.002 to 0.010; the
best adjusted p moves from 0.102 to **0.294**. Clustering costs roughly a factor
of five in the headline cell's p-value, which is the honest measure of how much
of its `n` was real.

Event-count-to-week-count ratios for every `|z| >= 2` cell, for anyone who wants
to check the clustering claim directly:

| field | events (20d labelled) | distinct weeks | max contracts in one week |
|---|---|---|---|
| mm_net | 190 | 89 | 6 |
| open_interest | 216 | 80 | 6 |
| swap_net | 148 | 73 | 5 |
| pm_net | 142 | 78 | 6 |
| conc_top4_long | 135 | 76 | 4 |
| conc_top4_short | 174 | 75 | 6 |
| mm_net_pct_oi | 123 | 67 | 5 |
| mm_weekly_flow | 114 | 67 | 5 |
| oi_change | 136 | 69 | 7 |

---

## 7. Limitations

This section is longer than the results section. That is the correct ratio for a
null this underpowered.

### 7.1 The test has about 10% power. This is the dominant limitation.

A t-statistic scales as the square root of sample size. Take a reference weekly
positioning effect of **t = 2.14 estimated on 1,451 weekly cross-sections** —
roughly the magnitude of the stronger published weekly-horizon results — and
scale it to our sample:

```
t_expected = 2.14 x sqrt(n / 1451)
  n = 116  ->  t = 0.61
  n = 160  ->  t = 0.71
```

Against a two-sided alpha = 0.05 critical value of 1.96, that gives power of
**9.3% to 11.0%**. In other words: **if an effect of that size were really there,
this study would have failed to detect it roughly nine times out of ten.**

Sensitivity across reference effect sizes, because the 2.14 figure is a stipulated
benchmark rather than a value we re-estimated ourselves — substitute your own:

| reference t (at n = 1,451) | expected t at n = 123 | power | expected t at n = 160 | power | n needed for 80% power |
|---|---|---|---|---|---|
| 1.50 | 0.44 | 7.2% | 0.50 | 7.9% | 5,062 |
| 2.00 | 0.58 | 9.0% | 0.66 | 10.2% | 2,847 |
| 2.14 | 0.62 | 9.6% | 0.71 | 11.0% | 2,487 |
| 2.50 | 0.73 | 11.3% | 0.83 | 13.2% | 1,822 |
| 3.00 | 0.87 | 14.1% | 1.00 | 16.9% | 1,265 |

Under every plausible assumption we would need on the order of **1,300 to 5,000
weekly observations** for a well-powered test. We have 123 events on 67
independent weeks. A null result at this power is close to preordained, and
should update your beliefs very little.

### 7.2 The specification cancels a signed effect by construction

This is the limitation a domain expert will raise first, and they are right to.

| | This study | The literature that finds an effect |
|---|---|---|
| Signal | `|z|` — **two-sided**, absolute | **signed** position change |
| Variable | **level** extremity of the position | **change** in the position |
| Trader class | primarily **managed money** / swap / concentration | primarily **commercial** hedgers |
| Horizon | 1, 5, **20** trading days | **one week** |

Pooling `|z|` means a week where managed money is extremely *long* and a week
where it is extremely *short* land in the same event bucket with the same sign of
expected return. If the true effect is signed — and the hedging-pressure
literature says it is — this specification averages it to approximately zero
**by construction**, regardless of whether the effect exists.

Kang, Rouwenhorst and Tang (2020) make this concrete: they decompose position
variation into a short-horizon component driven by non-commercial liquidity
provision and a long-horizon component driven by commercial hedging demand, and
find these two components have **contrasting directional effects** on expected
returns. A two-sided, level-based, class-pooled specification is close to a
worst case for detecting that structure.

We tested what the product shipped. We did not test what the literature endorses.

### 7.3 Sample and coverage

- **19 contracts**, all commodity futures. No financials, no currencies, no
  equity index. Nothing here generalises beyond commodities.
- **Approximately three years** of overlapping positioning and price coverage
  (2023-04 to 2026-06) — one macro regime, with a strong metals uptrend inside it.
- **33.0% of computable events dropped** for want of a price series. The dropped
  window is systematic (2022-01 to 2023-04, plus a 2026 tail), not random, so
  this is a coverage restriction rather than a random-missingness assumption.
- **Front-month price series** are used without an explicit roll-return
  adjustment. At a 20-day horizon this is a real source of noise in the return
  measurement.

### 7.4 Two residual leakage exposures we could not fully eliminate

Named because a reviewer would find them.

1. **Non-Tuesday as-of dates.** 3 of the 334 distinct as-of dates in the raw
   table are Mondays (2020-12-21, 2023-07-03, 2025-11-10) rather than Tuesdays.
   For those, `observed_at + 3 days` lands on a Thursday, one day before a normal
   Friday release, so up to one day of lookahead is possible. This touches **10
   event-observations across the whole study and 0 in the headline cell** — but
   note that our internal note previously asserted *every* as-of date was a
   Tuesday, and that was wrong.
2. **The three-day lag is a constant, not a per-report release timestamp.** We do
   not consume the CFTC's actual release-time metadata. Where a release was
   delayed past Friday (US federal holidays, the 2018-2019 shutdown backlog), a
   constant lag would understate the true delay. Both exposures push toward false
   *positives*, and we found none — so they cannot explain the null. They would
   matter to anyone reusing this harness to chase a positive.

### 7.5 Statistical fine print

- The p-value is a percentile bootstrap of the event mean against the population
  mean **treated as known without error**. With `n_pop` ~ 1,578 the baseline's own
  sampling error is small relative to the event mean's, but it is not zero, so
  the reported p-values are mildly anti-conservative.
- **The event set is a subset of the population.** In the best cell, events are
  7.79% of the population, so the baseline is contaminated with the very
  observations being tested. Correcting for this would *raise* the measured edge
  from 2.178% to 2.362% — the contamination is conservative, but it is there.
- **The confidence intervals use a circular block bootstrap; the p-values do
  not.** The p-value resamples i.i.d. That mismatch is precisely why Section 6
  exists, and Section 6 is the number we would defend.
- `B = 2,000` bootstrap draws makes p-values granular near zero. Re-running the
  best cell at `B = 20,000` gives p = 0.0031 rather than 0.002; the BH conclusion
  is unchanged.
- The z-score uses an expanding window with a 20-point minimum. Early-sample
  z-scores are estimated on as few as 20 prior observations and are noisy.

---

## 8. Relation to the literature

### 8.1 Our null replicates the published null

Sanders, Irwin and Merrin (2009) run bivariate Granger-causality tests on COT
trader positions across **ten agricultural futures markets** and conclude there is
"very little evidence that traders' positions are useful in forecasting (leading)
returns," while finding substantial evidence that traders *respond* to prices —
non-commercials in particular behaving as trend followers.

Their paper reports individual test p-values without any multiple-testing
correction. Applying Benjamini-Hochberg to the 30 published p-values from their
positions-to-returns direction (10 markets x 3 trader categories) leaves
**0 of 30 significant at alpha = 0.05, with a minimum adjusted p of 0.105**.

Set that beside our own minimum adjusted p of **0.102** and the picture is
consistent: on this class of specification, at this kind of sample size, neither
their data nor ours produces a finding that survives correction. Our contribution
is not overturning a positive result — it is that the published null holds up
under a stricter procedure, on different markets, in a different decade, with
publication lag explicitly handled.

*Note for replicators:* the paper is open access
(DOI [10.22004/ag.econ.54547](https://doi.org/10.22004/ag.econ.54547), full text
at `ageconsearch.umn.edu/record/54547`). The BH re-run is a one-liner —
`multipletests(published_p_values, alpha=0.05, method="fdr_bh")` — so the 0-of-30
figure takes about a minute to verify against their tables. We reproduce their
p-values here by reference rather than by transcription; check them yourself
before citing this claim.

### 8.2 Where the effect actually lives, if it lives anywhere

Kang, Rouwenhorst and Tang (2020, *Journal of Finance*) is the strongest modern
evidence that positioning carries information. Their finding is *structural*:
short-horizon position variation reflects non-commercial liquidity provision,
long-horizon variation reflects commercial hedging demand, the two have opposite
directional effects on expected returns, and the liquidity-provision income
largely offsets the insurance premium commercials pay.

Nothing in our study contradicts that, because our specification cannot see it.
A study designed to test it would use **signed** changes in **commercial** net
positions at a **one-week** horizon, decomposed into short- and long-horizon
components. That is a different paper, and one worth running.

### 8.3 Multiple testing in this field is not optional

Harvey, Liu and Zhu (2016) argue that most claimed cross-sectional return
findings are likely false because of the sheer number of specifications tested,
and propose a t-statistic threshold above 3.0 for a new factor. Our best
uncorrected t-equivalent, under week clustering, corresponds to p = 0.010 — well
short of that bar, before any correction. Any COT study that reports a single
"best" specification without disclosing the search that produced it should be
read with this in mind. We tested 54 cells and are telling you all 54.

---

## 9. Replication

### 9.1 Reproducing our exact numbers

The study is one read-only script. It queries a local SQLite pipeline database
built by this repository's collectors; the database is not distributed (it is
150 MB and contains 34 unrelated sources), but every input to it is free and
keyless.

```
git clone <this repository>
cd tirramind
git checkout 85d86b0
.venv/bin/python scripts/cftc_event_study.py
```

The script prints, in order: the 19-pair join table with per-pair point counts,
the attrition counts, the leakage audit with sample entry timestamps, and the
full 54-row results table with BH-adjusted p-values. Every number in Sections 3
and 5 of this document appears verbatim in that output.

### 9.2 Reproducing the study from scratch, without our database

You do not need our infrastructure. You need:

1. **Positioning.** Download `fut_disagg_txt_{year}.zip` from
   `https://www.cftc.gov/files/dea/history/` for 2022-2026 (public domain, no
   key). Filter to the 19 `CFTC_Contract_Market_Code` values in Section 3.2, and
   derive the nine fields with the formulas in Section 3.3.
2. **Prices.** Daily closes for the 19 tickers in Section 3.2 over the same
   window, from any free source.
3. **Method.** Implement Section 4 exactly: expanding-window z with a 20-point
   floor and `x[:-1]` as history; entry at the first close at or after
   as-of + 3 days with a 7-day tolerance; forward log returns at 1, 5, 20
   trading days; unconditional-return baseline; bootstrap p; BH across all
   testable cells.

Your event counts will differ from ours to the extent your price coverage
differs — recall that a third of our events were dropped for price-coverage
reasons, and that attrition is entirely a property of the price series you pick.
If you have price history back to 2022 or earlier, you will have a **larger and
better-powered** sample than we did, and we would genuinely like to know what
you find.

### 9.3 The specification we would run next

If you want to spend the effort well, do not re-run ours. Run the one the
literature endorses and we did not test:

- **signed** (not absolute) z on **changes** (not levels)
- **commercial / producer-merchant** net positions
- **one-week** forward horizon
- across as many contracts and years as you can join
- with publication lag handled as in Section 2.1, BH across every cell you test,
  and week-clustered resampling as in Section 6

That test would be informative whichever way it came out. Ours was not.

---

## 10. Conclusion

Across 19 commodity futures, 9 CFTC positioning fields, 3 horizons and 2 z-score
thresholds — 51 testable hypotheses in one Benjamini-Hochberg family — **no
specification survives correction**, with or without clustering for the fact that
correlated contracts fire in the same week. The smallest adjusted p-value is
0.102 under i.i.d. resampling and 0.294 under week clustering.

We used this to stop selling a product whose central claim was that these
anomalies precede price moves. For that decision the result was sufficient: an
underpowered null is not a licence to make a positive claim, and we could not
find the effect we were charging for.

For the broader question, the result is close to uninformative, and we will not
pretend otherwise. The test had roughly 10% power and used a two-sided,
level-based specification that cancels the signed, change-based effect the
literature actually reports. **The correct reading of this paper is "we could not
detect it, and we would not have expected to" — not "it is not there."**

The finding we will defend is narrower and more useful: **the publication-lag
handling, the unconditional-return null, the family-wide correction and the
week-clustered resampling are each individually capable of turning an apparent
COT edge into nothing.** If a COT study reports an edge and does not describe all
four, that is where to look first.

### Reproduce with

```
.venv/bin/python scripts/cftc_event_study.py
```

---

## References

- Benjamini, Y., and Y. Hochberg (1995). "Controlling the False Discovery Rate: A
  Practical and Powerful Approach to Multiple Testing." *Journal of the Royal
  Statistical Society, Series B* 57(1): 289-300.
  DOI [10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
- Harvey, C. R., Y. Liu, and C. Zhu (2016). "... and the Cross-Section of Expected
  Returns." *Review of Financial Studies* 29(1): 5-68.
  DOI [10.1093/rfs/hhv059](https://doi.org/10.1093/rfs/hhv059)
- Kang, W., K. G. Rouwenhorst, and K. Tang (2020). "A Tale of Two Premiums: The
  Role of Hedgers and Speculators in Commodity Futures Markets." *Journal of
  Finance* 75(1): 377-417.
  DOI [10.1111/jofi.12845](https://doi.org/10.1111/jofi.12845)
- Lahiri, S. N. (1999). "Theoretical Comparisons of Block Bootstrap Methods."
  *Annals of Statistics* 27(1).
  DOI [10.1214/aos/1018031117](https://doi.org/10.1214/aos/1018031117)
- Sanders, D. R., S. H. Irwin, and R. P. Merrin (2009). "Smart Money: The
  Forecasting Ability of CFTC Large Traders in Agricultural Futures Markets."
  *Journal of Agricultural and Resource Economics* 34(2): 276-296.
  DOI [10.22004/ag.econ.54547](https://doi.org/10.22004/ag.econ.54547) (open
  access)

---

## Appendix A — Corrections to the internal research note

`docs/research/cftc_forward_return_event_study.md` is the working note this paper
is built from. Two of its statements did not survive re-verification and are
corrected here. Both corrections make the result *weaker*, not stronger.

1. **Distinct-week count for the best cell.** The note states that the 123 events
   in the `mm_net_pct_oi`, `|z| >= 2`, 20-day cell "fall on only 116 distinct
   weeks." That figure belongs to the **251 pre-filter** events. The **123
   labelled** events fall on **67** distinct as-of weeks. Clustering is therefore
   materially worse than the note claimed, which is why Section 6 re-runs every
   cell at the week level.
2. **"Every as-of date is a Tuesday."** 331 of 334 distinct as-of dates are
   Tuesdays; 3 are Mondays. See Section 7.4 for the bounded exposure this creates.

Both were found by re-deriving the note's numbers from the database rather than
copying them forward. Everything else in the note reproduced exactly against
commit `85d86b0`.

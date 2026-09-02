---
title: "Interconnection Queue Attrition"
tags:
  - doc/research
  - topic/interconnection
  - topic/energy
  - status/active
---

# Interconnection Queue Attrition

## Question

Every forecast of the AI datacenter buildout quotes the interconnection queue —
466 GW waiting in ERCOT, 699 GW in MISO — as though it were a pipeline. Does a
request in that queue predict generation that actually gets built?

## Finding

No, and the gap is large enough to be the product. Across 10,996 requests with
known outcomes from MISO, NYISO, CAISO and ISO-NE, **17–21% of resolved
requests were built.** Attrition is monotone in size above 100 MW, and requests
of 1,000 MW or more — the entire AI datacenter story — were built **5.3% of the
time** (n=169, Wilson 95% CI 2.8–9.8).

It is a range rather than a point because naive built / (built + withdrawn)
gives 14.2%, which records censoring as failure: a 2024 request has not had
time to be built. Restricting to queue-year cohorts that are ≥90% resolved
gives the 17.2–21.4% band.

## Why it is defensible as a business

Grid operators publish current queue state and overwrite it in place. Nobody
retains yesterday's value, so revision history can only be accumulated forward
— it cannot be bought or back-scraped. A daily snapshot archive started now is
an asset that gets more valuable and cannot be caught up to.

It also sits outside the SEBI research-analyst perimeter that kills a
securities-prediction product: this is physical infrastructure, not advice on
securities.

## Method notes and traps hit

- **Mixed date formats** destroyed 84% of queue dates on the first pass. Four
  ISOs write dates four ways; inferring one format coerced the rest to NaT,
  indistinguishable from ordinary missingness. Caught only by cross-tabbing
  missingness against outcome *before* quoting any cohort.
- **Unnormalised fuel labels** invented a finding — "Solar 32.1%" vs
  "Photovoltaic 13.0%" measured ISO composition, not technology.
- **Queue ID is not a primary key** in three of four ISOs. NYISO reuses "0031"
  across an Astoria phase that completed and one that withdrew.
- **Non-stationarity was tested and ruled out**: build rate is flat, 22.2%
  before 2012 to 21.3% after 2017.

## Model

`logistic-v1`, validated walk-forward by queue vintage and scored against the
base rate rather than against zero: Brier 0.1540 vs 0.1668, **+7.7% skill.**
Calibration is the open problem — only 2 of 7 bins cover, and the direction is
over-spreading: above 50% the model says 65 and means 49. It ranks; it does not
yet quantify. Isotonic recalibration fixed one bin and cost skill, so this is
not a mapping problem.

## Portfolio risk

Outcomes are not independent, and this was measured before it was modelled.
Against a permutation null that keeps every predicted probability and every
group size and shuffles only membership, every grouping level clusters
significantly (p < 0.00025 at the 4,000-permutation floor): fuel 4.09x, state
2.62x, ISO study cycle 1.88x, county 1.80x, ISO 1.76x, transmission owner
1.72x. **Technology is a stronger shared shock than geography** — tariffs, cell
prices and turbine lead times hit every project of a type at once.

The permutation null was necessary rather than decorative: the underlying model
is miscalibrated, miscalibration inflates raw dispersion, and the permuted
nulls come back at 1.19–8.77 rather than the 1.0 a chi-square table assumes.

Modelled as a random-intercept logit fitted by marginal ML with Gauss–Hermite
quadrature; study cycle tau = 1.06, ICC 25.5%. Structure chosen by out-of-sample
portfolio coverage, not by largest tau. On a book concentrated in one
technology, independence's 80% interval covers only **62.6%** of the time
against 81.5% with shared shocks. On the live book the spread is **4.3x wider**
and the 1-in-10 case **21.6 GW worse**.

Caveat carried on the page itself: calibrating on projects is not calibrating
on capacity (+20.8% capacity bias, +6.1% after MW-weighted calibration), and the
live book is 68% MISO by capacity against 19% in training. The dispersion is a
ratio and survives both; the central estimate does not.

## Where it lives

Code, data and the snapshot archive: `savabs/queue_attrition`.
Published base rates: <https://tirramind.com/queue>.
Open forward ledger: <https://tirramind.com/predictions>.
Portfolio risk: <https://tirramind.com/portfolio>.

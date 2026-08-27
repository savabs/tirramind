---
title: "Spec: Ship the $19 tier as integration infrastructure"
tags:
  - doc/spec
  - topic/product
  - topic/api
  - status/active
date: 2026-08-27
---

# Spec: Ship the $19 tier as integration infrastructure

Research: [[tier_ladder_inversion]]

Positioning fixed by the owner 2026-08-27: **we sell the integration** — the
maths, finance, systems and architecture composed into one surface. Not
predictions. The $19 tier therefore becomes the entry rung of an API ladder,
with a digest as a convenience layer on top, rather than a standalone
newsletter.

## Step 1 — Make tier access monotonic in price

`agent/brief_server.py:52-54`. Replace the three ad-hoc sets with a ladder in
which each tier reaches everything below it:

```python
_BRIEF_TIERS         = {"brief", "scheduler", "entity", "data"}
_SCHEDULER_TIERS     = {"scheduler", "entity", "data"}
_ENTITY_GRAPH_TIERS  = {"entity", "data"}
_DATA_PLATFORM_TIERS = {"data"}
```

Gate `/brief.*` on `_BRIEF_TIERS` instead of `_valid_key`, so the $19 tier
becomes a real tier rather than a free add-on.

**Consequence to state plainly:** `scheduler` ($50) loses entity-graph and
data-platform access it has today. No subscriber exists, so nobody is affected.

Verifiable: a table test asserting, for every ordered pair of tiers, that the
higher-priced tier's reachable set is a superset of the lower's.

## Step 2 — Retire the two false claims

`products/brief_subscription/pricing.html:283,285`.

- "a Beta-Binomial model that learns win probability as outcomes accumulate" —
  `.tirra_opportunities/win_learner.jsonl` does not exist, so
  `WinProbabilityLearner._load` returns immediately and `probability_of`
  returns the bare prior `0.5/(0.5+1.0) = 0.3333` for every award. Rewrite to
  describe an uninformed prior that will learn once bid outcomes are recorded.
- "Federal contract opportunities" — `agent/tools/gov_contracts.py:52` calls
  USASpending `spending_by_award`, which returns contracts that have **already
  been awarded**, with the winner named. Say "recent federal awards".

Verifiable: grep for "learns"/"opportunities" in `pricing.html`; every survivor
must map to code that does it.

## Step 3 — Fix the field mappings that discard 168,275 observations per run

`scripts/live_intelligence_digest.py:33-39`. `_SCORABLE` declares field names
that do not exist in the stored payloads:

| declared | actual field | rows scanned | series built |
|---|---|---|---|
| `volatility`, `value` | `realized_vol_20d`, `intraday_range` | 4,294 | 0 |
| `tvl_change`, `value` | `tvl_usd` | 162,251 | 0 |
| `probability`, `value` | `yes_price`, `no_price` | 8,730 | 0 |
| `mm_net`, `open_interest` | (correct) | 5,488 | 80 |

This is why the digest is 100% CFTC. Correcting three strings adds 89
instrument-volatility series, 50 DeFi TVL series (median 3,214 points each) and
1,493 prediction markets to the same z-score + BOCPD machinery already trusted.

This is the single highest-value change in the spec, and it is exactly the
"integration" being sold.

Verifiable: generate a digest and assert findings from **more than one**
source.

## Step 4 — Render what is already computed

`scripts/intelligence_brief.py:103-138`. `render_markdown` drops `entity_id`,
`latest_value`, `n_points` and `flagged_ts`, which
`live_intelligence_digest.py:114-122` already computes. Today a subscriber sees:

```
`cftc` futures_positioning · mm_net → z=+3.09
```

when the same run knows:

```
COTTON NO. 2 — ICE Futures U.S. · open interest
z=+3.14 vs 169-week baseline · structural break 6 weeks ago
```

Also add a date and an edition id — the rendered output currently contains
neither, so two editions are indistinguishable. The delivery log shows checksum
`5f0d435ed7c9ebc0` delivered three times.

Verifiable: an edition rendering five identical `P(win)=33%` rows and eight
anomalies from one source is a FAIL. Assert distinct sources and a rendered
date.

## Step 5 — Reconcile cadence, then schedule

Copy promises weekly in four places (`index.html:283`, `pricing.html:8`,
`:183`, `terms.html:19`, `README.md:18`). `deploy/systemd/tirra-brief.timer` is
`OnCalendar=*:0/30` — every 30 minutes, overwriting one mutable file. Set it to
weekly and write a dated archive.

Add `tirra-brief.timer` to `deploy/provision_vm.sh` — it appears nowhere in
that script, not even in the explicit "skip this one" note that
`tirra-collect.timer` gets, which is why production `total_deliveries` is 0.

Before scheduling, remove the redundant fetch: `build_brief()`
(`intelligence_brief.py:50-51`) re-fetches what `refresh_fast_data` already
pulled (`tirra_engine.py:53-56`), against a `TimeoutStartSec=120` oneshot.

## Step 6 — Email delivery

`agent/payments/delivery.py` exists as deliberate inert groundwork and sends
only API keys. Add a digest sender beside it, still disabled. Add `last_sent_at`
and an unsubscribe token to the subscriber record — neither exists.

Blocked on the owner: Cloudflare Email Routing plus an outbound sender.

## Out of scope

Predictive-edge validation, the GNN retrain, SAM.gov solicitations. Step 2
makes the copy honest without any of them.

## Related

- [[tier_ladder_inversion]]
- [[deep_intelligence_roadmap]] — argued the $19 brief is "too thin to justify
  a subscription" under a *content* thesis. The integration thesis changes the
  question, not that analysis.

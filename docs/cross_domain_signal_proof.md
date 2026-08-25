---
title: "Cross-Domain Signal Proof — the Tender Alpha moat"
tags:
  - doc/research
  - topic/signals
  - topic/cross-domain
  - phase/1
  - status/active
---

# Cross-Domain Signal Proof

**Date:** 2026-08-23
**Purpose:** prove the cross-domain signal engine actually works on real data — the mechanism that would make a contract-intelligence product defensible ("the same company appears in contracts *and* shipping *and* filings").

## What runs

TirraMind's pipeline DB (`.tirra_pipeline/pipeline.db`, 124 MB) contains **real accumulated data**:

| Asset | Count |
|---|---|
| Entity observations | **353,064** (all with timestamps) |
| Unique entities | **3,195** |
| Entity links | **12,271** |
| Signal source tools with data | **22** |

## Proof 1 — Cross-domain links with real names

The `entity_links` table holds real, typed links across domains:

```
Reyes Javier A            --[works_for]-->  City
DONGFANG LIU              --[works_for]-->  nuvation bio
United Kingdom ETF        --[tracks_issuer]-->  blackrock
US country                --[exchange_country]-->  (many instruments)
```

## Proof 2 — One entity, many independent signal domains

The **US country entity** is observed simultaneously across 5 independent macro signals:

| Signal tool | Observation type | Obs count |
|---|---|---|
| `sovereign_debt` | sovereign_yield | 966 |
| `energy_supply` | petroleum_inventory | 352 |
| `global_pmi` | economic_activity | 4 |
| `capital_flows` | capital_flow | 1 |
| `central_bank_balance` | cb_policy_rate / cb_balance_sheet | 2 |

This is the core moat mechanism: the *same real-world thing* producing signal from multiple independent data domains in the same window.

## Proof 3 — Government contract recipients ARE cross-domain linked

The **Tender Alpha base** already connects to the entity graph. 12 distinct contract-recipient companies exist as entities, including house-hold names:

```
sikorsky aircraft, fluor intercontinental, clark construction,
georgia tech applied research, lawrence livermore national security,
rogue valley h2o, the regents of the university of california, ...
```

And these contract companies carry **24 cross-domain links** of types `awarded_by`, `operates_in` — i.e., the graph already knows who awards them and where they operate, ready to connect to shipping/filings/macro signals.

## Honest gap (the real build for Tender Alpha)

- **Company-level temporal overlap** (e.g., "a contract recipient company also has a vessel in port / an insider filing / a GDELT conflict event within 30 days") is **NOT yet populated** — the `detect_insider_gdelt` / `detect_vessel_sanctions` runs returned 0 for current entity coverage.
- This is the missing piece that would make the product genuinely cross-domain. It is **exactly what Tender Alpha should build**: connect `gov_contracts` recipients → same company across `ais_vessel`, `insider_filings`, `gdelt`, `form144`.

So: the **machinery is proven real** (12,271 links, cross-domain footprint, contract entities linked). The **product-relevant overlap is the unfilled build**.

## Command reference (how this was verified)

```python
from agent.pipeline.store import PipelineStore
s = PipelineStore('.tirra_pipeline/pipeline.db')
# link types + counts, entity overlaps, contract-recipient entities
```

## Related
- [[signals_primer]]
- [[checkpoint_2026-08-23_foundation_verified]]
- [[revenue_plan_2026-05-08]]
- [[quant_training_ground]]
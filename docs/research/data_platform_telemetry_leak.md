---
title: "Research: Data Platform API leaks internal DAG telemetry"
tags:
  - doc/research
  - topic/security
  - topic/api
  - status/active
date: 2026-08-27
---

# Research: Data Platform API leaks internal DAG telemetry

## 1. The leak, confirmed live

`agent/pipeline/store.py`'s `pipeline_data` table is shared, untyped, and
written to by two unrelated producers under the same `source` column:

- Genuine Layer 1 external data (`cftc`, `finra`, `gdelt`, ...) — what the
  Data Platform tier ($500/mo) markets and sells.
- Internal DAG-stage execution telemetry, stored by pipeline operators under
  their own node/operator name as `source` (`train_gnn`, `gnn_inference`,
  `score_entities`, `generate_features`, `run_detection`, `scan_adversarial`,
  `sac_inference`, `train_rl_policy`, `emit_portfolio`, `update_beliefs`,
  `load_models`, `component_perf_gnn_epochs`).

`agent/brief_server.py`'s `/api/v1/sources` and `/api/v1/data` endpoints read
this table with no distinction between the two. Verified directly against the
live `.tirra_pipeline/pipeline.db`:

```
GET /api/v1/data?source=train_gnn
→ {"trained": false, "loss_ewc": 579753920.0, ...}
```

A paying Data Platform subscriber could read the model's own untrained-state
defect telemetry through the API they pay for.

## 2. Root cause

No `is_internal` / `is_customer_facing` flag exists anywhere on
`pipeline_data`. The serving layer (`brief_server.py`) trusted the store's
`list_sources()` catalog as customer-safe without knowing which sources are
DAG-internal bookkeeping vs. genuine fetched data.

## 3. Fix (this pass)

Denylist at the serving layer, not the store — `_INTERNAL_TELEMETRY_SOURCES`
in `agent/brief_server.py`, filtered out of both `/api/v1/sources` and
`/api/v1/data`. Chosen over a store-level schema change because:

- It's the customer-facing boundary, matching the pattern already used for
  `_ENTITY_GRAPH_TIERS` gating.
- A store-level flag would require a migration and touch ~40 existing
  internal callers for no additional safety at this scope.

## 4. What's still needed (not done here)

A structural fix — e.g. a `source_kind` column set at write time — so a new
internal DAG stage can't reintroduce this by omission. The denylist is a
manual list that must be updated whenever a new internal DAG stage is added;
flagged here so it isn't forgotten silently.

## Related

- [[entity_graph_tier_mismatch]] — same audit pass, same class of issue
  (customer-facing over-exposure of internal store contents).

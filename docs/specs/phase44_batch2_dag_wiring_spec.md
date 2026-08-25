---
title: "Spec: Phase 44 — Batch 2 DAG Wiring"
tags:
  - doc/spec
  - phase/44
  - topic/pipeline
  - topic/diversity
  - layer/surveillance
---

# Spec: Phase 44 — Batch 2 DAG Wiring

## Goal

Wire 5 more L2-ready surveillance tools into `daily_collection`. No new tool code required.

## Files Affected

1. `agent/pipeline/dags/daily_collection.py` — add 5 Phase 44 nodes
2. `tests/test_pipeline_registry.py` — update 3 count assertions (22→27) + add 5 node config tests

## Implementation Steps

### 44.1 — Add Phase 44 block to daily_collection.py

Insert immediately before the `fetch_macro` node (matching Phase 43 pattern).

```python
# ═══════════════════════════════════════════════════════════════
# Phase 44 — Batch 2 Entity DAG Wiring
# ═══════════════════════════════════════════════════════════════

dag.add(
    "fetch_regulatory_gazette",
    operator="regulatory_gazette",
    table_name="regulatory_gazette",
    params={"days_back": 7, "limit": 50},
    timeout=120,
    retries=2,
)

dag.add(
    "fetch_form144",
    operator="form144",
    table_name="form144",
    params={"days_back": 14},
    timeout=180,
    retries=2,
)

dag.add(
    "fetch_supply_chain",
    operator="supply_chain_monitor",
    table_name="supply_chain_monitor",
    params={"mode": "producer_prices"},
    timeout=120,
    retries=2,
)

dag.add(
    "fetch_political_risk",
    operator="political_risk",
    table_name="political_risk",
    params={"mode": "candidates"},
    timeout=120,
    retries=2,
)

dag.add(
    "fetch_comtrade",
    operator="comtrade",
    table_name="comtrade",
    params={"mode": "partners", "reporter": "USA"},
    timeout=120,
    retries=2,
)
```

Notes:
- `form144` timeout=180: SEC EDGAR can be slow under load; matches `insider_filings` precedent
- All retries=2 consistent with Phase 43 pattern

### 44.2 — Update test_pipeline_registry.py count assertions

Three assertions change `22 → 27`:
- `test_node_count`: `assert len(dag.nodes) == 22` → `== 27`
- layer 0 assertion: `assert len(layers[0]) == 22` → `== 27`
- roots assertion: `assert len(dag.roots()) == 22` → `== 27`

### 44.3 — Add 5 per-node config tests

Append to the `TestDailyCollectionNodes` class, after the last Phase 43 test:

```python
def test_fetch_regulatory_gazette_config(self, dag):
    n = dag.nodes["fetch_regulatory_gazette"]
    assert n.operator == "regulatory_gazette"
    assert n.params["days_back"] == 7
    assert n.params["limit"] >= 25
    assert n.timeout > 0
    assert n.retries >= 1

def test_fetch_form144_config(self, dag):
    n = dag.nodes["fetch_form144"]
    assert n.operator == "form144"
    assert n.params["days_back"] == 14
    assert n.timeout > 0
    assert n.retries >= 1

def test_fetch_supply_chain_config(self, dag):
    n = dag.nodes["fetch_supply_chain"]
    assert n.operator == "supply_chain_monitor"
    assert n.params["mode"] == "producer_prices"
    assert n.timeout > 0
    assert n.retries >= 1

def test_fetch_political_risk_config(self, dag):
    n = dag.nodes["fetch_political_risk"]
    assert n.operator == "political_risk"
    assert n.params["mode"] == "candidates"
    assert n.timeout > 0
    assert n.retries >= 1

def test_fetch_comtrade_config(self, dag):
    n = dag.nodes["fetch_comtrade"]
    assert n.operator == "comtrade"
    assert n.params["mode"] == "partners"
    assert n.params["reporter"] == "USA"
    assert n.timeout > 0
    assert n.retries >= 1
```

## Edge Cases

- `regulatory_gazette` Federal Register API may be slow/unavailable → retries=2 handles
- `form144` SEC EDGAR rate limit 10 req/s → tool handles internally with backoff
- `supply_chain_monitor` BLS API may return 429 → tool returns partial result, non-fatal
- `political_risk` FEC API has paging; `candidates` mode fetches recent cycle → bounded
- `comtrade` UN Comtrade free tier may return 429 on heavy use → cache TTL avoids re-fetching

## Testing Plan

1. `pytest tests/test_pipeline_registry.py -v` — all assertions pass at 27 nodes, 5 new tests pass
2. Full regression: `pytest tests/ -x -q` — 0 new failures vs Phase 43 baseline (23 pre-existing failures)
3. Smoke: `python -c "from agent.pipeline.dags.daily_collection import build_daily_collection_dag; d=build_daily_collection_dag(); print(len(d.nodes))"`

## Related

- [[phase44_batch2_dag_wiring]]
- [[phase44_batch2_dag_wiring_task]]
- [[phase43_high_volume_dag_wiring_spec]]

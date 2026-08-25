---
title: "Spec: Phase 43 — High-Volume DAG Wiring"
tags:
  - doc/spec
  - phase/43
  - topic/pipeline
  - topic/diversity
  - layer/surveillance
---

# Spec: Phase 43 — High-Volume DAG Wiring

## Goal

Wire four already-L2-ready, high-volume surveillance tools into the `daily_collection` DAG to
accelerate entity observation accumulation and begin closing the entropy gap. No new tool code
required — DAG nodes + test updates only.

## Files Affected

1. `agent/pipeline/dags/daily_collection.py` — add 4 Phase 43 nodes
2. `tests/test_pipeline_registry.py` — update 3 count assertions (18→22) + add 4 node config tests

## Implementation Steps

### 43.1 — Add `fetch_ais_vessel` node

Location: after the Phase 42 `fetch_lobbying` block, before `fetch_macro`.

```python
dag.add(
    "fetch_ais_vessel",
    operator="ais_vessel",
    table_name="ais_vessel",
    params={"mode": "area", "area": "full_baltic", "limit": 500},
    timeout=180,
    retries=2,
)
```

- `full_baltic` covers 18K+ vessel source pool
- Persists vessel entities with `vessel_position` obs, up to 500/run
- timeout=180 accounts for metadata enrichment API calls per vessel
- retries=2 matches all other Phase 42 nodes

### 43.2 — Add `fetch_gov_contracts` node

```python
dag.add(
    "fetch_gov_contracts",
    operator="gov_contracts",
    table_name="gov_contracts",
    params={"mode": "recent", "limit": 100},
    timeout=120,
    retries=2,
)
```

- `recent` mode returns latest 100 award records
- Persists company + organization entities with `contract_award` obs

### 43.3 — Add `fetch_sanctions_monitor` node

```python
dag.add(
    "fetch_sanctions_monitor",
    operator="sanctions_monitor",
    table_name="sanctions_monitor",
    params={"mode": "recent", "days_back": 90, "limit": 100},
    timeout=120,
    retries=2,
)
```

- `recent` mode scans OFAC + UN for designations in last 90 days
- Persists person/company entities with `sanctions_listing` obs

### 43.4 — Add `fetch_patent_filings` node

```python
dag.add(
    "fetch_patent_filings",
    operator="patent_filings",
    table_name="patent_filings",
    params={"mode": "search", "cpc_class": "G06N", "limit": 50},
    timeout=120,
    retries=2,
)
```

- AI/ML patent class; creates company entities for top tech assignees
- `search` is the only mode that calls `_persist_entities`

### 43.5 — Update test_pipeline_registry.py

Update three hardcoded count assertions:
- `test_node_count`: `== 18` → `== 22`
- layer 0 assertion: `len(layers[0]) == 18` → `== 22`
- roots assertion: `len(dag.roots()) == 18` → `== 22`

Add 4 per-node config tests (matching Phase 42 pattern):
- `test_fetch_ais_vessel_config`
- `test_fetch_gov_contracts_config`
- `test_fetch_sanctions_monitor_config`
- `test_fetch_patent_filings_config`

Each test verifies: node exists, operator matches, required params present, timeout/retries set.

## Edge Cases

- `ais_vessel` area mode may return 0 vessels if AIS feed is temporarily unavailable → retries=2 handles
- `gov_contracts` USASpending.gov may time out under load → timeout=120 is conservative; tool has cache
- `sanctions_monitor` OFAC CSV parse may fail on unexpected XML → tool catches exceptions, returns partial
- `patent_filings` USPTO PatentsView returns 400 on malformed CPC class → `G06N` is a valid class

## Testing Plan

1. Run `pytest tests/test_pipeline_registry.py -v` — all count assertions must pass at 22
2. Confirm 4 new node config tests pass
3. Run full regression: `pytest tests/ -x -q` — no new failures
4. Smoke test (optional): `python scripts/run_collection.py --dry-run` if dry-run flag exists,
   or check that `build_daily_collection_dag()` returns a 22-node DAG without error

## Related

- [[phase43_high_volume_dag_wiring]]
- [[phase43_high_volume_dag_wiring_task]]
- [[phase42_entity_diversity_expansion_spec]]

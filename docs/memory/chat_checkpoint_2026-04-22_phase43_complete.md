---
title: "Checkpoint: Phase 43 Complete — High-Volume Entity DAG Wiring"
tags:
  - doc/checkpoint
  - phase/43
  - topic/pipeline
  - topic/diversity
  - layer/surveillance
  - status/done
---

# Checkpoint: Phase 43 Complete

**Date:** 2026-04-22  
**Session summary:** Verified Phase 42 regression finding, corrected checkpoint, then implemented Phase 43.

---

## What Was Done

### Phase 42 Checkpoint Correction (pre-work)
The Phase 42 checkpoint falsely claimed "tool coverage is no longer the bottleneck." Verified live:
- **35 of 51 tools (68.6%) are unwired from the DAG** — this remains the primary bottleneck
- `job_postings` is JOLTS/BLS aggregate L1 macro data, NOT entity-level employer records (common misclassification trap)
- Correction appended to `[[chat_checkpoint_2026-04-21_phase42_complete]]`

### Phase 43 — 4 New DAG Nodes
Added 4 high-volume, L2-ready, already-registered tools to `agent/pipeline/dags/daily_collection.py`:

| Node ID | Operator | Params | Est. entities/run |
|---|---|---|---|
| `fetch_ais_vessel` | `ais_vessel` | `mode=area, area=full_baltic, limit=500` | 500+ vessel_position obs |
| `fetch_gov_contracts` | `gov_contracts` | `mode=recent, limit=100` | 100 company + org nodes |
| `fetch_sanctions_monitor` | `sanctions_monitor` | `mode=recent, days_back=90, limit=100` | 100 person/company nodes |
| `fetch_patent_filings` | `patent_filings` | `mode=search, cpc_class=G06N, limit=50` | 50 company nodes |

DAG node count: **18 → 22**.

### Tests
- `test_pipeline_registry.py`: 45 → **49 PASS** (3 count assertions updated, node name set updated, 4 per-node config tests added)
- `test_defi_flows_edge.py` bonus fix: `cache.set.assert_called()` → `cache.put.assert_called()` — Phase 42 updated defi_flows.py to use the correct `DataCache.put()` API but the test was never updated.
- No new tool code needed — all 4 tools already L2-ready and wired into `cli.py` with `pipeline_store`.

### No schema changes
All observation types used by the 4 tools (`vessel_position`, `contract_award`, `sanctions_hit`, `patent_filing`) were already in the entity graph schema.

---

## Codebase State After Phase 43

### `agent/pipeline/dags/daily_collection.py`
22 nodes (single parallel layer). Phase 43 block inserted immediately before `fetch_macro`:
```python
# ═══════════════════════════════════════════════════════════════
# Phase 43 — High-Volume Entity DAG Wiring
# ═══════════════════════════════════════════════════════════════
dag.add("fetch_ais_vessel", operator="ais_vessel", table_name="ais_vessel",
    params={"mode": "area", "area": "full_baltic", "limit": 500}, timeout=180, retries=2)
dag.add("fetch_gov_contracts", operator="gov_contracts", table_name="gov_contracts",
    params={"mode": "recent", "limit": 100}, timeout=120, retries=2)
dag.add("fetch_sanctions_monitor", operator="sanctions_monitor", table_name="sanctions_monitor",
    params={"mode": "recent", "days_back": 90, "limit": 100}, timeout=120, retries=2)
dag.add("fetch_patent_filings", operator="patent_filings", table_name="patent_filings",
    params={"mode": "search", "cpc_class": "G06N", "limit": 50}, timeout=120, retries=2)
```

### Unwired tool count
- Before Phase 43: 35/51 unwired (68.6%)
- After Phase 43: **31/51 unwired (60.8%)**
- Still the primary bottleneck. See correction in [[chat_checkpoint_2026-04-21_phase42_complete]] for full priority table.

### Pre-existing test failures (safe to ignore in regression)

**Full regression result (excluding known feature_generation_dag + entity_linking files): 23 failed, 9540 passed.**

Phase 43 introduced 0 new failures (verified by git stash comparison + file diff). All 23 failures are pre-Phase-43 in origin:

| File | Failures | Origin |
|------|----------|--------|
| `tests/test_feature_generation_dag.py` | 5 | Phase 29 era (documented) |
| `tests/test_entity_linking.py` | 1 | Phase 29 era (documented) |
| `tests/test_world_model_discovery.py` | 2 | Pre-Phase-29 commit |
| `tests/test_world_model_update_fitting.py` | 1 | Pre-Phase-29 commit |
| `tests/test_walkforward_multi.py` | 2 | Post-Phase-29 uncommitted: `walkforward_runner.py` was changed to accept `instrument_daily` obs type alongside `daily_return` — `test_non_daily_return_obs_ignored` was never updated to match |
| Other files (18 remaining) | 18 | Post-Phase-29 uncommitted changes from earlier phases |

**The `test_walkforward_multi.py` failures** stem from an uncommitted change to `agent/quant/walkforward_runner.py` that added `instrument_daily` to the accepted obs types. The test expects the old `daily_return`-only behavior. Fix: update `test_non_daily_return_obs_ignored` to use a truly irrelevant obs type instead of `instrument_daily`.

---

## Key Technical Facts (for cold-start)

- **Cache API:** `DataCache.get(source, params)` / `DataCache.put(source, params, data)` — NOT `.get(url)` / `.set(url, data)`. The `set` form was the old pre-refactor API. Any test using `cache.set.assert_called()` is a bug.
- **L2 persistence pattern:** Tools with `pipeline_store` call `self._persist_entities(records)` → `self._persist_entities_inner()` → `store.add_entity()` / `store.add_observation()` / `store.link_entities()`.
- **Entity ID:** `entity_id_from_key(entity_type, key)` → SHA-256[:16].
- **DAG `add()` signature:** `dag.add(node_id, operator, table_name, params, timeout, retries)`.
- **`build_tool_registry()` in `agent/cli.py`:** Injects `pipeline_store` into all L2 tools at lines ~129–149.

---

## Next Phase Candidates

Priority: continue wiring unwired tools into the DAG. Remaining 31 unwired tools span all 7 layers.

**Highest-priority wiring candidates (L2-ready, not yet in DAG):**

1. `cert_transparency` — SSL cert issuance events for company entities (high frequency)
2. `dns_monitor` — Domain registration/change events for company entities
3. `regulatory_gazette` — Regulatory filings, entity-level
4. `political_risk` — Country-level events, could be conditioned on GNN evaluation
5. `comtrade` — Trade flow data, country-node observations

**GNN-guided decision:** After 1 week of `ais_vessel` data accumulation (~3,500+ vessel observations), re-run GNN attention diagnostics to check if vessel nodes are data-starved. That result should gate:
- Adding `fetch_ais_vessel_port_calls` (separate mode, separate node)
- Whether to prioritize more vessel data vs. a different entity type

**Historical backfill:**
- `sovereign_debt` and `central_bank_balance` historical rows still recommended (~3-8k rows)
- Low entropy impact but fills structural gaps in macro-condition nodes

---

## Artifacts Created/Modified This Session

| File | Action |
|------|--------|
| `[[phase43_high_volume_dag_wiring]]` | Created |
| `[[phase43_high_volume_dag_wiring_spec]]` | Created |
| `[[phase43_high_volume_dag_wiring]]` | Created (moved from active/) |
| `agent/pipeline/dags/daily_collection.py` | Modified — 4 nodes added |
| `tests/test_pipeline_registry.py` | Modified — 4 tests updated, 4 added |
| `tests/test_defi_flows_edge.py` | Modified — cache.set→cache.put fix |
| `[[chat_checkpoint_2026-04-21_phase42_complete]]` | Modified — correction appended |

## Related

- [[phase43_high_volume_dag_wiring]]
- [[phase43_high_volume_dag_wiring_spec]]
- [[chat_checkpoint_2026-04-21_phase42_complete]]

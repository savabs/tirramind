---
title: "Spec: Deep Surveillance Tools — Phase 10b (insider_filings L2)"
tags:
  - doc/spec
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Spec: Deep Surveillance Tools — Phase 10b.1 (insider_filings L2 Upgrade)

## Goal

Upgrade `insider_filings` from L1 (ephemeral aggregate clusters) to L2 (entity-resolved, persistent per-insider observations). After this upgrade:

1. Each insider transaction is recorded as an `entity_observation` with `depth_level=2`.
2. Companies and insiders are registered in the entity registry with CIK-based aliases.
3. Cluster deduplication uses CIK instead of fragile name matching.
4. The tool works identically when no PipelineStore is provided (backward compatible).
5. MI gain from L2 vs L1 is measurable via the depth evaluation framework.

## Files Affected

| File | Action |
|------|--------|
| `agent/tools/insider_filings.py` | **Modify** — add PipelineStore wiring, reporter_cik extraction, entity persistence, CIK dedup |
| `tests/test_insider_filings_l2.py` | **Create** — L2 upgrade tests (entity registration, observation storage, CIK dedup, backward compat) |
| `tests/test_insider_filings_mi.py` | **Create** — MI measurement integration test (L1 vs L2 depth eval) |

## Implementation Steps

### Step 10b.1.1: Add reporter_cik to transaction dicts

In `_parse_filings()`, add `reporter_cik` to the dict passed to `_parse_form4_xml()` and ensure it propagates to the returned transaction list. The reporter CIK is already extracted from EFTS `ciks[0]` but currently discarded.

Also pass `issuer_cik` through to each transaction dict (already available as local variable).

**Changes:**
- In `_parse_filings()`: add `reporter_cik = ciks[0]` to each transaction dict
- In `_parse_form4_xml()`: accept new `reporter_cik` and `issuer_cik` params, include in output dicts
- **No behavior change** — existing cluster detection ignores unknown keys.

**Test:** Verify transaction dicts have `reporter_cik` and `issuer_cik` fields.

### Step 10b.1.2: Accept optional PipelineStore in constructor

Modify `__init__()` to accept `pipeline_store: PipelineStore | None = None` alongside existing `cache` parameter. Store as `self._store`.

**Changes:**
- `__init__(self, cache=None, *, pipeline_store=None)` (keyword-only for store)
- Add `from __future__ import annotations` (already present)
- Add TYPE_CHECKING import for PipelineStore

**Test:** Construct tool with and without pipeline_store; both initialize without error.

### Step 10b.1.3: Entity registration + observation storage

Add private method `_persist_entities(self, transactions: list[dict]) -> None` that:

1. Skips if `self._store is None`
2. Collects unique companies by `issuer_cik` and unique insiders by `reporter_cik`
3. For each company: `register_entity(type="company", entity_id=entity_id_from_key("company", issuer_cik), canonical_name=normalize_company_name(company))` + `add_entity_alias(eid, "sec_cik", issuer_cik)` + `add_entity_alias(eid, "ticker", ticker)` (if available)
4. For each insider: `register_entity(type="person", entity_id=entity_id_from_key("person", reporter_cik), canonical_name=name)` + `add_entity_alias(eid, "sec_cik", reporter_cik)`
5. For each transaction: `store_entity_observation(entity_id=insider_eid, source_tool="insider_filings", observed_at=<date_as_timestamp>, observation_type="purchase", depth_level=2, value={ticker, company, shares, price, role})`
6. All wrapped in try/except to prevent persistence errors from breaking tool results

Call `_persist_entities(transactions)` in `execute()` after `_parse_filings()` returns.

**Test:** With PipelineStore, verify entities registered and observations stored. Without PipelineStore, verify no errors.

### Step 10b.1.4: CIK-based dedup in _find_best_cluster()

In `_find_best_cluster()`, change dedup logic:
- If transaction has `reporter_cik` and it's truthy, use `reporter_cik` as dedup key
- Otherwise fall back to `name.upper().strip()` (existing behavior)

This fixes the bug where two insiders with identical names would be collapsed into one.

**Test:** Two transactions with same name but different reporter_cik → counted as 2 distinct insiders. Same reporter_cik, different names → counted as 1 insider.

### Step 10b.1.5: Add entity_ids to cluster data

Enrich the cluster dicts in `_detect_clusters()` / `_find_best_cluster()` with an `entity_ids` mapping: `{insider_name: entity_id}` for each insider in the cluster. This enables downstream consumers to trace clusters to the entity graph.

Only populated when `reporter_cik` is available in the transaction dicts.

**Test:** Cluster data contains `entity_ids` dict with correct entity_id values.

### Step 10b.1.6: Edge case test suite

Comprehensive edge case tests covering:
- Malformed EFTS entries (missing CIKs, empty display_names)
- Reporter CIK is None/empty → falls back to name dedup
- Duplicate entity registration (idempotent)
- PipelineStore write failure → tool still returns ToolResult
- Unicode insider names
- Same insider filing at multiple companies
- Very large scan (boundary: 500+ transactions)
- Transaction with zero shares or zero price → still persisted as observation

### Step 10b.1.7: MI measurement integration test

Standalone test that:
1. Creates PipelineStore in `:memory:`
2. Runs insider_filings with PipelineStore to populate L2 observations
3. Creates simulated L1 observations (aggregate cluster count per day)
4. Computes conditional MI of L2 observations vs L1, against a synthetic target
5. Stores depth_evaluation result
6. Asserts MI(L2|L1) > 0 (entity-level data adds signal beyond aggregates)

## Edge Cases

1. **No PipelineStore** — tool works exactly as before, no entity persistence attempted.
2. **EFTS entry with < 2 CIKs** — already skipped in `_parse_filings()`. No change needed.
3. **reporter_cik missing in some transactions** — `_persist_entities()` skips insider registration for those, `_find_best_cluster()` falls back to name dedup.
4. **Entity registration race condition** — INSERT OR IGNORE makes this safe.
5. **PipelineStore write failure** — caught by try/except, logged, tool continues.
6. **Company name normalization failure** — caught by try/except around `normalize_company_name()`, uses raw name as fallback.
7. **Duplicate transactions** — same insider, same date, same shares → stored as separate observations (intentional — may represent amendments).
8. **Empty transaction list** — `_persist_entities()` returns immediately, no-op.

## Testing Plan

### Unit Tests (per step)
- **10b.1.1:** Parse mock EFTS response, verify reporter_cik and issuer_cik in transaction dicts.
- **10b.1.2:** Construct tool with/without pipeline_store, verify attribute set.
- **10b.1.3:** Mock PipelineStore, call `_persist_entities()` with sample transactions, verify entity registration and observation storage calls.
- **10b.1.4:** Cluster detection with mixed CIK/name dedup scenarios.
- **10b.1.5:** Cluster data includes entity_ids mapping.
- **10b.1.6:** All edge cases above.
- **10b.1.7:** Full MI measurement loop.

## Related

- [[deep_surveillance_tools]]
- [[deep_surveillance_tools_spec]]
- [[deep_surveillance_10b]]
- [[project_memory]]

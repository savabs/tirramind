---
title: "Spec: Form 144 L2 Upgrade"
tags:
  - doc/spec
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Spec: Form 144 L2 Upgrade

## Goal

Upgrade `Form144Tool` to L2 entity-resolved observations. Same pattern as [[deep_surveillance_10b|insider_filings L2]], applied to sell-intent data.

## Files Affected

| File | Action |
|------|--------|
| `agent/tools/form144.py` | Modify: CIK threading, PipelineStore, persistence, CIK dedup, entity_ids |
| `tests/test_form144_l2.py` | Create: L2-specific test suite |

## Implementation Steps

### 10b.2.1: Extract reporter_cik in `_parse_filings()`

In the metadata extraction loop, after the swap logic resolves `issuer_cik`:
- Derive `reporter_cik` from the remaining CIK in the `ciks` array
- Add `reporter_cik` and `issuer_cik` to every filing dict (both metadata-only and XML-parsed paths)
- For the XML-parsed path, inject `issuer_cik` and `reporter_cik` into the parsed record after `_parse_form144_xml()` returns

**Verification:** Unit test that EFTS CIK pairs appear in all parsed filing dicts.

### 10b.2.2: Optional PipelineStore in constructor

- Add `*, pipeline_store: PipelineStore | None = None` to `__init__()`
- Store as `self._store`
- TYPE_CHECKING import for PipelineStore

**Verification:** Constructor accepts both paths (with/without store). Existing tests still pass.

### 10b.2.3: Implement `_persist_entities()` after parsing

- New method `_persist_entities(filings)` → `_persist_entities_inner(filings)`
- Register companies by `issuer_cik` with aliases (`sec_cik`, `ticker`)
- Register insiders by `reporter_cik` with alias (`sec_cik`)
- Store observations: `source_tool="form144"`, `depth_level=2`, `observation_type="sell_intent"`
- Value dict: `ticker`, `company`, `shares_to_sell`, `dollar_value`, `acquisition_type`, `urgency`, `relationship`
- Call from `execute()` after `_parse_filings()`, wrapped in try/except
- Skip when `self._store is None`

**Verification:** With PipelineStore: entities registered, aliases created, observations stored. Without: no-op.

### 10b.2.4: CIK-based dedup in `_find_best_sell_cluster()`

- Use `reporter_cik` for `seen_names` dedup when present
- Fall back to `_normalize_name(insider_name)` for backward compatibility
- Track CIK-to-name mapping for entity_ids output

**Verification:** Two filings with same reporter_cik but different display names → deduplicated to one insider.

### 10b.2.5: Add `entity_ids` to cluster output

- In `_find_best_sell_cluster()`, build `entity_ids` dict mapping `insider_name → entity_id`
- Use `entity_id_from_key("person", reporter_cik)` when CIK available
- Include in cluster dict

**Verification:** Cluster data includes `entity_ids` mapping.

### 10b.2.6: Edge case test suite

Cover:
- CIK extraction after swap logic (both orderings)
- Missing CIKs (< 2 in array)
- Metadata-only records get CIKs from EFTS
- XML-parsed records get CIKs injected
- Persistence with None store (no-op)
- Persistence failure doesn't break execute()
- CIK dedup vs name dedup fallback
- entity_ids present/absent based on CIK availability

### 10b.2.7: MI measurement integration test

Same pattern as insider_filings MI test:
- Synthetic data: signal + noise at L1/L2
- Store L2 observations via form144 persistence
- Measure MI(L2; target | L1) > 0

## Edge Cases

1. EFTS hit with `ciks` array of length 1 → skip (existing guard)
2. CIK swap changes which CIK is reporter → must extract reporter AFTER swap
3. Metadata-only records (`_metadata_only=True`) → still persist entities from EFTS CIKs
4. XML fetch returns None → fallback record still has EFTS CIKs
5. `_parse_form144_xml` returns None (gift, no shares) → no observation stored
6. All filings are gifts → no entities persisted
7. PipelineStore raises during persistence → swallowed by try/except, ToolResult unaffected

## Testing Plan

- `tests/test_form144_l2.py`: Comprehensive L2 test suite (~50 tests)
- Existing `tests/test_form144_edge.py`: Must still pass (backward compatibility)
- MI integration: Separate test class proving L2 adds signal

## Related

- [[form144_l2]]
- [[deep_surveillance_10b2]]
- [[deep_surveillance_tools]]
- [[deep_surveillance_tools_10b_spec]]

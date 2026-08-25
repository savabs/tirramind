---
title: "Spec: GNN-Guided Expansion Round 2"
tags:
  - doc/spec
  - phase/23
  - topic/surveillance
  - topic/gnn-expansion
  - layer/surveillance
---

# Spec: GNN-Guided Expansion Round 2

## Goal

Upgrade 3 Tier-1 L1 tools to L2 (entity persistence + entity linking) and expand graph_builder to support the new observation types: `short_interest`, `creditor_filing`, `drug_approval`.

## Files Affected

### Modified
| File | Change |
|------|--------|
| `agent/tools/finra_short_volume.py` | Add `pipeline_store` kwarg, `_persist_entities()`, `_persist_entities_inner()` |
| `agent/tools/creditor_filings.py` | Same pattern + debtor-creditor entity links |
| `agent/tools/drug_regulatory.py` | Same pattern + company→country market links |
| `agent/models/gnn/graph_builder.py` | Add 3 obs types + 2 link types to constants |

### Created
| File | Purpose |
|------|---------|
| `tests/test_finra_l2.py` | L2 persistence + edge case tests for finra_short_volume |
| `tests/test_creditor_l2.py` | L2 persistence + edge case tests for creditor_filings |
| `tests/test_drug_regulatory_l2.py` | L2 persistence + edge case tests for drug_regulatory |
| `tests/test_gnn_expansion_r2.py` | Integration: all 3 L2 tools → store → graph_builder → HeteroData |

## Implementation Steps

### 23a: Graph Builder Expansion

**23a.1: Add new observation types to graph_builder.py**

Insert 3 new observation types into `OBSERVATION_TYPES` (alphabetical order):
- `creditor_filing` (between `cross_entity_pattern` and `dns_change`)
- `drug_approval` (between `dns_change` and `form144_filing`)
- `short_interest` (between `sell_intent` and `tvl_change`)

This changes `len(OBSERVATION_TYPES)` from 18→21.

**Impact:** `ENRICHMENT_DIM` includes an obs_type distribution of size `len(OBSERVATION_TYPES)`. This must update from 18→21. The base enrichment computation in `_build_node_features_enriched()` uses `_OBS_TYPE_TO_IDX` for indexing, which auto-rebuilds from the list. No logic change needed — just the constant expansion.

Verify: `ENRICHMENT_DIM` is computed or hardcoded. If hardcoded (currently 27), update to 30 (27 - 18 + 21 = 30). If computed from `len(OBSERVATION_TYPES)`, no change needed.

**23a.2: Add new link types**

No constant list of link types needs updating — `_build_edge_data()` dynamically groups by `(src_type, link_type, dst_type)` from the store. New link types (`debtor_of`, `market_authorized_in`) will flow through automatically.

Verify: confirm `_build_edge_data()` is fully dynamic and doesn't filter by a hardcoded list.

### 23b: finra_short_volume L2

**23b.1: Add pipeline_store to constructor**

```python
def __init__(
    self,
    cache: DataCache | None = None,
    *,
    pipeline_store: PipelineStore | None = None,
) -> None:
    self._cache = cache
    self._store = pipeline_store
```

Add imports (same pattern as sanctions_monitor):
```python
if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key, normalize_company_name
except ImportError:
    entity_id_from_key = None
    normalize_company_name = None
```

**23b.2: Implement _persist_entities()**

For each ticker result:
1. Register entity: `entity_type="company"`, `canonical_name=ticker.upper()`, `entity_id=entity_id_from_key("company", ticker.upper())`
2. Store observation: `observation_type="short_interest"`, value dict containing:
   - `short_ratio`, `total_volume`, `short_volume`, `zscore`, `trend`, `is_anomaly`
   - `days_to_cover` (from short_interest mode)
   - `settlement_date` (if available)
3. No entity linking (single-entity tool, market data)

**23b.3: Call from _run()**

After results are assembled but before return, call `self._persist_entities(results)`.

**23b.4: Write test_finra_l2.py**

Tests:
- Store is None → no crash, no persistence
- Store present → entities registered, observations stored
- Multi-ticker results → each ticker gets its own entity + observations
- Empty results → no-op
- Missing fields → graceful handling
- Anomaly detection preserved in stored value

### 23c: creditor_filings L2

**23c.1: Add pipeline_store to constructor**

Same pattern as 23b.1.

**23c.2: Implement _persist_entities()**

For SEC EDGAR 8-K results:
1. Register entity: `entity_type="company"`, `canonical_name=normalize_company_name(name)`, `entity_id=entity_id_from_key("company", canon)`
2. Store observation: `observation_type="creditor_filing"`, value dict:
   - `cik`, `form`, `file_date`, `items` (list of 8-K items)
   - `is_stress_signal` (boolean, true if items contain 1.01/2.03/2.04)
3. Entity linking: when `persons_entitled` (creditor names) are available from UK charges:
   - Register creditor as company entity
   - `store.link_entities(debtor_eid, creditor_eid, link_type="debtor_of", source="creditor_filings", confidence=0.8)`

For UK Companies House charges:
1. Register entity: `entity_type="company"`, from company_name/company_number
2. Store observations per charge: `observation_type="creditor_filing"`, value dict:
   - `charge_number`, `status`, `created_on`, `classification`, `persons_entitled`
   - `is_red_flag` (status == "outstanding" or "part-satisfied")

**23c.3: Call from _run()**

Call `self._persist_entities(results)` from both search and stress_scan result paths.

**23c.4: Write test_creditor_l2.py**

Tests:
- SEC EDGAR results → company entities + creditor_filing observations
- UK charges → debtor-creditor entity links
- Stress cluster detection → multiple filings for same entity
- Empty results → no-op
- Missing CIK → entity still registered with name only
- Self-link guard (debtor == creditor name → skip)

### 23d: drug_regulatory L2

**23d.1: Add pipeline_store to constructor**

Same pattern.

**23d.2: Implement _persist_entities()**

For approvals mode:
1. Register sponsor entity: `entity_type="company"`, `canonical_name=normalize_company_name(sponsor_name)`
2. Store observation: `observation_type="drug_approval"`, value dict:
   - `application_number`, `brand_names`, `submission_type`, `submission_date`, `review_priority`
3. Entity linking: `store.link_entities(company_eid, country_eid, link_type="market_authorized_in", source="drug_regulatory", confidence=1.0)` — US market authorization (FDA data is US-only)

For adverse_events mode:
1. Same sponsor registration
2. Store observation: `observation_type="drug_approval"` (reuse type for all FDA data), value dict:
   - `drug_names`, `reactions`, `serious`, `receive_date`, `seriousness_ratio`

**23d.3: Call from _run()**

Call `self._persist_entities(results)` from all three mode result paths.

**23d.4: Write test_drug_regulatory_l2.py**

Tests:
- Approval results → company entity + drug_approval observation
- Adverse events → company entity + observation with seriousness flag
- Country link created for FDA approvals (US market)
- Missing sponsor_name → skip entity registration
- Empty results → no-op
- Labels mode → company entity, no approval observation (informational only)

### 23e: Integration + Edge Cases

**23e.1: Write test_gnn_expansion_r2.py**

Integration tests:
1. All 3 L2 tools → store → query entities → verify correct types and counts
2. Store → graph_builder → HeteroData → verify new obs types in feature vectors
3. Entity links (debtor_of, market_authorized_in) → verify edge_index in HeteroData
4. Mixed old + new L2 tools → single graph → verify backward compatibility
5. ENRICHMENT_DIM update verified (39 → 42 if hardcoded, or auto if computed)

Edge cases:
- Duplicate entity registration (idempotent)
- Observation with unknown obs_type → fallback behavior
- Entity link confidence < 1.0 → preserved in edge_attr
- Empty store → graph builds with zero nodes (no crash)

## Testing Plan

| Test File | Scope | Estimated Tests |
|-----------|-------|-----------------|
| test_finra_l2.py | finra_short_volume L2 persistence | ~12 |
| test_creditor_l2.py | creditor_filings L2 + entity links | ~15 |
| test_drug_regulatory_l2.py | drug_regulatory L2 + country links | ~12 |
| test_gnn_expansion_r2.py | Integration + edge cases | ~15 |
| **Total** | | **~54** |

## Edge Cases

- Entity name normalization edge cases (unicode, case, whitespace)
- Duplicate observations (same entity, same timestamp) → idempotent
- Missing optional fields in API response → graceful degradation
- Self-link guard → creditor == debtor → skip link
- Store.link_entities ValueError for self-links → caught in try/except
- Graph builder with 21 obs types → verify feature dimension correctness
- Existing GNN model with 18-dim obs type head → requires retraining (not backward compatible for saved models, but we have no saved models yet)

## Related

- [[gnn_guided_expansion_r2]] — Research doc
- [[gnn_guided_tool_expansion]] — Phase 16 first-round research
- [[l2_tool_expansion]] — Phase 13 L2 patterns
- [[quant_training_ground]] — Master tracker

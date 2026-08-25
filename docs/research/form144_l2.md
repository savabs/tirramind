---
title: "Research: Form 144 L2 Upgrade"
tags:
  - doc/research
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Research: Form 144 L2 Upgrade

## Goal

Upgrade `form144` tool from L1 aggregate sell-intent clusters to L2 entity-resolved observations, following the exact pattern established by [[deep_surveillance_10b|insider_filings L2 (Phase 10b.1)]].

---

## Current Architecture

`Form144Tool` (~850 lines, `agent/tools/form144.py`):

- **Data source:** SEC EDGAR EFTS (free, 10 req/sec) + Form 144 XML archives
- **Pipeline:** EFTS search → two-phase parse (metadata only for singletons, XML fetch only for cluster candidates) → cluster detection → ToolResult
- **Cluster definition:** 2+ distinct insiders filing Form 144 at same company within 14 days
- **Dedup:** By `_normalize_name()` (uppercase + suffix stripping) — no CIK-based dedup
- **Constructor:** `__init__(self, cache=None)` — no PipelineStore

### CIK Handling — Current State

In `_parse_filings()`, the EFTS `ciks` array is used but only partially:

```python
issuer_cik = ciks[0]        # used for XML fetch URL
filer_name = names[1]       # insider name from display_names
# reporter_cik is available as ciks[1] but NEVER extracted
```

The code has swap logic: if the ticker is found in `names[1]` instead of `names[0]`, it swaps `issuer_cik = ciks[1]`. After the swap, the reporter CIK is the *other* element.

The XML parser (`_parse_form144_xml`) is a module-level function that receives `ticker`, `company`, and `file_date` — but NOT CIKs. The XML itself has `issuerCik` but not always reporter CIK.

### CIK Extraction Fix

After the swap logic resolves which CIK is the issuer, the reporter CIK is deterministic:

```python
# After swap logic completes:
reporter_cik = ciks[1] if issuer_cik == ciks[0] else ciks[0]
```

However, `ciks` may have only 1 element (edge case already handled by `len(ciks) < 2` guard). When 2+ CIKs exist, both are always available.

---

## Observations

### Structural Similarity to insider_filings

| Aspect | insider_filings | form144 |
|--------|----------------|---------|
| Data source | EFTS + Form 4 XML | EFTS + Form 144 XML |
| CIK source | `ciks[0]` = reporter, `ciks[1]` = issuer | `ciks[0]` = issuer (with swap), other = reporter |
| Entity types | company (issuer_cik) + person (reporter_cik) | company (issuer_cik) + person (reporter_cik) |
| Observation type | `purchase` | `sell_intent` |
| Cluster detection | 3+ insiders, 14-day window | 2+ insiders, 14-day window |
| Dedup method | CIK-based (after L2 upgrade) | Name-based (needs CIK upgrade) |
| PipelineStore | ✅ Optional kwarg | ❌ Not present |

### Key Differences from insider_filings

1. **Two-phase parse optimization:** Form144 only fetches XML for cluster candidates (2+ filers per ticker). Metadata-only records exist for singletons. Entity persistence must handle both: full records with CIKs and metadata-only records.
2. **Module-level XML parser:** `_parse_form144_xml()` is a free function, not a method. CIKs need to be passed to it as additional args or injected after parsing.
3. **`_metadata_only` flag:** Some filing dicts have `_metadata_only=True` — these lack XML-derived fields but still have `issuer_cik` from EFTS and `filer_name` from display_names. Reporter CIK is still extractable from EFTS for these.
4. **Observation value:** Different fields — `shares_to_sell`, `dollar_value`, `acquisition_type`, `urgency`, `has_10b5_1_plan`, `relationship` instead of `shares`, `price`, `role`.

### Risks

1. **CIK swap logic complicates reporter extraction.** Must derive reporter_cik *after* the swap resolves issuer_cik. Needs careful testing.
2. **Metadata-only records have no XML-parsed CIKs.** Entity persistence must use EFTS CIKs for these, which is fine since EFTS CIKs are the authoritative source anyway.
3. **`_parse_form144_xml` is module-level.** Adding CIK params changes a public function signature. Must remain backward-compatible.

---

## L2 Upgrade Design

### Change 1: Extract reporter_cik in `_parse_filings()`

After swap logic resolves `issuer_cik`, derive:
```python
reporter_cik = next((c for c in ciks if c != issuer_cik), ciks[-1])
```

Add both `issuer_cik` and `reporter_cik` to every filing dict (both metadata-only and XML-parsed records).

### Change 2: Optional PipelineStore in constructor

```python
def __init__(self, cache=None, *, pipeline_store=None):
```

Identical pattern to insider_filings.

### Change 3: `_persist_entities()` after parsing

After `_parse_filings()` returns, call `_persist_entities(filings)`:
- Register companies by `issuer_cik`
- Register insiders by `reporter_cik`
- Store observations with `depth_level=2`, `observation_type="sell_intent"`
- Value dict: `ticker`, `company`, `shares_to_sell`, `dollar_value`, `acquisition_type`, `urgency`, `relationship`
- Wrapped in try/except — persistence failure must not break ToolResult

### Change 4: CIK-based dedup in `_find_best_sell_cluster()`

Use `reporter_cik` for dedup when present, fall back to `_normalize_name()`.

### Change 5: `entity_ids` in cluster output

Add `entity_ids: {insider_name: entity_id}` to cluster dicts.

---

## Related

- [[deep_surveillance_tools]]
- [[form144_l2_spec]]
- [[deep_surveillance_10b2]]
- [[deep_surveillance_10b]]
- [[project_memory]]

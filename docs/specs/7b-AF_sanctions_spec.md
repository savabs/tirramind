---
title: "Spec: sanctions_monitor"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/sanctions
---

# Spec: sanctions_monitor

## Goal
Monitor global sanctions lists (OFAC SDN + UN Security Council) for entity screening, recent additions, and program analysis. Provides structured access to sanctioned entity data as a surveillance surface signal.

## Files Affected
1. **CREATE** `agent/tools/sanctions_monitor.py` — Tool implementation
2. **MODIFY** `agent/cli.py` — Register tool
3. **MODIFY** `agent/learning/bandit.py` — Add bandit arm
4. **CREATE** `tests/test_sanctions_monitor_edge.py` — Edge case tests

## Implementation Steps

### 3.1: Create sanctions_monitor.py skeleton
- Module docstring with signal theory
- Constants: URLs, timeout, user-agent
- SanctionsMonitorTool class with name, description, parameters, execute()
- 3 modes: search, recent, programs

### 3.2: Implement OFAC SDN CSV parser
- `_fetch_ofac_sdn() -> tuple[list[dict], str | None]`
- Download sdn.csv, parse with csv module (handle no-header, `-0-` nulls)
- Normalize to records: {source, entity_id, name, type, programs, remarks, aliases}
- Cache parsed records with 6h TTL

### 3.3: Implement UN SC XML parser
- `_fetch_un_consolidated() -> tuple[list[dict], str | None]`
- Download consolidated.xml, parse with xml.etree.ElementTree
- INDIVIDUAL: FIRST_NAME + SECOND_NAME → name, LISTED_ON → listed_date, UN_LIST_TYPE → programs
- ENTITY: similar structure
- Normalize + cache with 6h TTL

### 3.4: Implement search mode
- Input: query (name substring), source (ofac/un/all), entity_type (individual/entity/all), program (filter by program)
- Case-insensitive substring match on name + aliases
- Return matched entities with source, program, type, remarks
- Limit results (default 25, max 100)

### 3.5: Implement recent mode
- Input: days_back (default 90, max 365), source (un/all)
- UN: filter by LISTED_ON or LAST_DAY_UPDATED within days_back
- OFAC: no per-entry dates → skip (note limitation in output)
- Sort by date descending
- Return recently listed/updated entities

### 3.6: Implement programs mode
- Aggregate unique program codes across OFAC + UN
- Count entities per program per source
- Return program → {count, source, example_entities}
- Sort by count descending

### 3.7: Register in cli.py
- Import SanctionsMonitorTool
- Add `registry.register(SanctionsMonitorTool(cache=cache))`

### 3.8: Add bandit arm
- Add `sanctions_screening` arm to DEFAULT_ARMS
- Tools: `["sanctions_monitor", "gdelt", "market_data"]`
- Examples: sanctions search, recent additions, program analysis

### 3.9: Write edge case tests
- Invalid mode, missing query, empty results
- OFAC CSV parsing edge cases: special chars in names, multi-line remarks, `-0-` handling
- UN XML parsing: missing fields, empty aliases, malformed dates
- Search: partial match, case insensitivity, unicode names
- Programs: empty list, single-entry programs
- Recent: future dates, boundary dates, no recent entries
- Network errors: timeout, connection error, malformed response
- Cache hit/miss paths
- Limit/pagination bounds

## Edge Cases
- Names with commas, quotes, unicode (Arabic, Cyrillic, Chinese)
- Entities with multiple programs (comma-separated in OFAC)
- Empty/missing fields in UN XML
- Very large result sets → enforce limit
- Network timeout on 5.5MB download → configurable timeout (30s)
- Malformed CSV/XML → graceful error, not crash

## Testing Plan
- All tests use mocked HTTP responses (no real API calls)
- Mock OFAC CSV with representative entries (individuals, entities, multi-program, unicode, `-0-` fields)
- Mock UN XML with representative entries (INDIVIDUALS section, ENTITIES section, LISTED_ON dates)
- Verify each mode independently
- Verify cross-source search (all)
- Verify error paths produce ToolResult(success=False, ...) not exceptions

---

## Related

- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]

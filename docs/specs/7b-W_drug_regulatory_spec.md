---
title: "Spec: 7b-W — Drug Regulatory Tool (OpenFDA)"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/drug-regulatory
---

# Spec: 7b-W — Drug Regulatory Tool (OpenFDA)

## Goal
Fetch and parse FDA drug regulatory data via the OpenFDA API. Provide drug approvals (Drugs@FDA), adverse event queries (FAERS), and label data. Compute quant signals: adverse event spike detection, approval rate by sponsor, seriousness ratio trends.

## Files Affected
- **Create:** `agent/tools/drug_regulatory.py`
- **Modify:** `agent/cli.py` — add import + registration
- **Modify:** `agent/learning/bandit.py` — add GoalArm
- **Create:** `tests/test_drug_regulatory_edge.py`

## Implementation Steps

### 2.1: Create `agent/tools/drug_regulatory.py`
- Class `DrugRegulatoryTool(Tool)`
- `name = "drug_regulatory"`
- 3 modes: `approvals`, `adverse_events`, `labels`
- Parameters: `mode`, `search` (Elasticsearch query string), `drug_name` (convenience filter), `date_start`/`date_end` (YYYYMMDD), `count_field` (for faceted counts), `limit` (max 100 for results, 1000 for counts)
- `__init__(self, cache: DataCache | None = None)`
- Use `httpx.Client` with `timeout=20`, User-Agent header
- Cache with `DataCache` (key by mode + search + dates)
- Parse JSON response, extract `meta` + `results`
- Compute signals: event counts, seriousness ratio, approval summaries
- Return `ToolResult` with formatted output + structured `data`

### 2.2: API Implementation Details
- Endpoints:
  - `https://api.fda.gov/drug/drugsfda.json` (approvals mode)
  - `https://api.fda.gov/drug/event.json` (adverse_events mode)
  - `https://api.fda.gov/drug/label.json` (labels mode)
- Query params: `search=`, `count=`, `limit=`, `skip=`
- No API key (use free tier: 240/min, 1000/day) — sufficient for our usage
- Elasticsearch syntax: `field:value`, `+AND+`, `[date1+TO+date2]`
- Response shape: `{"meta": {"results": {"total": N}}, "results": [...]}`

### 2.3: Register in `agent/cli.py`
- Import `DrugRegulatoryTool` from `agent.tools.drug_regulatory`
- Add `registry.register(DrugRegulatoryTool(cache=cache))` after treasury_receipts

### 2.4: Add GoalArm in `agent/learning/bandit.py`
- `name="drug_regulatory_monitor"`
- `tools=["drug_regulatory", "web_search"]`
- Examples: recent FDA approvals, adverse event spike for a drug, label changes with warnings, approval rate for a sponsor

### 2.5: Write edge-case tests
- Cover: all 3 modes, invalid mode, empty results, HTTP errors, malformed JSON, cache, Elasticsearch syntax errors (API returns 400/404), count mode, drug_name convenience filter, date range filtering, seriousness ratio computation, tool schema validation

## Edge Cases
- API returns `{"error": {...}}` for bad queries (Elasticsearch parse errors)
- Empty `results: []` for narrow queries → handle gracefully
- Rate limit hit → 429 response → graceful error message
- FAERS data has duplicate reports → note in output, don't deduplicate (API-level issue)
- Drug name spelling variations → drug_name filter is substring match, user can also use raw search
- `count` mode returns `[{"term": "...", "count": N}]` — different shape than regular results
- Date format: FAERS uses YYYYMMDD, Drugs@FDA uses YYYY-MM-DD in submission dates

## Testing Plan
- Mock `httpx.Client.get` responses with synthetic JSON matching each endpoint's schema
- Test each mode independently
- Test count mode separately (different response shape)
- Test drug_name convenience filter → generates correct search string
- Test error paths: 400 (bad query), 404 (not found), 500, timeout, rate limit (429)
- Validate `ToolResult` structure
- Test cache integration

---

## Related

- [[7b-W_drug_regulatory|Research: 7B-W Drug Regulatory]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]

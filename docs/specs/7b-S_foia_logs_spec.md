---
title: "Spec: 7b-S — FOIA / FOI Request Logs Tool"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/foia
---

# Spec: 7b-S — FOIA / FOI Request Logs Tool

## Goal
Build a surveillance tool that monitors FOIA/FOI request activity to detect
investigation formation: when multiple requests cluster around the same entity,
agency, or topic, it's a leading indicator of upcoming disclosures.

## Files Affected
- **CREATE:** `agent/tools/foia_requests.py` — tool implementation
- **MODIFY:** `agent/cli.py` — import + registration (tool #37)
- **MODIFY:** `agent/learning/bandit.py` — new arm `investigation_signals` (#25)
- **CREATE:** `tests/test_foia_requests_edge.py` — edge case test suite
- **MODIFY:** 9 test files — update tool count 36→37, arm count 24→25

## Implementation Steps

### Step 1: Core tool skeleton
- Class `FoiaRequestsTool(Tool)` with name="foia_requests"
- 3 modes: search, agency_activity, entity_cluster
- Parameters: mode, query, agency, days_back, jurisdiction, limit
- Standard __init__(cache=None), execute() dispatch

### Step 2: MuckRock fetcher
- `_fetch_muckrock(endpoint, params)` → JSON
- Base URL: `https://www.muckrock.com/api_v1/`
- httpx.Client, UA="TirraMind/0.1", timeout=20s
- Pagination: follow `next` links up to max pages
- Error handling: 404, 429, 500, timeout, JSON parse

### Step 3: WhatDoTheyKnow fetcher
- `_fetch_wdtk(query)` → JSON
- Base URL: `https://www.whatdotheyknow.com/`
- Alaveteli API: `/api/v2/requests.json?query=<term>`
- Same error handling pattern

### Step 4: Mode — search
- Search MuckRock `/api_v1/foia/?q=<query>&page_size=<limit>`
- Optionally include WDTK results for UK coverage
- Normalize results: title, agency, status, date_filed, jurisdiction
- Sort by date descending
- Format as structured text output

### Step 5: Mode — agency_activity
- Fetch recent requests for an agency from MuckRock
- Count requests in recent window vs baseline
- Flag surge if recent > 2× baseline rate
- Output: request count, trend, surge flag, recent request titles

### Step 6: Mode — entity_cluster
- Search for entity name across both MuckRock + WDTK
- Group results by agency
- Count distinct agencies + jurisdictions requesting about same entity
- Flag convergence: requests from 3+ agencies or 2+ jurisdictions
- Output: agency breakdown, convergence flag, timeline

### Step 7: Cache integration
- Cache key: `foia:<mode>:<query_hash>`
- TTL: 1800s (30 min) — FOIA data doesn't change frequently
- Use self._cache.get_or_fetch() pattern

### Step 8: Registration
- Import in cli.py, register as FoiaRequestsTool(cache=cache)
- New bandit arm `investigation_signals` with tools=[foia_requests, web_search]

## Edge Cases
- Empty search results
- MuckRock API down / rate-limited
- WDTK API down (graceful degradation — MuckRock-only fallback)
- Invalid mode, missing required params
- Very long query strings
- Agency not found in MuckRock
- Pagination edge (0 results, 1 page, many pages)
- Unicode in request titles
- Date parsing failures
- Network timeout

## Testing Plan
- Unit test each helper function
- Mock httpx for all API calls
- Test each mode with mock data
- Test error paths (timeout, 404, 429, 500)
- Test cache hit/miss
- Test normalization of MuckRock + WDTK data
- Test surge detection logic
- Test convergence detection logic
- Integration: tool count = 37, arm count = 25

---

## Related

- [[7b-S_foia_logs|Research: 7B-S Foia Logs]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]

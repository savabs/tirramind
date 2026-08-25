---
title: "Spec: Phase 6a — GDELTTool (Geopolitical Event Surveillance)"
tags:
  - doc/spec
  - topic/gdelt
---

# Spec: Phase 6a — GDELTTool (Geopolitical Event Surveillance)

## Goal

Give TirraMind a tool that fetches structured geopolitical events from GDELT's raw event files — every conflict, sanction, protest, and military action worldwide, coded with CAMEO taxonomy, geocoordinates, and intensity scores. Secondary mode: keyword/theme article search via DOC API. Zero cost, no auth.

## Research

See: `[[gdelt]]`

## Files Affected

### New files
- `agent/tools/gdelt.py` — GDELTTool class

### Modified files
- `agent/cli.py` — register GDELTTool in `build_tool_registry()`
- `agent/learning/bandit.py` — add `geopolitical_intelligence` arm to DEFAULT_ARMS

### No new dependencies
Uses httpx (existing), zipfile + io (stdlib).

---

## Implementation Steps

### 6a.2: Create `agent/tools/gdelt.py` skeleton

- Class `GDELTTool(Tool)` with name `"gdelt"`
- Accept `cache: DataCache | None = None` in `__init__`
- Parameters schema:
  - `mode` (string, "events" | "articles", default "events")
  - `hours_back` (int, 1-24, default 1) — events mode only, how many hours of 15-min batches
  - `country_filter` (string, optional) — FIPS 2-letter country code for ActionGeo filtering
  - `min_goldstein` (number, optional) — only return events with Goldstein ≤ this (e.g., -5.0 for conflict)
  - `quad_class` (string, optional, "conflict" | "cooperation" | "all", default "conflict") — shorthand filter
  - `event_codes` (string, optional) — comma-separated root codes to include (e.g., "18,19,20")
  - `query` (string, required for articles mode) — search keywords for DOC API
  - `limit` (int, default 50) — max events/articles to return
- Description for LLM: geopolitical event monitoring, 300+ event types, 15-min updates
- Test: import succeeds, `to_openai_tool()` returns valid schema

### 6a.3: Implement `_fetch_event_batches()` helper

- Compute batch timestamps: round current UTC time down to nearest 15-min mark, iterate backwards for `hours_back * 4` batches
- URL pattern: `http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.export.CSV.zip`
- Fetch each zip with `httpx.Client(timeout=15, follow_redirects=True)`
- Extract CSV from zip using `zipfile.ZipFile(io.BytesIO(data))`
- Cache each batch individually: key `("gdelt_events", {"batch": "YYYYMMDDHHMMSS"})` — batches are immutable, long TTL is fine
- Rate politeness: 0.2s sleep between batch downloads
- Handle: 404 (batch not yet published — skip silently, GDELT sometimes lags a few minutes), network errors (log + skip)
- Return: list of raw CSV text content (one per successful batch)
- Test: mock httpx, verify batch URL computation, verify zip extraction

### 6a.4: Implement `_parse_events()` helper

Column map (tab-separated, no header, 61 columns):
```
[0] GlobalEventID, [1] Day, [4] FractionDate,
[5] Actor1Code, [6] Actor1Name, [7] Actor1CountryCode, [12] Actor1Type1Code,
[15] Actor2Code, [16] Actor2Name, [17] Actor2CountryCode, [22] Actor2Type1Code,
[25] IsRootEvent, [26] EventCode, [27] EventBaseCode, [28] EventRootCode,
[29] QuadClass, [30] GoldsteinScale, [31] NumMentions, [32] NumSources,
[33] NumArticles, [34] AvgTone,
[51] ActionGeo_Type, [52] ActionGeo_FullName, [53] ActionGeo_CountryCode,
[56] ActionGeo_Lat, [57] ActionGeo_Long,
[59] DATEADDED, [60] SOURCEURL
```

For each row:
- Split by `\t`, skip if < 61 columns
- Parse numeric fields safely (Goldstein, Tone, Lat, Long — default to None on parse failure)
- Build event dict:
```python
{
    "id": str,
    "date": str (YYYYMMDD),
    "actor1": {"name": str, "country": str, "type": str},
    "actor2": {"name": str, "country": str, "type": str},
    "event_code": str,
    "event_root": str,
    "event_description": str,  # lookup from CAMEO_ROOT_CODES dict
    "quad_class": int,
    "quad_label": str,  # "Verbal Cooperation" / "Material Conflict" / etc
    "goldstein": float,
    "num_mentions": int,
    "num_sources": int,
    "avg_tone": float,
    "location": {"name": str, "country": str, "lat": float, "lon": float},
    "source_url": str,
}
```
- Define `CAMEO_ROOT_CODES` dict mapping "01"-"20" → human-readable labels
- Define `QUAD_LABELS` dict: {1: "Verbal Cooperation", 2: "Material Cooperation", 3: "Verbal Conflict", 4: "Material Conflict"}
- Test: parse a known fixture row → verify all fields

### 6a.5: Implement `_fetch_articles()` helper

- DOC API: `GET https://api.gdeltproject.org/api/v2/doc/doc`
- Params: `query=<query>`, `mode=artlist`, `format=json`, `maxrecords=min(limit, 250)`, `timespan=1w`, `sort=datedesc`
- Cache key: `("gdelt_articles", {"query": query, "timespan": "1w"})`
- Rate caution: DOC API returns 429 on burst. Single request per invocation, no rapid retries.
- Parse JSON response: `data["articles"]` → list of `{"url", "title", "seendate", "domain", "language", "sourcecountry", "tone"}`
- Handle: 429 → `ToolResult(success=False, output="GDELT rate limited, try again in a minute")`
- Test: mock httpx, verify param construction and JSON parsing

### 6a.6: Implement `execute()` method

Events mode flow:
1. Validate params
2. Call `_fetch_event_batches(hours_back)`
3. Call `_parse_events(raw_batches)` → all events
4. Apply filters in order:
   - `quad_class`: if "conflict" → keep QuadClass 3+4. If "cooperation" → keep 1+2. If "all" → no filter.
   - `country_filter`: match against `location["country"]` (case-insensitive)
   - `min_goldstein`: keep events where goldstein ≤ threshold
   - `event_codes`: keep events whose event_root is in the comma-separated list
5. Sort by `abs(goldstein)` descending (most impactful first), then by `num_mentions` descending as tiebreaker
6. Truncate to `limit`
7. Build summary: total events fetched, filtered count, batches fetched, top conflict countries (Counter on location.country for conflict events)
8. Format output text: summary header + table of top events (actor1 vs actor2, event description, Goldstein, location)
9. Return `ToolResult(success=True, output=formatted_str, data={"events": list, "summary": dict})`

Articles mode flow:
1. Require `query` param, fail if empty
2. Call `_fetch_articles(query, limit)`
3. Format output: list of articles with title, tone, source, date
4. Return `ToolResult(success=True, output=formatted_str, data={"articles": list})`

Error handling: wrap in try/except, log, return `ToolResult(success=False, output=error_msg)`

### 6a.7: Register in `cli.py` and add bandit arm

- `agent/cli.py`: import `GDELTTool`, add `GDELTTool(cache=cache)` to `build_tool_registry()`
- `agent/learning/bandit.py`: add to `DEFAULT_ARMS`:
```python
GoalArm(
    name="geopolitical_intelligence",
    description="Monitor geopolitical events and their market implications",
    tools=["gdelt", "market_data", "macro_data"],
    examples=["Track conflict escalation in oil-producing regions", "Monitor sanctions events"],
)
```

### 6a.8: Live test

- Run `python -m agent` or direct invocation of `execute(mode="events", hours_back=1, quad_class="conflict")`
- Verify: real conflict events returned with CAMEO codes, Goldstein scores, geocoordinates
- Test country filter: `country_filter="IR"` should return only Iran-related events
- Test articles mode: `execute(mode="articles", query="oil sanctions iran")`
- Verify caching: second call with same batch timestamps returns cached data

---

## Edge Cases

1. **Batch not yet published** — GDELT sometimes lags 1-5 min. The most recent batch URL may 404. Skip silently and return the batches that succeeded.
2. **Empty batch** — Some 15-min windows have very few events. Return what we get with a note.
3. **Malformed rows** — Some rows have < 61 columns or non-numeric fields where numbers expected. Skip the row, log a warning.
4. **DOC API rate limit** — 429 response. Return friendly error, don't retry in a loop.
5. **Very large result set** — 24 hours × 96 batches × 1200 events = 115K events. Filter early (QuadClass check during parse, not after) to keep memory reasonable.
6. **Network timeout** — Individual batch timeout. Skip the failed batch, continue with others.
7. **Actor fields empty** — Many events have Actor1 or Actor2 as empty strings. Still valid events (e.g., "unknown attacker"). Default name to "UNKNOWN".

## Testing Plan

1. **Unit**: Parse a hardcoded TSV row → verify all 61 columns mapped correctly
2. **Unit**: `_fetch_event_batches()` with mocked httpx → verify URL computation and zip extraction
3. **Unit**: Filter chain — verify QuadClass, country, Goldstein, event_code filters work independently and combined
4. **Integration**: Live fetch → real events returned with valid structure
5. **Cache**: Second call returns cached batches without HTTP requests

---

## Related

- [[gdelt|Research: Gdelt]]

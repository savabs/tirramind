---
title: "Feature: GDELT (Global Database of Events, Language, and Tone)"
tags:
  - doc/research
  - layer/world-model
  - topic/gdelt
---

# Feature: GDELT (Global Database of Events, Language, and Tone)

## What GDELT Is

GDELT monitors global news media in 65 languages, machine-translates everything to English, and produces structured geopolitical event records every 15 minutes. It extracts actors (countries, organizations, groups), event types (300+ CAMEO codes), locations (geocoded), sentiment (tone), and intensity (Goldstein scale). Free, no API key, no auth. Updated since February 2015, historical data back to 1979.

This is the "poor man's Palantir" — the same type of structured event intelligence that governments pay millions for, available as open data files.

## Why It Matters for TirraMind

Geopolitical events drive commodity prices, currency moves, and equity risk premiums. GDELT gives us:
- **Real-time conflict escalation** — Russia-Ukraine, Iran-Israel, China-Taiwan events coded as they happen
- **Commodity supply disruption signals** — conflict in oil-producing regions (Iraq, Libya, Venezuela) → supply risk
- **Sanctions/embargo tracking** — CAMEO codes 163 (impose embargo), 172 (impose sanctions)
- **Protest/instability monitoring** — CAMEO code 14x events in any country
- **Diplomatic cooperation/breakdown** — tracks the full cooperation-conflict spectrum

When cross-referenced with CFTC positioning, insider clusters, and Polymarket odds, GDELT events become nodes in the world model's causal graph.

## Data Access Methods

### Primary: Raw Event Files (No Auth, No Rate Limit)

Every 15 minutes, GDELT publishes 3 files:

| File | Content | Size |
|------|---------|------|
| `YYYYMMDDHHMMSS.export.CSV.zip` | Events table (structured geopolitical events) | ~70KB |
| `YYYYMMDDHHMMSS.mentions.CSV.zip` | Mentions table (tracks event discussion progression) | ~90KB |
| `YYYYMMDDHHMMSS.gkg.csv.zip` | Global Knowledge Graph (themes, persons, orgs, tone, GCAM) | ~4MB |

**Access URLs:**
- Latest batch: `http://data.gdeltproject.org/gdeltv2/lastupdate.txt`
- Specific batch: `http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.export.CSV.zip`
- Master file list: `http://data.gdeltproject.org/gdeltv2/masterfilelist.txt`

**No API key. No rate limit on file downloads. No authentication.** Just download the zip, extract the CSV, parse the tab-separated rows.

**Volume:**
- ~1,200 events per 15-min batch
- ~115,000 events/day
- ~70KB compressed per batch → fetching 1 hour (4 batches) = ~280KB

### Secondary: DOC API (Full-Text Article Search)

Base URL: `https://api.gdeltproject.org/api/v2/doc/doc`

| Parameter | Values | Notes |
|-----------|--------|-------|
| `query` | keywords, phrases, operators | supports `theme:TERROR`, `sourcecountry:US`, `tone<-5`, `near20:"trump putin"` |
| `mode` | artlist, timelinevol, timelinetone, tonechart, wordcloud | artlist = article list, timelinevol = coverage volume over time |
| `format` | json, csv, html | json for programmatic use |
| `maxrecords` | 1-250 | default 75 |
| `timespan` | 15min-3months | e.g., `1d`, `1w`, `2months` |
| `startdatetime` / `enddatetime` | YYYYMMDDHHMMSS | precise time range within last 3 months |
| `sort` | datedesc, dateasc, tonedesc, toneasc | default: relevance |

**No API key required.** Has IP-based rate limiting (429 on rapid successive calls). Use conservatively — add 1-2s delay between requests.

**JSON response** (artlist mode): `{"articles": [{"url": "...", "title": "...", "seendate": "...", "domain": "...", "language": "...", "sourcecountry": "...", "socialimage": "...", "tone": float}]}`

**Limitations:**
- Searches last 3 months only
- Max 250 articles per query
- Rate limited (429 on burst)
- Returns articles, NOT structured events — no CAMEO codes, no actors, no Goldstein

### Tertiary: BigQuery (Historical Analysis)

Tables: `gdelt-bq:gdeltv2.events`, `gdelt-bq:gdeltv2.eventmentions`, `gdelt-bq:gdeltv2.gkg`

Google BigQuery free tier: 1TB/month query processing. Needs a Google Cloud account (free). Useful for historical backtesting but NOT needed for real-time surveillance. Defer to later phase.

## Events Table Schema (61 Columns)

The raw export CSV is tab-separated with no header. Columns by index:

### Identity (0-4)
| Index | Name | Type | Example |
|-------|------|------|---------|
| 0 | GlobalEventID | int | 1295798132 |
| 1 | Day | int (YYYYMMDD) | 20260324 |
| 2 | MonthYear | int (YYYYMM) | 202603 |
| 3 | Year | int | 2026 |
| 4 | FractionDate | float | 2026.2301 |

### Actor1 (5-14)
| Index | Name | Description |
|-------|------|-------------|
| 5 | Actor1Code | CAMEO actor code |
| 6 | Actor1Name | Full text name |
| 7 | Actor1CountryCode | 3-letter country |
| 8 | Actor1KnownGroupCode | Group affiliation |
| 9 | Actor1EthnicCode | Ethnic affiliation |
| 10 | Actor1Religion1Code | Primary religion |
| 11 | Actor1Religion2Code | Secondary religion |
| 12 | Actor1Type1Code | GOV, MIL, REB, OPP, etc. |
| 13 | Actor1Type2Code | Secondary type |
| 14 | Actor1Type3Code | Tertiary type |

### Actor2 (15-24) — Same structure as Actor1

### Event (25-34) — THE MOST IMPORTANT FIELDS
| Index | Name | Description | Values |
|-------|------|-------------|--------|
| 25 | IsRootEvent | Is this the root event or a sub-event? | 0 or 1 |
| 26 | EventCode | Full CAMEO event code | 3-4 digits (e.g., 190 = Fight) |
| 27 | EventBaseCode | Base code (first 3 digits) | e.g., 190 |
| 28 | EventRootCode | Root code (first 2 digits) | 01-20 |
| 29 | QuadClass | Cooperation/conflict quadrant | 1-4 (see below) 
| 30 | GoldsteinScale | Event impact score | -10.0 to +10.0 |
| 31 | NumMentions | Number of article mentions | int |
| 32 | NumSources | Number of distinct sources | int |
| 33 | NumArticles | Number of distinct articles | int |
| 34 | AvgTone | Average tone of source articles | float (~-10 to +10) |

### Geography — Actor1Geo (35-42), Actor2Geo (43-50), ActionGeo (51-58)
Each set has: Type, FullName, CountryCode, ADM1Code, ADM2Code, Lat, Long, FeatureID

### Metadata (59-60)
| Index | Name | Description |
|-------|------|-------------|
| 59 | DATEADDED | Timestamp batch was processed (YYYYMMDDHHMMSS) |
| 60 | SOURCEURL | URL of the source article |

## QuadClass — The Four Types of Events

| QuadClass | Meaning | Description | Typical Volume |
|-----------|---------|-------------|----------------|
| 1 | Verbal Cooperation | Diplomacy, statements of support, agreements | ~58% |
| 2 | Material Cooperation | Aid, trade, economic support | ~8% |
| 3 | Verbal Conflict | Demands, accusations, threats, protests | ~16% |
| 4 | Material Conflict | Military action, violence, sanctions, warfare | ~18% |

**For TirraMind: QuadClass 3+4 events are the primary signals.** When verbal conflict (3) transitions to material conflict (4), escalation is happening. This transition pattern is detectable via Hawkes process.

## CAMEO Root Event Codes (20 Categories)

| Code | Event Type | Goldstein | QuadClass |
|------|-----------|-----------|-----------|
| 01 | Make Public Statement | 0.0 | 1 |
| 02 | Appeal | +3.0 | 1 |
| 03 | Express Intent to Cooperate | +4.0 | 1 |
| 04 | Consult | +5.0 | 1 |
| 05 | Engage in Diplomatic Cooperation | +6.0 | 1 |
| 06 | Engage in Material Cooperation | +7.0 | 2 |
| 07 | Provide Aid | +7.4 | 2 |
| 08 | Yield | +5.0 | 2 |
| 09 | Investigate | -2.0 | 1 |
| 10 | Demand | -5.0 | 3 |
| 11 | Disapprove | -2.0 | 3 |
| 12 | Reject | -4.0 | 3 |
| 13 | Threaten | -7.0 | 3 |
| 14 | Protest | -6.5 | 3 |
| 15 | Exhibit Military Posture | -7.2 | 4 |
| 16 | Reduce Relations | -4.0 | 3 |
| 17 | Coerce | -5.0 | 4 |
| 18 | Assault | -9.0 | 4 |
| 19 | Fight | -10.0 | 4 |
| 20 | Engage in Unconventional Mass Violence | -10.0 | 4 |

**Key sub-codes we care about:**
- **163**: Impose embargo/boycott/sanctions (commodity supply disruption)
- **172**: Impose administrative sanctions (trade impact)
- **173**: Arrest/detain (political instability signal)
- **190-195**: Fight sub-codes (armed conflict intensity)
- **141-145**: Protest sub-codes (civil unrest monitoring)
- **152-155**: Military posture (tensions escalation)

## Verified Live Data (Today's Batch)

From the 10:00 UTC batch (20260324100000):
- **1,205 total events** in 15 minutes
- **699 Verbal Cooperation** (58%)
- **102 Material Cooperation** (8.5%)
- **187 Verbal Conflict** (15.5%)
- **217 Material Conflict** (18%)
- **74 conflict events with named actors in first 500 rows**

Example real events:
- FIGHTER vs IRAN | code=190 (Fight) | Goldstein=-10.0 | Iran
- MILITIA vs INTELLIGENCE | code=190 (Fight) | Goldstein=-10.0 | Iraq/Iran
- ARMENIAN vs ARMENIA | code=203 | Goldstein=-10.0 | Yerevan
- AFGHAN vs ADMINISTRATION | code=172 (Coerce) | Goldstein=-5.0 | Cuba

## Tool Design Decision

**Two modes for the GDELTTool:**

### Mode 1: `events` (Primary — Structured Events)
- Download latest N 15-min batches from raw files
- Parse the 61-column tab-separated events
- Filter by: country, region, QuadClass, EventRootCode, GoldsteinScale threshold, actor types
- Return structured events with full CAMEO codes, actors, geo, and intensity metrics
- This feeds into Hawkes process for escalation detection

### Mode 2: `articles` (Secondary — Full-Text Search)
- Use DOC API for keyword/theme search
- Return articles with tone scores, URLs, dates
- Search GKG themes: `theme:TERROR`, `theme:ENV_OIL`, `theme:FOOD_SECURITY`
- Useful for deeper investigation of specific topics
- Max 250 results, last 3 months

**Primary mode is `events`** because:
1. No rate limiting on raw file downloads
2. Full structured data (CAMEO codes, Goldstein, actors, geo)
3. Feeds directly into the Hawkes process (Phase 7)
4. Can aggregate multiple batches for time-windowed analysis

## Implementation Notes

### Parameters Schema
```
mode: "events" | "articles" (default: "events")
hours_back: 1-24 (for events mode, how many hours of batches to fetch, default: 1)
country_filter: optional country code (e.g., "IR", "RU", "CN") — filter ActionGeo
min_goldstein: optional float, only return events below this threshold (for conflict detection)
quad_class: optional list [3, 4] — filter to conflict events only
event_codes: optional list of root codes to include (e.g., ["19", "18", "17"])
query: required for articles mode — search keywords
```

### Caching Strategy
- Cache key: `("gdelt_events", {"batch_timestamp": "YYYYMMDDHHMMSS"})` — one cache entry per batch
- Once a batch is downloaded, it never changes (GDELT publishes immutable snapshots)
- TTL can be very long (24h+) since batches are immutable
- For articles mode: standard 6hr TTL

### Rate Limiting
- Raw files: no rate limit, but add 0.2s delay between batch downloads to be polite
- DOC API: 1-2s delay between requests to avoid 429

### Dependencies
- No new dependencies needed — `urllib.request` + `zipfile` + `io` from stdlib
- Or use existing `httpx` for consistency with other tools

### Output Format
Events mode returns:
```python
{
    "events": [
        {
            "id": "1295798132",
            "date": "20260324",
            "actor1": {"name": "IRAN", "country": "IRN", "type": "GOV"},
            "actor2": {"name": "ISRAEL", "country": "ISR", "type": "MIL"},
            "event_code": "190",
            "event_root": "19",
            "event_description": "FIGHT",
            "quad_class": 4,
            "goldstein": -10.0,
            "num_mentions": 25,
            "num_sources": 12,
            "avg_tone": -7.3,
            "location": {"name": "Tehran", "country": "IR", "lat": 35.6892, "lon": 51.389}
        }
    ],
    "summary": {
        "total_events": 4800,
        "conflict_events": 1200,
        "batches_fetched": 4,
        "time_range": "09:00-10:00 UTC",
        "top_conflict_countries": {"IR": 45, "RU": 38, "UA": 35, ...}
    }
}
```

## Risks

1. **Volume management** — 1,200 events/batch × 4 batches/hour = 4,800 events. Need efficient filtering.
2. **False positives** — GDELT's NLP isn't perfect. Some events are mis-coded. Use NumMentions and NumSources as confidence proxies — high-mention events are more likely real.
3. **Temporal lag** — Despite "15-min updates," events may reference articles from hours ago. The DATEADDED is the processing time, not the event time.
4. **Noise ratio** — Verbal cooperation is 58% of events. Most events are noise. Filtering to QuadClass 3+4 and known conflict root codes is essential.
5. **DOC API rate limiting** — 429 on burst. The raw files have no such issue. Prefer raw files for primary access.
6. **Geographic ambiguity** — Some locations are incorrectly geocoded (common city names, translation errors). Use ActionGeo_Type to filter for higher-confidence geolocations (Type 3 = city, Type 4 = landmark are higher precision than Type 1 = country).

## Market-Relevant GKG Themes (for DOC API)

These themes are available in the DOC API via `theme:THEME_NAME`:
- `ENV_OIL` — Oil-related coverage
- `FOOD_SECURITY` — Food supply/price
- `ECON_TRADE` — Trade agreements/disputes
- `MILITARY` — Military operations
- `TERROR` — Terrorism events
- `SANCTIONS` — Sanctions coverage
- `PROTEST` — Protests and civil unrest
- `NATURAL_DISASTER` — Disasters affecting supply chains

Full theme list: `http://data.gdeltproject.org/api/v2/guides/LOOKUP-GKGTHEMES.TXT`

## Commodity-Country Mapping (For Cross-Referencing)

When conflict events occur in these countries, they have direct commodity implications:

| Country | Commodity Impact | Mechanism |
|---------|-----------------|-----------|
| Saudi Arabia (SA) | Oil, Energy | Production disruption |
| Iran (IR) | Oil | Sanctions, Strait of Hormuz |
| Russia (RS) | Oil, Gas, Wheat, Palladium | Sanctions, supply disruption |
| Ukraine (UP) | Wheat, Corn, Sunflower | Production disruption |
| Iraq (IZ) | Oil | Conflict-driven supply risk |
| Libya (LY) | Oil | Civil instability |
| Venezuela (VE) | Oil | Political crisis |
| China (CH) | Rare earths, Manufacturing | Trade war, Taiwan tensions |
| Taiwan (TW) | Semiconductors | Invasion risk → chip supply |
| Australia (AS) | Iron ore, Coal, LNG | China trade tensions |
| Chile (CI) | Copper | Labor unrest, nationalization |
| DRC (CG) | Cobalt | Conflict minerals |
| South Africa (SF) | Platinum, Palladium | Labor strikes |
| Nigeria (NI) | Oil | Niger Delta instability |

## Files Affected (Implementation Phase)

- `agent/tools/gdelt.py` — New file: GDELTTool class
- `agent/cli.py` — Register GDELTTool in build_tool_registry()
- `agent/learning/bandit.py` — Add `geopolitical_intelligence` arm to DEFAULT_ARMS

---

## Related

- [[gdelt_spec|Spec: Gdelt]]

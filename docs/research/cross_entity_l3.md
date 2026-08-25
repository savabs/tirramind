---
title: "Research: L3 Cross-Entity Pattern Detection"
tags:
  - doc/research
  - phase/11
  - topic/surveillance
  - topic/convergence
  - layer/feature-engineering
---

# Research: L3 Cross-Entity Pattern Detection

## Goal

Move from L2 (entity-resolved observations per tool) to L3 (cross-domain entity combinations). L2 answers "what is entity X doing?" L3 answers "what hidden pattern emerges when we combine what entities X, Y, Z are doing across different data domains?"

This is the moat. Nobody else combines insider filings + vessel tracking + crypto flows + geopolitical events at entity resolution.

---

## Current L2 Inventory

| Tool | Entity Type | Key | Observation Type | Example |
|------|------------|-----|-----------------|---------|
| insider_filings | company, person | CIK | insider_trade | CEO of Exxon sells $2M |
| form144 | company, person | CIK | form144_filing | CFO of Chevron files intent to sell |
| whale_alert | wallet | BTC address | btc_transfer | 1000 BTC moved from wallet X |
| ais_vessel | vessel | IMO/MMSI | vessel_position, port_call | Tanker diverts from Russian port |
| gdelt | country | FIPS code | geopolitical_event | US sanctions Russia (Goldstein -9.0) |

## What's Missing for L3

### 1. No Entity Link Table

The current schema has `entities`, `entity_aliases`, `entity_observations`. There is **no way to express "entity A is related to entity B"** — e.g., "company:Exxon operates vessel:IMO9000001" or "country:US sanctions country:RU which affects company:Lukoil".

**Need:** An `entity_links` table:
```sql
CREATE TABLE entity_links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id_a TEXT NOT NULL REFERENCES entities(entity_id),
    entity_id_b TEXT NOT NULL REFERENCES entities(entity_id),
    link_type TEXT NOT NULL,      -- "operates", "headquartered_in", "sanctioned_by"
    confidence REAL DEFAULT 1.0,
    source TEXT NOT NULL,          -- where the link came from
    created_at REAL NOT NULL,
    metadata_json TEXT,
    UNIQUE(entity_id_a, entity_id_b, link_type)
);
```

### 2. No Temporal Co-occurrence Query

Currently `query_entity_observations()` takes a single `entity_id`. There is **no way to ask "find time windows where entity A and entity B both have observations within ±N hours"**.

**Need:** A cross-entity temporal query:
```python
def query_co_occurrences(
    entity_id_a: str,
    entity_id_b: str,
    window_hours: float = 24.0,
    since: float | None = None,
) -> list[dict]:
    """Find observation pairs from different entities within a time window."""
```

### 3. No Entity-to-Entity Resolution Across Domains

Example: company "EXXON MOBIL" (CIK 34088) in insider_filings needs to be linked to:
- Vessels it operates (via IMO owner databases)
- Country it's headquartered in (US)
- GDELT events mentioning it

Currently these are separate entity universes. L3 bridges them.

---

## Concrete L3 Patterns (Ordered by Expected Signal Value)

### Pattern 1: Insider Selling × GDELT Conflict Escalation

**Hypothesis:** When insiders at defense/energy companies sell shares AND GDELT shows escalating conflict in regions those companies operate in → potential foreknowledge of supply disruption or contract loss.

**Entities involved:** person (CIK) → company (CIK) → country (FIPS) → GDELT events
**Link needed:** company → country (headquartered_in, operates_in)
**Signal:** Temporal co-occurrence of insider_trade observations + geopolitical_event observations within 72h window, filtered to negative Goldstein scores.

**Why this is valuable:** This pattern is invisible to anyone looking at only SEC filings OR only GDELT. The signal is in the junction.

### Pattern 2: Vessel Rerouting × Sanctions Escalation

**Hypothesis:** When vessels divert away from sanctioned-nation ports AND GDELT shows sanctions-related events → early detection of sanctions enforcement or evasion.

**Entities involved:** vessel (IMO) → country (FIPS port) → GDELT events
**Link needed:** vessel → country (via port_call destination or position geo-fence)
**Signal:** Vessel changes destination away from a country that has recent negative GDELT events (quad_class=4, event_root 17-19).

**Why this is valuable:** Physical signal (vessel rerouting) is T0 — it happens before news. GDELT is T+15min. Together they detect sanctions evasion in near-real-time.

### Pattern 3: Whale Crypto Transfers × Geopolitical Events

**Hypothesis:** Large BTC movements correlate with geopolitical instability — capital flight signal.

**Entities involved:** wallet (BTC address) → country (FIPS) via geo-IP or exchange attribution
**Link needed:** wallet → country (exchange_country attribution)
**Signal:** Spike in BTC transfer volume within 24h of high-impact GDELT events (Goldstein ≤ -5).

**Why this is valuable:** Crypto flows are harder to trace than wire transfers. Cross-referencing with GDELT reveals which geopolitical events drive capital flight, which predicts FX and commodity moves.

---

## Implementation Strategy

### Phase 11a: Cross-Entity Infrastructure

Build the plumbing L3 patterns need:

1. **`entity_links` table** — schema + CRUD in PipelineStore
2. **`query_co_occurrences()`** — temporal co-occurrence query across entity observation streams
3. **`link_entities()`** — API to create typed links between entities
4. **`query_entity_links()`** — query links by entity, type, or confidence
5. **Company → Country seed links** — use SEC tickers data (already loaded in entity.py) to link companies to their HQ country

### Phase 11b: First L3 Pattern — Insider × GDELT

Implement Pattern 1 as the proof-of-concept:

1. **Link companies to countries** using metadata from SEC filings (state of incorporation → FIPS code mapping)
2. **Build a co-occurrence detector** that finds (insider_trade, geopolitical_event) observation pairs within a configurable time window for linked entities
3. **Score the pattern** using conditional MI: does knowing about both events together predict price moves better than either alone?
4. **Store L3 observations** at depth_level=3 with the combined evidence

### Phase 11c: Additional L3 Patterns

Implement Patterns 2 and 3 once the infrastructure from 11a is validated.

---

## Infrastructure Design Decisions

### Entity Link Types (Initial Set)

| Link Type | A → B | Example |
|-----------|-------|---------|
| `headquartered_in` | company → country | Exxon → US |
| `operates_in` | company → country | Shell → NL, UK, NG |
| `registered_under` | vessel → country | Tanker → flag state |
| `port_call_to` | vessel → country | Derived from port_call observations |
| `exchange_based_in` | wallet → country | Coinbase wallet → US |

### Co-occurrence Window

Default: 72 hours. Configurable per pattern. Insider filings have T+2 disclosure lag, so the window must accommodate that. GDELT is near-real-time (15min). AIS is real-time.

### Scoring

Use the existing `depth_eval.py` conditional MI estimator:
- X_existing = L2 observations (single-domain)
- X_new = L3 co-occurrence features (cross-domain)
- Y = target variable (price move, spread change)
- If MI(L3; Y | L2) > 0 → L3 adds signal beyond L2

---

## Risks

1. **Spurious correlations** — Cross-domain co-occurrences will generate many false patterns. Need statistical significance testing (permutation tests).
2. **Sparse data** — Entity linkage across domains may have low coverage initially. Start with company→country links which have the highest coverage.
3. **Time alignment** — Different tools have different latency (AIS: real-time, GDELT: 15min, SEC: T+2 days). Windows must be asymmetric.
4. **Schema migration** — Adding `entity_links` table requires careful migration for existing DBs.
5. **Combinatorial explosion** — N entities × M entities × T time windows = huge search space. Must be selective about which pairs to check.

## Data Requirements

- No new external data sources needed
- All queries are over existing entity_observations table
- Company→country mapping available from SEC EDGAR metadata
- Vessel→country mapping available from port_call data already stored

## Related

- [[gdelt_l2]]
- [[ais_vessel_l2]]
- [[whale_alert_l2]]
- [[deep_surveillance_tools]]
- [[cross_entity_l3_spec]]

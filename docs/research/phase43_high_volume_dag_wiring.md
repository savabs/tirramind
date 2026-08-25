---
title: "Research: Phase 43 — High-Volume DAG Wiring"
tags:
  - doc/research
  - phase/43
  - topic/pipeline
  - topic/diversity
  - layer/surveillance
---

# Research: Phase 43 — High-Volume DAG Wiring

## Context

Phase 42 lifted entity observation entropy from 0.17 → 0.29 nats by wiring 8 dormant L2 tools.
The target of ≥1.0 nats was not reached. Post-session analysis confirmed the correct diagnosis:

- Volume asymmetry is structural: `instrument_universe` persists ~68k OHLC rows/run; all new tools
  combined add tens of rows per snapshot.
- Tool coverage is also a gap: 35 of 51 data tools (68.6%) are still unwired in `daily_collection`.
- The highest-leverage next move is NOT backfill — it is wiring the high-volume entity-creating tools
  that are already L2-ready.

## Target Tools

All four tools are already registered with `pipeline_store` in `build_tool_registry` in `agent/cli.py`.
No new Python code is needed — only DAG nodes + test updates.

### ais_vessel (priority 1 — highest volume)

- **Source:** Finnish Transport Infrastructure Agency Digitraffic AIS
- **URL:** https://meri.digitraffic.fi/api/ais/v1/vessels
- **Rate limits:** None detected; sub-second responses
- **Modes:** area, vessel, port_calls, destination_flow
- **Chosen mode:** `area`, `area=full_baltic`, `limit=500`
  - `full_baltic` bbox: (54.0°N–66.0°N, 9.0°E–31.0°E)
  - Source pool: 18K+ vessels
  - L2 obs types: `vessel_position` (per vessel per run), `port_call` (via port_calls mode — separate call)
  - Entity type created: `vessel` (new entity type — not yet in graph)
- **Volume estimate:** 100–500 vessel entities per run, each with a `vessel_position` observation
- **Expected entropy impact:** High — 500 new entity rows/run × daily accumulation; after 30 days:
  ~15k vessel observations vs current ~200-obs non-instrument pool

### gov_contracts (priority 2)

- **Source:** USASpending.gov API (federal contract awards)
- **Modes:** recent, top, agency, search
- **Chosen mode:** `recent`, `limit=100`
  - Returns latest 100 federal contract awards
  - L2 entities: `company` (recipient), `organization` (agency)
  - L2 obs type: `contract_award` per company
  - Also creates `awarded_by` company→organization links
- **Volume:** ~100 award records/run, new company/org entities accumulate over days

### sanctions_monitor (priority 3)

- **Source:** OFAC SDN list + UN Security Council consolidated list
- **Modes:** recent, search, programs
- **Chosen mode:** `recent`, `days_back=90`, `limit=100`
  - Returns most-recently-added designations across OFAC + UN
  - L2 entities: `person`, `company`
  - L2 obs type: `sanctions_listing`
  - Also creates `located_in` country links
- **Volume:** Sanctions lists change slowly (~5–30 new entries/day across sources). Over 90 days: ~100 records

### patent_filings (priority 4)

- **Source:** USPTO PatentsView API
- **Modes:** search, assignee, trends
- **Constraint:** `search` mode (the only entity-persisting mode) requires at least one of `query`,
  `assignee`, or `cpc_class`.
- **Chosen mode:** `search`, `cpc_class=G06N` (machine learning / AI patent class)
  - Broad AI/ML class — captures patents from major tech companies (Alphabet, Microsoft, Samsung, etc.)
  - L2 entities: `company` (assignee), US `country` node
  - L2 obs type: `patent_filing`
  - Limit: 50 (PatentsView default max per page)
- **Volume:** ~50 records/run, creates company entities for top tech assignees (relatively stable set)
- **Note:** This mode is narrower than the others — it covers AI/ML patents only. The GNN-guided
  expansion doctrine applies: after running, check if company nodes from this source are data-starved
  before broadening to additional CPC classes.

## Observation Type Registry

All four tools use pre-existing obs types already in `OBSERVATION_TYPES`:
- `vessel_position` — ais_vessel
- `contract_award` — gov_contracts
- `sanctions_listing` — sanctions_monitor
- `patent_filing` — patent_filings

No schema changes required.

## Test Impact

- `tests/test_pipeline_registry.py`: 3 node-count assertions change (18 → 22)
- 4 new per-node config tests needed (matching the Phase 42 pattern)

## Files to Modify

- `agent/pipeline/dags/daily_collection.py` — add 4 nodes
- `tests/test_pipeline_registry.py` — update 3 count assertions + add 4 tests

## Related

- [[phase43_high_volume_dag_wiring_spec]]
- [[phase43_high_volume_dag_wiring_task]]
- [[phase42_entity_diversity_expansion]]
- [[chat_checkpoint_2026-04-21_phase42_complete]]

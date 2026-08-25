---
title: "Research: Phase 18 — Tier 1 L2 Tool Expansion"
tags:
  - doc/research
  - phase/18
  - topic/surveillance
  - topic/world-model
  - layer/surveillance
  - layer/world-model
---

# Research: Phase 18 — Tier 1 L2 Tool Expansion

## Motivation

Phase 16's GNN diagnostic revealed that the single most critical gap after
entity linking (Phase 17, now complete) is the **organization entity type: zero
tools produce it**. The Tier 1 tools ranked by the diagnostic are:

| Rank | Tool | Score | Primary Contribution |
|------|------|-------|---------------------|
| 1 | sanctions_monitor | 0.87 | Fills organization gap, cross-domain links |
| 2 | gov_contracts | 0.82 | Reinforces company+organization, procurement signal |
| 3 | supply_chain_monitor | 0.76 | Company→company edges via supply chain |

All three are already implemented as L1 tools. The upgrade adds:
- `pipeline_store` kwarg to constructors
- `_persist_entities()` / `_persist_entities_inner()` methods
- Entity registration, observation storage, and `link_entities()` calls
- New observation types where existing ones don't fit

## Current L1 Tool Architecture

All three tools follow the same pattern:
- Constructor: `__init__(self, cache=None)` — no pipeline_store
- `execute(**kwargs) → ToolResult` with mode dispatch
- Return structured dicts with results
- No entity persistence, no graph integration

### sanctions_monitor.py
- **Modes:** search, recent, programs
- **Data sources:** OFAC SDN (CSV, 5.5MB), UN Consolidated List (XML, 2MB)
- **Record fields:** source, entity_id, name, type (individual/entity/vessel),
  programs[], listed_date, last_updated, nationality, aliases[], remarks
- **Update cadence:** Weekly (OFAC), irregular (UN)

### gov_contracts.py
- **Modes:** recent, top, agency, search
- **Data sources:** USASpending.gov (US), Contracts Finder (UK)
- **Record fields:** award_id, recipient, amount_usd, agency, sub_agency,
  award_type, start_date, end_date, description
- **Update cadence:** Daily (USASpending), daily (UK)

### supply_chain_monitor.py
- **Modes:** producer_prices, import_prices, pressure_index
- **Data source:** BLS PPI + Import Prices (free, 25 req/day without key)
- **Record fields:** series_id → label, sector, values[{period, value}], signals
- **Update cadence:** Monthly (BLS publishes ~2 weeks after month end)

## L2 Upgrade Design

### Entity Type + Observation Type Mapping

| Tool | Entity Type(s) | Observation Type | New Types Needed |
|------|---------------|-----------------|-----------------|
| sanctions_monitor | `organization` (entities), `person` (individuals), `vessel` (vessels) | `sanctions_listing` (NEW) | observation: `sanctions_listing` |
| gov_contracts | `company` (recipients), `organization` (agencies) | `contract_award` (NEW) | observation: `contract_award` |
| supply_chain_monitor | `topic` (sector indices) | `price_movement` (NEW) | observation: `price_movement` |

**New observation types needed (3):**
- `sanctions_listing` — a sanctioned entity being listed/updated
- `contract_award` — a government contract being awarded to a company
- `price_movement` — a producer price index change for a sector

These must be added to `OBSERVATION_TYPES` in `graph_builder.py`.

### Entity Link Design

| Tool | Link Type | Direction | Confidence | Metadata |
|------|-----------|-----------|------------|----------|
| sanctions_monitor | `sanctioned_under` | person/org/vessel → country | 0.95 | `{programs, source}` |
| gov_contracts | `awarded_by` | company → organization (agency) | 1.0 | `{award_id, amount_usd}` |
| gov_contracts | `operates_in` | company → country | 0.9 | `{region}` |
| supply_chain_monitor | *(no entity links)* | — | — | Sector-level, no natural cross-entity |

**Rationale:**
- `sanctioned_under`: Programs map to countries (IRAN→Iran, CUBA→Cuba, UKRAINE→Ukraine).
  The entity → country link encodes "this entity operates in / is associated with
  jurisdiction X". Confidence 0.95 because OFAC/UN attribution is authoritative
  but programs sometimes span multiple countries.
- `awarded_by`: Contract award is an explicit factual relationship between
  recipient company and awarding agency. Confidence 1.0 — the USASpending
  API is authoritative.
- `operates_in`: If a company receives a US government contract, it operates
  in the US. Same for UK contracts. Confidence 0.9 — the company has a
  US/UK presence but may be headquartered elsewhere.
- supply_chain_monitor has no natural entity links — it tracks sector-level
  price indices, not entity-to-entity relationships.

### Sanctions → Country Mapping

Need a program-to-country lookup for `sanctioned_under` links:

```python
_PROGRAM_COUNTRY = {
    "IRAN": "IR", "CUBA": "CU", "UKRAINE-EO13662": "UA",
    "SYRIA": "SY", "DPRK": "KP", "SDGT": None,  # global terrorism, no single country
    "RUSSIA": "RU", "CHINA": "CN", "VENEZUELA": "VE",
    "MYANMAR": "MM", "MALI": "ML", "CAR": "CF", "DRC": "CD",
    "SOL": "SO", "YEM": "YE", "LBY": "LY", "HTI": "HT",
    "ISIL": None,  # transnational
    "BALKANS": None,  # multi-country
}
```

Only create links when a program maps to a specific country. Skip multi-country
/ global programs (SDGT, ISIL, BALKANS).

### Entity ID Strategy

| Tool | Entity Type | ID Key |
|------|------------|--------|
| sanctions_monitor | organization | `entity_id_from_key("organization", normalize_company_name(name))` |
| sanctions_monitor | person | `entity_id_from_key("person", name.lower().strip())` |
| sanctions_monitor | vessel | `entity_id_from_key("vessel", name.lower().strip())` |
| gov_contracts | company (recipient) | `entity_id_from_key("company", normalize_company_name(recipient))` |
| gov_contracts | organization (agency) | `entity_id_from_key("organization", normalize_company_name(agency))` |
| supply_chain_monitor | topic (sector) | `entity_id_from_key("topic", series_id)` |

**Note:** sanctions entities use `organization` not `company` — they are often
state-owned enterprises, front companies, or non-corporate entities. The
`organization` type was designed for this exact use case.

### Graph Builder Changes

Add 3 new observation types to `OBSERVATION_TYPES`:
```python
"sanctions_listing",
"contract_award",
"price_movement",
```

No new entity types needed — all types exist.

### Edge Type Mapping (GNN)

After Phase 18, the GNN will have these new edge types:
- `("person", "sanctioned_under", "country")`
- `("organization", "sanctioned_under", "country")`
- `("vessel", "sanctioned_under", "country")`
- `("company", "awarded_by", "organization")`
- `("company", "operates_in", "country")`

This addresses the critical diagnostic finding: `organization` goes from 0 tools
→ 2 tools (sanctions + gov_contracts agencies).

## Depth Roadmap (Signal Depth Doctrine)

### sanctions_monitor
- **L1 (current):** Aggregate counts, program summaries
- **L2 (Phase 18):** Entity-resolved SDN/UN entries per person/org/vessel,
  timestamped observations, cross-country program links
- **L3 (future):** Cross-domain: sanctioned entity appears in insider_filings
  (person) or AIS vessel tracking (vessel) or gov_contracts (company)

### gov_contracts
- **L1 (current):** Aggregate awards, top contracts, agency summaries
- **L2 (Phase 18):** Entity-resolved recipients + agencies, per-company
  contract award observations, company→agency links
- **L3 (future):** Company winning contracts across US+UK = multinational
  gov coordination; Cross with lobbying (company lobbies then wins contract)

### supply_chain_monitor
- **L1 (current):** Sector-level PPI series, signals
- **L2 (Phase 18):** Sector-as-topic entity, per-sector time-series observations
- **L3 (future):** PPI sector spikes correlated with company earnings in that sector

## Files Affected

### New/Modified Implementation Files
- `agent/tools/sanctions_monitor.py` — add pipeline_store, _persist_entities, links
- `agent/tools/gov_contracts.py` — add pipeline_store, _persist_entities, links
- `agent/tools/supply_chain_monitor.py` — add pipeline_store, _persist_entities
- `agent/models/gnn/graph_builder.py` — add 3 observation types

### New Test Files
- `tests/test_sanctions_monitor_l2.py`
- `tests/test_gov_contracts_l2.py`
- `tests/test_supply_chain_monitor_l2.py`

### Existing Test Files (potential regression)
- `tests/test_gnn_integration.py` — may need update for new obs types
- `tests/test_graph_builder.py` — may need update for new OBSERVATION_TYPES count
- `tests/test_graph_builder_expanded.py` — may need update

## Risks

1. **sanctions_monitor name normalization:** Sanctioned entity names are often
   non-Western (Arabic, Cyrillic transliterations). `normalize_company_name()`
   may strip too aggressively. Mitigation: use it only for SDN_Type "entity",
   use simpler normalization for individuals/vessels.
2. **gov_contracts UK data model:** UK Contracts Finder uses OCDS standard,
   which has different field names than USASpending. Normalization already
   handled in L1 execute(), so L2 just reads the normalized output.
3. **supply_chain_monitor has no natural entity links:** This is fine — not
   every tool needs links. The sector-as-topic entities are standalone
   observation channels that enrich the GNN's topic node type.
4. **New observation types require graph_builder update:** Must update
   OBSERVATION_TYPES list before any L2 tool can successfully persist.
   This is the first implementation step.

## External References

- OFAC SDN format: https://home.treasury.gov/policy-issues/financial-sanctions/specially-designated-nationals-list-sdn-list/sdn-data-formats
- UN Consolidated List: https://www.un.org/securitycouncil/content/un-sc-consolidated-list
- USASpending API: https://api.usaspending.gov/docs/
- UK Contracts Finder: https://www.contractsfinder.service.gov.uk/apidocumentation/home
- BLS Public API: https://www.bls.gov/developers/api_signature_v2.htm

## Related

- [[entity_linking_layer]] — Phase 17 (prerequisite)
- [[l2_tool_expansion]] — Phase 13 (pattern reference)
- [[gnn_guided_tool_expansion]] — Phase 16 (diagnostic ranking)
- [[tier1_tool_expansion_spec]] — implementation spec
- [[7b-AF_sanctions_monitor]] — original tool research
- [[7b-G_gov_contracts]] — original tool research
- [[7b-AO_supply_chain_monitor]] — original tool research

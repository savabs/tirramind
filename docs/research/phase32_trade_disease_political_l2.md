---
title: "Research: Phase 32 — Trade + Disease + Political L2"
tags:
  - doc/research
  - phase/32
  - topic/l2-expansion
  - layer/surveillance
---

# Research: Phase 32 — Trade + Disease + Political L2

## Goal

Add L2 entity persistence to four tools: `comtrade`, `transport_throughput`,
`disease_surveillance`, `political_risk`.  This phase addresses cross-domain
entity gaps identified in [[l2_expansion_roadmap]] Phase 32.

## Current State

All four tools are L1-only: `__init__(self, cache=None)` with no
`pipeline_store`, no `_persist_entities`.  Each has well-structured response
`data` dicts with clear entity candidates.

## Tool Audit

### comtrade (ComtradeTool)

- **Modes:** flows, commodity, partners
- **Data:** `_parse_trade_records` → `records[]` with `reporter`, `reporter_code` (M49), `partner`, `partner_code`, `flow` (X/M), `trade_value_usd`, `commodity_code`, `period`
- **Entity mapping:** country entities for reporter and partner.  ISO-3 codes available via `_M49_TO_ISO`.
- **Result data shape:** `{"mode": ..., "reporter": <ISO3>, "partner": <ISO3>, "records": [...], "record_count": ...}`
- **Decision:** Persist on reporter country.  Bilateral pair info goes into the observation value dict.  Comtrade uses ISO-3 codes (USA, CHN) but our country entities use ISO-2 (US, CN).  Need ISO3→ISO2 conversion.
- **Obs type:** `trade_flow`

### transport_throughput (TransportThroughputTool)

- **Modes:** recent, trend, port, compare
- **Data:** `records[]` / `series[]` / `ports[]` / `comparison[]` — all have `border` field ("US-Canada Border" or "US-Mexico Border"), `measure`, and volume data.
- **Entity mapping:** Country entities for US, CA, MX.  Border string maps directly.
- **Decision:** Persist on country entities.  Each border observation → two country obs (US + partner).  Recent/port modes have per-border records; trend/compare have time series.
- **Obs type:** `border_throughput`

### disease_surveillance (DiseaseSurveillanceTool)

- **Modes:** wastewater, outbreaks, eu_surveillance, genomics
- **Data shapes vary by mode:**
  - wastewater: `summaries[]` with `state`, `detection_rate`, `mean_concentration`
  - outbreaks: `entries[]` with `country_parsed`, `disease_parsed`
  - eu_surveillance: `records[]` with `country_code`/`country`
  - genomics: flat dict with `organism`, `current_count`, `yoy_ratio`, `signal`
- **Entity mapping:** US (wastewater is US-only, aggregate to country US), country entities from WHO DON / ECDC country fields, US (genomics is typically US-relevant but organism-specific, not country-specific)
- **Decision:** Persist on country entities.  Wastewater → "US".  Outbreaks → extract country from `country_parsed`.  EU surveillance → use `country_code`.  Genomics → skip (no country dimension, organism-level signal).
- **Obs type:** `pathogen_level`

### political_risk (PoliticalRiskTool)

- **Modes:** candidates, filings, expenditures
- **Data:** `records[]` with FEC-specific fields.  Candidates have `candidate_id`, `name`, `party`, `office`, `state`.  Expenditures have `candidate_id`, `committee_id`, `support_oppose`, `expenditure_amount`.
- **Entity mapping:** Per [[l2_expansion_roadmap]], persist on **person** entities (candidate).  Candidate_id is the stable entity key.
- **Decision:** Persist on person entities keyed by `candidate_id`.  Candidates mode → one obs per candidate with fundraising status.  Expenditures mode → one obs per targeted candidate with aggregated spend.  Filings mode → committee-level, skip person persistence (committee is not a person entity).
- **Obs type:** `campaign_finance`

## ISO-3 → ISO-2 Mapping (comtrade)

Comtrade uses ISO-3 alpha codes (USA, CHN, DEU).  Our entity graph uses ISO-2
(US, CN, DE) for country entities.  Need a lookup in the tool.  Other L2 tools
with ISO-3 (migration_flows) already solved this with a `_normalize_country_code`
mapping.  Reuse that pattern.

## Graph Builder Updates

Current: 39 OBSERVATION_TYPES, ENRICHMENT_DIM = 48.
Add 4: `border_throughput`, `campaign_finance`, `pathogen_level`, `trade_flow`.
New: 43 OBSERVATION_TYPES, ENRICHMENT_DIM = 52.

## Risks

- **Comtrade free tier:** Only 10 records per request.  L2 persistence may have
  very few records to work with.  Not a blocker — sparse observations are fine.
- **Disease surveillance multi-source:** 4 different data sources with different
  schemas.  Persistence logic must handle mode-specific data shapes.
- **Political risk FEC DEMO_KEY:** Rate-limited.  Persistence must be non-fatal.
- **Transport throughput:** Only US-Canada/US-Mexico data.  Only 3 country entities
  ever created (US, CA, MX).

## Existing L2 Pattern (Reference: food_security.py)

```python
# Constructor: pipeline_store kwarg
def __init__(self, cache=None, pipeline_store=None):
    self._store = pipeline_store

# TYPE_CHECKING import guard
if TYPE_CHECKING:
    from agent.pipeline.store import PipelineStore

try:
    from agent.pipeline.entity import entity_id_from_key as _entity_id_from_key
except ImportError:
    _entity_id_from_key = None

# Non-fatal wrapper
def _persist_entities(self, data, mode):
    if self._store is None or _entity_id_from_key is None:
        return {"obs_count": 0}
    try:
        return self._persist_entities_inner(data, mode)
    except Exception:
        log.exception("... persistence failed (non-fatal)")
        return {"obs_count": 0}

# Inner implementation
def _persist_entities_inner(self, data, mode):
    # register entity, store observation, return count dict
```

## Related

- [[l2_expansion_roadmap]]
- [[phase32_trade_disease_political_l2_spec]]
- [[phase32_trade_disease_political_l2]]
- [[chat_checkpoint_2026-04-17_phase31_complete]]

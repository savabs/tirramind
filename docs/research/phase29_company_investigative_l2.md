---
title: "Research: Phase 29 — Company + Investigative L2"
tags:
  - doc/research
  - phase/29
  - topic/entity-linking
  - topic/bankruptcy
  - topic/foia
  - topic/academic-preprints
  - layer/surveillance
  - layer/world-model
---

# Research: Phase 29 — Company + Investigative L2

**Date:** 2026-04-16
**Prior art:** [[7b-E_bankruptcy_court]], [[7b-S_foia_logs]], [[7b-M_academic_preprints]]

## Goal

Upgrade three existing surveillance tools to L2 entity-resolved persistence so company and person nodes receive investigative-quality observations. After Phase 28 enriched country nodes (6 obs types), company/person nodes are the next starved entity class.

## Current Architecture

### Tools to upgrade

| Tool | File | Modes | Entity targets | Obs type |
|------|------|-------|----------------|----------|
| `bankruptcy_court` | `agent/tools/bankruptcy_court.py` | us_bankruptcy, sec_enforcement, sec_bankruptcy, uk_insolvency | company | `bankruptcy_status` |
| `foia_requests` | `agent/tools/foia_requests.py` | muckrock, whatdotheyknow | company, person | `investigation_signal` |
| `academic_preprints` | `agent/tools/academic_preprints.py` | papers, trials, trending | company, person, topic | `research_velocity` |

All three already accept `pipeline_store` in `__init__` (added during Phase 7b tool creation). The `_persist_entities` / `_persist_entities_inner` pattern is identical across all L2 tools (sovereign_debt, capital_flows, global_pmi from Phase 28).

### Entity ID generation

```
company:  entity_id_from_key("company", ticker_or_name)
person:   entity_id_from_key("person", full_name_normalized)
topic:    entity_id_from_key("topic", arxiv_category_or_term)
```

### Graph builder changes

- OBSERVATION_TYPES: 32 → 35 (add `bankruptcy_status`, `investigation_signal`, `research_velocity`)
- ENRICHMENT_DIM: 41 → 44 (9 base stats + 35 obs_type_dist)

## Observations

### bankruptcy_court L2

**Entity resolution strategy:** debtor name from PACER RSS / SEC filings → company entity. PACER RSS entries contain case titles like "In re: Acme Corp" — extract debtor name, normalize, register as company entity.

**Observation value schema:**
```python
{
    "source": "pacer|sec_enforcement|sec_8k|uk_gazette",
    "chapter": "7|11|13|15|None",
    "court": "sdny|del|...",
    "filing_date": "2026-04-15",
    "docket_type": "petition|motion|order",
    "severity": "terminal|restructuring|enforcement|investigation",
}
```

**Mode mapping:**
- `us_bankruptcy` → extract debtor name → company entity → `bankruptcy_status` obs
- `sec_enforcement` → extract respondent → company entity → `bankruptcy_status` obs
- `sec_bankruptcy` → extract filer name → company entity → `bankruptcy_status` obs
- `uk_insolvency` → extract company name → company entity → `bankruptcy_status` obs

### foia_requests L2

**Entity resolution strategy:** FOIA subjects and requestee agencies → company/person entities. MuckRock API returns structured fields: `agency`, `title`, `user.username`. Subject extraction from title text.

**Observation value schema:**
```python
{
    "source": "muckrock|whatdotheyknow",
    "agency": "SEC|FBI|FTC|...",
    "status": "submitted|processing|completed|rejected",
    "cluster_intensity": 3,  # number of requests about same entity in window
    "request_date": "2026-04-10",
}
```

**Entity targets:**
- Company names extracted from request titles → company entity → `investigation_signal`
- Person names from request titles → person entity → `investigation_signal`

### academic_preprints L2

**Entity resolution strategy:** arXiv authors and ClinicalTrials.gov sponsors/collaborators → company/person/topic entities.

**Observation value schema:**
```python
{
    "source": "arxiv|clinicaltrials",
    "category": "q-fin|cs.AI|...",   # arxiv mode
    "phase": "Phase 1|Phase 2|...",    # trials mode
    "paper_count": 15,
    "velocity_pct": 0.35,  # surge relative to baseline
    "date": "2026-04-15",
}
```

**Entity targets:**
- Clinical trial sponsors (pharma companies) → company → `research_velocity`
- arXiv authors with corporate affiliation → person → `research_velocity`
- arXiv categories → topic → `research_velocity`

## Risks

1. **Debtor name normalization** — PACER RSS titles are inconsistent ("In re: Acme Corp", "Acme Corporation, Debtor"). Fuzzy matching needed but keep it simple: strip common prefixes/suffixes, uppercase, use as entity key.
2. **FOIA subject extraction** — request titles are free-text. Entity extraction is heuristic (keyword matching on known company/person names). False positives acceptable — graph learns to weight.
3. **arXiv author → company** — author affiliations not always parseable. Start with clinical trial sponsors (structured) and arXiv category → topic (deterministic).
4. **Observation volume** — bankruptcy and FOIA are low-frequency signals. Graph builder handles sparse observations well.

## Data Requirements

- All three tools already fetch and return structured data via free APIs
- No new API keys or data sources needed
- Entity ID generation: `entity_id_from_key` from `agent/pipeline/entity.py` (SHA-256[:16])

## Implementation Pattern

Identical to Phase 28 L2 upgrades:
1. Add `TYPE_CHECKING` import guard for `PipelineStore`
2. Add `_persist_entities()` / `_persist_entities_inner()` methods
3. Call `_persist_entities()` from `execute()` after successful data fetch
4. Register observation types in `OBSERVATION_TYPES` (graph_builder.py)
5. Write edge-case tests per tool
6. Write integration diagnostic tests
7. Update stale count assertions across test suite

## Depth Roadmap

- **L1 (current):** Aggregate counts (e.g., "47 bankruptcy filings this week")
- **L2 (Phase 29):** Entity-resolved observations (specific company → bankruptcy_status, specific person → investigation_signal)
- **L3 (future):** Cross-entity patterns (company A's bankruptcy + company B's FOIA cluster + pharma preprint surge = sector distress signal)

## Related

- [[phase29_company_investigative_l2_spec]]
- [[phase29_company_investigative_l2]]
- [[7b-E_bankruptcy_court]]
- [[7b-S_foia_logs]]
- [[7b-M_academic_preprints]]
- [[phase28_country_macro_enrichment]]
- [[chat_checkpoint_2026-04-16_phase28_complete]]

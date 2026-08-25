---
title: "Research: Phase 33 — Organization + Grid Enrichment L2"
tags:
  - doc/research
  - phase/33
  - topic/l2-expansion
  - layer/surveillance
---

# Research: Phase 33 — Organization + Grid Enrichment L2

## Context

Phase 33 adds L2 entity persistence to the last two tools in the L2 expansion roadmap before the final diagnostic sweep (Phase 34). These tools address the most under-served entity types: `organization` (currently 1 obs source) and balancing authorities mapped to `organization`.

## Tool Audit

### `regulatory_gazette` (713 lines)

- **Class:** `RegulatoryGazetteTool(Tool)`
- **Constructor:** `__init__(self, cache=None)` — no `pipeline_store`
- **Modes:** `recent`, `search`, `agency`, `upcoming`
- **Data source:** US Federal Register API (free, no auth). UK legislation.gov.uk (secondary).
- **Key data:** `MARKET_AGENCIES` dict maps 20 aliases (sec, fed, cftc, ftc, epa, fda, etc.) to slugs and sector tags.
- **Entity extraction:** Each document has `agencies` list with agency names. The `_format_doc()` extracts `agency_names = [a.get("name") or a.get("raw_name")]`. The `MARKET_AGENCIES` dict provides clean organization IDs.
- **Obs type:** `regulatory_velocity` — per-agency rule count, types, significance flags.
- **Entity target:** `organization` — each agency becomes an organization entity.
- **Extraction strategy:** From `data["documents"]`, collect unique agency names, resolve to MARKET_AGENCIES key where possible, register as organization, store count/types/significance per agency.

### `electricity_monitor` (500 lines)

- **Class:** `ElectricityMonitorTool(Tool)`
- **Constructor:** `__init__(self, *, cache=None)` — no `pipeline_store`, keyword-only cache
- **Modes:** `demand`, `generation`, `interchange`
- **Data source:** EIA API v2 (free, key required via `TIRRA_EIA_API_KEY`)
- **Key data:** `KNOWN_REGIONS` dict maps 14 BA codes (PJM, CISO, ERCO, MISO, etc.) to names. `EIA_FUEL_TYPES` maps 8 fuel codes.
- **Entity extraction:** Each call has a `region` parameter (BA code). The BA code is a natural organization key.
- **Obs type:** `grid_demand` — per-BA demand stats (peak/trough/avg MW), generation mix, or interchange flows.
- **Entity target:** `organization` — each BA is an organization entity.
- **Important note:** `__init__` uses keyword-only args (`*, cache`). Must keep this when adding `pipeline_store`.
- **Extraction strategy:** The `region` parameter is the entity key. All 3 modes return data for that region. Register BA as organization, store mode-specific summary.

## Entity Mapping Decisions

### Organization Entity (regulatory_gazette)

- **Key:** Lowercase agency alias from `MARKET_AGENCIES` (e.g., `sec`, `fed`, `epa`). For agencies NOT in `MARKET_AGENCIES`, use first agency name lowercased with spaces→underscores.
- **Rationale:** `MARKET_AGENCIES` provides stable, curated keys. Unknown agencies get a best-effort key.
- **Obs value:** `{mode, doc_count, significant_count, types, top_topics}`.

### Organization Entity (electricity_monitor)

- **Key:** BA code uppercased (e.g., `PJM`, `CISO`, `ERCO`). These are stable identifiers from EIA.
- **Rationale:** BA codes are the canonical identifiers. The `KNOWN_REGIONS` dict maps them to names for registration.
- **Obs value:** Mode-specific — demand: `{peak_mw, trough_mw, avg_mw}`; generation: `{renewable_pct, fossil_pct, total_mwh}`; interchange: `{total_export, total_import, net_mwh}`.

### No new entity type needed

Per roadmap step 33.4: "Decide if `region` needs to be a new entity type or maps to existing `organization`." Answer: map to `organization`. Balancing authorities are organizations. The ENTITY_TYPES list already includes `organization`. No changes to entity types needed.

## Graph Builder Updates

- Add 2 new obs types: `grid_demand`, `regulatory_velocity` (alphabetical insertion)
- OBSERVATION_TYPES: 43 → 45
- ENRICHMENT_DIM: 52 → 54

## Existing L2 Pattern

Same pattern as Phases 29-32: `TYPE_CHECKING` import guard, try-import `entity_id_from_key`, `pipeline_store` kwarg, `_persist_entities` wrapper (non-fatal), `_persist_entities_inner` (actual logic).

## Risks

1. `electricity_monitor` caches plain text strings (`result` is a string, not a dict with `.data`). The `_persist_entities` call needs to extract entity info from the parameters, not from the result data, since mode handlers return formatted text.
2. `regulatory_gazette` documents have variable `agencies` lists — some docs have 0 or 3+ agencies. Need to aggregate per-agency.
3. `regulatory_gazette.execute()` returns `ToolResult` with `data={"documents": [...]}`. Can extract agencies from `data["documents"]`.
4. `electricity_monitor._demand/._generation/._interchange` return formatted strings cached in `_cache`. The result data isn't in a structured dict. Need to extract region/mode from the parameters passed to execute, not from the result.

## Related

- [[l2_expansion_roadmap]]
- [[phase32_trade_disease_political_l2]]
- [[quant_training_ground]]

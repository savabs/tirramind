---
title: "Research: Phase 31 — Remaining Country Signals"
tags:
  - doc/research
  - phase/31
  - topic/entity-linking
  - topic/consumer-sentiment
  - topic/food-security
  - topic/internet-infrastructure
  - topic/migration
  - layer/surveillance
  - layer/world-model
---

# Research: Phase 31 — Remaining Country Signals

## Goal

Upgrade four existing L1 surveillance tools to persist country-level L2 observations onto country nodes:

- `consumer_sentiment` → `consumer_confidence`
- `food_security` → `food_security`
- `internet_outages` → `internet_disruption`
- `migration_flows` → `migration_pressure`

After Phase 29, the country entity class is still the densest place to compound graph value quickly. Phase 28 added macro/monetary structure; Phase 31 adds household, food, connectivity, and population-stress structure.

## Search Log

- Local prior-art docs reviewed: `[[phase28_country_macro_enrichment]]`, `[[phase28_country_macro_enrichment_spec]]`, `[[7b-AM_consumer_sentiment]]`, `[[7b-AI_internet_infrastructure]]`, `[[starved_class_audit]]`, `[[l2_expansion_roadmap]]`
- Local code reviewed: `agent/tools/consumer_sentiment.py`, `agent/tools/food_security.py`, `agent/tools/internet_outages.py`, `agent/tools/migration_flows.py`, `agent/tools/global_pmi.py`, `agent/tools/capital_flows.py`
- External source surfaces verified from tool integrations already in code:
  - Eurostat dissemination API (`ei_bsco_m`)
  - World Bank Open Data API (`api.worldbank.org/v2`)
  - OONI API (`api.ooni.io/api/v1`)
  - RIPE Atlas API (`atlas.ripe.net/api/v2`)
  - UNHCR population API (`api.unhcr.org/population/v1`)

## Current Architecture

### Tools to upgrade

| Tool | File | Current State | Country payload already available | Proposed obs type |
|------|------|---------------|-----------------------------------|-------------------|
| `consumer_sentiment` | `agent/tools/consumer_sentiment.py` | L1, no PipelineStore | EU country series + US sentiment/CPI payloads | `consumer_confidence` |
| `food_security` | `agent/tools/food_security.py` | L1, no PipelineStore | single-country World Bank indicator payload | `food_security` |
| `internet_outages` | `agent/tools/internet_outages.py` | L1, no PipelineStore | country-scoped OONI / RIPE payloads | `internet_disruption` |
| `migration_flows` | `agent/tools/migration_flows.py` | L1, no PipelineStore | country-scoped UNHCR / WB payloads | `migration_pressure` |

### Proven L2 pattern to preserve

Phase 28 and Phase 29 already established the standard persistence pattern:

1. `TYPE_CHECKING` import for `PipelineStore`
2. guarded import of `entity_id_from_key`
3. constructor accepts `pipeline_store: PipelineStore | None = None`
4. `execute()` captures a `ToolResult` and calls `_persist_entities(result.data, mode)` after success
5. `_persist_entities()` no-ops when store/entity helper is unavailable and wraps inner persistence in non-fatal try/except
6. `_persist_entities_inner()` handles entity registration + `store_entity_observation(..., depth_level=2)`

### Country code constraints

- `consumer_sentiment.eu_confidence` uses Eurostat geo codes. Actual countries are already ISO-2; aggregates (`EU27_2020`, `EA20`) must be skipped.
- `consumer_sentiment.us_sentiment` and `inflation_reality` map directly to `US`.
- `food_security` uses ISO-2/ISO-3 plus `WLD`; `WLD` must be skipped.
- `internet_outages` uses ISO-2; aggregate `ALL` / empty country must be skipped.
- `migration_flows.displacement` / `asylum` use ISO-3 for UNHCR; these need ISO-3 → ISO-2 mapping before persistence.
- `migration_flows.remittances` already uses ISO-2/3 World Bank country codes; if a 3-letter code appears, normalize to ISO-2 before persistence.

## Observations

### consumer_sentiment

The tool already returns everything needed for L2 persistence:

- `eu_confidence`: `data[geo]` series + per-country signals (`latest`, `mom_change`, `trend`, `consecutive_decline`)
- `us_sentiment`: `signals` dict with headline sentiment and inflation-expectation fields
- `inflation_reality`: `signals` dict with CPI and expectation-gap fields

**Observation value schema:**

```python
{
    "mode": "eu_confidence|us_sentiment|inflation_reality",
    "source": "eurostat|fred|bls",
    "latest": float | None,
    "period": str | None,
    "mom_change": float | None,
    "trend": str | None,
    "expectation_gap": float | None,
}
```

### food_security

The tool already normalizes single-country World Bank indicator results and computes trend/vulnerability signals.

**Observation value schema:**

```python
{
    "mode": "production|cereal_yield|food_trade",
    "indicator": str,
    "latest_value": float | None,
    "latest_year": str | None,
    "yoy_change_pct": float | None,
    "trend_direction": "up|down|None",
    "stress_alert": str | None,
    "vulnerability": str | None,
}
```

### internet_outages

Each mode already produces a country-scoped `signals` payload:

- `censorship`: anomaly / confirmed / failure counts and anomaly-rate alerting
- `network_health`: disconnect-rate and ASN concentration metrics
- `outage_detection`: aggregate anomaly/failure rates over a date window

**Observation value schema:**

```python
{
    "mode": "censorship|network_health|outage_detection",
    "test_name": str | None,
    "anomaly_rate_pct": float | None,
    "disconnect_rate_pct": float | None,
    "confirmed_count": int | None,
    "failure_count": int | None,
    "alert": str | None,
}
```

### migration_flows

The tool spans three country-stress modes:

- `displacement`: displaced-person stock pressure
- `asylum`: asylum acceptance / closure policy pressure
- `remittances`: diaspora transfer dependence and shock

**Observation value schema:**

```python
{
    "mode": "displacement|asylum|remittances",
    "role": "asylum|origin|None",
    "year": int | None,
    "total_displaced": int | None,
    "acceptance_rate": float | None,
    "latest_value": float | None,
    "yoy_change_pct": float | None,
    "trend": str | None,
    "alert": str | None,
}
```

## Risks

1. `consumer_sentiment` mixes EU aggregates with true countries. Persisting `EU27_2020` or `EA20` would contaminate the country graph.
2. `food_security` accepts `WLD`; world aggregates must remain L1-only.
3. `internet_outages` can run without a country for OONI modes. Those global scans should not create country observations.
4. `migration_flows` requires ISO normalization across UNHCR ISO-3 and World Bank alpha-2 payloads.
5. Observation-type count changes will break stale graph-builder assertions (`OBSERVATION_TYPES`, `ENRICHMENT_DIM`) unless updated together.

## Data Requirements

- No new data sources or credentials are needed.
- All four tools already produce structured payloads suitable for L2 persistence.
- One new diagnostic file is needed to verify the full store → graph-builder path.

## Math/Algorithm Survey

This phase is structural, not algorithmically novel. The important decision is observation design:

- Prefer one observation type per tool rather than one type per tool mode.
- Preserve tool-mode detail in the observation payload.
- Keep the graph surface compact: 4 new observation types instead of 10+ mode-specific types.

That choice keeps `OBSERVATION_TYPES` high-signal and avoids fragmenting country evidence into ultra-sparse mode-specific buckets.

## Depth Roadmap

- **L1 (current):** country summaries in tool output only
- **L2 (Phase 31):** structured observations on country nodes
- **L3 (future):** multi-country motifs such as consumer collapse + food stress + internet disruption + migration surge on the same regional corridor

## Implementation Intent

- Approved: follow the exact L2 pattern used in Phases 28–29, add 4 observation types, and write both edge-case and integration diagnostics.
- Rejected: creating aggregate pseudo-country nodes like `EU27_2020`, `ALL`, or `WLD`.

## Related

- [[phase31_remaining_country_signals_spec]]
- [[phase31_remaining_country_signals]]
- [[phase28_country_macro_enrichment]]
- [[7b-AM_consumer_sentiment]]
- [[7b-AI_internet_infrastructure]]
- [[starved_class_audit]]
- [[l2_expansion_roadmap]]
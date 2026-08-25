---
title: "Spec: Phase 34 — Commodity Country Links + Diagnostic Sweep"
tags:
  - doc/spec
  - phase/34
  - topic/l2-expansion
  - layer/surveillance
  - topic/entity-linking
---

# Spec: Phase 34 — Commodity Country Links + Diagnostic Sweep

## Goal

1. Link all 20 commodity futures to their exchange country so they are no longer
   isolated nodes in the entity graph.
2. Build a reusable graph diagnostic utility for verifying entity graph health.

## Files Affected

| File | Change |
|------|--------|
| `agent/tools/instrument_universe.py` | Add `primary_exchange_country` field to `InstrumentDef`, set on all 20 commodities, extend `_persist_instrument_links` |
| `agent/models/gnn/graph_diagnostics.py` | **NEW** — diagnostic health-check utility |
| `tests/test_phase34_commodity_links.py` | **NEW** — comprehensive edge case tests |

## Implementation Steps

### 34.1: Add `primary_exchange_country` to InstrumentDef

In `instrument_universe.py`:

1. Add `primary_exchange_country: str | None = None` to `InstrumentDef` dataclass,
   after `protocol`.  Comment: `# ISO code for the country where the exchange is domiciled`.
2. Set `primary_exchange_country="US"` on all 20 commodity future definitions.
   They all trade on US exchanges (CME Group / ICE Futures US).

### 34.2: Extend `_persist_instrument_links` for `exchange_country`

In `_persist_instrument_links`:

1. Add `"exchange_country": 0` to the `counts` dict.
2. After the `tracks_protocol` block, add a new block:
   ```python
   # ── instrument → country (exchange_country) ──
   if inst.primary_exchange_country:
       exc_eid = entity_id_from_key("country", inst.primary_exchange_country)
       store.register_entity(
           entity_type="country",
           canonical_name=inst.primary_exchange_country,
           entity_id=exc_eid,
       )
       link_id = store.link_entities(
           entity_id_a=inst_eid,
           entity_id_b=exc_eid,
           link_type="exchange_country",
           source="instrument_universe",
           confidence=1.0,
           metadata={"ticker": inst.ticker},
       )
       if link_id:
           counts["exchange_country"] += 1
   ```
3. Update the `log.info` format string to include `exchange_country` count.

### 34.3: Graph Diagnostic Utility

Create `agent/models/gnn/graph_diagnostics.py` with a single function:

```python
def diagnose_graph(store: PipelineStore) -> dict[str, Any]
```

Returns a dict with:
- `entity_counts`: dict[str, int] — count per entity type
- `observation_counts`: dict[str, int] — count per observation type
- `link_counts`: dict[str, int] — count per link type
- `orphan_entities`: list[dict] — entities with zero links
- `entity_types_without_obs`: list[str] — entity types with zero observations
- `obs_types_without_instances`: list[str] — observation types with zero instances
- `total_entities`, `total_observations`, `total_links`: int

### 34.4: Tests

Cover:
- InstrumentDef: new field defaults to None, commodity futures have it set
- _persist_instrument_links: creates `exchange_country` links for commodities
- _persist_instrument_links: doesn't create `exchange_country` when field is None
- _persist_instrument_links: total link counts with new link type
- diagnose_graph: correct counts with empty store
- diagnose_graph: correct counts with populated store
- diagnose_graph: orphan detection
- diagnose_graph: obs type gap detection

## Edge Cases

1. `primary_exchange_country=None` (most non-commodity instruments) — no link created
2. Instrument already has `located_in` link AND `exchange_country` — both should exist
   (conceptually: "domiciled in US" and "traded on exchange in US" are both valid)
3. Graph builder handles new `(instrument, exchange_country, country)` edge type
   dynamically — no code change needed in graph_builder.py

## Testing Plan

1. Unit tests for InstrumentDef field presence
2. Integration tests for _persist_instrument_links with mock PipelineStore
3. Unit tests for diagnose_graph
4. Full regression suite: prior phases unaffected

## Related

- [[phase34_commodity_links_diagnostic]] — research
- [[phase34_commodity_links_diagnostic]] — task
- [[phase33_org_grid_l2_spec]] — previous spec

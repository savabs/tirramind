---
title: "Spec: L2 Tool Expansion — Entity Persistence for GNN"
tags:
  - doc/spec
  - phase/13
  - topic/surveillance
  - topic/world-model
  - layer/surveillance
  - layer/world-model
---

# Spec: L2 Tool Expansion

## Goal

Upgrade 7 VERY HIGH priority data tools from L1 (aggregate) to L2 (entity persistence) so their data feeds the Temporal Heterogeneous GNN. Expand the graph builder's type registries to accommodate new entity and observation types. Fix the existing insider_filings observation_type mismatch.

## Files Affected

### Modified
- `agent/pipeline/entity.py` — expand `EntityType` Literal
- `agent/models/gnn/graph_builder.py` — expand `ENTITY_TYPES`, `OBSERVATION_TYPES`, add unknown-type handling
- `agent/tools/insider_filings.py` — fix observation_type `"purchase"` → `"insider_trade"`
- `agent/tools/cert_transparency.py` — add L2 persistence
- `agent/tools/dns_monitor.py` — add L2 persistence
- `agent/tools/wikipedia_pageviews.py` — add L2 persistence
- `agent/tools/lobbying.py` — add L2 persistence
- `agent/tools/patent_filings.py` — add L2 persistence
- `agent/tools/defi_flows.py` — add L2 persistence
- `agent/tools/interconnection_queue.py` — add L2 persistence

### Created
- `tests/tools/test_cert_transparency_l2.py`
- `tests/tools/test_dns_monitor_l2.py`
- `tests/tools/test_wikipedia_pageviews_l2.py`
- `tests/tools/test_lobbying_l2.py`
- `tests/tools/test_patent_filings_l2.py`
- `tests/tools/test_defi_flows_l2.py`
- `tests/tools/test_interconnection_queue_l2.py`
- `tests/models/gnn/test_graph_builder_expanded.py`

## Implementation Steps

### Phase 13a: Graph Builder + Entity Module Expansion

1. **13a.1** — Fix `insider_filings.py` line ~426: change `observation_type="purchase"` to `observation_type="insider_trade"`.
2. **13a.2** — In `entity.py`, expand `EntityType` Literal to include `"domain"`, `"protocol"`, `"topic"`.
3. **13a.3** — In `graph_builder.py`, expand `ENTITY_TYPES` to: company, country, domain, organization, person, protocol, topic, vessel, wallet (sorted alphabetically).
4. **13a.4** — In `graph_builder.py`, expand `OBSERVATION_TYPES` to include: btc_transfer, cert_issued, cross_entity_pattern, dns_change, form144_filing, geopolitical_event, insider_trade, lobbying_spend, pageview_spike, patent_filing, port_call, project_status, sell_intent, tvl_change, vessel_position (sorted alphabetically).
5. **13a.5** — In `graph_builder.py` `_build_node_features()`, add fallback for unknown entity types: log warning, use index 0 instead of crashing.
6. **13a.6** — In `graph_builder.py` `build()`, iterate over all types present in `id_map.type_local` (not just hardcoded `ENTITY_TYPES`) so dynamically-added types still produce node features.
7. **13a.7** — Write `tests/models/gnn/test_graph_builder_expanded.py` testing: new types produce correct one-hot encoding, unknown types handled gracefully, insider_trade observations encode correctly.

### Phase 13b: L2 Digital Infrastructure (cert_transparency, dns_monitor, wikipedia_pageviews)

For each tool, apply the established L2 pattern:
1. Add `from typing import TYPE_CHECKING` and `if TYPE_CHECKING:` import for `PipelineStore`
2. Add `pipeline_store: PipelineStore | None = None` to `__init__`
3. Add `self._store = pipeline_store`
4. Try-import `entity_id_from_key` (with `None` fallback)
5. Add `_persist_entities()` wrapper (catch + log errors)
6. Add `_persist_entities_inner()` with entity extraction logic
7. Call `self._persist_entities(results)` in execute() before returning

**Entity mapping per tool:**

| Tool | Entity Registration | Observation |
|------|-------------------|-------------|
| cert_transparency | domain entity (key=domain name) | cert_issued: entry_timestamp, active/expired counts, issuer |
| dns_monitor | domain entity (key=domain name) | dns_change: record types, providers, ttl, low_ttl_warning |
| wikipedia_pageviews | topic entity (key=article name) | pageview_spike: z_score, latest_views, spike_ratio, date |

8. Write L2 tests per tool: mock PipelineStore, verify register_entity/store_entity_observation calls, verify error isolation.

### Phase 13c: L2 Corporate Intelligence (lobbying, patent_filings)

Same L2 pattern. Entity mapping:

| Tool | Entity Registration | Observation |
|------|-------------------|-------------|
| lobbying | company entity (key=registrant_name, normalized) | lobbying_spend: income, filing_year, filing_period, issues list |
| patent_filings | company entity (key=assignee_organization, normalized) | patent_filing: patent_number, patent_date, cpc_subgroup_id, title |

Tests per tool.

### Phase 13d: L2 Energy + DeFi (defi_flows, interconnection_queue)

Same L2 pattern. Entity mapping:

| Tool | Entity Registration | Observation |
|------|-------------------|-------------|
| defi_flows | protocol entity (key=protocol name lowercase) | tvl_change: tvl_usd, chain, change_1d_pct, change_7d_pct, category |
| interconnection_queue | company entity (key=entity_name, normalized) | project_status: nameplate_capacity_mw, energy_source_code, state, status |

Note: interconnection_queue returns text-only ToolResult. Extract from `records` list before text formatting.

Tests per tool.

### Phase 13e: Integration Verification

Write integration test that:
1. Creates a PipelineStore with all 12 L2 tools' worth of entities
2. Builds HeteroData via GraphBuilder
3. Verifies all 9 entity types produce node features
4. Verifies all 15 observation types are represented in events
5. Verifies entity cross-linking (company appears from multiple tools)

## Edge Cases

- Tool API fails: persistence must not block ToolResult return
- Empty results: _persist_entities handles empty lists gracefully
- Duplicate entities across tools: entity_id_from_key deterministic dedup
- Missing fields: skip entity registration if key field is absent
- Unicode in entity names: normalize_company_name handles NFKD
- interconnection_queue text-only return: structured data available pre-formatting

## Testing Plan

- Per-tool L2 unit tests: mock PipelineStore, verify calls, error isolation
- Graph builder expansion tests: new type encodings, fallbacks
- Integration test: multi-tool → store → graph → HeteroData pipeline
- Regression: existing 172 GNN tests must continue passing

## Related

- [[l2_tool_expansion]] — Phase 13 research
- [[l2_tool_expansion]] — Phase 13 task
- [[temporal_het_gnn_spec]] — Phase 12 spec (GNN architecture)
- [[temporal_het_gnn]] — Phase 12 research

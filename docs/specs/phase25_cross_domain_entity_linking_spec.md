---
title: "Spec: Phase 25 Cross-Domain Entity Linking"
tags:
  - doc/spec
  - phase/25
  - topic/gnn-expansion
  - topic/entity-linking
  - topic/instruments
  - layer/world-model
  - layer/surveillance
---

# Spec: Phase 25 Cross-Domain Entity Linking

## Goal

Connect the dense instrument subgraph created in Phase 24 to the existing company, country, person, topic, and wallet graph so the HetTGN can propagate surprise and attention across domains instead of learning on isolated instrument nodes. This phase also upgrades the highest-value remaining L1 tools that directly feed those new cross-domain links.

## Trusted Inputs

- [[phase25_gnn_diagnostic]] — current graph-state diagnosis and Phase 25 recommendation
- [[entity_linking_layer_spec]] — established link-creation pattern and guardrail: only encode explicit factual relationships
- [[e2e_global_integration_spec]] — instrument node creation and diagnostic requirements from Phase 24

## Guardrails

- Only persist relationships that are explicit in source data or in deterministic instrument metadata already maintained by the repo.
- Do not hand-code predictive weights, signal scores, or directionality into links.
- Keep tool-layer logic in tool modules and graph/persistence logic in pipeline/world-model modules.
- Every sub-phase must finish with targeted edge-case tests before it is marked complete.

## Files Affected

Expected primary files:

- `agent/tools/instrument_universe.py`
- `agent/pipeline/store.py`
- `agent/pipeline/entity.py` or existing entity helper module if new link helpers are needed
- `agent/tools/cftc.py`
- `agent/tools/polymarket.py`
- `agent/tools/polymarket_whales.py`
- `agent/models/gnn/graph_builder.py`
- `tests/test_instrument_universe.py`
- `tests/test_cftc.py` and/or a dedicated L2 test file
- `tests/test_polymarket*.py`
- `tests/test_graph_builder_expanded.py` or a dedicated integration test file

## Implementation Steps

### 25a: Instrument Graph Wiring

**25a.1: Define deterministic instrument issuer/link metadata**

Add explicit metadata needed to connect instruments to real entities without guessing:

- stock/ETF issuer or sponsor organization where explicit
- primary country for issuer/market where explicit
- contract-family mapping for futures/FX where explicit and maintained locally

This metadata belongs with the instrument registry, not scattered across inference code.

**25a.2: Persist `instrument -> company` and `company -> country` links**

During instrument entity registration or daily ingest, create link rows for supported instruments:

- `tracks_issuer` or repo-approved equivalent for instrument/company
- `located_in` or repo-approved equivalent for company/country where the country is explicit

Reuse Phase 17 link persistence patterns and dedup behavior.

**25a.3: Verify graph builder emits the new edge types cleanly**

Confirm the graph builder includes the new link types without hard-coded breakage in edge indexing, enrichment dims, or node/edge counts.

### 25b: CFTC L2 Upgrade for Instrument Connectivity

**25b.1: Persist contract or instrument entities from CFTC records**

Extend the CFTC tool so records produce entities/observations tied to tradeable instrument nodes instead of remaining aggregate-only.

**25b.2: Link positioning data to instrument entities**

Create explicit links between the CFTC contract representation and the corresponding instrument entity maintained in the instrument universe.

**25b.3: Add edge-case coverage**

Cover missing mappings, duplicate rows, unknown contracts, partial datasets, malformed numeric fields, and empty fetch responses.

### 25c: Polymarket L2 Upgrade for Topic/Participant Connectivity

**25c.1: Persist topic or market entities from Polymarket markets**

Upgrade market records so the tool creates entities that can connect prediction markets to existing topic/person/world-state nodes.

**25c.2: Persist participant or wallet links from whale/large-bet flows**

Upgrade `polymarket_whales` so large participant activity materializes as entity-level observations/links rather than text-only summaries.

**25c.3: Add edge-case coverage**

Cover missing participant identity, repeated market snapshots, closed markets, null liquidity/price fields, malformed timestamps, and duplicate wallets.

### 25d: Retrain and Re-Diagnose the Connected Graph

**25d.1: Re-run targeted graph/integration tests**

Validate that new entities and links survive store persistence and graph building.

**25d.2: Re-run GNN diagnostics with instrument focus**

Measure whether instrument neighborhoods are less isolated and whether cross-type attention is reaching company/country/topic nodes.

**25d.3: Record the outcome for the next expansion decision**

Write a checkpoint or report stating which instrument classes remain starved after the new links are live.

## Edge Cases

- Instruments that have no explicit issuer or country metadata must remain unlinked rather than guessed.
- Multiple instruments may map to the same issuer; dedup company entities and links.
- Futures/FX instruments may map to a country or contract family but not a company; do not force a company edge.
- Cross-listed or multi-country issuers must follow a deterministic single-source-of-truth rule documented in code/tests.
- CFTC symbols that do not map to the instrument universe must degrade gracefully and log/skip.
- Polymarket records with anonymous or incomplete participant fields must still preserve valid market/topic structure without fabricating person entities.

## Testing Plan

- Instrument-link unit tests for explicit issuer/country mappings, dedup, and unlinked fallback.
- CFTC L2 tests for persistence, mapping correctness, malformed/partial data, and idempotency.
- Polymarket L2 tests for entity persistence, repeated snapshots, missing identities, and duplicate market rows.
- Graph-builder integration tests asserting new edge types are present and correctly typed.
- Diagnostic re-run proving reduced instrument isolation versus the prior Phase 24 baseline.

## Related

- [[phase25_gnn_diagnostic]] — Phase 25 research input
- [[phase25_cross_domain_entity_linking]] — Active task
- [[entity_linking_layer_spec]] — Prior link-pattern spec
- [[e2e_global_integration_spec]] — Phase 24 instrument integration spec
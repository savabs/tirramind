---
title: "Task: Entity Linking Layer"
tags:
  - doc/task
  - status/done
  - phase/17
  - topic/surveillance
  - topic/world-model
  - layer/surveillance
  - layer/world-model
---

# Task: Entity Linking Layer

Status: completed
Research: [[entity_linking_layer]]
Spec: [[entity_linking_layer_spec]]

---

## Overview

Add `link_entities()` calls to 9 of 12 existing L2 tools. The GNN currently has 9 entity types and 15 observation types but **zero edges**. This phase wires the graph so the HetTGN can propagate signal across connected entities.

Constraint: hard-code only explicit factual edges from source data. Do not hard-code predictive logic, learned relations, or market-direction rules in this phase.

## Steps

### Phase 17a: Person→Company links (insider_filings + form144)

- [x] 17a.1: Add `works_for` link to insider_filings `_persist_entities_inner()`
- [x] 17a.2: Add `works_for` link to form144 `_persist_entities_inner()`
- [x] 17a.3: Write tests for 17a (normal, missing CIK, dedup) — 24 tests, all passing

### Phase 17b: Same-tool intra-entity links (whale_alert + gdelt)

- [x] 17b.1: Add `transacts_with` link to whale_alert `_persist_entities_inner()` (sender→receiver wallets)
- [x] 17b.2: Add `event_involves` link to gdelt `_persist_entities_inner()` (country→country bilateral)
- [x] 17b.3: Write tests for 17b (normal, same-country skip, empty addr) — 20 tests, all passing

### Phase 17c: Cross-entity links with new entity creation (ais_vessel + lobbying)

- [x] 17c.1: Add `port_call_to` link to ais_vessel (vessel→country); add `_DEST_COUNTRY` mapping (~50 ports)
- [x] 17c.2: Add `lobbies_for` link to lobbying (registrant→client company); restructured dedup to store observations for all filings; added "Self" client guard
- [x] 17c.3: SKIPPED `cert_for` for cert_transparency — crt.sh JSON API does not expose certificate subject Organization field (only common_name/issuer_name)
- [x] 17c.4: Write tests for 17c — 18 new tests (10 ais_vessel + 8 lobbying), 62 total, all passing; fixed 6 mock regressions in test_ais_vessel_l2.py + 1 integration test assertion

### Phase 17d: Fixed-country links (interconnection_queue + patent_filings)

- [x] 17d.1: Add `located_in` link to interconnection_queue (company→country US); metadata includes state
- [x] 17d.2: Add `patents_in` link to patent_filings (company→country US); metadata includes patent_number
- [x] 17d.3: Write tests for 17d — 17 new tests (8 IQ + 9 patent), 79 total, all passing; fixed 6 mock regressions in test_corporate_energy_defi_l2.py

### Phase 17e: Integration tests + edge case suite

- [x] 17e.1: Cross-tool consistency tests — 4 tests (same company from lobbying+patents, patent+IQ, same US country, cross-tool links)
- [x] 17e.2: Graph builder integration — 5 tests (works_for/located_in/patents_in edge_index, edge_attr confidence, multi-type coexistence)
- [x] 17e.3: Full edge case suite — 7 tests (self-link guard, idempotent insertion, long metadata, no-pipeline all tools, different link types same pair, confidence filter, direction filter)

---

## Link Type Summary

| Link Type | Source Tool | Entity A → Entity B | Confidence |
|-----------|-----------|---------------------|------------|
| `works_for` | insider_filings | person → company | 1.0 |
| `works_for` | form144 | person → company | 1.0 |
| `transacts_with` | whale_alert | wallet → wallet | 1.0 |
| `event_involves` | gdelt | country → country | 0.9 |
| `port_call_to` | ais_vessel | vessel → country | 0.8 |
| `lobbies_for` | lobbying | company → company | 0.9 |
| `located_in` | interconnection_queue | company → country | 1.0 |
| `patents_in` | patent_filings | company → country | 1.0 |

## Related

- [[entity_linking_layer]] — research doc
- [[entity_linking_layer_spec]] — implementation spec
- [[tool_priority_ranking]] — Phase 16 ranking (motivation)
- [[l2_tool_expansion]] — Phase 13 L2 tool task
- [[quant_training_ground]] — master task
- [[checkpoint_archive_2026]] — archived prior checkpoint material

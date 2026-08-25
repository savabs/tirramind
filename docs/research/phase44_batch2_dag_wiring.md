---
title: "Research: Phase 44 — Batch 2 DAG Wiring"
tags:
  - doc/research
  - phase/44
  - topic/pipeline
  - topic/diversity
  - layer/surveillance
---

# Research: Phase 44 — Batch 2 DAG Wiring

## Context

Phase 43 added 4 nodes (22-node DAG). 31 of 51 data tools (60.8%) are still unwired.
Phase 44 wires the next 5 L2-ready tools that require no domain/entity-specific params.

Excluded from this batch (need domain params): `cert_transparency` (requires `domain`),
`dns_monitor` (requires `domain` or `domains` list). These need a domain-list strategy first.

## Target Tools

All 5 tools are already:
- Registered in `build_tool_registry()` with `pipeline_store` injected
- Have `_persist_entities` / `_persist_entities_inner` implemented
- Use observation types already in the schema (`graph_builder.py` + `trainer.py` confirmed)

No new Python code needed — DAG nodes + test updates only.

### regulatory_gazette (priority 1)

- **Source:** Federal Register API (regulations.gov) — free, no auth
- **Required params:** none (`days_back=7`, `limit=25` defaults)
- **Chosen params:** `days_back=7`, `limit=50`
- **L2 obs type:** `regulatory_velocity` on organization entities
- **Volume:** ~50 regulatory documents/run, creates org entities per agency

### form144 (priority 2)

- **Source:** SEC EDGAR EFTS — free, 10 req/s limit, User-Agent required
- **Required params:** none (`days_back=14` default)
- **Chosen params:** `days_back=14` (default, covers recent filer activity)
- **L2 obs type:** `sell_intent` on company + person entities
- **Volume:** ~20–80 Form 144 filings per 14-day window, creates company/person nodes

### supply_chain_monitor (priority 3)

- **Source:** BLS PPI — free, no auth
- **Required params:** `mode`
- **Chosen params:** `mode=producer_prices`
- **L2 obs type:** `price_movement` on organization entities (industry sectors)
- **Volume:** Small — BLS sector-level series, creates industry-sector org nodes

### political_risk (priority 4)

- **Source:** FEC API — free, no auth
- **Required params:** `mode`
- **Chosen params:** `mode=candidates` (broadest FEC coverage, creates candidate person entities)
- **L2 obs type:** `campaign_finance` on person entities
- **Volume:** ~20 candidate entities/run; FEC data is relatively stable

### comtrade (priority 5)

- **Source:** UN Comtrade API — free tier, no auth for recent data
- **Required params:** `mode` (+ `reporter` for flows mode)
- **Chosen params:** `mode=partners`, `reporter=USA` (top US trading partners by value)
  - `partners` mode only requires `reporter`; `flows` mode requires both reporter + partner
- **L2 obs type:** `trade_flow` on country entities
- **Volume:** ~10–20 country-pair records/run

## Observation Type Verification

All 5 obs types confirmed present in codebase:
- `regulatory_velocity` — `graph_builder.py:96`, `trainer.py:516`
- `sell_intent` — `graph_builder.py:99`, `trainer.py:481`
- `price_movement` — `graph_builder.py:94`, `trainer.py:514`
- `campaign_finance` — `graph_builder.py:63`, `trainer.py:506`
- `trade_flow` — `graph_builder.py:102`, `trainer.py:503`

No schema changes required.

## Test Impact

- `tests/test_pipeline_registry.py`: 3 count assertions: `22 → 27`
- 5 new per-node config tests

## Files to Modify

- `agent/pipeline/dags/daily_collection.py` — add 5 Phase 44 nodes
- `tests/test_pipeline_registry.py` — update 3 count assertions + add 5 tests

## Related

- [[phase44_batch2_dag_wiring_spec]]
- [[phase44_batch2_dag_wiring_task]]
- [[phase43_high_volume_dag_wiring]]
- [[chat_checkpoint_2026-04-22_phase43_complete]]

---
title: "Checkpoint: Phase 23 GNN-Guided Expansion R2 Complete"
tags:
  - doc/checkpoint
  - phase/23
  - topic/gnn-expansion
  - topic/surveillance
  - layer/surveillance
---

# Checkpoint — 2026-04-13

## Session Summary

Phase 23 (GNN-Guided Expansion R2) is **fully complete**. All 23 phases of the quant computation stack are done.

## What Was Completed

### Phase 23: GNN-Guided Expansion R2

Upgraded 3 Tier-1 L1 tools to L2 entity persistence based on GNN diagnostic evaluation.

| Sub-phase | Tool | Tests | Key Details |
|-----------|------|-------|-------------|
| 23a | Graph builder expansion | 23/23 | 3 new obs types (short_interest, creditor_filing, drug_approval), ENRICHMENT_DIM 27→30 |
| 23b | finra_short_volume L2 | 16/16 | Company entities per ticker, short_interest observations |
| 23c | creditor_filings L2 | 19/19 | SEC EDGAR + UK Companies House, debtor_of links (confidence=0.8) |
| 23d | drug_regulatory L2 | 22/22 | FDA approvals + adverse events, market_authorized_in links (company→US) |
| 23e | Full regression | 174/174 | All GNN + L2 tests green |

### Bug Caught & Fixed

- **Store API kwarg mismatch** in drug_regulatory.py: used `source=` instead of `source_tool=` for observations, and `source_id`/`target_id` instead of `entity_id_a`/`entity_id_b` for links. Caught during test writing, fixed before it could become a runtime bug.

## Current State

- **23 phases complete** (Layers 1-6 of the 7-layer computation stack)
- **18 L2 tools** total (15 prior + finra_short_volume + creditor_filings + drug_regulatory)
- **21 observation types** in graph_builder (was 18)
- **ENRICHMENT_DIM = 30** (was 27)
- **New link types**: `debtor_of` (creditor_filings), `market_authorized_in` (drug_regulatory)
- **~20 remaining L1 tools** that could be upgraded in future rounds

## Files Modified This Session

| File | Change |
|------|--------|
| `agent/models/gnn/graph_builder.py` | 3 new obs types, ENRICHMENT_DIM 27→30 |
| `agent/tools/finra_short_volume.py` | L2 persistence (pipeline_store, _persist_entities) |
| `agent/tools/creditor_filings.py` | L2 persistence + debtor-creditor entity links |
| `agent/tools/drug_regulatory.py` | L2 persistence + market_authorized_in links |
| `tests/test_graph_builder_expanded.py` | Updated obs type count assertion 18→21 |
| `tests/test_finra_l2.py` | Created — 16 tests |
| `tests/test_creditor_l2.py` | Created — 19 tests |
| `tests/test_drug_regulatory_l2.py` | Created — 22 tests |
| `[[gnn_guided_expansion_r2]]` | Marked complete |
| `[[quant_training_ground]]` | Phase 23 added to master tracker |

## Strategic Next Steps

With all 23 phases complete, the candidate next moves (from start of session):

1. **End-to-end integration test** — wire all layers together, run a full pipeline from data fetch to portfolio allocation
2. **Backtest with real data** — walk-forward validation on historical data
3. **Layer 7: LLM Support** — text parsing, narrative synthesis (lowest priority per architecture)
4. **GNN-Guided Expansion R3** — next tier of L2 upgrades based on fresh GNN diagnostics
5. **Production hardening** — config management, monitoring, error recovery

## L2 Persistence Pattern (for future upgrades)

Standard pattern established across all 3 tools:
1. `TYPE_CHECKING` import for `PipelineStore`, try/except for `entity_id_from_key`/`normalize_company_name`
2. Constructor: `pipeline_store: PipelineStore | None = None` kwarg
3. `_persist_entities()` wrapper: guard on `self._store is None`, try/except non-fatal
4. `_persist_entities_inner()`: `register_entity()` + `store_entity_observation()` + optional `link_entities()`
5. Store API: `source_tool=` (not `source=`), `entity_id_a`/`entity_id_b` (not `source_id`/`target_id`)

## Related

- [[gnn_guided_expansion_r2]] — Research doc
- [[gnn_guided_expansion_r2_spec]] — Spec doc
- [[gnn_guided_expansion_r2|Task file]] — Task (completed)
- [[quant_training_ground]] — Master phase tracker
- [[gnn_guided_tool_expansion]] — Phase 16 (first-round GNN expansion)

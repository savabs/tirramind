---
title: "Task: GNN-Guided Expansion Round 2"
tags:
  - doc/task
  - status/done
  - phase/23
  - topic/surveillance
  - topic/gnn-expansion
  - layer/surveillance
---

# Task: GNN-Guided Expansion Round 2

Status: completed
Research: [[gnn_guided_expansion_r2]]
Spec: [[gnn_guided_expansion_r2_spec]]

---

## Phase 23a: Graph Builder Expansion

- [x] **23a.1**: Add 3 new obs types to `OBSERVATION_TYPES` in graph_builder.py (short_interest, creditor_filing, drug_approval)
- [x] **23a.2**: Update `ENRICHMENT_DIM` if hardcoded (18→21 in obs_type distribution)
- [x] **23a.3**: Verify `_build_edge_data()` is fully dynamic (no link-type filtering)

## Phase 23b: finra_short_volume L2

- [x] **23b.1**: Add pipeline_store constructor kwarg + entity imports
- [x] **23b.2**: Implement `_persist_entities()` + `_persist_entities_inner()`
- [x] **23b.3**: Call from `_run()` result paths
- [x] **23b.4**: Write `tests/test_finra_l2.py` — run + pass (16/16)

## Phase 23c: creditor_filings L2

- [x] **23c.1**: Add pipeline_store constructor kwarg + entity imports
- [x] **23c.2**: Implement `_persist_entities()` + `_persist_entities_inner()` with debtor-creditor links
- [x] **23c.3**: Call from `_run()` result paths
- [x] **23c.4**: Write `tests/test_creditor_l2.py` — run + pass (19/19)

## Phase 23d: drug_regulatory L2

- [x] **23d.1**: Add pipeline_store constructor kwarg + entity imports
- [x] **23d.2**: Implement `_persist_entities()` + `_persist_entities_inner()` with market links
- [x] **23d.3**: Call from `_run()` result paths
- [x] **23d.4**: Write `tests/test_drug_regulatory_l2.py` — run + pass (22/22)

## Phase 23e: Integration + Edge Cases

- [x] **23e.1**: Full regression test suite — 174/174 passed (graph_builder + HetTGN + pattern_extractor + surprise + all 3 L2 tools)
- [x] **23e.2**: Fixed store API kwarg mismatch (source→source_tool, source_id/target_id→entity_id_a/entity_id_b)
- [x] **23e.3**: Update master tracker

## Related

- [[gnn_guided_expansion_r2]] — Research doc
- [[gnn_guided_expansion_r2_spec]] — Spec doc
- [[quant_training_ground]] — Master phase tracker
- [[gnn_guided_tool_expansion]] — Phase 16 first-round

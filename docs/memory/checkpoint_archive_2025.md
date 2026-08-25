---
title: "Checkpoint Archive 2025"
tags:
  - doc/memory
---

# Checkpoint Archive 2025


## 2025-01-27 — chat_checkpoint_2025-01-27_7bT_complete

Chat Checkpoint — 2025-01-27 — 7b-T Sovereign Debt Complete
# Chat Checkpoint — 2025-01-27 — 7b-T Sovereign Debt Complete

## Status
7b-T Sovereign Debt is **fully complete**. All green: 2194 passed, 0 failed, 6 skipped.

## Current Counts
- **35 tools** registered in cli.py
- **23 bandit arms** in DEFAULT_ARMS (bandit.py)

## What Was Done This Session
1. Discovered research, spec, implementation, and registration all existed from a prior session
2. Found and fixed cache API bug: `cache.put(..., ttl=...)` → `cache.put(source, params, data)` (4 occurrences)
3. Wrote 94 edge case tests (`tests/test_sovereign_debt_edge.py`) — all passing
4. Updated 19 stale count assertions (34→35 tools, 22→23 arms) across 11 test files

---

## 2025-05-12 — chat_checkpoint_2025-05-12_tier6

Checkpoint: Tier 6 — Learned Feature Selection & Tool Routing
# Checkpoint: Tier 6 Completion

**Date**: 2025-05-12
**Session**: Continued from [[chat_checkpoint_2026-04-14_learned_architecture]]

## What Was Done

Completed Tier 6 (Changes 11 + 12) of the [[learned_vs_handcoded_architecture_spec]], moving the system from **75% → 82% learned**.

### Change 11: Learned Feature Selection (FeatureGate)

**Concept**: Regime-conditioned soft gating over the 5 feature groups (surprise, belief, market, entity_count, adversarial). An MLP maps the HMM regime posterior to per-group gate values in [floor, 1.0], with entropy regularization to prevent collapse.

**Files created/modified**:

---

## 2025-06-28 — chat_checkpoint_2025-06-28

Chat Checkpoint — 2025-06-28 (Batch 5 Complete)
# Chat Checkpoint — 2025-06-28 (Batch 5 Complete)

## What Was Done

Built 3 data tools in batch (Phase 7b, Batch 5):

### Tool #39: UN Comtrade (`agent/tools/comtrade.py`, ~370 lines)
- **7b-Y** — Global bilateral trade flow monitor
- 3 modes: `flows`, `commodity`, `partners`
- Free preview API (comtradeapi.un.org) + premium switching via `TIRRA_UN_COMTRADE_KEY`
- 34 M49 country codes, 18 strategic HS commodities (crude oil, semiconductors, rare earths, wheat, etc.)
- Bug fix: empty string in `_resolve_country()` matched everything
- Bandit arm: `global_trade`


---

## 2025-07-08 — chat_checkpoint_2025-07-08

Chat Checkpoint — 2025-07-08
# Chat Checkpoint — 2025-07-08

## Session Summary

Built **7b-V: UCC/Creditor Filings** tool end-to-end.

## What Was Done

### 7b-V: Creditor Filings Tool (#38)
- **File:** `agent/tools/creditor_filings.py` (~490 lines)
- **Modes:**
  1. `search` — Search SEC EDGAR 8-K filings for credit-event language (security interest, pledge, lien, credit facility, collateral) by entity name. Also searches UK Companies House charges if `TIRRA_COMPANIES_HOUSE_KEY` env var is set.
  2. `uk_charges` — List charges for a UK company by company number or name search. Shows charge status, classification (debenture/floating/fixed/mortgage), creditors, satisfaction dates. Flags high-stress (3+ outstanding) and moderate (1+).
  3. `stress_scan` — Broad scan across SEC EDGAR for recent 8-K credit-event filings. Detects entity-level filing clusters (2+ filings = potential distress signal).

---

## 2025-07-16 — chat_checkpoint_2025-07-16_phase12

Checkpoint: Phase 12 Complete — Temporal Heterogeneous GNN
# Checkpoint: Phase 12 Complete

**Date:** 2025-07-16
**Status:** All 6 sub-phases complete. 172/172 tests passing.

## What Was Built

A complete Temporal Heterogeneous Graph Network (HetTGN) pipeline that:
1. Converts the PipelineStore entity graph into PyG HeteroData
2. Encodes temporal features via Time2Vec + TemporalEncoder
3. Learns cross-entity patterns via self-supervised next-event prediction
4. Extracts discovered patterns from model attention/embeddings
5. Crystallizes patterns into production rules compatible with cross_entity.py
6. Runs auto-discovered patterns alongside hand-crafted L3 patterns

---

## 2025-07-18 — chat_checkpoint_2025-07-18

Checkpoint: Phase 22 Adversarial Complete
# Checkpoint: Phase 22 Adversarial Intelligence Layer — COMPLETE

**Date:** 2025-07-18
**Session:** Phase 22c finalization (edge case + validation tests)

## What Was Done

Phase 22 (Adversarial Intelligence Layer) is now fully implemented and tested: **148/148 tests pass across 9 test files.**

### Phase 22a — Core Infrastructure (79 tests)
- `agent/adversarial/config.py` — EdgeDecayConfig, VPINConfig, CrowdingConfig, AdversarialConfig
- `agent/adversarial/flags.py` — AdversarialFlag frozen dataclass with validation
- `agent/adversarial/edge_decay.py` — EdgeDecayMonitor using BOCPD on rolling Sharpe
- `agent/adversarial/vpin.py` — VPINEstimator (BVC method, Easley et al. 2012)

---

## 2025-07-23 — chat_checkpoint_2025-07-23

Checkpoint: Phase 13 L2 Tool Expansion Complete
# Checkpoint: Phase 13 L2 Tool Expansion Complete

**Date:** 2025-07-23
**Session focus:** Upgrade 7 VERY HIGH priority tools from L1 → L2 entity persistence so the GNN can see their data.

## What Was Done

### Phase 13a: Graph Builder + Entity Module Expansion
- Fixed insider_filings obs_type "purchase" → "insider_trade"
- Expanded EntityType Literal: +domain, +protocol, +topic (6 → 9)
- Expanded ENTITY_TYPES and OBSERVATION_TYPES in graph_builder.py (6→9, 8→15)
- Made graph builder iterate `union(ENTITY_TYPES, id_map.type_local.keys())` for dynamic type support
- Added unknown-type fallback (log warning, default to index 0)
- 42 new tests in `test_graph_builder_expanded.py`

---

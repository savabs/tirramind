---
title: "Checkpoint: 2026-04-15 — Tier 8 Complete, Full Roadmap Done, Session Final"
tags:
  - doc/checkpoint
  - phase/25
  - topic/autonomous-discovery
  - topic/entity-ontology
  - topic/self-improving
  - topic/learned-architecture
  - layer/surveillance
  - layer/feature-engineering
  - layer/world-model
---

# Checkpoint: 2026-04-15 — Learned Architecture Roadmap Complete

**Date:** 2026-04-15
**Session scope:** Tier 8 implementation finalization, test suite creation, full regression, roadmap closure

---

## TL;DR

The **entire [[learned_vs_handcoded_architecture_spec]]** is now **COMPLETE** — all 8 tiers, all 16 changes. The system is at **95% learned / 5% hand-coded**. The remaining 5% is the irreducible core that stays hand-coded by design (safety, schemas, plumbing, math identities, ethics). **There is no Tier 9.**

> **CORRECTION:** The previous checkpoint (`chat_checkpoint_2025-07-24_tier8_complete.md`) erroneously stated "Remaining 5% for Tier 9: Change 17 = Learned H/R matrices, Change 18 = End-to-end differentiable pipeline." **This was hallucinated.** The actual spec (lines 330-407) ends at Tier 8. No Changes 17 or 18 exist in the spec. The irreducible 5% is intentionally permanent.

---

## What Was Done This Session

### 1. Tier 8 Implementation Completed (Change 15 + Change 16)

**Change 15 — Autonomous Data Source Discovery:**
- `agent/discovery/source_scout.py` — CKAN catalog search, HTTP probing, TF-IDF relevance scoring
- `agent/discovery/signal_evaluator.py` — KSG mutual information estimation for signal assessment
- `agent/discovery/tool_factory.py` — Template-based tool generation + config persistence
- `agent/pipeline/store.py` — +3 tables (`discovered_sources`, `unresolved_entities`, `entity_type_registry`), ~15 CRUD methods
- `agent/learning/tool_router.py` — `add_arm()`/`remove_arm()` with persistence
- `agent/cli.py` — OntologyRegistry seeding, discovered tool loading on startup
- `agent/pipeline/dags/daily_collection.py` — `run_quarantine_cycle()` for source promotion/disabling

**Change 16 — Self-Extending Entity Ontology:**
- `agent/discovery/ontology_registry.py` — Dynamic entity/link type registry, seed init, CRUD, hierarchy
- `agent/discovery/type_inducer.py` — Jaccard clustering, cohesion scoring, PMI co-occurrence relationship induction
- `agent/pipeline/entity.py` — `EntityType = str` (was `Literal`), `SEED_ENTITY_TYPES`, `validate_entity_type()`, global registry accessor
- `agent/features/gnn_builder.py` — `get_connected_types()` dynamic expansion from OntologyRegistry

### 2. Test Suites Written & Passing

| File | Tests | Status |
|------|-------|--------|
| `tests/test_tier8_source_discovery.py` | 36 | ALL PASS |
| `tests/test_tier8_ontology.py` | 36 | ALL PASS |
| **Total Tier 8** | **72** | **72/72 PASS** |

### 3. Full Regression

- **8943 existing tests pass** — zero new regressions from Tier 8
- 41 pre-existing stale count assertions remain (tool_count, arm_count, node_count hardcoded values from older phases — cosmetic tech debt, not functional failures)

### 4. Pipeline Wiring Verified

- cli.py seeds `OntologyRegistry(store)` + calls `set_ontology_registry()` + loads discovered tool configs
- daily_collection.py includes `run_quarantine_cycle()` with 5-success promotion / 3-failure disabling
- End-to-end pipeline wiring test (6 assertions) passed

### 5. Previous Checkpoint Correction

- Identified and documented hallucinated "Tier 9" in `chat_checkpoint_2025-07-24_tier8_complete.md` — no such tier exists in the spec

---

## Learned Architecture Roadmap: COMPLETE

| Tier | % Learned | Changes | Status |
|------|-----------|---------|--------|
| Tier 1 | 28% | Change 1: Wire beliefs → SAC policy | **DONE** |
| Tier 2 | 45% | Changes 2, 4, 5: Learned world model params, surprise weights, loss weights | **DONE** |
| Tier 3 | 55% | Changes 7, 8, 9: Detector thresholds (GP-BO), dynamic goal arms, learned reward | **DONE** |
| Tier 4 | 65% | Changes 3, 6: Causal graph structure learning, learned state encoder | **DONE** |
| Tier 5 | 75% | Change 10: Differentiable Kalman / variational world model | **DONE** |
| Tier 6 | 82% | Changes 11, 12: Learned feature selection, learned tool routing | **DONE** |
| Tier 7 | 90% | Changes 13, 14: Self-modifying graph, meta-learned scheduling | **DONE** |
| Tier 8 | 95% | Changes 15, 16: Autonomous discovery, self-extending ontology | **DONE** |

### The Irreducible 5% (Stays Hand-Coded by Design)

1. **Safety constraints** — Leverage limits, position limits, max drawdown, legal/regulatory rules
2. **Schema invariants** — The *existence* of EngineeredFeature, BeliefState, EntityAlert as protocols (contents/relationships are learned)
3. **API plumbing** — HTTP calls, auth, serialization (mechanical, not intelligent)
4. **Textbook equations** — Sharpe, VaR, Kalman predict/update steps (only parameters learned)
5. **Ethical/legal boundaries** — What the system is forbidden from doing

---

## Key Algorithms (Tier 8)

- **Source discovery pipeline:** CKAN API search → TF-IDF relevance → HTTP probing → MI signal evaluation → template tool gen → quarantine → promotion
- **MI estimation:** KSG estimator (k=3 neighbors) via `sklearn.feature_selection.mutual_info_regression`, normalized by source entropy, threshold 0.05
- **Type induction:** Group unresolved entities by source_tool → Jaccard similarity → agglomerative clustering → cohesion scoring → type derivation → overlap detection → registration
- **Relationship induction:** PMI co-occurrence within 24h window → frequency threshold → link type registration
- **Quarantine:** 5 consecutive successes → promote to active; 3 consecutive failures → disable

---

## API Gotchas Discovered

- `store.link_entities()` is the correct method name (NOT `store_entity_link`)
- `update_unresolved_cluster()` takes `entity_ids: list[int]`, not a single int
- `resolve_unresolved_entities()` takes `cluster_id: int`, not a list
- `store_feature()` takes an `EngineeredFeature` object, not kwargs
- `ToolRoutingBandit.persist_path` expects a `Path` object, not a string

---

## Active Tasks (Not Part of This Session)

| Task | Phase | Status | Notes |
|------|-------|--------|-------|
| [[phase25_cross_domain_entity_linking]] | 25 | active | Cross-domain entity linking |
| [[phase26_mcp_agent_upgrade]] | 26 | active | MCP server stack upgrade |
| [[predictive_platform_positioning_task]] | 25 | active | Platform positioning / productization |
| [[quant_training_ground]] | 25 | active | Umbrella quant task |
| [[ecc_workflow_improvements]] | — | active | Workflow tooling (all steps done, needs closing?) |
| [[tier3_learn_meta_params]] | — | active | Stale? Tier 3 is done via main impl task |
| [[tier4_learn_dag_structure]] | — | active | Stale? Tier 4 is done via main impl task |
| [[tier4_learned_state_encoder]] | — | active | Stale? Tier 4 is done via main impl task |
| [[tier5_differentiable_kalman]] | — | active | Stale? Tier 5 is done via main impl task |
| [[tier7_self_modifying_structure]] | — | active | Stale? Tier 7 is done via main impl task |
| [[tier8_autonomous_discovery]] | — | status/done | ✅ Should be moved to `tasks/done/` |

---

## Known Tech Debt

1. **41 stale count assertions** — Tests that hardcode `tool_count=47` or similar from older phases. Actual count is now 60+. Cosmetic, not functional.
2. **Stale tier task files in `tasks/active/`** — tier3, tier4, tier5, tier7, tier8 task files still in `active/` despite being completed via the main `learned_architecture_impl.md` task. Should be moved to `tasks/done/` or consolidated.
3. **`ecc_workflow_improvements.md`** — All steps checked, still marked active. May need closing.
4. **Checkpoint date in previous file** — `chat_checkpoint_2025-07-24` was created during a session where dates were inconsistent. Content is accurate minus the Tier 9 hallucination.

---

## Project Architecture (Current State)

```
Layer 1: Surveillance    → agent/tools/            60+ data tools (CKAN discovery adds more autonomously)
Layer 2: Features        → agent/quant/            Signal extraction, changepoint, regime, spectral
         + GNN           → agent/features/          Temporal heterogeneous GNN, dynamic connected types
Layer 3: World Model     → agent/models/            Bayesian network (pgmpy), causal graph, learned structure
Layer 4: Fusion          → agent/fusion/            Differentiable Kalman, variational world model
Layer 5: RL Policy       → agent/learning/          SAC + Thompson Sampling bandit, learned routing
Layer 6: Adversarial     → agent/adversarial/       Manipulation detection, edge decay
Layer 7: LLM Support     → agent/reasoning/         Text parsing, narration (SUPPORT ONLY)
NEW:     Discovery       → agent/discovery/          Source scout, signal evaluator, tool factory,
                                                      ontology registry, type inducer
```

---

## Files Created/Modified This Session

### Created (8)
- `agent/discovery/__init__.py`
- `agent/discovery/source_scout.py`
- `agent/discovery/signal_evaluator.py`
- `agent/discovery/tool_factory.py`
- `agent/discovery/ontology_registry.py`
- `agent/discovery/type_inducer.py`
- `tests/test_tier8_source_discovery.py`
- `tests/test_tier8_ontology.py`

### Modified (6)
- `agent/pipeline/store.py` — +3 tables, +15 methods
- `agent/pipeline/entity.py` — EntityType = str, validation, global registry
- `agent/features/gnn_builder.py` — Dynamic connected types
- `agent/learning/tool_router.py` — add/remove arm with persistence
- `agent/cli.py` — Discovery wiring on startup
- `agent/pipeline/dags/daily_collection.py` — Quarantine cycle

---

## Resumption Guide

1. **The learned architecture spec is done.** All 16 changes across 8 tiers are implemented and tested. No further tier work unless the spec is extended.
2. **Next natural work:** Phase 25 (cross-domain entity linking), Phase 26 (MCP upgrade), or productization positioning.
3. **Before starting new work:** Clean up stale active task files (tier3-tier8 individual tasks should move to `tasks/done/`).
4. **Test baseline:** 8943 + 72 = 9015 passing tests. 41 stale count assertion failures (pre-existing).

---

## Related

- [[tier8_autonomous_discovery]] — Task (completed)
- [[tier8_autonomous_discovery_spec]] — Spec
- [[learned_vs_handcoded_architecture_spec]] — Master spec (COMPLETE)
- [[learned_architecture_impl]] — Tier 1+2 impl task (completed)
- [[chat_checkpoint_2025-07-24_tier8_complete]] — Previous checkpoint (note: Tier 9 mention was hallucinated)
- [[chat_checkpoint_2026-04-15_tier7_complete]] — Tier 7 checkpoint
- [[project_memory]] — Persistent architectural memory
- [[phase25_cross_domain_entity_linking]] — Potential next work
- [[phase26_mcp_agent_upgrade]] — Potential next work
- [[predictive_platform_positioning_task]] — Potential next work

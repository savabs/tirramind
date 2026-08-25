---
title: "Checkpoint: Tier 8 — Autonomous Discovery Complete"
tags:
  - doc/checkpoint
  - phase/25
  - topic/autonomous-discovery
  - topic/entity-ontology
  - topic/self-improving
  - layer/surveillance
  - layer/feature-engineering
  - layer/world-model
---

# Checkpoint: Tier 8 — Autonomous Discovery Complete

**Date:** 2025-07-24
**Task:** [[tier8_autonomous_discovery]]
**Spec:** [[tier8_autonomous_discovery_spec]]
**Research:** [[tier8_autonomous_discovery]]

---

## Summary

Tier 8 of the [[learned_vs_handcoded_architecture_spec]] is complete. Both changes implemented:

**Previous checkpoint:** [[chat_checkpoint_2026-04-15_tier7_complete]]

- **Change 15 — Autonomous Data Source Discovery**: The pipeline can discover, evaluate, and wire new data sources from API catalogs without human intervention.
- **Change 16 — Self-Extending Entity Ontology**: Entity types and relationship types emerge from data. The fixed `Literal` type vocabulary is replaced by a dynamic runtime registry.

System moves from **90% → 95% learned**.

---

## What Was Built

### New Files (6)
| File | Purpose |
|------|---------|
| `agent/discovery/__init__.py` | Package init |
| `agent/discovery/source_scout.py` | CKAN catalog search, HTTP probing, TF-IDF relevance scoring, `run_source_discovery()` orchestration |
| `agent/discovery/signal_evaluator.py` | KSG mutual information estimation (sklearn fallback to correlation), numeric series extraction |
| `agent/discovery/tool_factory.py` | `DiscoveredJsonApiTool`, `DiscoveredCsvFeedTool`, config persistence + round-trip |
| `agent/discovery/ontology_registry.py` | Dynamic entity type/link type registry, seed initialization, CRUD, hierarchy |
| `agent/discovery/type_inducer.py` | Jaccard clustering, cohesion scoring, type name derivation, PMI co-occurrence relationship induction |

### Modified Files (5)
| File | Change |
|------|--------|
| `agent/pipeline/store.py` | +3 tables (`discovered_sources`, `unresolved_entities`, `entity_type_registry`), +15 CRUD methods |
| `agent/pipeline/entity.py` | `EntityType = str` (was `Literal`), `SEED_ENTITY_TYPES` frozenset, `validate_entity_type()`, global registry accessor |
| `agent/features/gnn_builder.py` | `_SEED_CONNECTED_TYPES` + `get_connected_types()` dynamic expansion from OntologyRegistry |
| `agent/learning/tool_router.py` | `add_arm()`/`remove_arm()` with persistence |
| `agent/cli.py` | OntologyRegistry seeding on startup, discovered tool loading from disk configs |
| `agent/pipeline/dags/daily_collection.py` | `run_quarantine_cycle()` for source promotion/disabling |

### Test Files (2)
| File | Tests |
|------|-------|
| `tests/test_tier8_source_discovery.py` | 36 tests: store CRUD, SourceScout search/probe, SignalEvaluator MI, ToolFactory create/serialize, bandit arm mgmt, discovery orchestration, quarantine cycle |
| `tests/test_tier8_ontology.py` | 36 tests: OntologyRegistry seed/CRUD/hierarchy/validation, entity type validation, TypeInducer clustering/proposal/overlap/relationships, GNN dynamic types, global accessor |

---

## Test Results

- **72/72 Tier 8 tests pass**
- **8943 existing tests pass** (0 new regressions)
- 41 pre-existing stale count assertions remain (tool_count, arm_count, node_count hardcoded values from older phases)

---

## Key Algorithms

- **Source discovery**: CKAN API search → TF-IDF relevance scoring → HTTP probing → MI signal evaluation → template tool generation → quarantine → promotion
- **MI estimation**: KSG estimator (k=3 neighbors) via `sklearn.feature_selection.mutual_info_regression`, normalized by source entropy. Threshold: 0.05 (5% of source entropy is predictive)
- **Type induction**: Group unresolved entities by source_tool → Jaccard similarity of context field keys → agglomerative clustering → silhouette-like cohesion scoring → type name derivation → overlap detection → registration
- **Relationship induction**: PMI-based co-occurrence analysis within time window (default 24h) → frequency threshold → link type registration
- **Quarantine**: 5 successful cycles → promote to active. 3 consecutive failures → disable.

---

## Architecture State After Tier 8

The learned_vs_handcoded_architecture_spec is now at **95% learned**:
- Tier 1-4: Foundations (convergence, signals, world model, fusion)
- Tier 5: Learned observation model (attention-weighted entities)
- Tier 6: Learned observation matrices
- Tier 7: Self-modifying graph + meta-learned scheduling
- **Tier 8: Autonomous discovery + self-extending ontology** ← DONE

The remaining 5% is the **Irreducible 5%** that stays hand-coded by design (safety constraints, schema invariants, API plumbing, textbook equations, ethical/legal boundaries). There is no Tier 9 — the spec ends at Tier 8.

---

## Related

- [[tier8_autonomous_discovery]] — Research
- [[tier8_autonomous_discovery_spec]] — Spec
- [[learned_vs_handcoded_architecture_spec]] — Master spec
- [[tier7_self_modifying_structure]] — Previous tier
- [[chat_checkpoint_2026-04-15_tier7_complete]] — Previous checkpoint

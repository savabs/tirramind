---
title: "Task: Tier 8 — Autonomous Discovery & Self-Extending Ontology"
tags:
  - doc/task
  - status/done
  - phase/25
  - topic/autonomous-discovery
  - topic/entity-ontology
  - topic/self-improving
  - layer/surveillance
  - layer/feature-engineering
  - layer/world-model
---

# Task: Tier 8 — Autonomous Discovery & Self-Extending Ontology

Status: completed
Research: [[tier8_autonomous_discovery]]
Spec: [[tier8_autonomous_discovery_spec]]

---

## Change 15: Autonomous Data Source Discovery

- [x] 15.1: Add discovery tables to PipelineStore (discovered_sources, unresolved_entities, entity_type_registry)
- [x] 15.2: Create DataSourceCandidate + SourceScout (catalog search + probe)
- [x] 15.3: Create SignalEvaluator (MI-based signal assessment)
- [x] 15.4: Create ToolFactory (template-based tool generation + config persistence)
- [x] 15.5: Add dynamic arm support to ToolRoutingBandit
- [x] 15.6: Wire discovery into pipeline (cli.py tool loading, DAG quarantine)
- [x] 15.7: Create discovery orchestration function (run_source_discovery)
- [x] 15.T: Write and run Change 15 test suite (36/36 pass)

## Change 16: Self-Extending Entity Ontology

- [x] 16.1: Create OntologyRegistry (dynamic entity type + link type registry)
- [x] 16.2: Replace Literal EntityType with runtime validation
- [x] 16.3: Create TypeInducer (clustering + type proposal)
- [x] 16.4: Add relationship induction to TypeInducer
- [x] 16.5: Make GNN _CONNECTED_TYPES dynamic
- [x] 16.6: Wire ontology into pipeline startup
- [x] 16.T: Write and run Change 16 test suite (36/36 pass)

## Regression

- [x] 72/72 Tier 8 tests pass
- [x] 8943/8943 existing tests pass (41 pre-existing stale count assertions unchanged)

---

## Related

- [[tier8_autonomous_discovery]] — Research
- [[tier8_autonomous_discovery_spec]] — Spec
- [[learned_vs_handcoded_architecture_spec]] — Master spec
- [[tier7_self_modifying_structure]] — Previous tier

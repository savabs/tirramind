---
title: "Task: Phase 36 — Connect Disconnected Entity Types"
tags:
  - doc/task
  - status/done
  - phase/36
  - topic/entity-linking
  - topic/gnn
  - layer/world-model
  - layer/surveillance
---

# Task: Phase 36 — Connect Disconnected Entity Types

Status: completed
Research: [[phase36_connect_disconnected_entities]]
Spec: [[phase36_connect_disconnected_entities_spec]]

## Steps

- [x] 36.1: Add `_TOPIC_INSTRUMENT_MAP` dict to `polymarket.py`
- [x] 36.2: Add `topic_relates_to_instrument` links in polymarket `_persist_entities_inner()`
- [x] 36.3: Add `build_domain_company_map()` helper to `instrument_universe.py`
- [x] 36.4: Add `domain_owned_by` links in `cert_transparency._persist_entities_inner()`
- [x] 36.5: Add `domain_owned_by` links in `dns_monitor._persist_entities_inner()`
- [x] 36.6: Update SyntheticGraphGenerator (defaults + link generation for domain/topic)
- [x] 36.7: Write edge case tests (`tests/test_phase36_entity_linking.py`)
- [x] 36.8: Run full regression — 188/188 affected tests pass; pre-existing failures unrelated

## Related

- [[phase36_connect_disconnected_entities]]
- [[phase36_connect_disconnected_entities_spec]]
- [[quant_training_ground]]
- [[phase35_gnn_retrain_expanded_graph]]

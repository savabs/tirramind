---
title: "Checkpoint: 2026-04-16 — Phase 28 Complete, Phase 29 Ready"
tags:
  - doc/checkpoint
  - phase/28
  - phase/29
---

# Checkpoint: 2026-04-16 — Phase 28 Complete, Phase 29 Ready

## Session Summary

Committed all outstanding work from Phases 27–28 and the learned architecture tiers. Working tree is now clean. Phase 29 preflight artifacts created.

## Commits This Session

| Hash | Description |
|------|-------------|
| `fd8df01` | Phase 27-28: L2 entity persistence + macro enrichment (42 files) |
| `66f7152` | Learned architecture tiers 3-8 (35 files) |
| `92a8be8` | Tier 8: Discovery module (7 files) |
| `c204586` | Pipeline DAGs, executor, CLI (13 files) |
| `cb1c447` | Automation scripts (4 files) |

## Current State

- **Tests**: 4028 pass, 1 known pre-existing failure (`test_feature_generation_dag` — stale count `assert 17 == 6`)
- **OBSERVATION_TYPES**: 32
- **ENRICHMENT_DIM**: 41
- **Tools**: 60
- **Bandit arms**: 48
- **Working tree**: clean

## Phase 29: Company + Investigative L2 (READY)

Preflight artifacts created:
- Research: [[phase29_company_investigative_l2]]
- Spec: [[phase29_company_investigative_l2_spec]]
- Task: [[phase29_company_investigative_l2]]

### Steps
- [ ] 29.1: bankruptcy_court L2 persistence + tests
- [ ] 29.2: foia_requests L2 persistence + tests
- [ ] 29.3: academic_preprints L2 persistence + tests
- [ ] 29.4: graph_builder: 3 new obs types (32→35, ENRICHMENT_DIM 41→44)
- [ ] 29.5: Integration diagnostic tests
- [ ] 29.6: Full regression + checkpoint

### New Observation Types
- `bankruptcy_status` → company nodes (from bankruptcy_court)
- `investigation_signal` → company/person nodes (from foia_requests)
- `research_velocity` → company/topic nodes (from academic_preprints)

## Other Completed This Session

- Positioning task 25P.8 marked done (deferred-by-design, moved to `tasks/done/`)
- Working tree audit documented

## Known Issues

- `test_feature_generation_dag`: stale count assertion (17 vs 6) — needs update when feature list stabilizes

## Related

- [[phase29_company_investigative_l2]]
- [[phase29_company_investigative_l2_spec]]
- [[chat_checkpoint_2026-04-16_phase27_complete]]
- [[phase28_country_macro_enrichment]]

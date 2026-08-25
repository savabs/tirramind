---
title: "Checkpoint: Phase 29 — Reviewed Memory Pipeline Complete"
tags:
  - doc/checkpoint
  - phase/29
  - topic/memory
  - layer/learning
---

# Checkpoint: Phase 29 — Reviewed Memory Pipeline Complete

**Date:** 2026-04-17
**Status:** Implementation complete, all tests passing

## What Was Done

Implemented a lesson candidate promotion pipeline + episodic decay for TirraMind's autonomous memory system, inspired by the agentic-stack candidate lifecycle concept but adapted for autonomous statistical operation (no CLI review, no human-in-the-loop).

### Files Modified

| File | Change |
|------|--------|
| `agent/memory/store.py` | Added `validated`/`run_id` fields to `LearningEntry`, `decay()` to `EpisodicMemory`, `get_validated_learnings()`/`get_unvalidated_learnings()`/`mark_validated()`/`_persist_learnings()` to `SemanticMemory` |
| `agent/memory/candidates.py` | **NEW** — `LessonCandidate`, `CandidateStore`, `ProcessResult` (~260 LOC). Cluster→stage→promote/reject lifecycle with statistical thresholds |
| `agent/config/settings.py` | Added `lesson_min_support` (3), `lesson_min_runs` (2), `episode_ttl_days` (30) with `TIRRA_` env var parsing |
| `agent/core/autonomous.py` | Wired `CandidateStore` into the autonomous loop: import, init, run_id generation, candidate processing after each learning store, episodic decay at end of run |
| `tests/test_candidates.py` | **NEW** — 30 tests across 14+ edge case classes |

### Promotion Rules (CandidateStore)

- **Accept when:** `support_count ≥ min_support` AND `distinct_runs ≥ min_runs` AND `sign_agreement ≥ 0.8` AND not duplicate of already-accepted
- **Reject when:** low avg reward OR stale (>90 days) OR contradicts accepted lesson
- **Anti-churn:** rejection count tracked per candidate

### Test Results

- **30/30 new tests passing** (0.14s)
- **Full regression:** 3772+ tests passing (1 pre-existing failure in `test_entity_linking.py::TestWhaleAlertTransactsWith::test_no_inputs_no_links` — unrelated, entity graph issue)
- **Additional pre-existing failures:** 6 in `test_feature_generation_dag.py` (source_signals validation) — also unrelated to our changes

### Bugs Fixed During Implementation

1. `mem_dir` scope bug in `autonomous.py` — used variable only defined in `__init__`, fixed with `Path(self._config.memory_dir)`
2. Test helper falsy bug — `lessons=[] or ["default"]` evaluates to `["default"]` because `[]` is falsy; fixed with `if lessons is not None`
3. Backward compat double-load — `SemanticMemory` constructor auto-loads, test was calling `_load_learnings()` again

## Artifacts

- Research: [[reviewed_memory]]
- Spec: [[reviewed_memory_spec]]
- Task: [[reviewed_memory]] (in `tasks/done/`, status: completed)

## Pre-existing Issues (Not Phase 29)

- `test_entity_linking.py::TestWhaleAlertTransactsWith::test_no_inputs_no_links` — entity graph bug
- 6 failures in `test_feature_generation_dag.py` — feature generation `source_signals` validation

## Next Steps

- The pipeline now auto-promotes lessons with statistical support — monitor in production runs
- Consider adding GNN-guided expansion to determine when to tighten/loosen promotion thresholds

## Related

- [[reviewed_memory]]
- [[reviewed_memory_spec]]
- [[store]]
- [[autonomous]]

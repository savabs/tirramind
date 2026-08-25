---
title: "Checkpoint 2026-08-24 — AWOS self-learning embedded into TirraMind"
tags:
  - doc/checkpoint
  - phase/1
  - topic/signals
  - status/active
---

# Checkpoint: AWOS Learning Runtime Embedded into TirraMind

**Date:** 2026-08-24

## Summary

Ported the best self-learning capabilities from the AWOS coding-agent project
into TirraMind's embedded `agent/awos/` runtime as a new `agent/awos/learning/`
subpackage. The signal-runtime now has genuine self-improvement memory + policy,
not just event routing.

## What was added

| Module | Capability (ported from AWOS) | Status |
|---|---|---|
| `error_pattern_store.py` | Failure-memory (Reflexion-style verbal critiques) | ✅ |
| `skill_library.py` | Success-memory — "what approach wins" per signal operation | ✅ |
| `reward_store.py` | Outcome ledger + asymmetric cost-proportional reward math | ✅ |
| `prompt_evolver.py` | OPRO-style guideline evolution from outcomes | ✅ |
| `live_tool_synth.py` | On-the-fly helper script synthesis on recurring failures | ✅ |
| `ml_router.py` | LinUCB contextual bandit — learned method-tier selection | ✅ |
| `learning_core.py` | `LearningCore` composition object for the runtime | ✅ |
| `actions/learning.py` | `record_learning` action wired into the runtime dispatcher | ✅ |

## Wire-up

- `agent/awos/orchestrator/dispatcher.py` now imports the learning action
  (side-effect registration consistent with existing actions).
- Policy rules can fire `record_learning` to write an outcome into the
  learning pipeline; `default_learning(state_dir)` instantiates the core
  bound to the runtime config's state directory.

## Verification

- `.venv/bin/python -m pytest tests/test_awos_learning.py -q` — **17 passed**
- `.venv/bin/ruff check agent/awos/learning/ agent/awos/actions/learning.py agent/awos/orchestrator/dispatcher.py tests/test_awos_learning.py` — clean
- Runtime integration: CLI + Daemon import OK; `record_learning` registered and runs end-to-end against a temp state dir.
- Broader slice: 146 passed; 8 pre-existing failures (node-count drift in `pipeline_registry` + L2 `gov_contracts` needing fresh live data) — unrelated to this change.

## How it fits the plan

The embedded AWOS runtime is now more than an event router — it can learn from
the signal operations it performs (tender fetching, scoring, fusion, alerts).
This is the "best of AWOS inside the product" capability the user requested.

## Next (your call)

- Wire `LearningCore` into a real signal action (e.g. record fetch/score/alert
  outcomes from the pipeline) so the loop learns from live operations.
- Or continue adding the remaining "agent in the software" capabilities
  (agent_memory / vector recall surface).

## Related

- [[signals_primer]]
- [[cross_domain_signal_proof]]
- [[checkpoint_2026-08-23_foundation_verified]]
- [[quant_training_ground]]
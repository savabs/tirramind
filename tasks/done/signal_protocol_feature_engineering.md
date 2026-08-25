---
title: "Task: signal_protocol_feature_engineering"
tags:
  - doc/task
  - layer/feature-engineering
  - layer/world-model
  - phase/9
  - status/active
  - topic/signal-protocol
---

# Task: signal_protocol_feature_engineering

Status: completed
Research: [[signal_protocol_feature_engineering]]
Spec: [[signal_protocol_feature_engineering_spec]]

## Goal
Start Phase 8 by defining the stable feature contract and the first engineered-feature pipeline slice.

## Steps

- [x] 8.1: Define engineered-feature protocol and validation rules
  Verification: `agent/features/protocol.py` — frozen `EngineeredFeature` dataclass, `validate_feature()`, 98/98 tests pass ✅
- [x] 8.2: Extend pipeline persistence for engineered features
  Verification: `features` table in PipelineStore with store/query/batch/latest methods, validation at write boundary, idempotent upsert, 56/56 tests pass ✅
- [x] 8.3: Implement first convergence-derived feature builder
  Verification: `ConvergenceFeatureBuilder` in `agent/features/builders.py` — stress_breadth, stress_intensity, regime_persistence; 53 tests pass ✅
- [x] 8.4: Implement first continuous-state feature builder
  Verification: `MacroStateFeatureBuilder` — rate_momentum, yield_curve_slope, liquidity_pressure; same 53 tests pass ✅
- [x] 8.5: Add DAG integration for feature generation
  Verification: `agent/pipeline/dags/feature_generation.py` — `run_feature_generation()` callback + `build_feature_generation_dag()`; registered in `get_default_dags`; builder failure resilience; batch fallback; 27 tests pass ✅
- [x] 8.6: Write and run edge-case test suite
  Verification: `tests/test_feature_generation_dag.py` — DAG structure, registration, callback happy/edge/failure paths, end-to-end persistence roundtrip, protocol validation, idempotency; all 294 tests pass across full feature stack ✅

## Notes

- Canonical phase ordering follows `[[quant_training_ground]]`: Phase 8 is Signal Protocol + Feature Engineering, Phase 9 is World Model.
- Older notes that refer to “Phase 8 world model” should be treated as historical numbering, not the current workflow source of truth.

---

## Related

- [[signal_protocol_feature_engineering|Research: Signal Protocol Feature Engineering]]
- [[signal_protocol_feature_engineering_spec|Spec: Signal Protocol Feature Engineering]]
- [[convergence_detection]]
- [[world_model]]
- [[backtest_performance]]

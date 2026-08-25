---
title: "Spec: Signal Protocol + Feature Engineering"
tags:
  - doc/spec
  - layer/feature-engineering
  - phase/9
  - topic/signal-protocol
---

# Spec: Signal Protocol + Feature Engineering

## Goal
Create the first stable downstream feature contract for the pipeline so Phase 9 can consume model-ready quantitative state variables instead of ad hoc raw signal payloads.

## Files Affected
- `[[signal_protocol_feature_engineering]]`
- `[[signal_protocol_feature_engineering]]`
- Planned implementation files for the first slice:
  - `agent/features/` new package
  - `agent/pipeline/store.py`
  - `agent/pipeline/dags/` feature DAG wiring
  - focused tests under `tests/`

## Implementation Steps
1. Define a canonical engineered-feature protocol.
   - Required fields: feature name, effective timestamp, source lineage, horizon, value, confidence/quality, freshness, version.
2. Extend persistence for engineered features.
   - Add a dedicated storage surface rather than overloading the existing `signals` table further.
3. Implement the first deterministic feature builder on top of convergence output.
   - Example: convergence regime state, event breadth, persistence-weighted stress.
4. Implement one continuous-state feature builder on top of an existing macro/liquidity source.
   - Example: rolling momentum / spread / anomaly features.
5. Add pipeline DAG integration for feature generation.
6. Add comprehensive edge-case tests and point-in-time leakage checks.

## Edge Cases
- Missing upstream signals should produce explicit missingness, not silent zeroes.
- Recomputed features must be versioned so schema changes do not corrupt historical interpretation.
- Event signals and continuous signals need separate freshness semantics.
- Storage writes must tolerate duplicate recomputation paths deterministically.
- Feature timestamps must reflect when the information became knowable, not when the DAG happened to run.

## Testing Plan
- Unit tests for protocol validation and serialization.
- Storage tests for engineered-feature persistence and querying.
- Feature-builder tests for invalid inputs, empty upstream data, stale data, NaNs, duplicate timestamps, and leakage boundaries.
- DAG integration tests to confirm deterministic feature emission.

---

## Related

- [[signal_protocol_feature_engineering|Research: Signal Protocol Feature Engineering]]
- [[convergence_detection]]
- [[world_model]]
- [[backtest_performance]]

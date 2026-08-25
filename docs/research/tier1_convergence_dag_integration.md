---
title: "Feature: Tier 1 Convergence DAG Integration Test"
tags:
  - doc/research
  - phase/7c
  - topic/convergence
---

# Feature: Tier 1 Convergence DAG Integration Test

## Current Architecture

- `agent/pipeline/dags/convergence_detection.py` loads stored tool payloads via `_load_evidence_from_store()`, builds a `SignalRegistry`, runs `ConvergenceDetector.detect()`, and emits `convergence.*` signals.
- `agent/convergence/extractors.py` now contains real extractors for `internet_infrastructure`, `power_grid`, and `defi_flows`.
- `tests/test_convergence_dag.py` mostly validates DAG structure and callback behavior with mocks.
- `tests/test_convergence_subphase_d_edge.py` has broad mocked integration coverage, but not a store-backed Tier 1 smoke test.

## Observations

- The main remaining regression risk after Tier 1 extractor work is wiring, not extractor math in isolation.
- The DAG callback currently has no test that stores real Tier 1 payloads in `PipelineStore`, lets `_load_evidence_from_store()` extract real evidence, and verifies those signals reach the detector callback path.
- A full mathematical end-to-end convergence assertion is brittle here because detector output depends on broader historical structure; the stable seam is to keep store + extractor + registry real and patch detector output at the final detection boundary.

## Risks

- If the test mocks `_load_evidence_from_store()`, it misses the new Tier 1 evidence path entirely.
- If the test relies on real detector convergence formation from a tiny synthetic store, it may become fragile and fail for reasons unrelated to Tier 1 wiring.
- If signal IDs in the synthetic detection result do not match actual extracted Tier 1 IDs, the smoke test will under-validate the registry path.

## Step-Local References

- Local code reference: `agent/pipeline/dags/convergence_detection.py`
- Local code reference: `agent/convergence/extractors.py`
- Local test reference: `tests/test_convergence_dag.py`
- Local test reference: `tests/test_convergence_subphase_d_edge.py`
- Existing workflow reference: `[[convergence_detection_spec]]` step `7c-D.4`

## Test Strategy

- Add a real-store evidence loader test using Tier 1 payloads from `internet_infrastructure`, `power_grid`, and `defi_flows`.
- Add a DAG callback smoke test that:
  - stores real Tier 1 payloads,
  - uses the real `_load_evidence_from_store()` and `build_registry_from_evidence()`,
  - patches only `ConvergenceDetector.detect()` / persistence state,
  - asserts a `convergence.*` signal is emitted to the store.

## Reuse / License Notes

- No external code reuse is needed for this step.
- This is repo-internal test coverage only.

---

## Related

- [[tier1_convergence_dag_integration_spec|Spec: Tier1 Convergence Dag Integration]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]

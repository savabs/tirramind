---
title: "Spec: Tier 1 Convergence DAG Integration Test"
tags:
  - doc/spec
  - phase/7c
  - topic/convergence
---

# Spec: Tier 1 Convergence DAG Integration Test

## Goal

Add a store-backed convergence DAG smoke test that proves new Tier 1 tool payloads (`internet_infrastructure`, `power_grid`, `defi_flows`) are converted into evidence and can flow through the DAG callback to emitted `convergence.*` signals.

## Files Affected

- `tests/test_convergence_dag.py` — extend with Tier 1 loader and DAG smoke tests
- `[[tier1_convergence_dag_integration]]` — research and rationale
- `[[tier1_convergence_dag_integration]]` — atomic task tracking

## Implementation Steps

### Step 1: Add store-backed Tier 1 evidence loader test

Store representative payloads for:
- `internet_infrastructure` in `outages` mode
- `power_grid` in pricing or demand mode
- `defi_flows` in `tvl` mode

Call `_load_evidence_from_store()` with a real in-memory `PipelineStore` and assert the expected Tier 1 signal IDs appear.

### Step 2: Add DAG callback Tier 1 smoke test

Use a real in-memory `PipelineStore` populated with Tier 1 payloads. Patch only `ConvergenceDetector` so:
- the constructor receives a real registry built from extracted evidence,
- `detect()` returns one deterministic `DetectionResult` referencing actual Tier 1 signal IDs,
- `persistence_history` contains the result fingerprint.

Then call `run_convergence_detection()` and assert:
- `detected == 1`
- `emitted == 1`
- one `convergence.*` signal exists in the store
- emitted metadata includes the expected event type and Tier 1 categories

### Step 3: Verification

Run focused tests covering:
- the new Tier 1 loader test
- the new DAG smoke test
- the existing convergence DAG suite

## Edge Cases

- Stored Tier 1 payload exists but some tools emit multiple evidence rows; test should assert inclusion, not strict equality.
- Query ordering in `PipelineStore` is newest-first; assertions must not assume insertion order.
- DetectionResult signal names must match actual extracted Tier 1 signal IDs.
- The smoke test should not depend on real detector math or historical convergence persistence beyond patched detector state.

## Testing Plan

- `pytest tests/test_convergence_dag.py -v`
- If needed, run specific new test selectors first for faster debugging.

---

## Related

- [[tier1_convergence_dag_integration|Research: Tier1 Convergence Dag Integration]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]

---
title: "Task: Phase 7c — Convergence Detection Layer"
tags:
  - doc/task
  - layer/feature-engineering
  - layer/world-model
  - phase/7c
  - phase/9
  - status/active
  - topic/convergence
---

# Task: Phase 7c — Convergence Detection Layer

Status: completed
Research: [[convergence_detection]]
Spec: [[convergence_detection_spec]]

## Goal

Build the convergence detection system that transforms 60 independent data pipes into an integrated intelligence layer, detecting when normally-uncorrelated signals begin moving together.

## Scope Notes

- Layer: L2 (Feature Engineering) / L3 (World Model boundary)
- New package: `agent/convergence/`
- New DAG: `agent/pipeline/dags/convergence_detection.py`
- Non-goals: LLM reasoning, RL integration, transfer entropy (deferred), Bayesian network (Phase 9)

---

## Sub-phase 7c-A: Evidence Protocol + Taxonomy + Extractors

- [x] 7c-A.1: Create `agent/convergence/__init__.py` + `evidence.py` — Evidence frozen dataclass, EvidenceBus (submit/flush/snapshot), validation
  Verification: `pytest tests/test_convergence_evidence.py -v` — 39/39 pass ✅

- [x] 7c-A.2: Create `agent/convergence/taxonomy.py` — 11 CATEGORIES, SignalMeta dataclass, SignalRegistry (register/get/by_source/by_category)
  Verification: `pytest tests/test_convergence_taxonomy.py -v` — 43/43 pass ✅

- [x] 7c-A.3: Create `agent/convergence/extractors.py` — extractor framework + first 10 tools (cftc, weather_alerts, sanctions_monitor, ais_vessel_tracking, finra_short_volume, disease_surveillance, earthquake_proximity, global_pmi, treasury_receipts, job_postings)
  Verification: `pytest tests/test_convergence_extractors.py -v` — 71/71 pass ✅

- [x] 7c-A.4: Extend `extractors.py` — remaining 31 tool extractors (26 real + 5 output-only stubs, all categories covered)
  Verification: `pytest tests/test_convergence_extractors.py -v` — 180/180 pass ✅

- [x] 7c-A.5: Sub-phase A edge-case test suite — NaN values, boundary confidence, bus with 10K items, deeply nested data, wrong types, case-sensitive categories
  Verification: `pytest tests/test_convergence_evidence.py tests/test_convergence_taxonomy.py tests/test_convergence_extractors.py tests/test_convergence_subphase_a_edge.py -v` — 386/386 pass ✅ (110 new edge-case tests in 20 classes)

---

## Sub-phase 7c-B: Temporal Alignment + Atomic Signals

- [x] 7c-B.1: Create `agent/convergence/alignment.py` — TimeGrid enum, align_pair(), align_to_grid(), LOCF fill, staleness (is_stale), coarser-grid rule, event→binary
  Verification: `pytest tests/test_convergence_alignment.py -v` — 53/53 pass ✅

- [x] 7c-B.2: Create `agent/convergence/atomic_signals.py` — RollingStats (z_score, percentile), compute_anomaly(), normalize_direction(), SignalStream (ingest/compute), AtomicSignalResult
  Verification: `pytest tests/test_convergence_atomic.py -v` — 58/58 pass ✅ (hand-computed values verified)

- [x] 7c-B.3: Sub-phase B edge-case suite — all-NaN series, σ=0, window=1, window>data, out-of-order timestamps, duplicate timestamps, ties in percentile, staleness boundary
  Verification: `pytest tests/test_convergence_alignment.py tests/test_convergence_atomic.py tests/test_convergence_subphase_b_edge.py -v` — 148/148 pass ✅ (37 new edge-case tests in 17 classes, 534/534 total convergence tests)

---

## Sub-phase 7c-C: Coincidence Detection + Graph + FDR

- [x] 7c-C.1: Create `agent/convergence/coincidence.py` — rolling_correlation_score(), joint_exceedance_score(), concordance_score(), combined_coincidence_score(), CoincidenceResult dataclass
  Verification: `pytest tests/test_convergence_coincidence.py -v` — 65/65 pass ✅

- [x] 7c-C.2: Create `agent/convergence/graph.py` — build_coincidence_graph(), detect_convergence_cliques(), score_clique(), ConvergenceClique dataclass
  Verification: `pytest tests/test_convergence_graph.py -v` — 41/41 pass ✅

- [x] 7c-C.3: Create `agent/convergence/fdr.py` — apply_bh_correction(), fisher_combined_test(), persistence_filter(), cross_category_filter(), apply_all_controls()
  Verification: `pytest tests/test_convergence_fdr.py -v` — 39/39 pass ✅

- [x] 7c-C.4: Sub-phase C edge-case suite — length-0 arrays, σ=0 signal, all-NaN, 1000-node graph perf, 10K p-values BH, Fisher with p=0, persistence min=1
  Verification: `pytest tests/test_convergence_coincidence.py tests/test_convergence_graph.py tests/test_convergence_fdr.py tests/test_convergence_subphase_c_edge.py -v` — 180/180 pass ✅ (35 new edge-case tests in 13 classes)

---

## Sub-phase 7c-D: Templates + Detector + DAG Integration

- [x] 7c-D.1: Create `agent/convergence/templates.py` — CausalTemplate + TemplateStep dataclasses, TEMPLATE_LIBRARY (12 templates), match_template(), match_all_templates(), TemplateMatchResult
  Verification: `pytest tests/test_convergence_templates.py -v` — 46/46 pass ✅

- [x] 7c-D.2: Create `agent/convergence/detector.py` — ConvergenceDetectorConfig, ConvergenceDetector (init, detect), smart pair selection (~100-300 pairs not 1770)
  Verification: `pytest tests/test_convergence_detector.py -v` — 24/24 pass ✅

- [x] 7c-D.3: Create `agent/convergence/signals.py` — ConvergenceSignal dataclass, emit_signals(), format_signal_name(), to_metadata_dict()
  Verification: `pytest tests/test_convergence_signals.py -v` — 24/24 pass ✅

- [x] 7c-D.4: Create `agent/pipeline/dags/convergence_detection.py` — build_convergence_detection_dag(), _run_convergence_detection callback, register in __init__.py
  Verification: `pytest tests/test_convergence_dag.py -v` — 36/36 pass ✅

- [x] 7c-D.5: Modify `agent/cli.py` + `pyproject.toml` — register convergence DAG, add networkx>=3.0 dep, verify imports
  Verification: `python -c "from agent.convergence.detector import ConvergenceDetector; print('OK')"` — prints OK ✅; 4 DAGs registered ✅

- [x] 7c-D.6: Sub-phase D edge-case suite + full integration — empty store, 1 tool only, signal serialization round-trip, template edge cases, detector config, DAG callback, registry builder, full pipeline mocked integration
  Verification: `pytest tests/test_convergence_subphase_d_edge.py -v` — 39/39 pass ✅; `pytest tests/test_convergence_*.py -v` — 883/883 pass in 4.5s ✅

---

## Completion Checklist

- [x] Research note exists and is current
- [x] Spec matches the actual implementation plan
- [x] Each completed step has a verification result
- [x] Edge-case testing was added and run for code changes (sub-phases A.5, B.3, C.4, D.6)
- [x] Checkpoint written at the end of the session or sub-phase
- [x] All 12+ test files pass — 883/883 in 4.60s (verified 2026-04-05)
- [x] No circular imports — `from agent.convergence.detector import ConvergenceDetector` OK
- [x] Convergence DAG registered and executable — 4 DAGs via get_default_dags()
- [x] Pipeline store signals table used correctly — emit_signals() → store.store_signal()
- [x] No LLM calls in convergence package (deterministic pipeline) — grep confirmed 0 matches

## Notes

- Step 7c-A.4 (50 remaining extractors) is the largest single step. It's repetitive but not complex — each extractor is 5-15 lines. Consider splitting into batches if context fills.
- Smart pair selection in 7c-D.2 is the key performance optimization. Without it, O(n²) per cycle. With it, O(n × categories).
- Persistence state (in 7c-C.3) is held in memory within the ConvergenceDetector instance. If the process restarts, persistence resets. This is acceptable for now — persisting to SQLite is a future enhancement.
- The 3 deferred coincidence methods (transfer entropy, mutual information, PELT batch changepoint) will be added as 7c+ when 30+ days of pipeline data exist.

---

## Related

- [[convergence_detection|Research: Convergence Detection]]
- [[convergence_detection_spec|Spec: Convergence Detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]

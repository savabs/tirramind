---
title: "Task: convergence_backtest"
tags:
  - doc/task
  - phase/7c
  - status/active
  - topic/backtest
  - topic/convergence
---

# Task: convergence_backtest

Status: completed
Research: [[convergence_backtest]]
Spec: [[convergence_backtest_spec]]

## Goal
Prove the convergence engine works via synthetic validation + macro backtest. Establish phase-gate backtest rule for all future phases.

## Steps

### Sub-phase A: Synthetic Validation Engine
- [x] A.1: Create `agent/convergence/synthetic.py` — SyntheticScenario, generate_planted_chain, generate_decoy_signals, generate_scenarios
- [x] A.2: Add run_synthetic_validation() — precision/recall/F1/template accuracy/direction accuracy
- [x] A.3: Edge-case tests for synthetic validation (36 tests, all passing)

### Sub-phase B: Historical Macro Backtest
- [x] B.1: Create `agent/convergence/backtest.py` — HistoricalEvidenceBuilder (FRED → Evidence, no look-ahead)
- [x] B.2: Add FRED history fetcher helper (cached, 13 series)
- [x] B.3: Create ConvergenceBacktestStrategy + variants (risk-off, directional, template-weighted)
- [x] B.4: Create run_macro_backtest() orchestrator (WalkForward + scoring + bootstrap CI)
- [x] B.5: Edge-case tests for backtest engine (51 tests, all passing)

### Sub-phase C: Results & Phase Gate
- [x] C.1: Run full backtest with FRED data and record baseline results in docs/baselines/convergence_backtest_baseline.json
- [x] C.2: Phase-gate implemented: validate_against_baseline() + CLI --validate flag
- [x] C.3: CLI entry point complete (--synthetic, --macro, --validate, --save-baseline)

## Baseline Metrics (Synthetic Validation, 100 scenarios, seed=42)
- TP=80, FN=20, FP=2
- Precision: 0.9756, Recall: 0.8000, F1: 0.8791
- Template accuracy: 0.4125
- Direction accuracy: 0.8625

## Validation
- 2026-04-06: `python -m agent.convergence.backtest --validate --targets SPY TLT GLD --start-year 2010 --end-year 2025`
- Result: phase gate passed against `docs/baselines/convergence_backtest_baseline.json`

---

## Related

- [[convergence_backtest|Research: Convergence Backtest]]
- [[convergence_backtest_spec|Spec: Convergence Backtest]]
- [[convergence_detection]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]

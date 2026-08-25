---
title: "Checkpoint 2026-05-26b: Deep Architecture Review Complete"
tags:
  - doc/checkpoint
  - phase/41
  - topic/architecture
  - topic/gnn
  - topic/ewc
  - status/active
---

# Checkpoint 2026-05-26b: Deep Architecture Review

**Date:** 2026-05-26  
**Session focus:** Deep architecture review — validate the full TirraMind stack before V16 Kaggle run  
**Prior checkpoint:** [[chat_checkpoint_2026-05-26]] (EWC sidecar fix, `scripts/kaggle_watch.py`)

---

## Session Outcome

Architecture review COMPLETE. Written to `[[architecture_review_2026]]` (12-section doc, all components validated with citations).

**Verdict: The architecture is sound. The loss function is the most important thing to change.**

---

## Key Decisions Written This Session

### 1. Enable ListNet for V16 (ZERO code change)

**Finding:** Current Huber loss does not optimize IC. LambdaRankIC (arXiv:2605.00501, Lin et al., May 2026) proves that directly optimizing Rank IC via ranking loss outperforms MSE/Huber on IC, ICIR, monthly return, AND Sharpe ratio. ListNet (already implemented in `trainer.py` as `_listnet_loss()`) is the available approximation.

**Action:** Set `use_listnet_return_loss=True` in V16 training config. This is a training flag, zero code change.

**Expected effect:** IC improvement from −0.033 toward +0.01 to +0.05 range.

### 2. EWC + Replay Buffer for Regime Changes (Phase 50)

**Finding:** EWC prevents catastrophic forgetting (old tasks being overwritten) but does NOT handle distributional shift (regime changes where relationships themselves change). Financial regime changes (2020 COVID, 2022 rate hike cycle) change the entity relationship graph, not just add new tasks. EWC alone is insufficient.

**Action (Phase 50):** Add regime-stratified replay buffer. Sample past training windows proportional to regime label (normal vs. high-changepoint). ~50 MB RAM overhead.

### 3. Fix time_delta NaN Before V16

**Finding:** time_delta loss = NaN every epoch (all 12 metric rows in v15 log). Root cause: likely log(0) or division-by-zero in delta computation when consecutive observations share the same `observed_at` timestamp.

**Action:** Add `torch.isnan(time_delta_loss).any()` guard before adding to total loss. If NaN → zero the component. Prevents silent propagation to total loss.

### 4. Over-squashing: Low-Medium Risk, Deferred

**Finding:** Spatiotemporal GNNs have proven over-squashing (NeurIPS 2025). With L=2 layers, we are not in the deep oversmoothing regime. Temporal over-squashing may limit long-range signals.

**Action:** Deferred. Monitor IC × temporal lag in Phase 40 backtest. Rewiring options available if needed.

### 5. Kalman Fusion: Build with UKF, Not Standard KF

**Finding:** Standard Kalman assumes linear dynamics + Gaussian noise. Financial time series are nonlinear with fat tails. When building Phase 20 Kalman fusion:

**Action:** Use UKF (Unscented Kalman Filter) + Student-t noise model. Consider Switching Kalman Filter to integrate with regime_gate (Phase 49b already built).

---

## Key Facts Table

| Fact | Value | Canonical owner |
|---|---|---|
| Best checkpoint | epoch_040.pt | kaggle_runbook.md |
| Best IC | −0.033 (WEAK) | tirramind_structure.md |
| EWC sidecar commit | 5ba8b2c | git log |
| EWC tests | 16/16 pass | tests/test_ewc.py |
| V16 key flag | `use_listnet_return_loss=True` | This checkpoint |
| LambdaRankIC paper | arXiv:2605.00501 | architecture_review_2026.md |
| time_delta NaN | Known bug, needs NaN guard | architecture_review_2026.md |
| Replay buffer for regime changes | Phase 50, not V16 | architecture_review_2026.md |
| Kaggle kernel | `deeperisbetter/tirramind-h-g` | kaggle_runbook.md |
| Working pytest | `PYTHONPATH=. ~/.local/bin/pytest` | This checkpoint |
| System pytest | BROKEN (AttributeError __spec__) | This checkpoint |

---

## What Is Validated

| Component | Status | Key source |
|---|---|---|
| HetTGN architecture | ✅ VALIDATED | MDGNN AAAI-24, THGNN 2021 |
| Cross-domain entity linking | ✅ VALIDATED (needed) | MSub-GNN PMC 2022 |
| EWC continual learning | ✅ VALIDATED (+ known limitation) | NeurIPS 2025, EVCL ICML 2024 |
| IC optimization target | ✅ VALIDATED | LambdaRankIC 2026, QuantBench 2024 |
| ListNet ranking loss | ✅ VALIDATED, **enable now** | Cao et al. ICML 2007, LambdaRankIC 2026 |
| Multi-task loss | ⚠️ time_delta NaN (fix needed) | v15 training logs |
| POMDP framing | ✅ VALIDATED | PO-Dreamer ICLR 2026, SWB ICML 2021 |
| SAC model-free RL | ✅ VALIDATED (current phase) | Model agnosticism doctrine |
| Bayesian world model (pgmpy) | ✅ VALIDATED (<500 nodes) | Informed Dreamer RLC 2024 |
| Kalman fusion | ✅ VALIDATED (use UKF for Phase 20) | IJACSA 2025, arXiv:2510.20952 |

---

## Immediate Next Steps (Priority Order)

1. **Fix time_delta NaN guard** (trainer.py, ~10 lines) — prevents silent NaN propagation
2. **V16 Kaggle prep**: stage epoch_040.pt, set `use_listnet_return_loss=True`, push kernel
3. **Validate V16**: confirm no EWC spikes in first 2 blocks, IC trending positive
4. **Phase 17 entity linking**: complete cross-domain edges before Phase 40 final run
5. **Phase 50 replay buffer**: after Phase 40 confirms IC baseline

---

## Files Changed This Session

| File | Change |
|---|---|
| `[[architecture_review_2026]]` | NEW — 12-section deep architecture review with citations |
| `[[quant_training_ground]]` | Updated GNN training status block (2026-05-26 entry) |
| `[[chat_checkpoint_2026-05-26b_architecture_review]]` | NEW — this file |

(Files changed in prior checkpoint `chat_checkpoint_2026-05-26`: `agent/models/gnn/trainer.py` EWC sidecar, `scripts/kaggle_watch.py`, `[[kaggle_runbook]]`)

---

## Related

- [[architecture_review_2026]] — the research doc produced this session
- [[quant_training_ground]] — active task file with V16 next steps
- [[chat_checkpoint_2026-05-26]] — prior checkpoint (EWC sidecar fix)
- [[kaggle_runbook]] — V16 upload checklist
- [[living_system_online_gnn]] — EWC research doc

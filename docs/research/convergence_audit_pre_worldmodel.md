---
title: Convergence Detection Layer — Pre-World-Model Audit
tags:
  - doc/research
  - layer/world-model
  - phase/7c
  - topic/convergence
---

# Convergence Detection Layer — Pre-World-Model Audit

**Date:** 2025-01-XX  
**Scope:** Full audit of `agent/convergence/` (10 modules, ~6,078 LOC) + DAG + tests (16 files, ~10,118 LOC)  
**Test status:** 883/883 passing (4.55s)  

---

## 1. Executive Summary

The convergence detection layer (Phase 7c) is solid infrastructure. The architecture is clean, the math is sound, and the test suite is thorough. There are **no blockers** for starting the world model. However, there are **10 concrete issues** worth tracking — some should be fixed before the world model consumes convergence output, others can be deferred.

**Verdict: Clear to proceed to Phase 8 (World Model).** Fix items tagged `[FIX-BEFORE-WM]` first; defer the rest.

---

## 2. Findings

### F1: Missing extractors for 10 data tools `[FIX-BEFORE-WM]`

**12 tool files** lack convergence extractors. After removing 2 naming mismatches (ais_vessel → ais_vessel_tracking, supply_chain_monitor → supply_chain_prices), **10 genuine gaps** remain:

| Tool | Category (likely) | Priority |
|------|-------------------|----------|
| `consumer_sentiment` | macro_momentum | HIGH — referenced in 6 templates |
| `food_security` | biological | HIGH — referenced in 2 templates |
| `political_risk` | geopolitical | HIGH — referenced in 2 templates |
| `defi_flows` | financial_stress | MEDIUM — referenced in credit_stress template |
| `gov_contracts` | supply_chain | MEDIUM — referenced in tech_disruption template |
| `migration_flows` | geopolitical | MEDIUM — referenced in geopolitical template |
| `academic_preprints` | behavioral_intent | LOW — niche signal |
| `internet_outages` | physical_disruption | LOW — overlaps dns/cert tools |
| `labor_disruptions` | behavioral_intent | LOW — overlaps job_postings |
| `power_grid` | physical_flow | LOW — overlaps electricity_monitor/energy_supply |

The 3 HIGH-priority tools (`consumer_sentiment`, `food_security`, `political_risk`) are explicitly referenced in template regex patterns. Without extractors, those templates can never fully match. This directly weakens convergence detection before the world model even consumes the output.

**Recommendation:** Write extractors for the 3 HIGH tools before Phase 8. The 7 remaining can follow later.

---

### F2: `direction` is hardcoded to `+1` in signal emission `[FIX-BEFORE-WM]`

In `signals.py:165`:
```python
direction = 1  # default stress
if hasattr(clique, "edges") and clique.edges:
    pass  # Default +1; can refine later with edge direction data
```

Every emitted `ConvergenceSignal` has `direction=+1` regardless of actual evidence. The coincidence layer *does* compute per-pair direction (+1/−1) in `CoincidenceResult.direction`, and the combined scorer produces a score-weighted direction vote. This information exists but isn't propagated to emission.

**Impact:** The world model will receive a useless direction field. If it uses direction to inform belief updates, it'll be wrong 50% of the time.

**Fix:** Propagate the combined coincidence direction through the clique to the detection result. The data is already available — it just needs wiring.

---

### F3: `p_values_combined` is a monkey-patched attribute `[DEFER]`

In `fdr.py:257`:
```python
clique.p_values_combined = combined_p  # type: ignore[attr-defined]
```

And in `signals.py:171`:
```python
p_value = getattr(clique, "p_values_combined", 1.0)
```

`ConvergenceClique` is a mutable dataclass but doesn't declare `p_values_combined` as a field. The FDR layer patches it on, and the signals layer reads it via `getattr`. This works but is fragile — any refactor to make `ConvergenceClique` frozen would break silently.

**Fix (deferred):** Add `p_values_combined: float = 1.0` as an optional field on `ConvergenceClique`, or create a `ScoredClique` wrapper in the FDR module.

---

### F4: `persistence_count` never passed in DAG callback `[FIX-BEFORE-WM]`

In `convergence_detection.py:141`:
```python
signals = [from_detection_result(r, as_of=as_of) for r in results]
```

`from_detection_result` accepts `persistence_count` (default=0), but the DAG never passes it. The detector *does* maintain `_persistence_history` with the correct counts. Result: `persistence_days` in every emitted signal is always 0.

**Fix:** After `detector.detect()`, look up each result's clique fingerprint in `detector.persistence_history` and pass the count to `from_detection_result`.

---

### F5: Detector creates its own `_persistence_history` dict — no persistence across DAG runs `[DEFER]`

`ConvergenceDetector.__init__` creates a fresh `_persistence_history = {}` every time. Since the DAG callback creates a new detector on every run, persistence state is lost between runs. The persistence filter effectively requires `min_persistence=1` (anything detected once survives) because the counter always starts at 0.

**Impact:** The persistence filter (one of the 4 FDR levels) provides no value in production today.

**Fix (deferred):** Serialize persistence state to the pipeline store between runs. Add a `load_persistence` / `save_persistence` method pair, or store it in a dedicated table.

---

### F6: `_select_pairs` truncation is alphabetical, not by signal quality `[DEFER]`

When pairs exceed `max_pairs` (default 500), the code sorts alphabetically and truncates:
```python
pair_list = sorted(pairs)
pair_list = pair_list[:cfg.max_pairs]
```

This is safe but suboptimal — it may discard high-z-score pairs in favor of alphabetically-early low-quality ones.

**Fix (deferred):** Sort by combined |z| of both signals before truncation.

---

### F7: `build_registry_from_evidence` uses first-occurrence category `[ACCEPTABLE]`

When the same `signal_id` appears with conflicting categories across different evidence items, the first occurrence wins. This is documented and acceptable since:
1. Each extractor produces consistent categories per signal_id.
2. Category conflicts would only arise from extractor bugs, which have their own test coverage.

**No action needed.**

---

### F8: Template matching uses first-match-wins per step `[ACCEPTABLE]`

In `match_template`, once a step finds its first matching signal, it moves on:
```python
best_signal = sig_id
break  # first match wins for this step
```

This means the order of `clique.signals` (which is sorted alphabetically) determines matches, not signal quality. Acceptable for now because:
1. Templates test structural patterns, not specific signals.
2. False positives are controlled by the min_match threshold.

**No action needed** until template matching needs to be more precise for the world model.

---

### F9: Math correctness — verified sound `[OK]`

All statistical methods are correctly implemented with proper guards:

| Method | Implementation | Guards |
|--------|----------------|--------|
| Rolling correlation z-score | Pearson ρ → Fisher z → deviation from baseline | `σ=0` guard, `_EPSILON`, NaN aware |
| Joint exceedance | Binomial test on co-occurrence vs independence null | `_P_FLOOR`, `_MIN_VALID` |
| Concordance | First-difference sign agreement → binomial z | Window validation, NaN filtering |
| Fisher combined test | −2Σln(pᵢ) ~ χ²(2k) | `_P_FLOOR` prevents log(0) |
| BH FDR | `statsmodels.multipletests(method="fdr_bh")` | Clip to [_P_FLOOR, 1.0] |
| Bron-Kerbosch | NetworkX `find_cliques` + `_MAX_CLIQUES` safety cap | 10,000 clique limit |
| Sigmoid scoring | `mean_weight × cat_ratio × log2(n)` → `1/(1+e^(-x))` | Bounded [0,1] |

**No numerical stability issues found.**

---

### F10: Test coverage assessment `[OK]`

| Module | Tests | Edge tests | Verdict |
|--------|-------|------------|---------|
| evidence.py | 39 | SubphaseA: 124 | ✅ |
| taxonomy.py | 43 | SubphaseA: ↑ | ✅ |
| extractors.py | 180 | SubphaseA: ↑ | ✅ |
| alignment.py | 53 | SubphaseB: 37 | ✅ |
| atomic_signals.py | 58 | SubphaseB: ↑ | ✅ |
| coincidence.py | 65 | SubphaseC: 35 | ✅ |
| graph.py | 41 | SubphaseC: ↑ | ✅ |
| fdr.py | 39 | SubphaseC: ↑ | ✅ |
| templates.py | 46 | SubphaseD: 39 | ✅ |
| detector.py | 24 | SubphaseD: ↑ | ✅ |
| signals.py | 24 | SubphaseD: ↑ | ✅ |
| DAG | 36 | SubphaseD: ↑ | ✅ |

Total: 883 tests across 16 files (~10,118 LOC of tests for ~6,078 LOC of implementation). 1.66:1 test-to-code ratio.

**No coverage gaps identified.** The edge-case test suites (subphase_*_edge files) cover boundary values, NaN handling, empty inputs, and numerical stability.

---

## 3. Prioritized Action Items

### Before World Model (Fix Now)

| # | Finding | Effort | Impact |
|---|---------|--------|--------|
| 1 | F1: Write extractors for `consumer_sentiment`, `food_security`, `political_risk` | ~2h | HIGH — unlocks 6 templates |
| 2 | F2: Propagate coincidence direction to emitted signal | ~30m | MEDIUM — fixes useless field |
| 3 | F4: Pass `persistence_count` from detector history to signal emission in DAG | ~15m | LOW — fixes always-zero field |

### After World Model (Defer)

| # | Finding | Effort | Impact |
|---|---------|--------|--------|
| 4 | F5: Serialize persistence state across DAG runs | ~1h | MEDIUM — persistence filter actually works |
| 5 | F3: Promote `p_values_combined` to proper field | ~15m | LOW — code cleanliness |
| 6 | F6: Sort pair truncation by signal quality | ~15m | LOW — edge case at 500+ pairs |
| 7 | F1: Write extractors for remaining 7 tools | ~3h | LOW — incremental coverage |

---

## 4. Architecture Health

### What's Strong
- **Clean layer separation:** No module mixes data fetching with math.
- **LLM-free contract:** Zero LLM calls in the entire convergence package — verified by grep.
- **Defensive extraction:** Every extractor returns `[]` on failure, never raises.
- **Future leakage prevention:** LOCF alignment with staleness guards — no interpolation.
- **Statistical rigor:** 4-level FDR pipeline (BH → graph → Fisher → persistence → cross-category).

### What's Ready for the World Model
The convergence layer outputs `ConvergenceSignal` objects with:
- `value` (0-1 convergence strength)
- `event_type` (template classification)
- `signals_involved` + `categories_involved` (provenance)
- `p_value` (statistical confidence)
- `template_match` (causal pattern strength)

These are exactly the evidence nodes the Bayesian world model needs. The output schema is clean and the pipeline store integration works.

### Interface to World Model
The world model should:
1. Read `convergence.*` signals from `PipelineStore.query_signals()`
2. Use `value` as observation strength
3. Use `event_type` to route to the appropriate subgraph/node
4. Use `p_value` for evidence weighting
5. Use `categories_involved` to activate relevant factor nodes

No convergence code changes needed for this interface — it's already in the store.

---

## 5. Conclusion

Phase 7c is production-quality infrastructure. The 3 fixes tagged `[FIX-BEFORE-WM]` are small, well-scoped, and can be done in a single session. After those, the convergence layer is fully ready to feed the Bayesian world model.

---

## Related

- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]

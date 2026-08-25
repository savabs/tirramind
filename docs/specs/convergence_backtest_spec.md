---
title: "Spec: convergence_backtest"
tags:
  - doc/spec
  - phase/7c
  - topic/backtest
  - topic/convergence
---

# Spec: convergence_backtest

## Goal
1. Prove the convergence detection engine produces statistically significant predictive signals via synthetic validation (Approach C) and hybrid macro backtest (Approach D).
2. Establish a phase-gate backtest rule: every future phase must pass the backtest baseline before moving forward.

## Files Affected

### New files
| File | Purpose |
|------|---------|
| `agent/convergence/backtest.py` | HistoricalEvidenceBuilder + ConvergenceStrategy classes |
| `agent/convergence/synthetic.py` | Synthetic scenario generator for ground-truth validation |
| `tests/test_convergence_backtest.py` | Tests for backtest engine |
| `tests/test_convergence_synthetic.py` | Tests for synthetic validation |
| `docs/baselines/convergence_backtest_baseline.json` | Phase-gate baseline metrics |

### Modified files
| File | Change |
|------|--------|
| `[[quant_training_ground]]` | Add phase-gate backtest rule to phase descriptions |

## Implementation Steps

### Sub-phase A: Synthetic Validation Engine (tests the math)

**A.1: Create `agent/convergence/synthetic.py` — scenario generator**

Build a `SyntheticScenario` dataclass and `generate_scenarios()` function:
- `SyntheticScenario`: name, planted_evidence (list[Evidence]), decoy_evidence (list[Evidence]), expected_template (str), expected_direction (int), expected_lead_signal (str)
- `generate_planted_chain(template_name, start_ts, num_points=30, noise_level=0.3)`: Creates Evidence items that follow a specific CausalTemplate's signal pattern, categories, and temporal ordering. Signal values embed a real anomaly (z > 2.0).
- `generate_decoy_signals(num_signals=5, num_points=30, start_ts)`: Creates random Evidence with no cross-correlation or temporal ordering.
- `generate_scenarios(n=100, seed=42)`: Produces N scenarios, each with 1 planted chain + 3-8 decoys.

Verification: unit test that generated scenarios have correct shapes, categories, timestamps.

**A.2: Create synthetic validation runner**

In `agent/convergence/synthetic.py`, add `run_synthetic_validation(scenarios, config=None)`:
1. For each scenario: create in-memory PipelineStore, store all evidence, run ConvergenceDetector.detect()
2. Check: was the planted chain detected? (true positive)
3. Check: was any decoy detected? (false positive)
4. Check: was the correct template matched?
5. Check: was the correct direction assigned?
6. Check: was the correct lead signal identified?
7. Return `SyntheticValidationResult`: precision, recall, F1, template_accuracy, direction_accuracy, confusion_matrix

Verification: run on 50 scenarios, measure precision/recall.

**A.3: Edge-case test suite for synthetic validation**

Cover: empty scenario, all decoys (no planted), conflicting templates, boundary z-scores, single-evidence scenarios, numerical stability with extreme values.

Verification: all tests pass.

### Sub-phase B: Historical Evidence Builder (feeds the macro backtest)

**B.1: Create `agent/convergence/backtest.py` — HistoricalEvidenceBuilder**

Class that converts FRED time series into Evidence objects:
```python
class HistoricalEvidenceBuilder:
    def __init__(self, fred_data: dict[str, pd.DataFrame]):
        """fred_data: {series_id: DataFrame with columns ['date', 'value']}"""
    
    def build_evidence(self, as_of: float) -> list[Evidence]:
        """Generate Evidence items from FRED data available as of timestamp.
        Uses only data with date <= as_of (no look-ahead).
        """
```

Map FRED series to Evidence signal_ids:
| FRED Series | signal_id | category | direction logic |
|---|---|---|---|
| WALCL | central_bank.fed.assets | monetary_policy | Δ > 0 → +1 (easing) |
| RRPONTSYD | central_bank.fed.rrp | monetary_policy | Δ > 0 → -1 (tightening) |
| DFF | rate_monitor.fed.rate | monetary_policy | Δ > 0 → -1 (tightening) |
| DGS10 | sovereign_debt.us.10y | financial_stress | Δ > 0 → +1 (stress) |
| T10Y2Y | sovereign_debt.us.curve | financial_stress | < 0 → +1 (inversion = recession) |
| BAMLH0A0HYM2 | creditor.us.hy_spread | financial_stress | Δ > 0 → +1 (stress) |
| UNRATE | jobs.us.unemployment | macro_momentum | Δ > 0 → +1 (stress) |
| ICSA | jobs.us.claims | macro_momentum | Δ > 0 → +1 (stress) |
| NAPM | pmi.us.manufacturing | macro_momentum | < 50 → +1 (contraction) |
| UMCSENT | consumer.us.sentiment | macro_momentum | Δ < 0 → +1 (stress) |
| PERMIT | building.us.permits | macro_momentum | Δ < 0 → +1 (contraction) |
| CPIAUCSL | cpi.us.headline | macro_momentum | Δ > 0 → +1 (inflation) |
| M2SL | monetary.us.m2 | monetary_policy | Δ > 0 → +1 (expansion) |

Verification: build evidence for 2020-03-01, check expected signals from COVID-era data.

**B.2: Create FRED data fetcher helper**

`fetch_fred_history(series_ids, start_date, end_date)` that uses the existing `MacroDataTool` or direct FRED API to pull long historical series. Cache results locally to avoid re-fetching.

Verification: fetch DFF from 2005-2025, confirm 5000+ data points.

**B.3: Create `ConvergenceBacktestStrategy` — adapts convergence signals to WalkForward**

```python
class ConvergenceBacktestStrategy(Strategy):
    """Walk-forward strategy that runs convergence detection at each test step."""
    
    name = "convergence_risk_off"
    
    def generate_weights(self, train_returns, test_length, train_extra, test_extra):
        """
        train_extra['evidence_builder']: HistoricalEvidenceBuilder
        train_extra['store']: PipelineStore
        test_extra['test_dates']: list of test period dates
        
        For each test date:
          1. Build evidence up to that date
          2. Run ConvergenceDetector.detect(as_of=date)
          3. If stress convergence detected (direction=+1, score > threshold) → weight=0
          4. Otherwise → weight=1
        """
```

Also implement:
- `ConvergenceDirectionalStrategy`: weight = -score × direction (proportional)
- `ConvergenceTemplateStrategy`: only act on high template_match (>0.5)

Verification: run on synthetic data, confirm weights change at expected points.

**B.4: Create `run_macro_backtest()` orchestrator**

```python
def run_macro_backtest(
    start_year: int = 2010,
    end_year: int = 2025,
    targets: list[str] = ["SPY", "TLT", "GLD"],
    min_train: int = 52,
    test_size: int = 12,
) -> dict[str, BacktestResult]:
```

1. Fetch FRED history for all 13 series
2. Fetch target returns via yfinance
3. Align to weekly frequency
4. Build HistoricalEvidenceBuilder
5. Run WalkForward with ConvergenceBacktestStrategy
6. Run WalkForward with BuyAndHoldStrategy (benchmark)
7. Compute block_bootstrap_ci for Sharpe difference
8. Return results per target

Verification: full run produces BacktestResult with aggregate metrics.

**B.5: Edge-case test suite for backtest engine**

Cover: missing FRED series, partial data, zero-variance periods, empty evidence windows, weight boundary values, NaN returns, single-fold backtest, misaligned dates.

### Sub-phase C: Results & Phase Gate

**C.1: Run full backtest and record results**

Execute `run_macro_backtest()` and `run_synthetic_validation()`. Record in `docs/baselines/convergence_backtest_baseline.json`.

**C.2: Add phase-gate backtest rule to workflow**

Update `[[quant_training_ground]]` to add a mandatory post-phase validation step for Phases 8-12. The rule:
- After each phase: run `python -m agent.convergence.backtest --validate`
- Compare to baseline; if any key metric degrades > 10% relative, block the phase.
- Store new baseline only when phase passes.

**C.3: Create CLI entry point**

Add `if __name__ == "__main__"` block to `agent/convergence/backtest.py`:
- `--synthetic`: run synthetic validation only
- `--macro`: run macro backtest only
- `--validate`: compare against baseline and report pass/fail
- `--save-baseline`: persist current results as new baseline

## Edge Cases
- **FRED API unavailable**: fall back to cached data; if no cache, raise clear error
- **Insufficient FRED history**: require 5+ years; skip series with gaps > 6 months
- **No convergence detected in any window**: valid result (engine finds nothing → weight=1 throughout → same as buy-hold)
- **All convergence = false positives**: strategy underperforms buy-hold → Sharpe negative → detection logged but not a failure of the test infrastructure
- **PipelineStore memory limits**: use on-disk SQLite for macro backtest (not in-memory); clean up between folds
- **Look-ahead bias**: strict `as_of` enforcement — HistoricalEvidenceBuilder MUST filter by date
- **Publication lag**: FRED official publication dates differ from data period dates. Use series observation date, not period date.

## Testing Plan
1. **A.3**: Synthetic edge cases (15+ tests)
2. **B.5**: Backtest engine edge cases (20+ tests)
3. **Full synthetic validation**: 100 scenarios, expect precision > 0.70, recall > 0.60
4. **Full macro backtest**: SPY 2010-2025, expect convergence strategy Sharpe ≥ buy-hold Sharpe, bootstrap CI computed
5. **Phase gate**: verify JSON baseline write/read/compare works

---

## Related

- [[convergence_backtest|Research: Convergence Backtest]]
- [[convergence_detection]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]

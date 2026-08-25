---
title: "Spec: Scoring & Validation Framework (Phase 3)"
tags:
  - doc/spec
  - topic/scoring
---

# Spec: Scoring & Validation Framework (Phase 3)

## Goal

Build a reusable backtesting and scoring engine that any strategy can be evaluated through. Replace Phase 2's ad-hoc validation with composable primitives. Expose it to the agent as a tool.

## Files Affected

### New files
- `agent/quant/backtest.py` — walk-forward backtester, strategy protocol, regime-conditional analysis
- `agent/tools/backtest.py` — BacktestTool for agent integration

### Modified files
- `agent/quant/scoring.py` — add Sortino, Calmar, VaR, CVaR, drawdown duration, turnover, bootstrap CIs
- `agent/cli.py` — register BacktestTool

## Implementation Steps

### Research & Spec (no code)
- [ ] 3.1: Write research doc (`[[scoring_validation]]`)
- [ ] 3.2: Write spec doc (`[[scoring_validation_spec]]`)

### Extended Scoring Metrics
- [ ] 3.3: Add `sortino_ratio()` to `scoring.py` — downside deviation denominator, test with known asymmetric returns
- [ ] 3.4: Add `calmar_ratio()` to `scoring.py` — ann. return / |max DD|, test with synthetic drawdown series
- [ ] 3.5: Add `value_at_risk()` and `cvar()` to `scoring.py` — historical percentile VaR + CVaR, test with known distributions
- [ ] 3.6: Add `drawdown_duration()` to `scoring.py` — max consecutive periods below peak, test with step-function series
- [ ] 3.7: Add `turnover()` to `scoring.py` — mean absolute weight change, test with known position series
- [ ] 3.8: Add `score_returns()` summary function — computes all metrics at once, returns a dict

### Bootstrap Confidence Intervals
- [ ] 3.9: Add `block_bootstrap_ci()` to `scoring.py` — block bootstrap for any metric function, configurable block length and n_bootstrap
- [ ] 3.10: Test bootstrap CIs — verify coverage on synthetic data (true Sharpe inside 95% CI ≥ 90% of the time)

### Strategy Protocol & Walk-Forward Engine
- [ ] 3.11: Create `agent/quant/backtest.py` — define `Strategy` protocol (abstract base): `generate_weights(train_data, test_dates) -> weights`
- [ ] 3.12: Implement `WalkForward` class — expanding-window backtester: takes Strategy + data, produces per-fold and aggregate results
- [ ] 3.13: Test WalkForward with a trivial strategy (always long) — verify returns match buy-and-hold
- [ ] 3.14: Implement `BacktestResult` dataclass — per-fold metrics, aggregate metrics, equity curve, regime breakdown
- [ ] 3.15: Implement `RegimeConditionalAnalysis` — given regime labels + returns, compute per-regime metrics using `score_returns()`

### Built-in Strategies (re-implement Phase 2 strategies as Strategy objects)
- [ ] 3.16: Implement `BuyAndHoldStrategy` — always weight=1.0, baseline reference
- [ ] 3.17: Implement `RegimeAvoidStrategy` — weight=0 during specified regime states, weight=1 otherwise
- [ ] 3.18: Implement `RegimeOnlyStrategy` — weight=1 during specified regime states, weight=0 otherwise

### Validation: Re-run Phase 2 through the engine
- [ ] 3.19: Re-run liquidity-regime backtest through WalkForward engine — verify results match Phase 2 (Sharpe within ±0.05)
- [ ] 3.20: Compute bootstrap CIs on Phase 2 strategy Sharpe ratios — document whether edge is statistically significant

### Agent Integration
- [ ] 3.21: Create `agent/tools/backtest.py` — BacktestTool that the agent can call with strategy type + parameters
- [ ] 3.22: Register BacktestTool in `agent/cli.py`
- [ ] 3.23: Test end-to-end: agent invokes BacktestTool → gets scored results

### Wrap-up
- [ ] 3.24: Update task file, mark Phase 3 complete

## Edge Cases

- **Insufficient data for a fold**: WalkForward must require minimum train size (e.g., 104 weeks = 2 years) and skip folds that don't meet it.
- **Zero-variance in fold**: scoring functions must handle σ=0 gracefully (return 0 or NaN, not crash).
- **Empty test period**: if regime strategy produces weight=0 for entire test window, return 0 metrics, not NaN.
- **Bootstrap block length > series length**: cap block length at series length / 2.
- **Look-ahead bias**: WalkForward must enforce that train data strictly precedes test data. No overlap. No future info.

## Testing Plan

Each step has its own inline test (synthetic data, known answers). Summary validation:
1. Extended metrics: compare against hand-calculated values on 3-element return series.
2. Bootstrap: verify 95% CI covers true parameter ≥ 90% over 100 Monte Carlo runs.
3. WalkForward: buy-and-hold strategy through engine must match manual calculation.
4. Phase 2 re-run: Sharpe ratios within ±0.05 of Phase 2 results.
5. Agent tool: call BacktestTool, verify structured output.

---

## Related

- [[scoring_validation|Research: Scoring Validation]]
- [[convergence_detection]]
- [[world_model]]
- [[backtest_performance]]

---
title: "Feature: Scoring & Validation Framework (Phase 3)"
tags:
  - doc/research
  - topic/scoring
---

# Feature: Scoring & Validation Framework (Phase 3)

## Current Architecture

### What exists (`agent/quant/scoring.py`)
- `sharpe_ratio(returns, risk_free, periods_per_year)` — annualized Sharpe from log returns
- `max_drawdown(returns)` — max peak-to-trough from log returns
- `information_ratio(returns, benchmark, periods_per_year)` — annualized IR vs benchmark
- `hit_rate(predictions, actuals)` — directional accuracy fraction

### What was done ad-hoc in Phase 2 (not in reusable code)
- Walk-forward backtest: train 2009-2020, test 2021-2024 — applied HMM regimes to SPY returns
- Regime-conditional return analysis: t-tests per regime for multiple assets
- Three strategy variants: buy-and-hold, avoid-crisis, only-neutral
- All done in inline scripts, nothing reusable

### What's missing for a serious scoring/validation engine
The current scoring module is a collection of standalone functions. There's no:
1. **Backtesting engine** — no reusable walk-forward or expanding-window backtester
2. **Strategy abstraction** — no way to define a strategy as an object and run it through validation
3. **Extended risk metrics** — no Sortino, Calmar, VaR, CVaR, drawdown duration, turnover
4. **Statistical significance** — no bootstrap CIs, no permutation tests, no multiple-testing corrections
5. **Signal evaluation** — no IC (information coefficient), IC-IR, quintile analysis, signal decay
6. **Agent tool** — no way for the agent to run a backtest or score a strategy via the tool interface

## Observations

### Phase 2 findings shape Phase 3 design
- Regime detection provides **situational awareness**, not direct signals. This means the scoring framework needs to evaluate *context-enriched* strategies, not just long/short regime signals.
- Walk-forward is essential — in-sample metrics are meaningless for a system claiming edge.
- The agent will eventually generate strategies autonomously. It needs a tool to evaluate them programmatically.

### What a RenTech-grade validation system actually needs
1. **Walk-forward engine** with configurable train/test splits (expanding, rolling, combinatorial purged cross-validation)
2. **Strategy protocol** — a simple interface: given data + signals, produce position weights. This lets any strategy run through the same validator.
3. **Comprehensive scoring** — risk-adjusted returns (Sharpe, Sortino, Calmar), tail risk (VaR, CVaR, max DD, DD duration), turnover, exposure metrics
4. **Statistical rigor** — bootstrap confidence intervals on Sharpe, permutation tests for signal validity, multiple-testing correction (Bonferroni/BH) when scanning many signals
5. **Signal diagnostics** — IC, IC-IR, factor exposure decomposition, signal decay analysis
6. **Agent integration** — a `BacktestTool` the agent can call to evaluate any strategy

### Scope control — what to build NOW vs. LATER
Phase 3 should build the **engine** — not every possible metric.

**NOW (Phase 3):**
- Walk-forward backtester (expanding window, at minimum)
- Strategy protocol (abstract base class)
- Extended scoring (Sortino, Calmar, VaR, CVaR, drawdown duration, turnover)
- Bootstrap confidence intervals on key metrics
- Regime-conditional performance breakdown (reusable, not ad-hoc)
- BacktestTool for agent integration
- Rerun Phase 2 liquidity-regime strategies through the new engine (proves it works)

**LATER (Phase 4+):**
- Combinatorial purged CV (López de Prado style)
- Signal decay analysis
- Factor exposure decomposition
- Multi-asset portfolio optimization
- Transaction cost modeling with realistic slippage

## Risks

1. **Over-engineering** — tempting to build a full backtesting platform. Stay focused on what the agent needs now.
2. **Look-ahead bias** — the walk-forward engine must enforce strict temporal separation. No future data leaking into training.
3. **Statistical traps** — bootstrap CIs on Sharpe are tricky because returns are autocorrelated. Need block bootstrap.
4. **Scope creep** — the scoring module could grow indefinitely. Phase 3 should produce a clean, extensible API, not every possible metric.

## Data Requirements

- Same data as Phase 2: FRED macro series (via MacroDataTool), market prices (via MarketDataTool)
- No new external data sources needed
- The framework operates on numpy arrays of returns + signals, so it's data-source agnostic

## Math/Algorithm Survey

### Walk-Forward Validation
Standard approach: split time series into sequential train/test blocks. Options:
- **Expanding window**: train on [0, t], test on [t, t+h]. t grows each fold.
- **Rolling window**: train on [t-w, t], test on [t, t+h]. Fixed window w.
- **Combinatorial purged CV** (López de Prado 2018): more sophisticated, deferred to Phase 4+.

For Phase 3, expanding window is the right starting point — it mirrors how a real system accumulates data over time.

### Extended Metrics

| Metric | Formula | Why |
|--------|---------|-----|
| Sortino ratio | $\frac{\mu - r_f}{\sigma_d}$ where $\sigma_d$ = downside deviation | Penalizes downside vol only, better for asymmetric strategies |
| Calmar ratio | $\frac{\text{Ann. return}}{|\text{Max DD}|}$ | Return per unit of worst-case drawdown |
| Value at Risk (VaR) | $F^{-1}(\alpha)$ at confidence level $\alpha$ | Tail risk threshold |
| Conditional VaR (CVaR) | $E[X | X \leq \text{VaR}(\alpha)]$ | Expected loss beyond VaR (coherent risk measure) |
| Drawdown duration | Max consecutive periods below previous peak | How long recoveries take |
| Turnover | $\frac{1}{T}\sum_t |w_t - w_{t-1}|$ | Trading cost proxy |

### Bootstrap Confidence Intervals
- **Block bootstrap** (Politis & Romano 1994): resample blocks of returns to preserve autocorrelation
- Generate B bootstrap samples → compute metric on each → take [α/2, 1-α/2] percentiles
- Block length: typically $T^{1/3}$ (cube root of sample size)

### Permutation Test for Strategy Significance
- Null hypothesis: strategy returns are not better than random
- Shuffle signal labels (preserve return sequence), recompute metric, repeat N times
- p-value = fraction of shuffled metrics ≥ observed metric

---

## Related

- [[scoring_validation_spec|Spec: Scoring Validation]]
- [[convergence_detection]]
- [[world_model]]
- [[backtest_performance]]

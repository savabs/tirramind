---
title: "Feature: RL Layer (Phase 4b)"
tags:
  - doc/research
  - topic/reinforcement-learning
---

# Feature: RL Layer (Phase 4b)

## Problem Statement

Phase 4 built the autonomy loop (reflect → generate → execute → evaluate → learn) but every decision is made by asking an LLM. Nothing learns. The LLM is a frozen model — it doesn't get better with experience. The "learning" is just a longer prompt.

Phase 4b replaces the decision-making core with a **Thompson Sampling multi-armed bandit** that makes the strategic choice ("what TYPE of thing to do next") based on observed rewards from past iterations. The LLM remains for filling in specifics ("WHICH exact goal within that type").

## Architecture

### Role Split

```
BEFORE (Phase 4):
  LLM decides everything → no learning

AFTER (Phase 4b):
  BANDIT decides: "what category of work" (learned from rewards)
      ↓
  LLM decides: "specific goal within category" (language generation)
      ↓
  ORCHESTRATOR: executes
      ↓
  REWARD: numeric score from evaluation (no LLM)
      ↓
  BANDIT: update(arm, reward) — actual parameter update
```

### Why Thompson Sampling

- **Simple.** Two numbers per arm (α, β). ~30 lines of core logic.
- **Principled exploration.** Arms with few observations have high variance → get explored. Arms with many observations converge → get exploited. No epsilon hyperparameter.
- **Converges fast.** 20-50 iterations is enough to identify good arms. Each iteration costs real time + API money, so sample efficiency matters.
- **Extensible.** Trivial to upgrade to contextual bandit (add state features) later without changing the interface.

### Action Space (Arms)

Five categories covering the agent's full capability set:

| Arm | Description | Primary Tools | Example Goal |
|-----|-------------|---------------|-------------|
| `backtest_strategy` | Test a new or modified strategy | backtest | "Backtest regime-avoid on TLT" |
| `tune_parameters` | Adjust existing strategy params | backtest | "Test avoid-crisis with HMM K=4 instead of K=3" |
| `explore_asset` | Regime detection on new asset | liquidity_regime, market_data | "Run regime detection on EUR/USD" |
| `fetch_macro_data` | Explore macro data series | macro_data | "Fetch and analyze ECBASSETSW trends" |
| `research_market` | Web research on conditions | web_search, web_browse | "Research latest Fed policy shift impact" |

Arms are configurable and extensible. New arms can be added without changing the bandit code.

### Reward Function

Pure numeric, no LLM. Combines:

1. **Evaluation score** (already 0-1 from the evaluator) — weighted 0.4
2. **Sharpe quality** — if backtest, normalize Sharpe to [0,1] — weighted 0.3
3. **Knowledge gain** — new_facts_count / 5, capped at 1.0 — weighted 0.2
4. **Novelty bonus** — small bonus for first time trying an arm — weighted 0.1
5. **Dead-end penalty** — subtract 0.3 if dead_end

Final reward clamped to [0, 1].

### Persistence

Bandit state (α, β per arm, pull counts, total reward) persists to `{memory_dir}/bandit_state.json`. Survives restarts. Accumulates across sessions — the agent genuinely improves over time.

## Observations

1. The bandit doesn't replace the LLM — it constraints it. "You will generate a goal of TYPE X" is a stronger prompt than "generate whatever you think is best."
2. Thompson Sampling is Bayesian — uncertainty drives exploration automatically. No tuning required.
3. The reward function has no LLM in the loop. Sharpe, facts count, dead-end — all extracted numerically. This is the critical difference from Phase 4.
4. Convergence speed depends on reward signal quality. The evaluator's `_extract_quant_metrics` already handles this for backtests. For non-backtest arms, the signal is weaker (fact counts) but still numeric.

## Risks

1. **Arm granularity.** If arms are too coarse ("do_stuff"), the bandit can't differentiate. If too fine ("backtest_SPY_regime_avoid_K3"), there are too many arms to converge. 5 categories is a reasonable starting point.
2. **Non-stationary rewards.** Market conditions change. An arm that paid off in volatile markets might not in calm ones. Solved later by contextual bandit (Phase 5).
3. **Cold start.** First 5-10 iterations are essentially random exploration. Acceptable — this is intrinsic to bandit algorithms.

---

## Related

- [[rl_layer_spec|Spec: Rl Layer]]

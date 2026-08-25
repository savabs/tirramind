---
title: "Research: Convergence Engine Backtest"
tags:
  - doc/research
  - layer/world-model
  - phase/7c
  - topic/backtest
  - topic/convergence
---

# Research: Convergence Engine Backtest

## Goal
Prove the convergence detection engine produces statistically significant predictive signals. Build a phase-gate backtest rule that applies after every major phase.

## Current Architecture

### Convergence Detector Pipeline
1. `ConvergenceDetector.detect(as_of)` reads from `PipelineStore.query_data(tool_name, since, until)`
2. Extracts `Evidence` via 49 per-tool extractors
3. Groups by `signal_id`, builds `SignalStream`, computes atomic anomaly scores
4. Smart pair selection → pairwise coincidence scoring (rolling-ρ, joint exceedance, concordance)
5. FDR controls (BH correction, Fisher combined, persistence, cross-category) → `ConvergenceClique`
6. Template matching against 50 causal chain templates → `DetectionResult`
7. Emits `ConvergenceSignal` with: value ∈ [0,1], event_type, direction (+1/-1/0), p-value, template_match

### Existing Backtest Infrastructure
- `WalkForward` (agent/quant/backtest.py): expanding-window backtester
- `Strategy` ABC: `generate_weights(train_returns, test_length, ...)` → weight array
- `score_returns()`: Sharpe, Sortino, Calmar, MDD, VaR, CVaR, drawdown_duration
- `block_bootstrap_ci()`: circular block bootstrap for confidence intervals
- `regime_conditional_analysis()`: per-regime return scoring
- `BuyAndHoldStrategy`, `RegimeAvoidStrategy`, `RegimeOnlyStrategy` built-in

### Data Availability
- yfinance: 20+ years daily OHLCV for major tickers (SPY, TLT, GLD, etc.)
- PipelineStore: `pipeline_data` table (tool outputs), `signals` table (computed signals)
- No historical pipeline data stored yet — system is not yet running live scheduled DAGs

## Core Problem

**The convergence engine has never run live.** There's no historical `pipeline_data` to replay. We cannot do a traditional historical backtest by replaying tool outputs because those outputs don't exist historically.

### The solutions, and why only one works now

| Approach | Feasibility | Why |
|----------|------------|-----|
| **A. Replay historical pipeline data** | Impossible | No historical DAG runs exist. Tools fetch live data. |
| **B. Back-fill tool outputs historically** | Partially possible but fragile | Most free APIs provide only current/recent data (BLS monthly, USASpending, etc.). yfinance has long history but it's Layer 3 (price). FRED series go back decades but are macro only. |
| **C. Synthetic evidence injection** | Fully feasible | Generate realistic multi-source evidence that embeds known causal chains. Run detector. Measure detection accuracy. This tests the *math*, not the data. |
| **D. Hybrid: macro series backtest** | Feasible | Use FRED series that DO have history (WALCL, M2SL, CFTC COT, PMI, yields, VIX, CPI, UNRATE, etc.) to backfill a subset of extractors. Run convergence on that subset. Measure against SPY/TLT/GLD returns. |
| **E. Forward-looking live validation** | Correct long-term answer | Run DAGs daily, accumulate evidence, test signal quality after 3-6 months of data. |

**Recommended: D (hybrid macro backtest) + C (synthetic validation)**

- **D** produces a real backtest result with real financial data, but uses only the ~15 extractors whose source data is historically available via FRED/yfinance.
- **C** validates the statistical machinery (FDR, templates, clique detection, scoring) against known ground truth.
- **E** should run in parallel once the system starts live collection.

## Approach D: Hybrid Macro Backtest — Detailed Design

### Historical Data Sources Available via FRED
| Signal Domain | FRED Series | Frequency | History |
|---|---|---|---|
| US Liquidity | WALCL, WTREGEN, RRPONTSYD | Weekly | 2003+ |
| Money Supply | M2SL | Monthly | 1959+ |
| Employment | UNRATE, ICSA, PAYEMS, JTSJOL | Monthly/Weekly | 1948+/1967+ |
| Inflation | CPIAUCSL, T10YIE, MICH | Monthly | 1947+/2003+ |
| Rates | DFF, DGS2, DGS10, DGS30 | Daily | 1962+/1990+ |
| Spreads | T10Y2Y (2-10 slope), BAMLH0A0HYM2 (HY OAS) | Daily | 1976+/1996+ |
| PMI proxy | NAPM (ISM Mfg PMI) | Monthly | 1948+ |
| Consumer | UMCSENT | Monthly | 1978+ |
| Housing | PERMIT, HOUST | Monthly | 1960+ |
| VIX | via yfinance (^VIX) | Daily | 1990+ |

### Targets (Layer 3 — what we predict)
| Ticker | Asset | Via |
|---|---|---|
| SPY | US equities | yfinance |
| TLT | US long bonds | yfinance |
| GLD | Gold | yfinance |
| DX-Y.NYB | Dollar index | yfinance |
| ^VIX | Volatility | yfinance |

### Architecture

```
┌─────────────────────────────────┐
│   HistoricalEvidenceBuilder     │
│                                 │
│  For each month t:              │
│    1. Fetch FRED series up to t │
│    2. Convert to Evidence via   │
│       simplified extractors     │
│    3. Store in PipelineStore    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   WalkForward loop              │
│                                 │
│  For each test window:          │
│    1. Run ConvergenceDetector   │
│       .detect(as_of=t)          │
│    2. Convert DetectionResults  │
│       to trading signal:        │
│       +1 = risk-on, -1 = risk-  │
│       off, 0 = neutral          │
│    3. Generate position weights │
│       for test period           │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Scoring                       │
│                                 │
│  score_returns() on:            │
│    - Convergence-long-SPY       │
│    - Convergence-avoid-SPY      │
│    - Benchmark: buy-and-hold    │
│  block_bootstrap_ci() for       │
│    statistical significance     │
└─────────────────────────────────┘
```

### Signal-to-Weight Translation Logic
The convergence detector outputs:
- `direction`: +1 (stress) or -1 (relief) or 0
- `boosted_score`: [0, 1] — strength of convergence
- `event_type`: template name

**Strategy 1 — Convergence Risk-Off:**
When stress convergence (direction=+1, score > threshold) detected → weight = 0 (exit).
Otherwise → weight = 1 (stay invested).
This is the simplest: does convergence == "get out" produce alpha?

**Strategy 2 — Convergence Directional:**
direction=+1 → weight = -score (short/reduce proportional to score)
direction=-1 → weight = +score (long/increase proportional to score)
direction=0 → weight = 0 (flat)

**Strategy 3 — Template-Weighted:**
Only act on high template_match (>0.5) detections. Different templates → different asset actions.

### What Constitutes "Proof"
| Metric | Threshold | Rationale |
|---|---|---|
| Sharpe vs buy-hold | > 0 (improvement over benchmark) | Basic sanity |
| Bootstrap CI for Sharpe | Lower bound > 0 at 95% | Statistical significance |
| Hit rate on direction | > 55% | Better than coin flip |
| Max drawdown reduction | > 10% relative to buy-hold | Risk management value |
| False positive rate | < 20% of signals | FDR is working |
| Detection lead time | Median > 5 trading days | Actionable, not stale |

If the engine FAILS these tests, that's valuable — it means the math needs tuning or the templates are wrong. We iterate, not ship.

## Approach C: Synthetic Validation — Detailed Design

### Purpose
Test the statistical machinery in isolation with known ground truth.

### Method
1. Generate N synthetic scenarios (N=100+) with known causal chains embedded
2. Each scenario: 3-7 signals, planted temporal ordering, known direction, known event type
3. Add noise signals (decoys) that should NOT trigger detections
4. Run full convergence detector on each scenario
5. Measure:
   - True positive rate (planted chains detected)
   - False positive rate (noise chains detected)
   - Template classification accuracy (correct event_type assigned)
   - Direction accuracy (correct +1/-1 assigned)
   - Temporal ordering accuracy (lead/lag correctly identified)

### Scoring
- Precision and recall on convergence detection
- Confusion matrix on template classification
- ROC curve varying z_threshold and p_threshold

## Phase-Gate Backtest Rule

### Proposed Workflow Addition
After every major phase that changes the computation stack (Phases 8-12):

```
Phase N implementation complete
        ↓
Run convergence backtest suite
        ↓
Compare metrics to previous phase baseline
        ↓
   ┌─── Metrics improved or neutral ───→ Phase N complete, save baseline
   │
   └─── Metrics degraded ───→ BLOCK: diagnose regression before Phase N+1
```

### What gets tested per phase gate:
1. **Synthetic validation** — precision/recall/F1 on planted scenarios (always runs, fast)
2. **Macro backtest** — Sharpe/MDD/hit rate on SPY/TLT/GLD (runs if data layer changed)
3. **Unit regression** — all existing convergence tests must pass (always runs, already mandatory)

### Baseline Storage
Store baseline metrics in `docs/baselines/convergence_backtest_baseline.json`:
```json
{
  "phase": "7c",
  "date": "2026-04-06",
  "synthetic": {
    "precision": 0.85,
    "recall": 0.80,
    "f1": 0.82,
    "template_accuracy": 0.75,
    "direction_accuracy": 0.90
  },
  "macro_backtest": {
    "spy_sharpe": 0.45,
    "spy_sharpe_ci_lower": 0.12,
    "tlt_sharpe": 0.30,
    "max_drawdown_reduction": 0.15,
    "hit_rate": 0.58,
    "false_positive_rate": 0.12
  }
}
```

## Risks
- FRED data has publication lag — must use point-in-time values (no look-ahead). FRED revisions are a risk; initial releases differ from final values. Mitigated by using `realtime_start`/`realtime_end` params in FRED API (vintage data).
- Only ~15/49 extractors have historical data → backtest covers partial evidence surface. This is OK — we're testing the *engine*, not every data source.
- Synthetic scenarios may not capture real-world complexity. Mitigated by using Approach D as ground truth.
- Overfitting risk if we tune detector params to maximize backtest Sharpe. Mitigated by WalkForward (expanding window, no future leak) + bootstrap CI.
- Monthly FRED data = low signal frequency → few independent observations per year. Bootstrap CI handles this but confidence intervals will be wide.

## Dependencies
- Existing: numpy, scipy, yfinance (installed), agent/quant/backtest.py, agent/quant/scoring.py
- New: FRED API access via `agent/tools/macro_data.py` (already exists, TIRRA_FRED_API_KEY or BLS fallback)
- No new packages needed

## References
- Walk-forward backtesting: de Prado, "Advances in Financial Machine Learning" (2018), Ch. 12 — combinatorial purged CV
- Block bootstrap: Politis & Romano (1994), "The Stationary Bootstrap" — autocorrelation-preserving resampling
- FDR validation: Benjamini & Hochberg (1995) — controlling false discovery rate in multiple testing
- Fisher's combined test: Fisher (1932), "Statistical Methods for Research Workers"
- Existing codebase: agent/quant/backtest.py (WalkForward), agent/quant/scoring.py (block_bootstrap_ci)

---

## Related

- [[convergence_backtest_spec|Spec: Convergence Backtest]]
- [[convergence_detection]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]

---
title: Liquidity Regime Detection — Results
tags:
  - doc/research
  - topic/liquidity
  - topic/regime
---

# Liquidity Regime Detection — Results

## Summary

Phase 2 of the quant training ground: build a global liquidity composite, detect
structural breaks (BOCPD) and regimes (HMM), validate against asset returns.

**Verdict:** The math layer works correctly. BOCPD finds real structural breaks.
HMM classifies variance regimes reliably. However, naive regime→return mapping
does not produce a tradable edge. The z-scored first-difference composite measures
*rate of change in policy*, not *absolute liquidity levels*, so extreme states
correspond to crisis dynamics (both up and down), not sustained bull/bear markets.

---

## Data Layer

### US Liquidity Composite
- **Formula:** `us_net = WALCL - WTREGEN - RRPONTSYD`
- **Detrending:** First difference → 52-week rolling z-score
- **Coverage:** 836 weekly obs, 2008-12-31 to 2025-01-01
- **Properties:** mean ≈ 0, std ≈ 1, COVID peak z = +3.67

### Global Composite (3-CB)
- **Central banks:** Fed + ECB + BOJ (BOE excluded — no usable FRED series)
- **FX conversion:** ECB (EUR→USD), BOJ (100M JPY→USD)
- **Correlation with US-only:** ρ = 0.517 (expected: >0.5, <1.0) ✅

---

## BOCPD Results (λ=200, recommended)

| Date       | Event                                    |
|------------|------------------------------------------|
| 2014-01-01 | Taper tantrum aftermath / QE3 wind-down  |
| 2020-03-18 | COVID liquidity flood                    |
| 2023-03-15 | SVB crisis / QT acceleration             |

Stable across λ ∈ [50, 100, 200, 400]. Additional CPs at shorter λ:
2021-03 (QE taper signal), 2024-04 (recent shift).

---

## HMM Results (K=3, chosen by BIC)

| Metric    | K=2     | K=3     |
|-----------|---------|---------|
| BIC       | 2330.3  | 2323.4  |

### Train-period regime statistics (2009-2020)

| State | Label       | Mean   | Variance | Weeks | % of Train |
|-------|-------------|--------|----------|-------|------------|
| 0     | Low-vol     | -0.055 | 0.456    | ~400  | ~64%       |
| 1     | High-vol    | 0.007  | 1.099    | ~220  | ~35%       |
| 2     | Extreme     | 1.365  | 7.804    | ~7    | ~1%        |

**Interpretation:** States split by *variance*, not by level. State 2 = crisis
events (COVID). States 0 and 1 = calm vs active policy periods.

### Transition matrix (train)
```
        To S0    To S1    To S2
S0    [ 0.962    0.032    0.005 ]
S1    [ 0.023    0.971    0.006 ]
S2    [ 0.002    0.278    0.720 ]
```

High diagonal → regimes are sticky (mean duration ~26 weeks for S0, ~34 for S1).

---

## Walk-Forward Backtest (train: 2009-2020, test: 2021-2024)

### Test period regime distribution
- State 0: 53 weeks (25.4%)
- State 1: 155 weeks (74.2%)
- State 2: 1 week (0.5%)

### Strategy Performance

| Strategy           | Ann. Return | Ann. Vol | Sharpe | Max DD  | IR vs B&H | Total Return |
|--------------------|-------------|----------|--------|---------|-----------|--------------|
| Buy-and-Hold SPY   | 12.73%      | 16.29%   | 0.782  | -24.46% | —         | +66.82%      |
| Avoid Crisis (s≠2) | 13.34%      | 16.23%   | 0.822  | -24.46% | 0.500     | +70.95%      |
| Only Neutral (s=1) | 5.81%       | 15.18%   | 0.383  | -24.46% | -1.145    | +26.30%      |

### Key Finding
The "contraction" state (state 0, low liquidity z-score) had the *best* SPY
returns in the test period (Sharpe 2.63). This is counterintuitive but explained
by the composite design: negative z-score = slowing liquidity *growth rate*, which
often occurs during healthy markets where the Fed isn't actively injecting.

**Conclusion:** Z-scored rate-of-change doesn't map to bull/bear. The regime
tool's value is *awareness* (knowing what policy phase you're in), not direct
long/short signals.

---

## Spectral Analysis

- **FFT:** Dominant cycle at ~1 cycle/year (annual budget/QE cycle)
- **CWT:** Confirms annual periodicity with time-localized power

---

## Diagnostic Plots

- `docs/research/us_composite_vs_spy.png` — composite overlaid with SPY price
- `docs/research/bocpd_liquidity.png` — BOCPD across 4 hazard lambdas
- `docs/research/hmm_regimes_liquidity.png` — 3-state regime coloring
- `docs/research/walkforward_backtest.png` — walk-forward cumulative returns

---

## Files Created

| File                          | Purpose                              |
|-------------------------------|--------------------------------------|
| `agent/quant/__init__.py`     | Package init                         |
| `agent/quant/liquidity.py`    | Liquidity composite (US + global)    |
| `agent/quant/changepoint.py`  | BOCPD changepoint detection          |
| `agent/quant/regime.py`       | HMM regime classification            |
| `agent/quant/spectral.py`     | FFT + CWT spectral analysis          |
| `agent/quant/scoring.py`      | Sharpe, max DD, IR, hit rate         |
| `agent/tools/liquidity_regime.py` | Agent tool wrapper (regime query) |

---

## Related

- [[convergence_detection]]
- [[world_model]]
- [[backtest_performance]]

---
title: "Research: Global Liquidity Regime Detection"
tags:
  - doc/research
  - topic/liquidity
  - topic/regime
---

# Research: Global Liquidity Regime Detection

Phase 2 of TirraMind Quant Training Ground.

---

## 1. What We're Building

A system that:
1. Constructs a **global liquidity composite** from central bank balance sheet data
2. Detects **regime changes** in that composite using real math (BOCPD, HMM, spectral)
3. **Validates** that detected regimes predict asset class returns out-of-sample

This is the first real intelligence layer — the foundation that contextualizes everything built on top of it.

---

## 2. Data Series Survey

### 2.1 US Fed Series (all on FRED, existing MacroDataTool)

| Series ID | Name | Frequency | Units | Lag | Notes |
|-----------|------|-----------|-------|-----|-------|
| **WALCL** | Fed total assets | Weekly (Wed) | Millions USD | ~2 days | The balance sheet. This IS the money printer |
| **WTREGEN** | Treasury General Account | Weekly (Wed) | Millions USD | ~2 days | Cash the Treasury holds at the Fed. When TGA rises, it drains liquidity from markets |
| **RRPONTSYD** | Overnight Reverse Repo | Daily | Billions USD | 1 day | Cash parked back at the Fed by money market funds. When high, liquidity is trapped |
| **M2SL** | M2 Money Supply | Monthly | Billions USD | ~3 weeks | Broad money. Slower-moving but fundamental |
| **DFF** | Fed Funds Rate | Daily | Percent | 1 day | The price of reserves. Useful as context, not a direct composite input |

**US Net Liquidity formula:** `WALCL - WTREGEN - RRPONTSYD`

This is the standard approximation. WALCL is total liquidity injected. TGA and RRP are liquidity that's been injected but isn't actually circulating in markets. The difference is what's actually "out there" driving asset prices.

**RRPONTSYD unit mismatch:** WALCL and WTREGEN are in millions, RRPONTSYD is in billions. Must multiply RRPONTSYD by 1000 before subtracting.

### 2.2 Non-US Central Banks (all on FRED — no new data tools needed)

| Series ID | Central Bank | Frequency | Units | History Start | Notes |
|-----------|-------------|-----------|-------|--------------|-------|
| **ECBASSETSW** | ECB (Euro Area) | Weekly | Millions EUR | 1999-01 | Verified on FRED. Weekly updates, ~4 day lag |
| **JPNASSETS** | Bank of Japan | Monthly | 100 Million JPY | 1998-04 | Verified. Monthly only — forward-fill to weekly |
| **BOEBSTAUKA** | Bank of England | Monthly | Millions GBP | Available | Referenced on FRED. Need to verify series start/frequency |

**PBOC (China):** No reliable free series on FRED. Chinese central bank data is opaque, delayed, and often revised. **Decision: exclude PBOC for Phase 2.** Note as a known gap. This is honest — bad data is worse than no data. Add when a reliable source is identified.

**Other central banks (SNB, RBA, BOK, Riksbank):** Small relative to the Big 4 (Fed, ECB, BOJ, BOE). Exclude for Phase 2 — diminishing returns. The Big 4 cover ~80% of global central bank assets.

### 2.3 FX Series for Currency Normalization

To sum balance sheets across currencies, we need USD conversion:
- **EURUSD=X** — Euro to USD (yfinance, daily)
- **JPYUSD** or **USDJPY=X** — Yen to USD (yfinance, daily, invert)
- **GBPUSD=X** — Pound to USD (yfinance, daily)

MarketDataTool already fetches these. Align to weekly (take Wednesday close to match WALCL).

### 2.4 Frequency Alignment Strategy

All series must be on the same calendar. Weekly (Wednesday) is the natural choice:

| Source frequency | Alignment method |
|-----------------|-----------------|
| Daily (RRPONTSYD, FX) | Resample: take Wednesday value (or last business day before Wed) |
| Weekly (WALCL, WTREGEN, ECBASSETSW) | Already aligned (Wed release) |
| Monthly (JPNASSETS, M2SL, BOEBSTAUKA) | Forward-fill: last known monthly value carried until next release |

**The monthly → weekly forward-fill is a real limitation.** BOJ and M2 data will be "stale" for up to 3 weeks between releases. This means: for the monthly series, the composite only updates when a new monthly release drops. This is a known information lag, not a bug — it reflects reality. But it means the composite's week-to-week changes are dominated by the weekly US series.

### 2.5 Historical Overlap

For all four central banks to contribute, we need data since at least 1999 (ECB start). But the most interesting period for regime detection is 2008+ (QE era). The meaningful sample:
- **Training:** 2008-01 to 2020-12 (12 years, captures QE1/2/3, taper tantrum, COVID QE)
- **Test:** 2021-01 to present (captures 2022 QT, 2023-24 slowdown/pivot)

This gives ~625 weekly observations for training, ~220 for test. Enough for HMM with 3 states. Tight for spectral methods (want long time series for low-frequency resolution).

---

## 3. Composite Construction

### 3.1 Formula

```
US_net = WALCL - WTREGEN - (RRPONTSYD * 1000)    [all in millions USD]
ECB_usd = ECBASSETSW * EURUSD                     [EUR millions → USD millions]
BOJ_usd = (JPNASSETS * 100) / USDJPY              [100M JPY → JPY → USD: multiply by 100M, divide by USDJPY]
BOE_usd = BOEBSTAUKA * GBPUSD                     [GBP millions → USD millions]

Global_composite = US_net + ECB_usd + BOJ_usd + BOE_usd
```

### 3.2 Normalization & Detrending

The raw composite is a massive number (~$25 trillion) with a secular uptrend (central banks have been expanding for decades). For regime detection, we care about **deviations from trend**, not the level itself.

Two approaches:
1. **Z-score of rate-of-change:** Compute week-over-week % change, then z-score (subtract mean, divide by std over a rolling window). This gives a stationary signal centered at 0. Problem: sensitive to the rolling window length.
2. **HP filter / bandpass:** Separate the trend from the cyclical component. Use the cycle. Problem: HP filter has well-known end-point bias (the trend estimate at the end of the sample is unreliable — exactly where you need it most for live trading).
3. **First difference then z-score:** Simple, no lookahead bias, stationary. Loses level information but that's OK — regime detection is about *changes in the rate of liquidity injection*, not the absolute level.

**Decision: Use approach 3 (first difference → z-score) as the primary signal.** It's clean, has no lookahead bias, and is online-computable. Add approach 1 as a secondary view. Avoid HP filter — the end-point bias is a dealbreaker for live use.

### 3.3 What the Composite Represents

When the composite is **rising** (positive z-score of ΔLiquidity):
- Central banks are net-expanding balance sheets
- Money is flowing into the financial system
- Risk assets should rally (equities, crypto, commodities)
- Bonds should be stable or rising (yields falling)
- Dollar should weaken (more USD supply)

When the composite is **falling** (negative z-score of ΔLiquidity):
- Central banks are net-contracting (QT) or growing slower than trend
- Money is being drained from the financial system
- Risk assets should sell off
- Flight to quality (bonds rally on fear, then sell off on higher-for-longer)
- Dollar should strengthen

The regime detector's job is to classify the current state and detect transitions between these states — ideally BEFORE asset prices fully reflect them.

---

## 4. Math Primitives Survey

### 4.1 BOCPD — Bayesian Online Changepoint Detection

**Paper:** Adams & MacKay (2007), "Bayesian Online Changepoint Detection"

**What it does:** At each timestep, maintains a posterior distribution over "run lengths" — how many observations since the last changepoint. When a new observation arrives, two things happen: (a) the existing run continues, or (b) a changepoint occurred and a new run starts. The algorithm computes the posterior probability of each possibility.

**Why it's right here:** Online (can process streaming data), probabilistic (gives confidence, not just binary "yes/no"), doesn't require pre-specifying the number of regimes.

**Mathematical core:**
- Hazard function `H(τ)`: prior probability that the current run ends at length τ. Simplest choice: constant hazard `H = 1/λ` where λ is expected run length.
- Observation model: probability of current observation given run length. Use Normal-Inverse-Gamma conjugate prior (handles unknown mean AND variance).
- Message passing: run length posterior `P(r_t | x_{1:t})` updated at each step via Bayes' rule.

**Implementation plan:** Pure numpy. ~100-150 lines. Key functions:
- `__init__(hazard_lambda, prior_mu, prior_kappa, prior_alpha, prior_beta)` — set priors
- `update(x)` → returns `P(changepoint_at_t)` and full run-length posterior
- `get_changepoints(threshold)` → return times where changepoint probability exceeds threshold

**Gotchas:**
- The run-length posterior matrix grows as O(T²) in memory. For 1000 observations it's ~8MB (fine). For 10000 it's 800MB (problem). Need to truncate old run lengths or use log-space tricks.
- Hazard rate λ is the most important hyperparameter. Too small → too many changepoints. Too large → misses real ones. May need to tune on synthetic data.
- The Normal-Inverse-Gamma conjugate makes life easy but assumes Gaussian observations. Our liquidity z-scores should be approximately Gaussian, so this is OK.

**Libraries:** None needed for BOCPD. The `ruptures` library has offline changepoint detection (PELT, binary segmentation) but NOT online BOCPD. We implement from scratch.

### 4.2 HMM — Hidden Markov Model (Gaussian Emission)

**What it does:** Models the liquidity composite as being generated by K hidden states (regimes), each with its own mean and variance. The system transitions between states according to a Markov chain with transition matrix A.

**Why it's right here:** Natural model for "expansion / contraction / transition" (K=3). The transition matrix gives expected regime durations. Can do Viterbi decoding (most likely state sequence) and forward-backward (posterior probability of each state at each time).

**Mathematical core:**
- Emission model: `P(x_t | z_t = k) = N(μ_k, σ²_k)` — Gaussian with state-specific parameters
- Transition model: `P(z_t = j | z_{t-1} = i) = A[i,j]`
- Parameter estimation: Baum-Welch (EM algorithm). E-step: forward-backward to get state posteriors. M-step: update μ, σ², A from posteriors.
- Decoding: Viterbi algorithm for most likely state sequence.

**Implementation decision: Use `hmmlearn` library.**

Reasons:
- Baum-Welch has well-known numerical issues (underflow in long sequences). hmmlearn handles this with log-space computation.
- We don't gain insight by reimplementing EM — the math is standard. The INSIGHT is in what we feed it and how we interpret the output.
- hmmlearn is small (~2MB), well-tested, pure Python/Cython.
- If we later need custom emission distributions (Student-t, mixture, etc.), we can subclass or implement.

**Gotchas:**
- **Choosing K:** Start with K=3 (expansion, contraction, transition). Also try K=2 (risk-on/risk-off) and K=4. Compare via BIC or log-likelihood on held-out data.
- **EM local optima:** Run multiple random initializations (n_init=10), take best log-likelihood.
- **Label switching:** HMM states are arbitrary — "state 1" in one run might be "state 2" in another. Post-hoc: label states by their mean (highest mean = expansion, lowest = contraction).
- **Regime persistence:** HMMs can chatter (switch states rapidly) if the signal is noisy. The transition matrix should have high diagonal (P(stay in state) > P(switch)). If not, the signal needs more smoothing.

**Dependency:** `hmmlearn>=0.3.0` — add to pyproject.toml.

### 4.3 Spectral Decomposition

**What it does:** Decomposes the liquidity composite into its frequency components.

**Two methods:**
1. **FFT (Fast Fourier Transform):** Gives global power spectrum — which frequencies (cycle lengths) dominate the signal. Quick sanity check: does the composite have a ~4-year QE cycle? A ~1-year seasonal? A ~6-week technical cycle?
2. **CWT (Continuous Wavelet Transform):** Gives time-frequency decomposition — not just WHAT frequencies, but WHEN they're active. This is critical because liquidity cycles are nonstationary — the QE cycle existed 2008-2014, disappeared briefly, then returned in 2020.

**Implementation: Use scipy.**
- `scipy.fft.rfft` / `scipy.fft.rfftfreq` for FFT
- `scipy.signal.cwt` with Morlet wavelet for time-frequency (or `scipy.signal.morlet2`)

**Why spectral matters for regime detection:**
- BOCPD and HMM operate on the raw signal. Spectral decomposition can PRE-FILTER the signal to isolate the timescale we care about.
- If we only care about macro regimes (months-to-years), we can bandpass filter to remove weekly noise and intra-month jitter before feeding to HMM. This dramatically improves HMM stability.
- The scalogram itself is a diagnostic: when specific frequency bands gain/lose power, that IS a regime change visible from a different angle.

**Gotchas:**
- FFT assumes stationarity → only useful for gross characterization, not regime detection directly.
- CWT is slow for long time series (O(N × num_scales)). For 1000 observations and 50 scales, it's fine.
- Wavelet choice matters. Morlet is standard for financial time-frequency analysis.

**Dependency:** `scipy>=1.11` — add to pyproject.toml.

### 4.4 Dependency Summary

New pyproject.toml additions needed:
```
scipy >= 1.11
hmmlearn >= 0.3
matplotlib >= 3.7    # for validation plots
```

Already available (transitive from yfinance):
- numpy 2.4.3
- pandas 3.0.1

---

## 5. Architecture

### 5.1 New Package: `agent/quant/`

Pure math, no tool/agent dependencies. Testable independently.

```
agent/quant/
    __init__.py
    liquidity.py       # LiquidityComposite: fetch, align, normalize, compute
    changepoint.py     # BOCPD implementation
    regime.py          # HMM wrapper (uses hmmlearn)
    spectral.py        # FFT + CWT utilities
    scoring.py         # Sharpe, drawdown, information ratio, hit rate
```

### 5.2 Design Principles

- **Each module has ONE job.** `liquidity.py` constructs the composite. `changepoint.py` detects changepoints. `regime.py` classifies regimes. They compose, they don't inherit.
- **No tool dependencies inside quant/.** These modules take numpy arrays and return numpy arrays. The tool layer (`agent/tools/`) handles fetching and formatting.
- **Each module is testable with synthetic data.** Before running on real liquidity data, every algorithm is verified on generated data with known ground truth.

### 5.3 Data Flow

```
MacroDataTool.fetch(WALCL, WTREGEN, ...) → raw series dict
    ↓
LiquidityComposite.from_series(raw_dict, fx_dict) → aligned DataFrame + composite array
    ↓
BOCPD.fit(composite) → changepoint probabilities
HMM.fit(composite) → regime labels + transition matrix
Spectral.analyze(composite) → power spectrum + scalogram
    ↓
Scoring.evaluate(regime_labels, asset_returns) → Sharpe, drawdown, hit rate
```

### 5.4 Tool Integration (step 2.25, later)

Thin wrapper: `LiquidityRegimeTool(Tool)` registered in CLI. Agent asks "what regime are we in?" → tool fetches latest data, runs composite + HMM, returns current regime label + confidence + last changepoint date.

---

## 6. Validation Plan

### 6.1 Benchmark Assets

| Ticker | Asset | Why |
|--------|-------|-----|
| SPY | US large cap equity | The canonical risk asset |
| TLT | US long-term Treasuries | Flight to quality / duration play |
| GLD | Gold | Inflation hedge, liquidity beneficiary |
| BTC-USD | Bitcoin | Pure liquidity proxy (no earnings, no dividends — just flows) |
| DX-Y.NYB | Dollar index | USD strength inversely related to liquidity |

All available via MarketDataTool (yfinance).

### 6.2 Regime-Conditional Analysis

For each HMM regime k:
- Mean weekly return of each benchmark
- Volatility (annualized std)
- Sharpe ratio (annualized return / annualized vol)
- Cross-asset correlation matrix (does it change across regimes?)

**What "success" looks like:**
- Expansion regime → positive mean returns for SPY, BTC, GLD; weaker dollar
- Contraction regime → negative mean returns for SPY, BTC; stronger dollar
- The DIFFERENCE in mean returns across regimes is statistically significant (t-test or bootstrap)
- Transition regime (if K=3) → high volatility, low/uncertain mean returns

### 6.3 Walk-Forward Protocol

1. Split data: train = 2008-01 to 2020-12, test = 2021-01 to present
2. Fit HMM on train data only
3. For each week in test: classify regime using trained HMM (filter, not smoother — no future data)
4. Compute regime-conditional returns on test data
5. Strategy: long SPY when expansion, flat (cash) when contraction/transition
6. Report: Sharpe, max drawdown, hit rate, total return vs. buy-and-hold SPY

**Critical: No peeking.** HMM parameters (μ, σ, A) are fixed at training-end values for the entire test period. If we refit monthly, we note it separately as "adaptive walk-forward" vs. "frozen walk-forward."

### 6.4 Overfitting Guards

- Compare K=2, K=3, K=4 — if K=4 is only marginally better than K=3 in-sample but worse out-of-sample, K=3 wins.
- Compare US-only composite vs. global composite — does adding ECB/BOJ/BOE actually improve out-of-sample prediction?
- Null hypothesis test: shuffle regime labels randomly, recompute conditional returns. Does the real labeling produce significantly better separation than random?

---

## 7. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Monthly series (BOJ, M2) dominate via stale forward-fill | False regime stationarity during weeks between releases | Weight weekly series higher; track "freshness" per observation |
| FX conversion adds noise | Currency moves contaminate the liquidity signal | Try both: USD-converted sum vs. z-score-then-sum (normalize each CB to its own std first, then sum z-scores) |
| Sample size too small for robust HMM | ~625 weekly training obs with K=3 gives ~200 obs per state (borderline) | Start with K=2 (more data per state); compare BIC across K |
| BOCPD hazard rate sensitivity | Different λ gives very different changepoints | Run with λ ∈ {50, 100, 200, 400} (expected weeks between changes), compare |
| HMM chattering on noisy signal | Rapid state switching makes the regime label unusable | Smooth composite before feeding to HMM (4-week moving average) |
| 2020 COVID is an extreme outlier | One event dominates the entire fit | Try with/without 2020-03 to 2020-06 as a robustness check |

---

## 8. Open Questions for Spec Phase

1. **Z-score window length for detrending:** 52 weeks (1 year)? 104 weeks (2 years)? Or expanding window?
2. **Smooth before HMM?** 4-week MA removes noise but delays detection. Or: feed raw data to HMM and let the emission model handle noise via σ?
3. **Alternative composites:** Instead of sum of USD-converted levels, try: sum of z-scored individual series (each normalized to its own mean/std). This avoids FX noise and gives each central bank equal weight regardless of absolute size.
4. **DFF / SOFR as additional inputs:** The price of money (interest rate) is a separate liquidity dimension from the quantity (balance sheet). Consider multivariate HMM: [composite_level, rate_level] as 2D observation. Phase 2 or save for later?

---

## 9. Summary

**What we know:**
- All 4 central bank balance sheet series are on FRED — no new data infrastructure needed
- FX for normalization available via yfinance — no new tools needed
- BOCPD: implement from scratch (~150 lines numpy)
- HMM: use hmmlearn (well-tested, handles numerical issues)
- Spectral: use scipy.fft + scipy.signal.cwt
- New deps: scipy, hmmlearn, matplotlib
- Architecture: `agent/quant/` package, pure math, composed with existing tools

**What we don't know yet (determines spec decisions):**
- Best detrending approach (decided by testing on real data in step 2.8-2.9)
- Optimal K for HMM (decided by BIC comparison in step 2.19)
- Whether global composite outperforms US-only (decided by validation in step 2.22)

**These unknowns are resolved by implementation steps, not by more research.** The research phase is done. Next: spec.

---

## Related

- [[liquidity_regime_detection_spec|Spec: Liquidity Regime Detection]]
- [[convergence_detection]]
- [[world_model]]
- [[backtest_performance]]

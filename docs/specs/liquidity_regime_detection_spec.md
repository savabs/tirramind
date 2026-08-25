---
title: "Spec: Global Liquidity Regime Detection"
tags:
  - doc/spec
  - topic/liquidity
  - topic/regime
---

# Spec: Global Liquidity Regime Detection

Research: `[[liquidity_regime_detection]]`
Task: `[[quant_training_ground]]` — Phase 2

---

## Goal

Build a global liquidity composite from central bank balance sheet data, detect regime changes with BOCPD and HMM, validate that detected regimes predict asset class returns out-of-sample.

---

## Files Affected

### New files

| File | Purpose |
|------|---------|
| `agent/quant/__init__.py` | Empty package init |
| `agent/quant/liquidity.py` | LiquidityComposite class — fetch, align, normalize, compute |
| `agent/quant/changepoint.py` | BOCPD (Bayesian Online Changepoint Detection) |
| `agent/quant/regime.py` | Gaussian HMM wrapper around hmmlearn |
| `agent/quant/spectral.py` | FFT power spectrum + CWT scalogram |
| `agent/quant/scoring.py` | Sharpe, max drawdown, information ratio, hit rate |

### Modified files

| File | Change |
|------|--------|
| `pyproject.toml` | Add scipy>=1.11, hmmlearn>=0.3, matplotlib>=3.7 |
| `agent/cli.py` | Register LiquidityRegimeTool (step 2.25) |

---

## Implementation Steps

Ordered. Each step: one change, one test, one thing proved.

### Infrastructure (steps 2.4-2.5)

**2.4: Create `agent/quant/__init__.py`**
- Empty file. Just establishes the package.
- Test: `python -c "import agent.quant"` succeeds.

**2.5: Create `agent/quant/liquidity.py` — skeleton**
- Class `LiquidityComposite` with:
  - `__init__(self, macro_tool: MacroDataTool, market_tool: MarketDataTool)`
  - `fetch_us(self, start: str, end: str) -> pd.DataFrame` — stub, returns empty
  - `fetch_global(self, start: str, end: str) -> pd.DataFrame` — stub, returns empty
  - `compute(self, raw: pd.DataFrame) -> pd.Series` — stub, returns empty
- Test: class instantiates, methods return empty results without error.

### Data Layer (steps 2.6-2.11)

**2.6: Implement `fetch_us()`**
- Fetch WALCL, WTREGEN, RRPONTSYD, M2SL via MacroDataTool/FRED
- Parse into pd.DataFrame with DatetimeIndex
- Handle: RRPONTSYD in billions → multiply by 1000 to match millions
- Return DataFrame with columns: `walcl`, `wtregen`, `rrp`, `m2`
- Test: call with `start="2020-01-01"`, verify all 4 columns have data, correct units.

**2.7: Test `fetch_us()` data quality**
- Verify no NaN gaps longer than 2 weeks for weekly series
- Verify date ranges are correct
- Verify WALCL values are in the right order of magnitude (~7-9 trillion millions for recent dates)
- Print summary stats (mean, min, max, count) for sanity

**2.8: Implement `compute()` — US net liquidity**
- Align all series to weekly (Wednesday): resample daily → weekly, forward-fill monthly
- Compute: `us_net = walcl - wtregen - rrp`
- Detrend: first difference → rolling z-score (52-week window)
- Return pd.Series of z-scored ΔLiquidity
- Test: plot composite vs SPY (visual sanity). Verify: z-scores centered near 0, std near 1.

**2.9: Visual validation of US composite**
- Fetch SPY daily via MarketDataTool, resample to weekly
- Plot on dual axis: composite z-score (left) vs SPY price (right)
- Visual check: 2020-03 liquidity collapse aligns with equity crash, 2020-04+ liquidity surge aligns with rally, 2022 contraction aligns with bear market
- Save plot to `docs/research/us_composite_vs_spy.png`
- This step produces no code change — it's validation of 2.8.

**2.10: Implement `fetch_global()` — add ECB, BOJ, BOE**
- Fetch ECBASSETSW, JPNASSETS, BOEBSTAUKA via MacroDataTool
- Fetch EURUSD=X, USDJPY=X, GBPUSD=X via MarketDataTool
- Convert to USD millions:
  - ECB: `ecb_eur * eurusd_rate`
  - BOJ: `(boj_100m_jpy * 100_000_000) / usdjpy_rate / 1_000_000` → millions USD
  - BOE: `boe_gbp * gbpusd_rate`
- Merge with US data into single DataFrame
- Test: all columns present, USD values in right magnitude (ECB ~$7T, BOJ ~$4T, BOE ~$1T)

**2.11: Implement global composite**
- Update `compute()` to accept flag `global_=True`
- Global formula: `us_net + ecb_usd + boj_usd + boe_usd`
- Same detrending: first difference → rolling z-score (52-week window)
- Test: compare US-only composite vs global composite. They should correlate but not be identical. Global should be smoother (more data averaging out noise).

### Math Layer (steps 2.12-2.19)

**2.12: Implement BOCPD in `agent/quant/changepoint.py`**
- Class `BOCPD`:
  - `__init__(self, hazard_lambda: float = 200, prior_mu: float = 0, prior_kappa: float = 1, prior_alpha: float = 1, prior_beta: float = 1)`
  - `fit(self, data: np.ndarray) -> BOCPDResult`
  - `BOCPDResult`: dataclass with `run_length_posterior: np.ndarray`, `changepoint_probs: np.ndarray`, `changepoints(threshold: float) -> list[int]`
- Algorithm: Adams & MacKay 2007 message-passing
  - Observation model: Normal with conjugate Normal-Inverse-Gamma prior
  - Hazard: constant `1/hazard_lambda`
  - Run length posterior updated via Bayes' rule at each timestep
  - Truncate run lengths at max 2*hazard_lambda to bound memory
- Pure numpy, no external dependencies

**2.13: Test BOCPD on synthetic data**
- Generate: 500 obs from N(0,1), then 500 obs from N(3,1) (one changepoint at t=500)
- Run BOCPD with hazard_lambda=200
- Verify: changepoint probability peaks near t=500 (within ±10)
- Second test: 3 segments of 200 obs each with means [0, 3, -2]. Verify 2 changepoints detected.
- Third test: pure noise N(0,1) for 1000 obs. Verify: no changepoints detected (all probs < threshold).

**2.14: Implement HMM in `agent/quant/regime.py`**
- Class `RegimeHMM`:
  - `__init__(self, n_states: int = 3, n_init: int = 10, max_iter: int = 100)`
  - `fit(self, data: np.ndarray) -> RegimeResult`
  - `predict(self, data: np.ndarray) -> np.ndarray` — classify new data with frozen params
  - `RegimeResult`: dataclass with `states: np.ndarray`, `means: np.ndarray`, `variances: np.ndarray`, `transition_matrix: np.ndarray`, `log_likelihood: float`
- Uses `hmmlearn.hmm.GaussianHMM` internally
- Post-fit: relabel states by ascending mean (state 0 = lowest mean = contraction)
- Handle: data must be 2D for hmmlearn → reshape

**2.15: Test HMM on synthetic data**
- Generate 3-state switching series: 300 obs per state, means [−2, 0, 3], variance 0.5
- Fit with n_states=3
- Verify: recovered means within ±0.5 of true values
- Verify: state sequence matches ground truth for >90% of observations
- Verify: transition matrix has high diagonal (persistence)

**2.16: Implement spectral analysis in `agent/quant/spectral.py`**
- Functions (not a class — these are stateless utilities):
  - `power_spectrum(data: np.ndarray, sampling_freq: float = 52.0) -> tuple[np.ndarray, np.ndarray]` — returns (frequencies_in_cycles_per_year, power)
  - `scalogram(data: np.ndarray, periods: np.ndarray | None = None, sampling_freq: float = 52.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — returns (times, periods, power_matrix)
- `sampling_freq=52.0` because our data is weekly (52 weeks/year)
- Uses scipy.fft for power spectrum, scipy.signal.cwt + morlet2 for scalogram

**2.17: Test spectral on synthetic data**
- Generate: sum of two sinusoids at periods 52 weeks (1 year) and 13 weeks (1 quarter) + noise
- FFT test: verify power spectrum peaks at 1/year and 4/year
- CWT test: verify scalogram shows both frequencies, localized correctly

**2.18: Apply BOCPD to real liquidity composite**
- Run BOCPD on US composite (from step 2.8) with hazard_lambda values [50, 100, 200, 400]
- Plot: composite timeseries with vertical lines at detected changepoints
- Inspect: do changepoints correspond to known events? (2008 crisis, 2013 taper tantrum, 2020-03 COVID, 2022-06 QT acceleration)
- Save diagnostic plots to `docs/research/`
- No code change to BOCPD — this is application and tuning.

**2.19: Apply HMM to real liquidity composite**
- Fit HMM with K=2 and K=3 on US composite
- Compare log-likelihood and BIC: `BIC = -2*LL + K*log(N)` where K = number of free parameters
- For chosen K: inspect regime labels. Plot: composite colored by regime.
- Inspect: expansion regime should correspond to QE periods, contraction to QT/tightening.
- Document: chosen K, regime means, regime durations, transition matrix in research notes.

### Validation Layer (steps 2.20-2.26)

**2.20: Fetch benchmark returns**
- Fetch SPY, TLT, GLD, BTC-USD, DX-Y.NYB via MarketDataTool
- Compute weekly log returns: `log(price_t / price_{t-1})`
- Align to composite dates (inner join on DatetimeIndex)
- Result: DataFrame with columns per asset, rows per week, aligned to liquidity data

**2.21: Regime-conditional return analysis**
- For each regime label (from step 2.19):
  - Mean weekly return per asset
  - Annualized volatility per asset
  - Annualized Sharpe per asset
- Present as table. Check: expansion regime positive Sharpe for risk assets? Contraction regime negative?
- Statistical test: t-test on mean returns between expansion vs contraction. Is p < 0.05?

**2.22: Walk-forward backtest**
- Train: fit HMM on 2008-01 to 2020-12
- Test: for each week 2021-01 to present:
  - Classify regime using trained HMM (predict, not fit)
  - Record regime label
- Compute regime-conditional returns on test set only
- Compare: test-set Sharpe of regime strategy vs. buy-and-hold SPY

**2.23: Implement `agent/quant/scoring.py`**
- Functions:
  - `sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 52) -> float`
  - `max_drawdown(returns: np.ndarray) -> float` — from cumulative return series
  - `information_ratio(returns: np.ndarray, benchmark: np.ndarray, periods_per_year: int = 52) -> float`
  - `hit_rate(predictions: np.ndarray, actuals: np.ndarray) -> float` — % directionally correct
- Test: known inputs with hand-calculated outputs

**2.24: Score the strategy**
- Strategy: long SPY when HMM says expansion, cash (0 return) otherwise
- Walk-forward returns from step 2.22
- Compute: Sharpe, max drawdown, hit rate, cumulative return
- Compare vs. buy-and-hold SPY (same metrics)
- Document results in `[[liquidity_regime_results]]`

**2.25: Register as agent tool**
- Create `agent/tools/liquidity_regime.py`:
  - `LiquidityRegimeTool(Tool)`:
    - name: `liquidity_regime`
    - parameters: `lookback_years` (default: 5), `global_` (default: false)
    - execute: fetch data → compute composite → fit HMM → return current regime, confidence, last changepoint, transition probabilities
- Register in `agent/cli.py`
- Test: run agent with "what liquidity regime are we in right now?"

**2.26: Update documentation**
- Mark Phase 2 complete in `[[quant_training_ground]]`
- Update `[[project_memory]]` with:
  - What K was chosen and why
  - Walk-forward Sharpe achieved
  - Whether global composite outperformed US-only
  - Key lessons learned

---

## Edge Cases

| Case | Handling |
|------|----------|
| FRED API returns empty for a series | `fetch_us()` raises ValueError with specific series ID |
| FX data has gaps (holidays differ across countries) | Forward-fill FX, max 5-day gap. Beyond 5 days → raise |
| RRPONTSYD unavailable before 2013 (series start) | Use 0 for RRP before series start. US net liquidity = WALCL - WTREGEN for earlier period |
| HMM fit doesn't converge | hmmlearn raises ConvergenceWarning. Catch it, increase max_iter to 500, retry once. If still fails, log warning and return best-so-far |
| BOCPD run-length matrix exceeds memory | Truncate at 2 * hazard_lambda. Proven in Adams & MacKay — long run lengths have negligible probability |
| Walk-forward test has too few observations per regime | Report sample size per regime. If any regime has <20 observations in test, note that statistical significance is questionable |

---

## Testing Plan

| Step | Test | Pass criteria |
|------|------|--------------|
| 2.4 | `python -c "import agent.quant"` | No error |
| 2.5 | Instantiate LiquidityComposite, call stubs | Returns empty without error |
| 2.6 | fetch_us("2020-01-01", "2024-01-01") | 4 columns, >150 rows, correct magnitude |
| 2.7 | Data quality checks | No NaN gaps >2 weeks, values in expected range |
| 2.8 | compute() on US data | Z-scores: mean ≈ 0, std ≈ 1 (after warm-up) |
| 2.9 | Visual: composite vs SPY | Major events align (2020 crash/rally, 2022 bear) |
| 2.10 | fetch_global() | All CB columns present, USD magnitudes correct |
| 2.11 | Global composite | Correlates with US composite (ρ > 0.5), not identical |
| 2.12 | BOCPD class exists, methods callable | No error on empty input |
| 2.13 | BOCPD on synthetic | Finds known changepoint within ±10. No false positives on pure noise |
| 2.14 | RegimeHMM class exists | Wraps hmmlearn correctly |
| 2.15 | HMM on synthetic | Recovers 3 states, means within ±0.5, accuracy >90% |
| 2.16 | Spectral functions exist | Returns arrays of correct shape |
| 2.17 | Spectral on synthetic | Finds known frequencies in power spectrum |
| 2.18 | BOCPD on real data | Detects 2020-03 and 2022 changepoints |
| 2.19 | HMM on real data | Regime labels are interpretable (expansion in QE, contraction in QT) |
| 2.20 | Benchmark returns | 5 assets, aligned dates, no NaN |
| 2.21 | Regime-conditional analysis | Expansion Sharpe > 0 for SPY. Contraction Sharpe < expansion |
| 2.22 | Walk-forward | Out-of-sample regime labels exist for 2021+ |
| 2.23 | Scoring functions | Hand-calculated examples match |
| 2.24 | Strategy score | Strategy Sharpe > 0 out-of-sample (even if modest) |
| 2.25 | Agent tool works | Agent answers "what regime are we in?" |
| 2.26 | Docs updated | Task file shows Phase 2 complete |

---

## Related

- [[liquidity_regime_detection|Research: Liquidity Regime Detection]]
- [[convergence_detection]]
- [[world_model]]
- [[backtest_performance]]

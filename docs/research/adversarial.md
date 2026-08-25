---
title: "Feature: Adversarial Intelligence Layer"
tags:
  - doc/research
  - phase/22
  - topic/adversarial
  - topic/manipulation-detection
  - topic/edge-decay
  - layer/adversarial
---

# Feature: Adversarial Intelligence Layer

## Goal

Build Layer 6 of the 7-layer computation stack: **manipulation detection, edge decay monitoring, and game-theoretic counterparty modeling**. The system must reason about other market participants — detecting when observed signals are artifacts of manipulation rather than genuine information, monitoring whether our own edge is eroding as others discover the same patterns, and estimating crowding risk.

**Core insight:** TirraMind already has the entity graph (L1-L3), surprise signals (L4), and RL policy (L5). The adversarial layer doesn't build new data pipelines — it sits *on top of* existing signals and asks: "Is this signal real, or is someone manufacturing it? And how long will it last?"

**Design principle from copilot-instructions:** "Assume smart adversaries. If you can see a signal, eventually someone else will too."

---

## Search Log

- GitHub keywords searched:
  - `market manipulation detection python` → 2 repos (xrp-watchdog: wash trading for XRP; OOP-Trading-Systems-Python: reference only)
  - `spoofing detection` → 28 repos, all non-financial (GPS, face, email, hardware spoofing). Zero financial market spoofing repos.
  - `PIN probability informed trading` → 10 repos: `sclearchan/InfoTrad` (Python, 1 star), `monty-se/PINstimation` (R, 40 stars, comprehensive), `shuangology/Probability-of-Informed-Trading` (Jupyter, 41 stars, A-shares)
  - `market microstructure order flow python` → 7 repos: `SpookyJumpyBeans/crypto-market-microstructure-analyzer` (OFI), `Rakeshks7/hawkes-process-hft-microstructure` (Hawkes for HFT)
  - `crowding risk alpha decay quantitative` → 0 repos
  - `edge decay signal alpha decay` → 0 repos
- Documentation keywords searched: Easley-O'Hara PIN model, VPIN, spoofing detection SEC
- Other search surfaces: academic paper references below

## External Repositories Reviewed

- `monty-se/PINstimation` (R, GPL-3.0, 40 stars):
  - Why relevant: Comprehensive PIN/MPin/VPIN estimation with multiple MLE initialization methods (Yan & Zhang 2012 improvement). Well-tested.
  - Useful implementation idea: Multi-start MLE for PIN to avoid local optima; hierarchical clustering for initial parameter guesses; VPIN as real-time proxy.
  - License: GPL-3.0 — **concept only** (GPL incompatible with commercial use).
  - Reuse conclusion: **concept only** — extract the algorithm and initialization strategy, implement independently.

- `sclearchan/InfoTrad` (Python, MIT, 1 star):
  - Why relevant: Direct Python PIN implementation.
  - Useful implementation idea: Basic PIN MLE. Very minimal.
  - License: MIT — reusable pattern if desired.
  - Reuse conclusion: **concept only** — too minimal, we need VPIN and multi-start initialization.

- `shuangology/Probability-of-Informed-Trading` (Jupyter, MIT, 41 stars):
  - Why relevant: PIN implementation with Yan & Zhang (2012) improved estimation method.
  - Useful implementation idea: Improved initial parameter estimation from aggregate buy/sell volume.
  - License: MIT — reusable pattern.
  - Reuse conclusion: **concept only** — notebook format, not production code.

- `realgrapedrop/xrp-watchdog` (Shell, 1 star):
  - Why relevant: Wash trading detection for crypto DEX.
  - Useful implementation idea: Self-trade detection via address pair analysis.
  - License: Unclear.
  - Reuse conclusion: **concept only** — crypto-specific, not applicable to equity/prediction markets.

## Documentation / Academic Sources Reviewed

### PIN Model
- **Easley, Kiefer, O'Hara, Paperman (1996)** — "Liquidity, Information, and Infrequently Traded Stocks", *Journal of Finance* 51(4).
  - What it clarified: Original PIN model. Poisson mixture: at each trading day, with probability α an information event occurs. Given an event, probability δ it's bad news, (1-δ) good news. Buy and sell orders follow Poisson(ε_b), Poisson(ε_s) for uninformed, plus Poisson(μ) for informed side. PIN = αμ / (αμ + ε_b + ε_s).
  - Key detail: MLE over daily buy/sell counts. Notoriously multimodal — needs multi-start.

- **Easley, López de Prado, O'Hara (2012)** — "Flow Toxicity and Liquidity in a High-Frequency World", *Review of Financial Studies* 25(5).
  - What it clarified: **VPIN** (Volume-synchronized PIN). Real-time measure. Classifies trades into buy/sell volume buckets, computes order imbalance. No MLE needed — can be computed streaming. Predicted Flash Crash of 2010 with 2-hour lead.
  - API/concept: V_buy(τ), V_sell(τ) per volume bucket τ. VPIN = Σ|V_sell − V_buy| / (n·V_bar). Updates every volume bucket, not every event.

- **Yan & Zhang (2012)** — "An improved estimation method for the probability of informed trading", *Journal of Banking & Finance* 36(2).
  - What it clarified: Improved initial parameter estimation for PIN MLE using aggregate statistics — avoids most local optima. Used by PINstimation R package.

### Edge Decay
- **McLean & Pontiff (2016)** — "Does Academic Research Destroy Stock Return Predictability?", *Journal of Finance* 71(1).
  - What it clarified: Anomaly returns decline 32% post-publication, 58% post-sample end. This motivates continuous signal health monitoring.
  - Design implication: Track rolling Sharpe per signal with structural break detection.

- **Chordia, Subrahmanyam, Tong (2014)** — "Have Capital Market Anomalies Attenuated in the Recent Era of High Liquidity and Trading Activity?", *Journal of Accounting & Economics*.
  - What it clarified: Many anomalies disappeared as markets became more efficient. Alpha decay is real and accelerating.

### Crowding Risk
- **Khandani & Lo (2011)** — "What Happened to the Quants in August 2007?", *Journal of Investment Management*.
  - What it clarified: Simultaneous factor unwind caused cascading losses. Crowding in momentum/mean-reversion strategies. Crowding risk is correlated exit risk, not just correlated entry.
  - Design implication: Monitor strategy overlap via factor exposure similarity, detect unwind cascades via spread compression.

### Market Manipulation Taxonomy
- **SEC Dodd-Frank Act §747** — spoofing definition: "bidding or offering with the intent to cancel the bid or offer before execution." Measurable via order-to-trade ratio (OTR) and time-to-cancel distributions.
- **Comerton-Forde & Putniņš (2015)** — "Measuring and explaining the dynamics of market manipulation", *Journal of Financial Economics*.
  - What it clarified: End-of-day price manipulation detectable via return reversal patterns. Order-based (spoofing/layering) vs. trade-based (wash trading, pump and dump) vs. information-based (false rumors).

---

## Current Architecture

### Relevant local modules
- **`agent/quant/scoring.py`**: Rolling Sharpe calculation — directly usable for edge decay monitoring.
- **`agent/quant/changepoint.py`**: BOCPD — can detect structural breaks in signal Sharpe (edge decay changepoints).
- **`agent/quant/regime.py`**: HMM — can model adversarial regime states (normal vs. manipulation regime).
- **`agent/fusion/alert.py`**: EntityAlert — surprise signals that could be triggered by manipulation. The adversarial layer needs to distinguish genuine surprises from manipulation-induced surprises.
- **`agent/fusion/convergence.py`**: ConvergenceCluster — coordinated entity movement could indicate coordinated manipulation.
- **`agent/learning/policy/weight_learner.py`**: Differentiable Sharpe — reusable for signal health scoring.
- **`agent/learning/policy/reward_fn.py`**: RewardFunction — adversarial flags should modify reward (penalize acting on manipulated signals).
- **`agent/quant/backtest.py`**: Strategy ABC, WalkForward — adversarial-filtered strategies should integrate via same ABC.

### Correct insertion points
- **New module: `agent/adversarial/`** — per README and copilot-instructions. NOT `agent/quant/adversarial.py` as the old roadmap suggested. Layer 6 is its own module with clean separation. The `agent/quant/` path in the Phase 12 text was a pre-architecture decision; the 7-layer stack is authoritative.
- Adversarial signals feed back into the reward function (penalize acting on detected manipulation) and the weight learner (auto-downweight decaying signals).

### Existing patterns to preserve
- Frozen dataclasses for records (like EntityAlert, BeliefState)
- Strategy ABC for walk-forward evaluation
- DAG integration for scheduled adversarial scans
- PipelineStore for persistence

---

## Observations

### What already exists
1. **Hawkes process** (`agent/fusion/hawkes.py`) — self-exciting point process already used for EntityAlert enrichment. Directly applicable to spoofing detection (burst of order/cancel events).
2. **CUSUM** (`agent/fusion/cusum.py`) — change detection. Useful for detecting sudden shifts in signal quality (edge decay).
3. **BOCPD** (`agent/quant/changepoint.py`) — Bayesian changepoint detection. The best tool for edge decay monitoring (detect when a signal's Sharpe undergoes a structural break).
4. **ConvergenceDetector** (`agent/fusion/convergence.py`) — detects coordinated entity behavior. Could surface coordinated pump-and-dump or wash trading.
5. **Scoring suite** (`agent/quant/scoring.py`) — Sharpe, Sortino, Calmar, VaR, CVaR already implemented. Rolling versions needed for edge decay.

### What is missing
1. **No microstructure module** — no OFI, no order book imbalance, no trade classification. TirraMind operates at daily/weekly frequency with public data (SEC, prediction markets, GDELT). Order-level microstructure data isn't in our surveillance surface.
2. **No HFT data** — spoofing detection (12.1) requires millisecond-level order book data. Our tools don't collect this. Can't do classical spoofing detection without limit order book data.
3. **No social media velocity tool** — pump-and-dump detection (12.3) requires real-time social media monitoring. Our GDELT tool (news) has daily granularity. Not sufficient for real-time P&D detection.

### Important constraints
- **Data availability drives scope.** TirraMind's surveillance surface is free public APIs at daily/weekly resolution. We don't have tick-level order book data or real-time social feeds. This fundamentally constrains what adversarial detection is feasible.
- **Focus on what we CAN detect with existing data:**
  - Edge decay: ✅ — only needs our own signal performance history
  - VPIN proxy: ✅ — can compute from daily volume data (aggregated VPIN per Easley et al. 2012)
  - Crowding risk estimation: ✅ — can infer from factor exposure correlations and convergence clusters
  - Pump-and-dump detection: partial — can detect via GDELT + SEC filing (delayed, not real-time)
  - Spoofing detection: ❌ — requires tick-level LOB data
  - Stop hunting detection: ❌ — requires intraday price data

---

## Risks

### Licensing risks
- PIN/VPIN: The PINstimation R package is GPL-3.0. We implement from the original papers (Easley et al. 1996, 2012) — algorithms are not copyrightable, only code is. **Implement independently from paper math.**
- No other licensing issues.

### Technical risks
- **VPIN without tick data**: VPIN was designed for volume-synchronized buckets from tick data. With daily aggregates, we get a much noisier estimate. Need to validate whether daily-VPIN retains predictive power.
- **PIN MLE convergence**: Notorious for multiple local optima. Multi-start initialization (Yan & Zhang 2012 method) mitigates but doesn't eliminate.
- **Edge decay false positives**: A signal's Sharpe can temporarily decline due to regime change (not competitor crowding). Need to distinguish structural decay from cyclical drawdown.

### Testing risks
- No real manipulation labels to train/test against. Must use synthetic manipulation injection + known historical events (Flash Crash, GameStop) for validation.

---

## Scope Decision: What to Build vs. Defer

Given TirraMind's data constraints (daily/weekly public APIs, no tick-level LOB), the adversarial layer should focus on **what we can actually detect with our existing surveillance surface**, not on textbook market manipulation that requires HFT data.

### Build Now (Phase 22)
| Component | Data Required | Already Available? | Why Build |
|-----------|---------------|-------------------|-----------|
| **Edge decay monitor** | Per-signal Sharpe history | ✅ scoring.py | Core: prevents acting on dead signals |
| **VPIN estimator** | Daily volume + price | ✅ market tools | Information asymmetry at daily resolution |
| **Crowding risk estimator** | Convergence clusters + factor exposures | ✅ convergence.py | Unwind cascade risk |
| **Manipulation flag protocol** | All of the above | ✅ | Unified output contract |
| **Adversarial reward modifier** | Adversarial flags + reward function | ✅ reward_fn.py | RL policy should penalize manipulated signals |
| **Historical validation** | Synthetic + historical replay | ✅ backtest.py | Prove the detectors work |

### Defer (Need New Data Tools First)
| Component | Data Required | Available? | Defer Until |
|-----------|---------------|-----------|-------------|
| Spoofing detection (12.1) | Millisecond LOB | ❌ | When LOB data tool is built |
| Stop hunting (12.2) | Intraday price + stop clusters | ❌ | When intraday data tool is built |
| Pump-and-dump (12.3) | Real-time social media | ❌ | When social media velocity tool is built |
| PIN full MLE (12.4) | Tick-level trade classification | ❌ | When trade-level data tool is built |

### Rationale
This is consistent with the architecture doctrine: "the cheapest data is often the most valuable because nobody else looks at it." Edge decay and crowding risk require *no new data* — they operate entirely on the system's own signal history and existing entity graph. That's zero marginal data cost for high-value adversarial intelligence.

---

## Data Requirements

### Required inputs (already available)
- Per-signal Sharpe time series → from `scoring.py` applied to each signal's contribution
- EntityAlert surprise vectors → from `agent/fusion/alert.py`
- ConvergenceCluster data → from `agent/fusion/convergence.py`
- Daily volume data → from market data tools
- BeliefState posteriors → from world model
- Portfolio position weights → from RL policy

### What still needs to be added
- Rolling Sharpe computation per signal (not just aggregate) — minor extension to scoring.py
- Volume-bucket classification for VPIN — new computation in adversarial module
- Factor exposure vectors per strategy — derivable from existing walk-forward results

---

## Math/Algorithm Survey

### 1. Edge Decay Monitor

**Objective:** Detect when a signal's predictive power is structurally declining (not just in a temporary drawdown).

**Method: BOCPD on rolling Sharpe.**

Given signal i's return series $r_i = \{r_{i,1}, \ldots, r_{i,T}\}$, compute the rolling Sharpe ratio with window $w$:

$$S_{i,t} = \frac{\bar{r}_{i,t:t-w}}{\sigma_{r_{i,t:t-w}}} \cdot \sqrt{52}$$

Apply BOCPD (already implemented in `changepoint.py`) to the time series $\{S_{i,t}\}$. A changepoint with post-change mean $< S_{pre}/2$ indicates structural decay.

**Why BOCPD over simpler methods:** BOCPD gives a posterior over the location of the changepoint, not just a binary "changed/not-changed." This allows the RL policy to gradually downweight a signal proportional to the decay probability rather than making a hard on/off decision.

**Decay score:** $d_i(t) = P(\text{changepoint in last } k \text{ periods} | S_{i,1:t})$ where $k$ is a lookback window. The RL reward function can scale the signal's contribution by $(1 - d_i(t))$.

**Trusted source:** Adams & MacKay (2007), "Bayesian Online Changepoint Detection" — the same reference already used in `changepoint.py`.

### 2. VPIN (Volume-Synchronized PIN)

**Objective:** Estimate information asymmetry from daily volume data.

**Method:** From Easley, López de Prado, O'Hara (2012).

Partition total volume $V$ into $n$ buckets of equal size $\bar{V}$. Within each bucket $\tau$, classify volume as:
- $V_{buy}(\tau)$: buy-initiated volume
- $V_{sell}(\tau)$: sell-initiated volume

At daily resolution without tick data, use the **bulk volume classification** (BVC) method (Easley et al. 2012 §3.2):

$$V_{buy,t} = V_t \cdot \Phi\left(\frac{r_t}{\sigma_r}\right), \quad V_{sell,t} = V_t - V_{buy,t}$$

where $\Phi$ is the standard normal CDF, $r_t$ is the log return, and $\sigma_r$ is the rolling return volatility.

$$\text{VPIN}_t = \frac{1}{n} \sum_{\tau=t-n+1}^{t} \frac{|V_{sell}(\tau) - V_{buy}(\tau)|}{\bar{V}}$$

**Output:** VPIN ∈ [0, 1]. High VPIN → high probability of informed trading → market is "toxic" to uninformed participants.

**Why VPIN over classical PIN:** PIN requires MLE over Poisson parameters — computationally expensive, multimodal, requires tick-level buy/sell classification. VPIN is a direct statistic computable from daily volume + returns. For our daily-frequency surveillance surface, VPIN is both feasible and better calibrated.

**Trusted source:** Easley, López de Prado, O'Hara (2012), "Flow Toxicity and Liquidity in a High-Frequency World", *Review of Financial Studies* 25(5), 1457–1493.

### 3. Crowding Risk Estimator

**Objective:** Estimate how many other market participants are likely exploiting the same signals, and the cascade risk if everyone exits simultaneously.

**Method: Factor exposure similarity + convergence cluster density.**

For each active strategy/signal $i$ with weight vector $w_i$ in the surprise space:

1. **Crowding score via convergence cluster density:** The ConvergenceDetector already groups entities with correlated surprise patterns. If a cluster is large and dense, many entities are showing the same anomaly — meaning many participants could be trading the same signal.

$$\text{crowd}_i(t) = \frac{|\text{cluster}_i(t)|}{\bar{|\text{cluster}|}} \cdot \rho_{\text{intra}}(t)$$

where $|\text{cluster}_i|$ is the cluster size and $\rho_{\text{intra}}$ is the mean intra-cluster correlation.

2. **Unwind risk score:** When crowding is high, the risk of a simultaneous exit is elevated. Following Khandani & Lo (2011), model the potential unwind impact as proportional to:

$$\text{unwind}_i(t) = \text{crowd}_i(t) \cdot \text{position}_i(t) \cdot \frac{1}{\text{liquidity}_i(t)}$$

where liquidity can be proxied by the rolling average daily volume.

**Why not agent-based simulation (as 12.6 suggested)?** Agent-based simulation of unwind scenarios requires calibrating agent populations, strategies, and market impact models — all requiring data we don't have. The statistical crowding score above is computable from existing data and degrades gracefully (it's a rough estimate, not a precise simulation, and it correctly acknowledges that).

**Trusted source:** Khandani & Lo (2011); Stein (2009) "Presidential Address: Sophisticated Investors and Market Efficiency", *Journal of Finance* 64(4).

### 4. Manipulation Flag Protocol

**Objective:** Unified output contract for adversarial signals, consumable by the RL policy's reward function and the weight learner's edge decay mechanism.

Design: Frozen dataclass mirroring EntityAlert's pattern:

```python
@dataclass(frozen=True)
class AdversarialFlag:
    entity_id: str | None          # None for market-wide flags
    flag_type: str                 # "edge_decay" | "vpin_spike" | "crowding_risk"
    severity: float                # 0.0 = benign, 1.0 = severe
    confidence: float              # statistical confidence
    signal_name: str | None        # which signal is affected
    evidence: dict                 # supporting data
    timestamp: float               # unix epoch
```

The adversarial layer produces a list of AdversarialFlags per evaluation cycle. These feed into:
- **RewardFunction**: penalty term proportional to Σ severity×confidence for active positions
- **WeightLearner**: decay weights on flagged signals
- **RL Policy**: state vector includes adversarial summary features

### Implementation Options Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Full PIN MLE | Gold standard | Needs tick data, multimodal MLE, slow | **Defer** — use VPIN |
| VPIN daily | Works with our data, streaming, simple | Noisier than tick-VPIN | **Build** |
| Agent-based crowding sim | Rich dynamics | Uncalibrable without real position data | **Defer** — use statistical proxy |
| Statistical crowding score | Uses existing convergence clusters | Rough estimate | **Build** |
| BOCPD edge decay | Already implemented, principled | Needs per-signal Sharpe series | **Build** |
| Simple threshold edge decay | Easy to implement | Arbitrary threshold, no uncertainty | **Reject** — BOCPD is better and already exists |

---

## Depth Roadmap

- **L1 (Aggregate):** Market-wide VPIN, aggregate edge decay score, overall crowding index. Infrastructure-level, not entity-specific.
- **L2 (Entity-level):** Per-entity VPIN (for entities with sufficient volume data), per-signal edge decay tracking, per-entity crowding risk. This is where actionable alpha lives.
- **L3 (Cross-entity):** Cross-domain crowding detection (e.g., the same insider filing cluster that triggers our surprise signals is also showing in Polymarket odds shifts — crowding is happening across prediction markets and equities simultaneously). This requires the entity linking layer (Phase 17) to be mature.

**Phase 22 targets L1 and L2.** L3 deferred until entity linking is dense enough to surface cross-domain patterns.

---

## Related

- [[adversarial_spec]] — Spec doc (to be created)
- [[rl_policy]] — Phase 21 research (RL policy consumes adversarial flags)
- [[rl_policy_spec]] — Phase 21 spec (reward function integration point)
- [[signal_fusion]] — Phase 20 research (surprise signals — what adversarial layer scrutinizes)
- [[quant_training_ground]] — Master phase tracker

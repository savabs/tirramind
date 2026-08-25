---
title: "Research: World-State Prediction Methodology — Beyond Sharpe"
tags:
  - doc/research
  - phase/48
  - topic/world-model
  - topic/evaluation
  - topic/convergence
  - layer/world-model
  - layer/surveillance
  - layer/fusion
---

# Research: World-State Prediction Methodology — Beyond Sharpe

## The Core Problem

Phase 40 exposed a methodological mistake that runs deeper than the GNN implementation.

We evaluated the GNN using portfolio Sharpe. That is the **wrong primary metric** — and not just because the signal is weak. It is wrong in principle.

Sharpe measures:

```
embedding → weight rule → portfolio return → Sharpe
```

This conflates three independent questions into one number:

1. **Signal quality:** Does the GNN embedding encode anything predictively useful?
2. **Signal-to-weight translation:** Does the weight rule (softmax of norms, value head scores) correctly extract that signal?
3. **Allocation quality:** Does the resulting weight distribution reduce portfolio variance?

When Sharpe is bad, you cannot tell which layer failed. GNN-EmbNorm failed on (2) — concentration of 44.65% in one asset, caused by using connectivity/activity as a return proxy. GNN-ValueHead failed on (1) — value head trained with in-sample MSE on full history, no walk-forward splits, in-sample bias baked in. Sharpe said both failed equally. They failed for completely different reasons.

But the deeper problem is that **Sharpe was never the right question.**

---

## What TirraMind Is Actually Building

TirraMind is not a portfolio optimizer. It is a **world-state inference machine.**

The correct framing:

> Given a stream of sensor readings from physical and behavioral reality, infer the hidden state of the world — and propagate that state forward to produce probability distributions over future observable events.

The financial signal — or any predictive signal — is a **readout from inferred world state.** Not the primary target.

This changes the architecture priority and the evaluation methodology completely.

---

## The Sensor Foundation Is Not Infrastructure — It Is the Moat

A nervous system with no sensory organs is blind. The math (HetTGN, Kalman, Bayesian propagation) can only detect relationships that exist within its current sensor space.

The relationship the system is trying to detect does not live in any one sensor. It lives in the **cross-tensor product of sensor space:**

A drought signal in Brazilian weather data (NOAA) + vessel rerouting in AIS tracking (OpenSky) + CFTC long buildup in soy futures + food security score deterioration in import-dependent countries (FAO) + GDELT political event frequency in those countries.

None of those five signals, individually, is an edge. The economist with a Bloomberg terminal sees the CFTC buildup, maybe the drought. They cannot simultaneously watch all five shift in concert over three weeks in a way that implies a specific food security crisis that will not appear in PMI or price action for another 6 weeks.

**The relationship only becomes visible when the sensors exist and are wired together.** This is the irreplicable moat — not the math. The math is learnable by anyone. The observation surface is not.

### Current Sensor Inventory (62 tools, as of 2026-04-28)

**L0 Physical (real-world state, hardest to manipulate):**
- `ais_vessel.py` — vessel positions, routing anomalies, port congestion
- `power_grid.py` — NYISO grid load, fuel mix, pricing (industrial activity proxy)
- `weather_alerts.py` — NOAA alerts, anomaly scoring (crop, energy, physical disruption)
- `earthquake_proximity.py` — seismic events near infrastructure
- `satellite_activity.py` — satellite-based activity proxies
- `food_security.py` — FAO food security indicators
- `energy_supply.py` — energy supply signals
- `transport_throughput.py` — transport volume signals
- `supply_chain_monitor.py` — supply chain stress indicators

**L1 Behavioral (committed human decisions, legally recorded):**
- `cftc.py` — futures positioning (COT data), institutional commitment
- `insider_filings.py` — SEC Form 4, insider transactions
- `form144.py` — planned insider sales
- `finra_short_volume.py` — short interest, DTC, squeeze risk
- `whale_alert.py` — large on-chain transactions
- `gdelt.py` — global event stream (L1 behavioral, not L0 physical)
- `polymarket.py`, `polymarket_whales.py` — prediction market probabilities + whale positioning
- `sovereign_debt.py`, `treasury_receipts.py` — government fiscal signals
- `sanctions_monitor.py`, `gov_contracts.py`, `lobbying.py` — political economy signals
- `disease_surveillance.py`, `drug_regulatory.py` — health system signals
- `patent_filings.py`, `academic_preprints.py` — technology frontier signals
- ... and ~40 more

**The sensor surface already exists.** The problem is not missing sensors. The problem is:

1. **Observation density is pathological** — 95.7% of DB observations are `instrument_daily`. The physical and behavioral sensors are built but sparse in the DB (limited backfill).
2. **Cross-domain entity links are thin** — the GNN edges between sensor domains are not yet dense enough to learn propagation.
3. **The evaluation framework measures the wrong thing** — Sharpe can't tell you if the GNN is detecting anything real.

---

## What "Predictiveness" Means in This Architecture

The system produces probability distributions over future entity states and events. Not price forecasts.

The prediction hierarchy is:

| Layer | What is being predicted | Example |
|---|---|---|
| L0 → L1 | Given physical observations, which behavioral events become likely? | Vessel diversion + grain port congestion → political pressure on food-importing country |
| L1 → L2 | Given behavioral events, which information shifts emerge? | Insider selling + short buildup + satellite parking lot empty → company distress narrative forming |
| L2 → L3 | Given information shifts, which prices/geopolitical states change? | Credit spread widening + capital flight → sovereign stress visible in markets |

The GNN embedding `h_i` for an entity is not "predict the price of X." It encodes the **latent state** of that entity — is this country under fiscal stress? Is this shipping company routing around something? Is this commodity supply chain degrading? Is this protocol accumulating unusual wallet activity?

The edges encode **how states propagate:** geopolitical stress in a wheat-producing country → commodity positioning shift in CFTC → shipping rerouting in AIS → food security pressure in import-dependent countries → political instability events in GDELT → sovereign debt spread widening.

None of that chain is price prediction. It is state inference and causal propagation. The financial signal — if it exists — emerges from correctly modeling that chain, not from directly regressing embeddings onto returns.

---

## The Correct Evaluation Framework

### Tier 1: Information Coefficient (IC) — Minimum Bar

$$IC_t = \text{Spearman}\left(\hat{s}_{i,t},\ r_{i,t+21d}\right)$$

Where $\hat{s}_{i,t}$ is the GNN's predicted score for instrument $i$ at time $t$ (value head output or embedding norm) and $r_{i,t+21d}$ is the actual 21-day forward return.

- Mean IC > 0.03 across folds with t-stat > 2.0: the GNN has a weak but real return signal
- Mean IC > 0.07: meaningful signal
- IC ≈ 0: the embedding carries no return information

**Why:** IC separates signal quality from allocation quality. A good IC with bad Sharpe means the allocation rule is the problem. A bad IC means the embedding itself carries no information.

**Limitation:** IC only measures return predictability. It does not measure whether the GNN correctly infers world state. A model can have IC=0 and still be learning real structure (the causal chain may not lead to price within 21 days).

### Tier 2: Event Prediction Accuracy — Real World-State Signal

For each entity type, the model should be able to predict: "will a significant event occur for this entity within N days?"

Concretely:
- **GDELT events:** Given the GNN state of a country node at time T, does the model assign higher probability to the country having a significant GDELT event in [T, T+7d]?
- **CFTC regime shifts:** Given the GNN state of a commodity instrument node, does the model predict a >2σ positioning change in [T, T+21d]?
- **Instrument price regime change:** Given the GNN state, does the model predict a breakout (>1.5× realized vol move) in [T, T+7d]?
- **Entity state transitions:** Does the model correctly predict when an entity transitions from a "stable" to a "stressed" regime?

**Metric:** Precision@K, Recall@K, AUC-ROC for binary event prediction. Evaluated in walk-forward fashion (train on [0, T], test on [T, T+N]).

**Why this matters:** A model that correctly predicts political instability events 10 days before they become GDELT events is a genuine world-state inference machine, regardless of what instrument prices do. That is the proof of concept.

### Tier 3: Causal Chain Signal — Cross-Domain Propagation

The irreplaceable signal: does a change in embedding state at entity A *precede* a measurable state change at causally downstream entity B?

**Measurement approach:**
- **Transfer entropy:** $TE_{A \to B} = H(B_t | B_{t-1}) - H(B_t | B_{t-1}, A_{t-\tau})$ — does knowing A's embedding history reduce uncertainty about B's future state?
- **Granger causality on embeddings:** Does lagged embedding of entity A Granger-cause embedding of entity B for causally linked entity pairs?
- **Lead-lag correlation:** For known causal pairs (wheat drought → food import country stress → political instability), does the embedding signal flow in the correct temporal direction?

**Known causal chains to test:**
1. Weather anomaly (NOAA) → crop yield signal (FAO) → grain vessel routing (AIS) → CFTC soy/wheat positioning → food security deterioration → GDELT political events in import countries
2. Corporate insider selling (Form 4) + short buildup (FINRA) → company GDELT mentions → price regime change
3. Port congestion (AIS) + supply chain stress → manufacturer GDELT events → industrial commodity price shift

**Why:** If the GNN correctly learns these chains from the data without being told about them, that is the proof that the architecture works as designed. That cannot be seen in portfolio Sharpe at all.

### Tier 4: Prediction Market Calibration — Against Market Consensus

Polymarket gives us probability distributions over geopolitical and economic events from a liquid prediction market. The GNN should, after sufficient training on physical and behavioral signals, produce probability estimates for those same events.

**Metric:** Brier score and calibration curve for GNN-predicted event probabilities vs Polymarket consensus. If GNN is well-calibrated, the prediction errors should be uncorrelated with market errors — meaning the GNN is adding information beyond what the market already knows.

---

## Why Cross-Domain Is the Moat

A single sensor = time series analysis. Commoditized.

Two sensors from the same domain = slightly richer quant. Still commoditized (everyone has Bloomberg + macro data).

**Cross-domain entity combinations are the moat** — because the relationships only emerge when you observe the same hidden state through independent physical and behavioral channels simultaneously.

### The Detection Threshold

A relationship becomes **machine-detectable but human-invisible** when:
1. It requires simultaneously observing >3 independent sensor domains
2. At least 1 domain is physical (L0), not just behavioral/financial
3. The causal chain has a time delay >2 weeks between physical signal and market consequence
4. No single human analyst has access to all sensor streams simultaneously

The 51-sensor surface that TirraMind has built crosses that threshold for many commodity, geopolitical, and macro relationships. The GNN just needs enough data density to learn them.

### Cross-Domain Pairs to Explicitly Test

| Sensor A | Sensor B | Hypothesized relationship | Measurable consequence |
|---|---|---|---|
| AIS vessel rerouting | CFTC commodity positioning | Supply chain stress → institutional positioning ahead of price | Commodity price move T+2w |
| Weather anomaly (NOAA) | Food security (FAO) | Crop stress → food security deterioration | GDELT political events in import countries T+3w |
| Power grid anomaly | Industrial sector GDELT | Industrial slowdown → sector events | Equity sector return T+1w |
| Satellite activity | Insider selling (Form 4) | Physical activity drop → insider aware before disclosure | Equity price T+4w |
| Sovereign debt spread | Currency pressure + capital flight signals | Fiscal stress accumulation | Political instability events T+2w |
| CFTC long buildup | Polymarket event probability | Institutional positioning ahead of known events | Event occurrence probability calibration |
| Earthquake proximity | Supply chain monitor | Physical infrastructure disruption | Sector/commodity price T+1w |

---

## The Density Problem (Prerequisite)

The evaluation framework above is meaningless without observation density. The GNN cannot learn cross-domain propagation if 95.7% of its observations are `instrument_daily`.

**Density requirements before meaningful evaluation:**

| Entity type | Minimum observations per entity | Current estimated state |
|---|---|---|
| country | ≥500 (weather + GDELT + sovereign + food + power) | Sparse — GDELT dense, others thin |
| instrument | ≥500 (daily prices + CFTC + finra + insider) | Instrument_daily dense, others sparse |
| company | ≥200 (insider + form144 + finra + GDELT) | Sparse |
| vessel | ≥100 (AIS position history) | Very sparse |
| cftc_contract | ≥100 (weekly COT positions) | Moderate |
| wallet | ≥50 (whale transactions) | Very sparse |

**Phase 47 (historical backfill) is the prerequisite.** Without 2-5 years of history per tool per entity, the GNN is training on noise, and the evaluation framework will return noise.

---

## Implementation Roadmap

**Step 1 (Phase 47, blocking):** Historical backfill all 51 tools for 2-5 years. Run density audit. Ensure ≥100 observations per entity per entity type. Patch sparse entities before proceeding.

**Step 2 (Phase 40 retrain, after backfill):** Retrain GNN on full historical data. Fix Kendall loss clamping (see [[phase41_model_refresh_hardening]] for the log-variance bound fix). Verify loss healthy across 10+ epochs.

**Step 3 (this methodology, Phase 41/evaluation):** Run the Tier 1–3 evaluation framework on the retrained model:
- Tier 1: IC across 40 walk-forward folds for all 89 instruments
- Tier 2: Event prediction for GDELT country events, CFTC regime shifts, price breakouts
- Tier 3: Transfer entropy for 3 known causal chains (weather→food→political, vessel→CFTC→price, insider→FINRA→price)
- Tier 4: Polymarket calibration (if Brier score < 0.25 vs market, GNN has independent signal)

**Step 4 (Phase 48):** Transformer world model + Dreamer model-based RL — only after density audit passes and Tier 2/3 evaluation confirms the GNN is learning real causal structure.

---

## Math References (for implementation)

**Information Coefficient:**
- De Prado, *Advances in Financial Machine Learning* (2018), Chapter 5 — IC computation and statistical significance testing for financial signals

**Transfer Entropy:**
- Schreiber 2000, *Measuring Information Transfer*, PRL 85:461 — original formulation
- Barnett et al. 2009 — equivalence of Granger causality and TE under Gaussian assumptions
- Implementation: `pyinform` library (MIT license) or manual KNN estimator via `sklearn`

**Granger Causality:**
- Hamilton 1994, *Time Series Analysis* — Chapter 11, standard reference
- Implementation: `statsmodels.tsa.stattools.grangercausalitytests`

**Brier Score / Calibration:**
- Gneiting & Raftery 2007, *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA
- Implementation: `sklearn.calibration.calibration_curve`

**Event Prediction / Binary Classification in Walk-Forward:**
- Standard: `sklearn.metrics.roc_auc_score` with temporal train/test splits
- Walk-forward split: same infrastructure as `MultiAssetWalkForward` in `agent/quant/backtest.py`

---

## Risks

- **Density prerequisite**: If Phase 47 backfill is incomplete, all Tier 2-4 evaluations will be noisy and inconclusive. Do not skip the density audit.
- **Label leakage in event prediction**: Event labels must be constructed strictly after the fold boundary. GDELT events are timestamped — use `observed_at` field, not insertion time.
- **Spurious Granger causality**: With many entity pairs, multiple testing correction is mandatory. Use Bonferroni or Benjamini-Hochberg on transfer entropy p-values.
- **Polymarket calibration**: Polymarket prices are noisy for illiquid contracts. Only use markets with >$10K volume for calibration.
- **AIS coverage gaps**: OpenSky has ~3-5% global vessel coverage. Rerouting signals are detectable for high-traffic corridors (Hormuz, Suez, GIUK gap) but not open ocean.

---

## Related
- [[phase41_model_refresh_hardening]]
- [[real_data_model_refresh]]
- [[world_model]]
- [[temporal_het_gnn]]
- [[living_system_online_gnn]]
- [[quant_training_ground]]
- [[cross_entity_l3]]
- [[vessel_sanctions_l3]]
- [[whale_geopolitical_l3]]
- [[gnn_guided_expansion_r2]]

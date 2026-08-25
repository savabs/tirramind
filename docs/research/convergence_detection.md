---
title: "Feature: Phase 7c — Convergence Detection Layer"
tags:
  - doc/research
  - phase/7c
  - phase/9
  - topic/convergence
---

# Feature: Phase 7c — Convergence Detection Layer

## Goal

Transform 60 independent data pipes into a single intelligence system that detects when normally-uncorrelated signals begin moving together — the mathematical signature of a hidden cause propagating through observable reality.

**The problem we solve:** A drought in Brazil (weather_alerts), a ship diversion through the Strait of Hormuz (ais_vessel), and a hiring freeze at Nestlé Germany (job_postings) are each noise alone. Together they are a supply chain crisis forming. No human can watch 60 dashboards. No tool can see across its own silo. The convergence layer sees the whole board.

**Business case:** This is where "collection of scrapers" becomes "intelligence edge." Without this layer, TirraMind is a well-organized RSS reader. With it, we are the only system on Earth that detects weak multi-channel coincidences before they compound into events that move markets.

---

## The Fundamental Insight

**Markets are multivariate. Causes are sparse. Observable consequences are diffuse.**

A single hidden cause (e.g., "China is about to restrict rare earth exports") produces dozens of diffuse, individually-noisy observable consequences:

- Rare earth spot prices tick up 2% (macro_data) — noise-level
- Chinese rare earth mining patents accelerate (patent_filings) — easily missed
- Shipping from Bayan Obo to Shanghai increases (ais_vessel) — one of 18,000 vessels
- Wikipedia page views for "neodymium" spike in Japanese (wikipedia_pageviews) — who notices?
- A Chinese government gazette mentions "strategic mineral reserves" (regulatory_gazette) — buried in 470 agencies
- Lobbying spend from US defense contractors on "critical minerals" increases 40% (lobbying) — 6-month lag data

Any ONE of these signals has a signal-to-noise ratio < 1. They are individually unactionable. But the JOINT probability of all 6 being noise simultaneously is astronomically low. The convergence layer computes this joint probability.

**The math:** If each signal has individual false-positive rate α ≈ 0.05, and they are truly independent under the null (no hidden cause), then observing k ≥ 4 of 6 flashing simultaneously has probability:

$$P(k \geq 4 | H_0) = \sum_{j=4}^{6} \binom{6}{j} \alpha^j (1-\alpha)^{6-j} \approx 1.4 \times 10^{-5}$$

That's the edge. One signal = noise. Five signals = knowledge.

---

## Search Log

### GitHub Keywords Searched
- `event fusion multi-source intelligence` — found: situational awareness systems, mostly military/NLP
- `bayesian evidence fusion sensor` — found: multi-sensor tracking (radar/lidar), relevant math
- `coincidence detection time series` — found: neuroscience spike-train analysis, directly applicable
- `causal discovery multivariate time series` — found: Granger causality, PCMCI, transfer entropy
- `information fusion heterogeneous` — found: Dempster-Shafer theory, fuzzy evidence combination
- `weak signal detection surveillance` — found: anomaly detection ensembles, mostly single-source
- `complex event processing temporal` — found: Esper/Flink CEP, relevant pattern language
- `bayesian network dynamic evidence propagation` — found: pgmpy, pymc, bnlearn, pomegranate

### Documentation Keywords Searched
- `pgmpy bayesian network tutorial` — DAG specification, CPD tables, belief propagation
- `pymc hierarchical bayesian model` — NUTS sampler, hierarchical priors for partial pooling
- `dempster shafer theory python` — pyds library (abandoned), manual implementation preferred
- `transfer entropy time series python` — jpype + Java JIDT library, or pure numpy implementation
- `PCMCI causal discovery tigramite` — Python package for causal discovery in time series
- `complex event processing python` — no strong Python CEP engine; build from scratch
- `false discovery rate benjamini hochberg` — statsmodels.stats.multitest
- `copula tail dependence python` — scipy.stats.gaussian_kde, openturns (heavy)

### Other Surfaces
- Academic: Adams & MacKay 2007 (BOCPD — already implemented), Runge et al. 2019 (PCMCI), Schreiber 2000 (Transfer Entropy), Pearl 2009 (Causality), Dempster 1967 / Shafer 1976 (evidence theory)
- Quant: López de Prado "Advances in Financial Machine Learning" Ch. 5 (meta-labeling, combinatorial purging), Ch. 17 (structural breaks)
- Intelligence community: PRISM (Pattern Recognition and Information Synthesis Model), ACH (Analysis of Competing Hypotheses) — Heuer 1999

---

## External Repositories Reviewed

### 1. tigramite (PCMCI causal discovery)
- **Why relevant:** Gold-standard causal discovery for multivariate time series. PCMCI algorithm handles contemporaneous + lagged causal links with conditional independence testing.
- **Useful idea:** The partial correlation framework for distinguishing direct vs. indirect causal links. The time-lagged cross-correlation with conditional independence testing.
- **License:** GPL-3.0 (copyleft — cannot directly use code in commercial project)
- **Reuse conclusion:** **Concept only.** Implement transfer entropy and conditional mutual information from scratch; use the PCMCI algorithmic framework (public academic paper) but not the GPL code.

### 2. pgmpy (Bayesian networks)
- **Why relevant:** Mature Python library for DAG-based probabilistic graphical models. Exact inference (variable elimination), approximate inference (belief propagation, sampling).
- **Useful idea:** The BayesianNetwork class, TabularCPD for discrete evidence, and the VariableElimination inference engine. Clean API for setting evidence and querying posteriors.
- **License:** MIT
- **Reuse conclusion:** **Reusable as dependency.** MIT-licensed, well-maintained. Use for the world model DAG and belief propagation. Don't wrap it — use it directly.

### 3. ruptures (changepoint detection)
- **Why relevant:** Fast offline changepoint detection (PELT, Binary Segmentation, Window-based). Complements our online BOCPD.
- **Useful idea:** The PELT algorithm for efficient offline multi-changepoint detection with linear complexity. Useful for batch re-analysis of historical convergence patterns.
- **License:** BSD-2-Clause
- **Reuse conclusion:** **Reusable as dependency.** BSD-licensed. Add for offline batch changepoint analysis on accumulated pipeline data. Complements our online BOCPD.

### 4. Esper / Flink CEP (Complex Event Processing)
- **Why relevant:** Industry-standard pattern matching on event streams. Temporal patterns like "A followed by B within 3 days, not preceded by C."
- **Useful idea:** The pattern language concept — declarative rules for temporal coincidences. We need this but in Python, operating on our evidence bus.
- **License:** N/A (Java ecosystem)
- **Reuse conclusion:** **Concept only.** Build a Python-native pattern matcher. The pattern language idea (sequence, within, not-preceded-by) is the key takeaway.

### 5. pomegranate (probabilistic models)
- **Why relevant:** Alternative to pgmpy. Bayesian networks, HMMs, GMMs, factor graphs. GPU-accelerated.
- **Useful idea:** Factor graph message passing for complex interdependencies. GPU acceleration for inference at scale.
- **License:** MIT
- **Reuse conclusion:** **Evaluate vs pgmpy.** May be faster for large networks. Keep as alternative. pgmpy has better documentation.

---

## Documentation Reviewed

### Dempster-Shafer Theory of Evidence
- **What it clarified:** DS theory assigns belief masses to sets of hypotheses, not just individual hypotheses. This handles "I don't know" (uncertainty) differently from "it's equally likely" (probability). For our case: a tool might provide evidence FOR a hypothesis without having any opinion about competing hypotheses.
- **Key concept:** Belief function Bel(A) ≤ Plausibility Pl(A). The gap Pl(A) - Bel(A) = uncertainty. Combined via Dempster's rule of combination (but beware! Dempster's rule fails when evidence highly conflicts — use Yager's modification or averaging for conflict resolution).
- **Carry forward:** Use DS combination for fusing binary/categorical evidence from heterogeneous sources. More appropriate than naive Bayes for our "some tools say yes, some say nothing" pattern.

### Transfer Entropy (Schreiber 2000)
- **What it clarified:** TE measures the amount of directed information transfer from one time series to another. It's a nonlinear generalization of Granger causality. $TE_{X \to Y} = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-l})$. If knowing X's past reduces uncertainty about Y's future beyond what Y's own past provides, there's information flow.
- **Carry forward:** Compute pairwise transfer entropy between all normalized signal streams to discover causal relationships the convergence layer should watch. This builds the causal graph empirically rather than hand-coding it.

### Pearl's Structural Causal Models
- **What it clarified:** The distinction between observational ($P(Y|X)$), interventional ($P(Y|do(X))$), and counterfactual ($P(Y_x|X',Y')$) queries. For convergence, we need interventional reasoning: if we OBSERVE signal X firing, what's the CAUSAL effect on hidden state Z, versus mere correlation?
- **Carry forward:** Design the world model as a structural causal graph, not just a correlation graph. Use d-separation to determine which signals are truly independent conditionalized on the hidden state.

### Complex Event Processing — Pattern Languages
- **What it clarified:** CEP systems detect patterns in event streams using operators: SEQUENCE (A then B), WITHIN (time window), AND (co-occurrence), NOT (absence), EVERY (recurring), FOLLOWED-BY, UNTIL. These compose into complex temporal patterns.
- **Carry forward:** Build a Python-native pattern language for the convergence detector. Patterns like: `SEQUENCE(sanctions_addition("Russia"), WITHIN(7d, shipping_diversion("Baltic")), WITHIN(30d, commodity_spike("palladium")))`.

### False Discovery Rate — Benjamini-Hochberg Procedure
- **What it clarified:** When testing many hypotheses simultaneously (we'll have thousands of potential convergence pairs), FDR control is essential. BH procedure: sort p-values, reject hypotheses where $p_{(i)} \leq \frac{i}{m} \cdot q$ for target FDR q. Controls the expected proportion of false discoveries.
- **Carry forward:** Every convergence detection cycle must apply BH correction before reporting. Target FDR q = 0.05.

---

## Current Architecture

### Relevant Local Modules

| Module | Role | Interface to 7c |
|--------|------|------------------|
| `agent/pipeline/store.py` | SQLite persistence (pipeline_data, signals, dag_runs) | Primary read/write store for convergence inputs and outputs |
| `agent/pipeline/dag.py` | DAG task graph definition | Convergence DAGs will depend on data collection DAGs |
| `agent/pipeline/executor.py` | Parallel DAG execution with upstream resolution | Will execute convergence detection as downstream DAG nodes |
| `agent/pipeline/operators.py` | ToolOperator, FunctionOperator | New ConvergenceOperator for running detection logic |
| `agent/pipeline/registry.py` | DAG registration | Register convergence DAGs |
| `agent/quant/changepoint.py` | BOCPD (bayesian online changepoint detection) | Detect regime shifts in individual signal streams |
| `agent/quant/regime.py` | HMM regime identification | Identify hidden states that generate observed signals |
| `agent/quant/spectral.py` | FFT + CWT spectral analysis | Find dominant frequencies, detect phase-locking between signals |
| `agent/quant/scoring.py` | Returns scoring (Sharpe, Sortino, etc.) | Score convergence signal predictive quality |
| `agent/memory/store.py` | Episodic/Semantic/Working memory | Store convergence episodes and learned facts |
| `agent/learning/bandit.py` | Thompson Sampling RL | Learn which convergence patterns produce edge |
| `agent/tools/*.py` (60 files) | Data collection tools | Upstream data sources. All return ToolResult(success, output, data) |

### Existing Patterns to Preserve
- **ToolResult protocol:** All tools return `ToolResult(success: bool, output: str, data: Any)`. Convergence must consume `.data` from tools, NOT parse `.output` strings.
- **Pipeline DAG pattern:** Nodes define `operator` (tool name or callable), `params`, `depends_on`. Results flow via `$upstream.node_id` parameter resolution.
- **Signal storage:** `store.store_signal(signal_name, value, metadata)` with `(signal_name, computed_at)` index. Convergence outputs must use this.
- **Cache discipline:** TTLs are tool-specific. Convergence layer consumes cached tool results from pipeline_data, does NOT re-fetch.
- **No LLM in Pipeline:** The pipeline layer is deterministic. Convergence detection must be pure math and rules — no LLM calls.
- **7-layer separation:** Convergence sits at Layer 2 (Feature Engineering) / Layer 3 (World Model) boundary. It reads Layer 1 (tools) outputs, produces Layer 2 signals, and feeds Layer 3 (Bayesian network — future Phase 9). Does NOT directly touch Layer 5+ (RL, adversarial).

### Correct Insertion Points
- **New package:** `agent/convergence/` — independent package, not stuffed into quant/ or pipeline/
- **New DAGs:** `agent/pipeline/dags/convergence_detection.py` — DAGs that depend on daily_collection outputs
- **Pipeline store extensions:** May need new table (`convergences`) or extend `signals` with richer metadata
- **CLI integration:** `agent/cli.py` — register convergence DAGs, potentially a convergence_query tool
- **Bandit arm:** `convergence_detection` arm in `agent/learning/bandit.py`

---

## Observations

### What Already Exists
1. **Data layer is complete.** 60 tools covering physical reality (L0) and human decisions (L1) across all major economies. This is the sensory surface.
2. **Pipeline layer is complete.** DAG executor, SQLite store, scheduler, operators. Infrastructure to run convergence detection on schedule.
3. **Quant primitives exist.** BOCPD (changepoint), HMM (regime), spectral (frequency), scoring (performance). Low-level building blocks.
4. **RL loop exists.** Thompson Sampling bandit can learn which convergence patterns produce alpha.

### What Is Missing — The 7 Gaps

**Gap 1: No Signal Taxonomy.**
Every tool returns a different schema. `cftc` returns `{contracts: [...]}`. `weather_alerts` returns `{alerts: [...]}`. `disease_surveillance` returns `{wastewater: {...}}`. There is no common signal format. You can't compute correlations between incompatible schemas.

**Gap 2: No Evidence Normalization.**
Tool outputs are raw facts. They are not normalized to a common scale, common temporal resolution, or common confidence metric. You can't compare a CFTC positioning z-score to a wastewater pathogen detection rate without normalization.

**Gap 3: No Temporal Alignment.**
Tools report at wildly different frequencies: CFTC = weekly, weather = real-time, job_postings = monthly, treasury_receipts = daily, earthquake = event-driven. You can't correlate signals at different clock rates without resampling.

**Gap 4: No Cross-Source Correlation.**  
No mechanism computes whether signal A and signal B are moving together — let alone whether they're UNUSUALLY moving together relative to their historical relationship.

**Gap 5: No Coincidence Detection.**
The core capability — detecting when multiple independent channels all shift simultaneously — does not exist. This is the entire purpose of 7c.

**Gap 6: No Causal Template Library.**
No pre-defined patterns encode domain knowledge (e.g., "sanctions + shipping diversion + commodity spike = supply disruption"). Everything would have to be discovered from scratch.

**Gap 7: No False Discovery Control.**
With 60 data sources and N_choose_2 = 1,770 pairwise combinations, random coincidences are guaranteed. No statistical correction mechanism exists to separate real convergences from noise.

### Important Constraints
- **No LLM in convergence computation.** The pipeline is deterministic. Math only.
- **Must handle missing data.** Not all tools succeed every run. Some APIs go down. Some have lag. Convergence cannot require all 60 channels to be present.
- **Must handle different update frequencies.** Weekly data (CFTC) mixed with hourly (electricity) mixed with event-driven (earthquakes). The temporal algebra must be explicit.
- **Must be computationally tractable.** 60 sources × history depth × pairwise correlations. Cannot blow up to O(n²) on every run. Must be O(n) per source with smart indexing.
- **Must degrade gracefully.** Day 1 with 3 weeks of pipeline data should produce useful output. Don't require 2 years of history.

---

## Risks

### Licensing
- **tigramite:** GPL-3.0. Cannot use code. Must implement transfer entropy and conditional MI from scratch using numpy/scipy. The PCMCI algorithm itself is published academic work (Runge 2018, Runge 2019 — published in Nature Communications). Algorithm is public; implementation must be original.
- **pgmpy:** MIT. Safe to use as dependency.
- **ruptures:** BSD-2. Safe to use as dependency.
- **scipy, numpy, statsmodels:** All BSD. Safe.

### Technical Risks
1. **Spurious convergence (Type I error).** The #1 risk. With 1,770 pairwise combinations tested frequently, we'll see "convergences" that are pure noise. **Mitigation:** BH FDR correction, minimum evidence count thresholds, persistence requirements (convergence must sustain for N periods).
2. **Computational blowup.** Pairwise analysis of 60 sources over T=365 time steps = O(60² × 365) per cycle. Not terrible, but grows fast if we add rolling windows or bootstrap CIs. **Mitigation:** Sparse computation — only compute correlations for sources within the same causal template neighborhood. Not all 1,770 pairs.
3. **Schema evolution.** The 60 tools have no contract guaranteeing schema stability. If a tool changes its `.data` format, the convergence extractor for that tool breaks silently. **Mitigation:** Evidence extractors must have defensive parsing with explicit field validation. Fail loudly (log error, skip source) not silently.
4. **Temporal alignment artifacts.** Resampling weekly data to daily (or vice versa) can create spurious correlations (aliasing, forward-fill bias). **Mitigation:** Always align to the COARSEST frequency in each comparison pair. Never upsample.
5. **Cold start.** Need historical data to establish baselines. Day 1 of pipeline = no baselines = no anomaly detection. **Mitigation:** Graceful degradation with minimum observation thresholds per source. Use parametric baselines (assume normal distribution) until empirical distribution converges (typically ~30 observations).
6. **Survivorship bias in causal templates.** If we hand-code templates based on past crises, we overfit to known patterns and miss novel event types. **Mitigation:** Split approach: hand-coded templates for known dynamics (80% of crises follow known patterns) PLUS a template-free statistical coincidence detector for novel patterns.

### Testing Risks
- Convergence detection is hard to test because real convergences are rare events. **Mitigation:** Synthetic event generators that inject known causal patterns into simulated multi-source data. The test verifies detection and measures false positive rate empirically.
- Integration tests require pipeline_data history. **Mitigation:** Build a `ConvergenceTestHarness` that pre-populates SQLite with realistic synthetic multi-source data.

---

## Data Requirements

### Required Inputs
1. **pipeline_data table rows** — historical tool outputs stored by the pipeline executor, keyed by `(source, fetched_at)`. This is the primary input.
2. **signals table rows** — previously computed atomic signals (if any). May be empty initially.
3. **Tool metadata** — each tool's name, update frequency, expected data schema. This needs to be codified (currently implicit).

### What Already Exists Locally
- Pipeline store with schema and CRUD methods ✓
- 60 tools producing ToolResult with `.data` field ✓
- Daily collection DAG fetching ~6 sources on schedule ✓
- Quant primitives (BOCPD, HMM, spectral, scoring) ✓

### What Still Needs to Be Added
1. **Evidence Protocol** — standardized intermediate representation between raw tool output and convergence input
2. **Signal Catalog** — registry of all extractable signals with metadata (source, frequency, dtype, direction_semantics, null_hypothesis)
3. **Historical pipeline data** — need to run the pipeline for a few weeks to accumulate baselines (or build a synthetic data generator for testing)
4. **Causal template library** — hand-coded domain knowledge about known causal chains

---

## Math/Algorithm Survey

### Layer 1: Evidence Normalization

**Problem:** Convert heterogeneous tool outputs → uniform evidence format.

**Approach: Per-Source Evidence Extractors**

Each tool gets a thin extractor function: `extract_evidence(tool_name, tool_data) → list[Evidence]`

```
Evidence:
  source: str           # Tool name
  signal_id: str        # Unique signal identifier (e.g., "cftc.crude_oil.mm_net_long")
  timestamp: float      # Unix time of observation
  value: float          # Numeric value (or NaN for categorical → encoded)
  direction: int        # +1 (bullish/expansion/stress), -1 (bearish/contraction/relief), 0 (neutral)
  confidence: float     # 0-1, source-quality weighted
  category: str         # Taxonomy bucket (see below)
  tags: list[str]       # Freeform metadata (country, sector, entity)
  ttl: int              # Seconds until this evidence is stale
```

**Signal Taxonomy Categories:**
- `physical_flow` — AIS vessels, transport throughput, energy supply, trade flows
- `physical_disruption` — weather, earthquake, fire, internet outage
- `financial_stress` — sovereign debt, creditor filings, bankruptcy, DeFi flows
- `monetary_policy` — central bank balance, rate monitor, capital flows
- `regulatory_action` — sanctions, drug_regulatory, regulatory_gazette, FOIA
- `behavioral_intent` — patent filings, lobbying, job postings, Wikipedia views, cert transparency
- `positioning` — CFTC, FINRA short, Polymarket whales, insider filings
- `macro_momentum` — PMI, consumer sentiment, building permits, tax receipts
- `biological` — disease surveillance, food security
- `geopolitical` — political risk, GDELT, migration flows
- `supply_chain` — supply chain monitor, interconnection queue, gov contracts

**Why this design:** Categories create neighborhoods for coincidence detection. We don't need to check all 1,770 pairs — we look for cross-CATEGORY convergences first (most informative) then within-category (supporting).

### Layer 2: Temporal Alignment & Resampling

**Problem:** Signals arrive at different cadences (hourly, daily, weekly, monthly, event-driven).

**Approach: Multi-Resolution Time Grid**

Define canonical time grids:
- `intraday` — hourly buckets (electricity, weather, internet outages)
- `daily` — daily close (treasury receipts, most pipeline runs)
- `weekly` — weekly (CFTC, FINRA, energy supply)
- `monthly` — monthly (job postings, PMI, building permits, consumer sentiment)

**Alignment Rules:**
1. For any pairwise comparison, align to the COARSER grid of the two signals. Never upsample.
2. For event-driven signals (earthquake, sanctions), convert to daily event-count or binary-flag-in-window.
3. Use last-observation-carried-forward (LOCF) for missing intermediate values, NEVER interpolation (interpolation leaks future information).
4. Each evidence object carries `ttl` — if the most recent observation is older than ttl, the signal is stale (treated as missing, not as "still the same value").

### Layer 3: Atomic Signal Computation

**Problem:** Convert raw evidence values into standardized anomaly scores.

**Approach: Rolling Z-Score + Empirical CDF**

For each signal stream:
1. Maintain a rolling window of last N observations (N = max(30, 2×period_length))
2. Compute rolling mean μ and std σ
3. z-score: $z_t = \frac{x_t - \mu_t}{\sigma_t}$ (guard: σ < ε → z = 0)
4. Empirical percentile: $p_t = \frac{\text{rank}(x_t)}{N}$
5. **Anomaly flag:** $|z_t| > 2.0$ or $p_t < 0.05$ or $p_t > 0.95$

**Direction normalization:** All signals are oriented so that positive z-scores mean "risk/stress/expansion" and negative mean "calm/contraction/relief." Some tools need sign-flipping (e.g., unemployment claims: higher = more stress = positive z in our convention).

### Layer 4: Pairwise Coincidence Scoring

**Problem:** Detect when two signals are unusually correlated within a time window.

**Approach: Multi-Method Coincidence Scoring**

For a pair of signals (A, B) with aligned observations:

**Method 1 — Rolling Correlation Deviation:**
Compute rolling Pearson correlation over window W. Compare current ρ_t to historical distribution of ρ. If |ρ_t - E[ρ]| > 2σ_ρ, the pair is exhibiting unusual co-movement. Handles both convergence (normally-uncorrelated signals becoming correlated) and divergence (normally-correlated signals breaking apart).

**Method 2 — Joint Exceedance:**
Count instances where both A and B exceed their respective z-score thresholds simultaneously. Under independence: $P(Z_A > 2 \cap Z_B > 2) = P(Z_A > 2) \times P(Z_B > 2) \approx 0.023 \times 0.023 \approx 5.3 \times 10^{-4}$. If observed frequency >> expected frequency, the pair is coinciding.

**Method 3 — Transfer Entropy (Directed):**
$TE_{A \to B}(lag) = H(B_t | B_{t-1:t-k}) - H(B_t | B_{t-1:t-k}, A_{t-lag:t-lag-l})$

Implemented using k-nearest-neighbor entropy estimator (Kraskov et al. 2004). Detects DIRECTED information flow: does A predict B? Asymmetric — TE(A→B) ≠ TE(B→A). The directionality tells us which signal leads.

**Method 4 — Mutual Information (Undirected):**
$MI(A, B) = H(A) + H(B) - H(A, B)$

Also KNN-based. Captures nonlinear dependencies that Pearson misses. High MI + low Pearson = nonlinear relationship worth investigating.

**Method 5 — Concordance Index:**
Simple binary: are A and B moving in the same direction this period? Hit rate over rolling window. Binomial test against H0: p=0.5.

**Which method when:**
- Rolling correlation: first pass, fast, catches linear co-movement
- Joint exceedance: extreme events, tail behavior
- Transfer entropy: directed causality, lag discovery
- Mutual information: nonlinear relationships
- Concordance: directional agreement, simplest signal

### Layer 5: Convergence Event Detection

**Problem:** Combine pairwise scores into multi-source convergence events.

**Approach: Graph-Based Cluster Detection**

1. Build a **coincidence graph** where nodes are signals and edges have weight = coincidence score (any of the 5 methods above, weighted).
2. At each time step, prune edges below a threshold → sparse graph.
3. Detect **connected components** or **dense subgraphs** (cliques) in the coincidence graph.
4. A convergence event = a clique of 3+ signals from 2+ different taxonomy categories, all exhibiting unusual co-movement within the same time window.

**Scoring a convergence event:**

$$\text{ConvergenceScore} = \frac{1}{|C|} \sum_{(i,j) \in C} w_{ij} \times \frac{\text{cross\_category\_count}}{|C|} \times \log_2(|C|)$$

where:
- $|C|$ = number of signals in the clique
- $w_{ij}$ = pairwise coincidence weight
- $\text{cross\_category\_count}$ = number of distinct taxonomy categories represented
- $\log_2(|C|)$ rewards larger convergences superlinearly

**Why graph-based:** It naturally handles the multi-source case without exhaustive enumeration. A clique of 5 signals is discovered via 10 pairwise edges, not $\binom{60}{5}$ combinations.

### Layer 6: Causal Chain Templates

**Problem:** Inject domain knowledge to prioritize known causal patterns over statistical noise.

**Approach: Template Library + Template Matching Score**

A causal chain template is a directed sequence of expected signal activations:

```
Template: "supply_chain_disruption"
  trigger: physical_disruption.* OR regulatory_action.sanctions.*
  within 7d: physical_flow.shipping.* (direction change)
  within 14d: positioning.cftc.* (crowding increase) OR financial_stress.*
  within 30d: macro_momentum.pmi.* (decline) OR supply_chain.*
  expected_amplitude: increasing
```

**Template matching:** When a convergence event is detected, compare it against all templates. If the event matches a template (ordered signals with correct timing and direction), the convergence confidence is boosted:

$$\text{AdjustedScore} = \text{ConvergenceScore} \times (1 + \alpha \times \text{TemplateMatchScore})$$

where α ≈ 0.5 (50% confidence boost for template match).

**Template Library (initial set — 12 core patterns):**

1. **Supply Chain Disruption** — physical_disruption → physical_flow change → positioning shift → price impact
2. **Monetary Policy Shift** — central_bank divergence → capital_flows → financial_stress → positioning
3. **Geopolitical Escalation** — sanctions/political_risk → physical_flow (shipping/flight) → commodity positioning → volatility
4. **Pandemic/Health Crisis** — biological signal → physical_flow (travel) → supply_chain → macro_momentum
5. **Agricultural Shock** — physical_disruption (weather) → food_security → macro_momentum (inflation) → monetary_policy
6. **Energy Crisis** — physical_disruption OR regulatory_action → energy supply → electricity demand → supply_chain → macro
7. **Credit Stress Cascade** — financial_stress (creditor filings/bankruptcy) → positioning (short interest) → sovereign_debt → capital_flows
8. **Tech/Innovation Disruption** — behavioral_intent (patents + hiring) → regulatory_action → positioning
9. **Labor Market Shift** — behavioral_intent (job_postings) → macro_momentum (PMI) → consumer sentiment → financial_stress
10. **Trade War Escalation** — regulatory_action (sanctions) → physical_flow (trade) → positioning (CFTC) → FX/commodity
11. **Real Estate / Construction Cycle** — macro_momentum (building_permits) → financial_stress → monetary_policy → consumer sentiment
12. **Digital Infrastructure Crisis** — physical_disruption (internet outage/BGP) → behavioral_intent (DNS/cert changes) → geopolitical (censorship)

**Template-free detection:** Templates catch known patterns. The graph-based coincidence detector catches unknown patterns. Both systems run in parallel. Novel convergences flagged as "unknown_pattern" for human review / autonomous learning.

### Layer 7: False Discovery Rate Control

**Problem:** With thousands of pairwise tests per cycle, false positives are guaranteed.

**Approach: Multi-Level FDR**

1. **Level 1 — Per-pair:** Each pairwise coincidence test produces a p-value. Apply Benjamini-Hochberg at q=0.05 across all pairs.
2. **Level 2 — Per-event:** Each convergence event (clique) has a combined significance. Apply Fisher's combined probability test: $\chi^2 = -2 \sum_{i} \ln(p_i)$ with $2k$ degrees of freedom.
3. **Level 3 — Persistence filter:** A convergence must persist for ≥ 2 consecutive observation periods to be emitted. Single-period flashes are logged but not promoted to signals.
4. **Level 4 — Minimum cross-category:** A convergence must include signals from ≥ 2 taxonomy categories. Within-category co-movement (e.g., two commodity signals moving together) has a much higher prior probability and lower information content.

#### Step-Local References for 7c-C.3 (FDR Implementation)

**Trusted sources:**
- **Benjamini & Hochberg (1995)** "Controlling the false discovery rate: a practical and powerful approach to multiple testing", *J. R. Statist. Soc. B*, 57(1), 289–300. The BH procedure: sort m p-values p₍₁₎ ≤ … ≤ p₍ₘ₎, find largest k s.t. p₍ₖ₎ ≤ (k/m)·q, reject H₍₁₎…H₍ₖ₎. Controls E[FDP] ≤ q under independence or positive regression dependency (PRDS). Our pairwise tests satisfy PRDS because they share data.
- **Fisher (1925)** "Statistical Methods for Research Workers". Fisher's combined test: $\chi^2 = -2 \sum_{i=1}^{k} \ln(p_i)$, distributed as $\chi^2_{2k}$ under joint null. Combined p-value = $\text{sf}(\chi^2, 2k)$ (scipy.stats.chi2.sf). Guard: clip $p_i$ to [1e-300, 1] before log to avoid -inf.
- **statsmodels.stats.multitest.multipletests**: Implementation used for BH. `method='fdr_bh'` returns `(reject_array, corrected_p, ...)`. BSD-3 licensed.
- **scipy.stats.chi2.sf**: Survival function for chi-squared distribution. Used for Fisher's combined p-value.

**Adjacent concepts considered:**
- Stouffer's Z method (alternative to Fisher's) — simpler (sum of Φ⁻¹(pᵢ)/√k) but assumes equal weight and normal approximation; Fisher's is more standard for heterogeneous p-values.
- Bonferroni — too conservative for ≥100 tests; BH is uniformly more powerful.
- BY (Benjamini-Yekutieli 2001) — controls FDR under arbitrary dependence; more conservative than BH. Unnecessary here since our pairwise tests satisfy PRDS.

**Repo-specific engineering decisions:**
- `persistence_filter` is in-memory state (dict of fingerprint → count). Acceptable for now; SQLite persistence deferred.
- `cross_category_filter` is redundant with graph.py's `min_categories` check but kept as defense-in-depth.
- `apply_all_controls` orchestrates the full pipeline: BH on pairs → rebuild graph from survivors → re-detect cliques → Fisher per clique → persistence → cross-category → emit.

### Layer 8: Convergence Signal Emission

**Problem:** Convert detected convergences into actionable signals for downstream consumption (Phase 8+).

**Output format (stored to pipeline signals table):**

```
ConvergenceSignal:
  signal_name: str       # "convergence.supply_chain_disruption.2026-04-03"
  computed_at: float      # Unix timestamp
  value: float            # ConvergenceScore ∈ [0, 1]
  metadata:
    event_type: str       # Template name or "unknown_pattern"
    signals_involved: list[str]  # Signal IDs in the clique
    categories_involved: list[str]  # Taxonomy categories
    cross_category_count: int
    p_value: float        # Fisher's combined p-value
    persistence_days: int # How many consecutive periods this convergence has been active
    template_match: float # Template match score (0 = no match, 1 = perfect match)
    direction: int        # +1 stress/risk-on, -1 relief/risk-off
    lead_signal: str      # Signal with earliest activation (potential cause)
    lag_signals: list[str] # Signals that responded (potential effects)
    evidence_summary: str  # Machine-readable summary for downstream
```

---

## Architecture Design

### Package Structure

```
agent/convergence/
├── __init__.py
├── evidence.py          # Evidence dataclass + evidence bus
├── extractors.py        # Per-tool evidence extraction functions
├── taxonomy.py          # Signal taxonomy categories + metadata registry
├── alignment.py         # Temporal alignment & resampling
├── atomic_signals.py    # Rolling z-score, percentile, anomaly flag
├── coincidence.py       # Pairwise coincidence scoring (5 methods)
├── graph.py             # Coincidence graph + clique detection
├── templates.py         # Causal chain template library + matcher
├── fdr.py               # FDR control (BH procedure, Fisher's, persistence)
├── detector.py          # Top-level ConvergenceDetector orchestrator
└── signals.py           # ConvergenceSignal emission + storage

agent/pipeline/dags/
├── convergence_detection.py  # DAG: depends on daily_collection → runs convergence
```

### Data Flow

```
                        ┌──────────────────┐
                        │  daily_collection │  (existing DAG)
                        │  DAG — 6+ nodes   │
                        └────────┬─────────┘
                                 │ pipeline_data rows
                                 ▼
                    ┌────────────────────────┐
                    │  convergence_detection  │  (new DAG)
                    │  DAG — 5 nodes          │
                    └────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
     ┌─────────────┐   ┌──────────────┐    ┌──────────────┐
     │   extract    │   │   align &    │    │   coincidence │
     │   evidence   │   │   normalize  │    │   scoring     │
     │   (Layer 1)  │   │   (Layer 2-3)│    │   (Layer 4)   │
     └──────┬──────┘   └──────┬───────┘    └──────┬───────┘
            │                  │                    │
            ▼                  ▼                    ▼
     ┌─────────────────────────────────────────────────────┐
     │            Convergence Graph (Layer 5)               │
     │  nodes = signals, edges = coincidence weights        │
     │  → clique detection → convergence events             │
     └──────────────────────┬──────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼                           ▼
     ┌─────────────────┐        ┌──────────────────┐
     │ Template Match   │        │  FDR Control     │
     │ (Layer 6)        │        │  (Layer 7)       │
     └────────┬────────┘        └────────┬─────────┘
              │                          │
              ▼                          ▼
     ┌──────────────────────────────────────────┐
     │       Signal Emission (Layer 8)          │
     │ → pipeline signals table                  │
     │ → semantic memory facts                   │
     │ → bandit reward (if predictive)           │
     └──────────────────────────────────────────┘
```

### Dependencies (new)

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| pgmpy | >=0.1.23 | Bayesian network structure for world model (Phase 9 prep, use in templates) | MIT |
| ruptures | >=1.1 | Offline batch changepoint detection (PELT) for historical convergence | BSD-2 |
| networkx | >=3.0 | Graph operations (already likely a transitive dep) — clique detection, connected components | BSD-3 |
| statsmodels | >=0.14 | BH FDR correction (`multipletests`), Fisher's method | BSD-3 |

**NOT added:**
- tigramite — GPL, concept only
- pomegranate — evaluate later, pgmpy sufficient for now

### Computational Complexity

| Operation | Complexity | Per-run cost (60 sources, 365 days) |
|-----------|-----------|-------------------------------------|
| Evidence extraction | O(S) per source | 60 extractors × ~10ms = 600ms |
| Temporal alignment | O(S × T) | 60 × 365 = ~22K operations, <100ms |
| Z-score computation | O(S × W) | 60 × 52 (weekly window) = ~3K, <50ms |
| Pairwise coincidence | O(P × T) where P = active pairs | ~200 active pairs × 365 = 73K, ~2s |
| Graph construction + clique | O(P + V) | ~200 edges, 60 vertices, <10ms |
| Template matching | O(E × T_lib) events × templates | ~5 events × 12 templates, <10ms |
| FDR correction | O(P × log P) | ~200 p-values, <1ms |
| **Total per cycle** | | **~3-5 seconds** |

This is extremely fast. Can run every 15 minutes without compute concern.

---

## Implementation Intent

### Concepts Approved for Implementation

1. **Evidence Protocol** (evidence.py) — standardized Evidence dataclass, evidence bus pattern
2. **Per-Tool Evidence Extractors** (extractors.py) — thin extractors for all 60 tools, with defensive parsing
3. **Signal Taxonomy** (taxonomy.py) — 11 categories, signal metadata registry (frequency, direction semantics, staleness TTL)
4. **Temporal Alignment** (alignment.py) — multi-resolution grids, LOCF, staleness logic
5. **Atomic Signal Computation** (atomic_signals.py) — rolling z-score, empirical percentile, anomaly flagging, direction normalization
6. **Pairwise Coincidence Scoring** (coincidence.py) — all 5 methods (rolling correlation, joint exceedance, transfer entropy, mutual information, concordance), with p-value output
7. **Coincidence Graph** (graph.py) — networkx graph construction, edge pruning, clique detection (Bron-Kerbosch or approximate)
8. **Causal Chain Templates** (templates.py) — 12 core templates, declarative pattern language, template matching with within-window and direction constraints
9. **FDR Control** (fdr.py) — BH procedure, Fisher's combined test, persistence filter, cross-category minimum
10. **Convergence Detector** (detector.py) — orchestrator, top-level `detect()` method, configuration
11. **Signal Emission** (signals.py) — ConvergenceSignal output, pipeline store integration
12. **Convergence DAG** (convergence_detection.py) — depends on daily_collection outputs, runs detection

### Concepts Approved but Deferred to Later Sub-Phase
- **Transfer entropy and mutual information:** Computationally heavier, requires more data history. Build rolling correlation + joint exceedance + concordance first. Add TE/MI in a second sub-phase when we have 30+ days of pipeline data.
- **pgmpy Bayesian network integration:** This is Phase 9 (World Model). Phase 7c builds the evidence protocol and coincidence detection. Phase 9 builds the causal graph on top.
- **Offline batch changepoint (ruptures):** Add when historical data accumulates. Initial detection uses online methods only.

### Concepts Rejected
- **LLM-based convergence reasoning:** Violates pipeline determinism constraint. The LLM can EXPLAIN convergences (Phase 7 LLM support) but cannot DETECT them.
- **Full Dempster-Shafer implementation:** Over-engineered for our needs. The evidence protocol with confidence scores and direction handles the "I have evidence for A but no opinion about B" case sufficiently. DS theory adds complexity without proportionate benefit at this stage.
- **Real-time streaming architecture:** Our data sources are batch-oriented (most update daily/weekly). Event-driven streaming (Kafka, Flink pattern) is premature. Batch processing on pipeline schedule is sufficient and much simpler.
- **Copula-based tail dependence:** Computationally expensive, requires large samples for stable estimation. Joint exceedance captures tail co-movement more robustly with less data.

### Notes for the Spec

The spec should decompose into 4 sub-phases:

**Sub-phase 7c-A: Evidence Protocol + Extractors + Taxonomy**
- evidence.py, extractors.py, taxonomy.py
- This is the foundation. Everything else depends on it.
- Test: synthetic tool outputs → evidence extraction → verify schema compliance, direction normalization, completeness

**Sub-phase 7c-B: Temporal Alignment + Atomic Signals**
- alignment.py, atomic_signals.py
- Test: inject multi-frequency synthetic data → verify alignment correctness, z-score computation, anomaly flagging, no future leakage

**Sub-phase 7c-C: Coincidence Detection + Graph + FDR**
- coincidence.py, graph.py, fdr.py
- The core math. Test with synthetic causal scenarios: inject known convergences into noise → verify detection with controlled false positive rate

**Sub-phase 7c-D: Templates + Detector + DAG Integration**
- templates.py, detector.py, signals.py, convergence_detection.py (DAG)
- Integration. Test: end-to-end with synthetic pipeline data → convergence detected → signal emitted → stored in pipeline store

Each sub-phase: implement → test (including edge cases) → mark done → next.

**Estimated total:** ~12-15 atomic implementation steps across 4 sub-phases.

---

## Related

- [[convergence_detection_spec|Spec: Convergence Detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]

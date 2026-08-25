---
title: "Feature: Signal Fusion — Dual-Resolution Entity Micro-Alpha + Macro Regime"
tags:
  - doc/research
  - phase/20
  - topic/signal-fusion
  - topic/entity-anomaly
  - topic/micro-alpha
  - layer/fusion
  - layer/feature-engineering
---

# Feature: Signal Fusion — Dual-Resolution Entity Micro-Alpha + Macro Regime

## Goal

Phase 20 builds the signal fusion layer: the mathematical engine that converts noisy, multi-source, multi-entity observations into actionable probability distributions at **two resolutions**:

1. **Macro-level:** Kalman/particle filter fuses 17 aggregate observations into 3 continuous latent states (stress, macro momentum, liquidity). This is the existing Phase 19 world model update, to be hardened.
2. **Entity-level (micro-alpha):** Per-entity sequential anomaly detection, changepoint monitoring, and event scoring that surfaces *specific* entities exhibiting behavior patterns predictive of material information leakage — the kind of signal that moves individual instruments, not just global regimes.

**Business problem:** The current pipeline produces probability distributions over 3 global regime states. That's useful but commodity-level. The real money is in entity-level micro-signals across ALL domains — finance, tech, geopolitics, supply chains, energy, maritime, health — anywhere our 57 tools observe non-commoditized raw data. Examples:

- **Finance:** A specific company's short-volume ratio spikes while its insider filings cluster → pre-event signal. A specific wallet accumulates governance tokens before a DeFi vote. CFTC positioning shows one managed-money cohort flipping conviction.
- **Geopolitics:** GDELT conflict intensity around a specific country accelerates while sanctions-monitor adds new entities in that country's program. A specific government contractor suddenly wins clustered awards from an agency that historically spread contracts.
- **Supply chain / physical:** A specific vessel deviates from its normal route near a sanctioned port. A specific grid zone's demand-forecast deviation persists for 3+ days (hidden industrial activity or shutdown). Earthquake swarm intensity near a specific semiconductor fab crosses threshold.
- **Tech / cyber:** A specific company's cert-transparency log shows burst of new subdomain registrations (stealth product launch or M&A infra). DNS A-record changes cluster for a specific domain (infrastructure migration before announcement).
- **Health / labor:** Wastewater pathogen concentration spikes in a region where a specific pharma company has trials. Job-posting velocity for a specific company collapses (restructuring signal).
- **Cross-domain convergence:** A specific person files Form 4 (insider) + the company they work for has a vessel reroute + a related wallet shows accumulation — three independent tools, one entity cluster, one signal.

These micro-changes are "big for the people involved" even when globally invisible. The micro-alpha layer must be domain-agnostic: the same CUSUM/BOCPD/Hawkes math applied to ANY entity type, fed by ANY tool.

**Architecture gap (identified this session):** The GNN already computes per-entity 128-dim embeddings and per-entity anomaly scores. But `GNNFeatureBuilder.build()` in [[gnn_pattern_and_finetuning]] averages them to 5 type-level scalar means, destroying all entity resolution before the world model sees them. Phase 20 must preserve entity-level signal through to output.

---

## Search Log

- **Wikipedia keywords searched:** CUSUM change detection, Event Study methodology, Abnormal Return finance, Information Asymmetry economics, Insider Trading detection, Probability of Informed Trading (404), Transfer Entropy, Hawkes Process
- **Documentation keywords searched:** PyOD outlier detection library, PyGOD graph anomaly detection, DOMINANT GCN autoencoder node anomaly, BOCPD Bayesian Online Changepoint Detection
- **arXiv keywords searched:** entity-level anomaly detection financial time series, node-level graph neural network anomaly, micro-alpha detection hedge fund (results were off-topic — video anomaly, ECG adversarial; pivoted to Wikipedia + OSS docs)

## External Repositories Reviewed

- **Repository:** [PyOD](https://github.com/yzhao062/pyod) (v2, 26M+ downloads)
  - Why it is relevant: Comprehensive anomaly detection library with 50+ tabular detectors, 7 time-series detectors, and 8 graph-based detectors. Graph detectors (DOMINANT, CoLA, AnomalyDAE, GUIDE, Radar, ANOMALOUS, CONAD, SCAN) operate on PyG Data objects and produce **per-node anomaly scores** — exactly what we need.
  - Useful implementation idea: DOMINANT (GCN AE, structure + attribute reconstruction, #1 on BOND benchmark) and CoLA (contrastive self-supervised, #2 on BOND) are both node-level anomaly detectors that score each node independently. We already have a GNN producing per-entity embeddings; we could use PyOD's per-node scoring as a reference or direct integration.
  - License: BSD-2-Clause (commercial use OK)
  - Reuse conclusion: **reusable pattern** — can use directly or adapt scoring methodology

- **Repository:** [PyGOD](https://pygod.org/) (dedicated graph outlier detection)
  - Why it is relevant: Focused library for graph-based outlier detection. Referenced by PyOD ecosystem.
  - License: BSD-2-Clause
  - Reuse conclusion: concept only (we already have HetTGN; use their scoring ideas, not their GNN)

## Documentation Reviewed

### 1. CUSUM (Cumulative Sum Control Chart) — E.S. Page, 1954
- **Source:** Wikipedia, "CUSUM"; Biometrika 1954
- **Key formula:** $S_{n+1} = \max(0, S_n + x_{n+1} - \omega)$ where $\omega$ is the target/allowance value
- **What it clarified:** CUSUM is a sequential analysis technique for monitoring change detection. When $S$ exceeds a threshold $h$, a change is declared. Average Run Length (ARL) is the key performance metric. It's **memoryless in the right direction**: the $\max(0, \cdot)$ resets after explaining away normal variation, so only sustained shifts accumulate.
- **Relevance to TirraMind:** Perfect for per-entity monitoring with minimal state. Each entity maintains a scalar CUSUM statistic. No need to store full history. Computational cost: $O(1)$ per entity per update. Can run across thousands of entities.
- **Limitation:** Detects mean shifts, not distributional changes. For richer changepoint detection, combine with BOCPD.

### 2. Event Study Methodology — MacKinlay, 1997
- **Source:** Wikipedia, "Event study"; MacKinlay, J. Financial Economics (1997)
- **Key formula:** $AR_{it} = R_{it} - E(R_{it})$ where expected return uses market model with 120-day estimation window. Cumulative abnormal return: $CAR_{i}(t_1, t_2) = \sum_{t=t_1}^{t_2} AR_{it}$
- **What it clarified:** Standard methodology for measuring whether a corporate event (merger, filing, announcement) produces statistically significant abnormal returns. The estimation window (e.g., days −150 to −30) establishes normal behavior; the event window (e.g., days −5 to +5) measures the abnormal response.
- **Relevance to TirraMind:** Directly applicable. For each entity, we can define "events" (new filing, large transaction, vessel arrival) and measure whether the entity's observational signature is abnormal relative to its own baseline. This is entity-level Event Study applied to non-price data.
- **API to carry forward:** Estimation window → per-entity rolling baseline. Event window → BOCPD/CUSUM alert zone. CAR → cumulative anomaly score.

### 3. Information Asymmetry — Akerlof (1970), Spence (1973), Stiglitz (1981)
- **Source:** Wikipedia, "Information asymmetry"
- **Core framework:** Adverse selection (Akerlof's "Market for Lemons"), signaling (Spence), screening (Stiglitz). In finance: insiders possess material non-public information; SEC Regulation FD attempted to level the field but information still leaks through behavioral traces.
- **Relevance to TirraMind:** This IS the theoretical foundation. TirraMind's edge is detecting information asymmetry *before it resolves into price*. Entity-level behavioral traces (filing timing, transaction size, vessel routing, wallet accumulation patterns) are the observable signatures of informed actors. The micro-alpha layer is a systematic, mathematical implementation of the information-asymmetry detection that investigative journalists do narratively.
- **Key insight:** "AI agents are able to reduce information asymmetry due to their access to big data to make more informed decisions" — Wikipedia. TirraMind operationalizes this across 57 data tools.

### 4. Abnormal Return
- **Source:** Wikipedia, "Abnormal return"
- **Key formula:** $AR_{it} = R_{it} - E(R_{it})$. Standardized: $SAR_{it} = AR_{it} / \hat{\sigma}_i$
- **Relevance:** Applied to entity observational signatures (not just price returns). An entity's "return" is its behavioral activity level; "abnormal return" is deviation from its own baseline. Standardization enables cross-entity comparison.

### 5. Insider Trading Detection Patterns
- **Source:** Wikipedia, "Insider trading"
- **Key patterns:** SEC monitors for unusual trading volume/returns in the days before material announcements. ~50 cases/year prosecuted. Misappropriation theory: trading on any material non-public information, even if you're not a corporate insider.
- **Relevance:** The SEC's detection methodology is the gold standard for entity-level micro-alpha detection. They look for: (a) abnormal returns timed to announcements, (b) unusual option activity before M&A, (c) patterns of trades by connected persons. TirraMind can detect similar patterns across entity types: filing timing, wallet flow timing, vessel routing changes.

### 6. Transfer Entropy — Schreiber, 2000
- **Source:** Wikipedia, "Transfer entropy"; Physical Review Letters 85(2):461-464
- **Key formula:** $T_{X \to Y} = H(Y_t | Y_{t-1:t-L}) - H(Y_t | Y_{t-1:t-L}, X_{t-1:t-L})$
- **What it clarified:** Non-parametric measure of directed information flow between two time series. Reduces to Granger causality for linear/Gaussian cases but handles nonlinear dependencies. Measures how much knowing past X reduces uncertainty about future Y, conditioned on Y's own past.
- **Relevance to TirraMind:** Entity-to-entity information flow detection. If wallet W3 consistently moves before exchange inflows spike, transfer entropy $T_{W3 \to ExchangeFlow}$ will be high. Can identify "leading" entities whose behavior predicts market-relevant outcomes.
- **Limitation:** Requires significant sample size for accurate estimation. Best applied to entity pairs with long observation histories.

### 7. Hawkes Process — Hawkes, 1971
- **Source:** Wikipedia, "Hawkes process"; Biometrika 58(1):83-90; Hawkes (2018) Quant Finance review
- **Key formula:** $\lambda_t = \mu(t) + \sum_{t_k < t} \phi(t - t_k)$ — intensity is baseline + self-exciting contribution from past events
- **What it clarified:** Self-exciting point process: each event increases the probability of future events. The branching ratio $\int_0^\infty \phi(t) dt$ controls whether activity is sub-critical (dies out) or critical (sustained bursts). Multivariate extension: $\lambda_t^i = \mu_i + \sum_j \sum_{t_k^j < t} \phi_{ij}(t - t_k^j)$
- **Relevance to TirraMind:** Models event clustering for entities. If a company receives multiple anomalous signals (insider filing + supply chain disruption + unusual vessel routing) in quick succession, the Hawkes intensity spikes — indicating a "hot" entity. The multivariate extension captures cross-entity contagion: activity at entity A exciting activity at entity B.
- **Application in finance:** Hawkes (2018) review shows widespread use in mathematical finance for order arrival modeling, market microstructure, and contagion. TirraMind applies the same framework to entity-level event streams.

### 8. BOCPD (Bayesian Online Changepoint Detection) — Adams & MacKay, 2007
- **Source:** Already implemented in `agent/quant/changepoint.py`
- **Key feature:** Conjugate prior (Normal-Inverse-Gamma), Student-t predictive distribution, online computation with $O(T)$ memory (can be bounded).
- **Relevance:** Apply per-entity to detect when an entity's behavioral signature undergoes a structural break (e.g., insider switches from selling to buying, wallet shifts from HODLing to active trading, vessel changes its regular route, grid zone demand deviates from forecast).

---

## Micro-Alpha Pattern Taxonomy (All Domains)

The entity scoring layer is **domain-agnostic** — the same math (CUSUM, BOCPD, Hawkes, Event Study baselines) runs on ANY entity type. What varies is the **observation signal** fed into the scorer. This section maps every major domain to the entity-level patterns our tools can detect.

### Domain 1: Corporate / Equity (Entity types: company, person)

| Tool | Entity | Micro-Pattern | Signal Type |
|------|--------|---------------|-------------|
| **insider_filings** | person, company | Insider buying/selling cluster (3+ insiders in 14d) | Filing burst → Hawkes |
| **form144** | person, company | Sell-intent pre-registration timing anomaly | Baseline deviation → Event Study |
| **finra_short_volume** | company | Short-volume ratio spike per ticker; dark pool concentration | Persistent shift → CUSUM |
| **patent_filings** | company | Patent velocity acceleration per assignee; CPC class pivot | Regime change → BOCPD |
| **creditor_filings** | company | UCC lien filing cluster; collateral pledge acceleration | Filing burst → Hawkes |
| **bankruptcy_court** | company | Chapter 11 filing + related creditor cluster | Event → CUSUM + Hawkes cross-entity |
| **cert_transparency** | company/domain | Subdomain burst (stealth launch, M&A infra prep) | Burst → Hawkes |
| **dns_monitor** | company/domain | A-record/NS migration (infra change before announcement) | Regime change → BOCPD |
| **job_postings** | company | Hiring velocity collapse or surge per company | Persistent shift → CUSUM |
| **lobbying** | company | Lobbying spend acceleration on new issue codes | Burst → Hawkes |
| **academic_preprints** | company/institution | Corporate lab paper surge on specific topic | Burst → Hawkes |

**Cross-signal convergence example:** Company X has (a) insider buying cluster from insider_filings, (b) patent velocity acceleration from patent_filings, (c) new subdomain burst from cert_transparency → all within 14 days → Hawkes cross-entity intensity spikes → "hot company" alert.

### Domain 2: Blockchain / DeFi (Entity types: wallet, protocol)

| Tool | Entity | Micro-Pattern | Signal Type |
|------|--------|---------------|-------------|
| **whale_alert** | wallet | Large BTC movement timing; custody↔exchange flow | Burst → Hawkes |
| **polymarket_whales** | wallet/trader | Whale accuracy trajectory; contrarian conviction bet | Regime change → BOCPD |
| **defi_flows** | protocol | TVL drain velocity (pre-exploit); stablecoin mint/burn cycle | Persistent shift → CUSUM |

**Cross-signal convergence example:** Wallet W accumulates governance tokens (whale_alert) + same wallet cluster bets on related Polymarket outcome (polymarket_whales) → Hawkes multivariate cross-entity intensity spikes → "informed wallet" alert.

### Domain 3: Geopolitics / Sanctions (Entity types: country, person, org)

| Tool | Entity | Micro-Pattern | Signal Type |
|------|--------|---------------|-------------|
| **gdelt** | country, org, person | Conflict intensity acceleration per actor; cooperation→conflict pivot | BOCPD regime shift |
| **sanctions_monitor** | entity, country | New entity listings per program; program scope expansion | Burst → Hawkes |
| **political_risk** | candidate, PAC | Campaign cash-on-hand drop; opposition spending surge | Persistent shift → CUSUM |
| **comtrade** | country, commodity | Bilateral trade flow collapse; mirror-trade asymmetry (smuggling) | BOCPD structural break |
| **migration_flows** | country | Displacement stock surge >20% YoY; asylum acceptance collapse | Persistent shift → CUSUM |
| **foia_requests** | entity, agency | Multi-agency FOIA clustering on one entity | Burst → Hawkes |

**Cross-signal convergence example:** GDELT conflict intensity for Country Y escalates (BOCPD) + sanctions_monitor adds 5 new entities from Y's regime (Hawkes burst) + comtrade shows bilateral trade collapse with Y (CUSUM) → "geopolitical escalation" entity-cluster alert.

### Domain 4: Supply Chain / Maritime / Physical (Entity types: vessel, port, facility)

| Tool | Entity | Micro-Pattern | Signal Type |
|------|--------|---------------|-------------|
| **ais_vessel** | vessel (MMSI) | Route deviation from historical pattern; unusual port call | BOCPD regime shift |
| **supply_chain_monitor** | sector, commodity | PPI acceleration by sector → margin squeeze per affected company | Persistent shift → CUSUM |
| **earthquake_proximity** | facility/infrastructure | Seismic swarm near semiconductor fab / mine / port | Burst → Hawkes |
| **weather_alerts** | region/facility | Severe alert clustering near critical infrastructure | Burst → Hawkes |
| **satellite_activity** | infrastructure | Fire radiative power near industrial site; NDVI crop decline | Regime change → BOCPD |
| **transportation_throughput** | border port | Truck/container volume MoM collapse at specific crossing | Persistent shift → CUSUM |

**Cross-signal convergence example:** Vessel V deviates to sanctioned port (ais_vessel BOCPD) + sanctions_monitor adds the port operator to SDN list (Hawkes) + the vessel's owning company has creditor filing cluster (creditor_filings Hawkes) → "sanctions evasion" cross-domain alert.

### Domain 5: Energy / Utilities (Entity types: grid zone, balancing authority, fuel type)

| Tool | Entity | Micro-Pattern | Signal Type |
|------|--------|---------------|-------------|
| **power_grid** | grid zone | Demand-forecast deviation persistence (hidden economic activity) | CUSUM persistent shift |
| **electricity_monitor** | balancing authority | Cross-region interchange flow reversal; renewable proportion collapse | BOCPD regime shift |
| **energy_supply** | commodity | 3+ week inventory decline streak; SPR drawdown acceleration | CUSUM persistent shift |
| **interconnection_queue** | company/project | Data center project pattern clustering; MW cancellation rate spike | Hawkes burst |

### Domain 6: Health / Pharma (Entity types: pathogen, drug, facility)

| Tool | Entity | Micro-Pattern | Signal Type |
|------|--------|---------------|-------------|
| **disease_surveillance** | pathogen, region | Wastewater PCR concentration spike; novel pathogen first detection | BOCPD changepoint |
| **drug_regulatory** | drug/company | FDA priority review acceleration; adverse event severity cluster | Hawkes burst |

### Domain 7: Fiscal / Sovereign (Entity types: country, central bank)

| Tool | Entity | Micro-Pattern | Signal Type |
|------|--------|---------------|-------------|
| **sovereign_debt** | country | Yield curve inversion per country; cross-country spread widening | BOCPD structural break |
| **capital_flows** | country | Foreign holdings selloff velocity; coordinated multi-country selling | CUSUM persistent shift |
| **central_bank_balance** | central bank | Policy divergence (expansion vs contraction); QE/QT pacing shift | BOCPD regime shift |
| **treasury_receipts** | fiscal category | Tax receipt category volatility; TGA balance swing persistence | CUSUM persistent shift |
| **global_pmi** | country | CLI turning point; BCI-CLI divergence per country | BOCPD changepoint |

### Cross-Sector Entity Convergence (L3 — The Moat)

**Individual entity anomalies are table stakes. The real alpha is in cross-sector convergence: when signals from unrelated data domains co-fire on entities linked through the entity graph within a tight time window.** No single tool can detect these. No single-sector analysis can surface them. The convergence itself IS the signal — it emerges only because our entity linking infrastructure connects people→companies→wallets→vessels→countries→domains→facilities into one temporal graph.

This section describes the **named convergence archetypes** — repeatable cross-sector patterns that the entity scorer must be architecturally capable of detecting. Each archetype specifies the tools involved, the entity link path that connects them, the expected temporal window, and the actionable signal.

#### Archetype 1: Corporate Pre-Announcement Cluster
**Sectors combined:** Corporate filings × Tech infrastructure × IP × Labor
**Why it matters:** Before a major announcement (M&A, product launch, strategic pivot), organizations cannot avoid leaving traces across multiple data domains simultaneously. No single trace is conclusive; the cluster is.

| Signal Source | Entity | Observable |
|---------------|--------|------------|
| insider_filings | person → works_for → company | Buying cluster: 3+ insiders in 14d |
| patent_filings | company | Patent velocity acceleration by CPC class |
| cert_transparency | domain → operated_by → company | Subdomain burst (new infra being stood up) |
| job_postings | company | Hiring surge in specific function (ML, legal, ops) |
| interconnection_queue | company/project | Data center MW application filed |
| lobbying | company | New issue-code lobbying spend |
| academic_preprints | institution → affiliated_with → company | Pre-print surge on specific topic |

**Entity link path:** `person --works_for--> company --operates--> domain --registered_in--> interconnection_queue`
**Temporal window:** 7–30 days
**Signal:** High confidence of imminent corporate event (product launch, M&A, restructuring)
**Why nobody else sees this:** Requires SEC filings + CT logs + USPTO data + Indeed/LinkedIn + FERC + LA filings in one temporal join. No vendor aggregates these.

#### Archetype 2: Sovereign Stress Cascade
**Sectors combined:** Geopolitics × Trade × Finance × Migration × Maritime
**Why it matters:** Country-level crises unfold through a predictable sequence across domains. GDELT conflict precedes sanctions, sanctions precede capital flight, capital flight precedes trade collapse, trade collapse precedes migration. Each stage arrives in a different data domain.

| Signal Source | Entity | Observable |
|---------------|--------|------------|
| gdelt | country | Conflict intensity acceleration (Goldstein scale) |
| sanctions_monitor | person/org → located_in → country | New sanctions listings per program |
| capital_flows | country | Foreign holdings selloff velocity |
| sovereign_debt | country | Yield curve inversion; spread widening |
| comtrade | country × country | Bilateral trade flow collapse |
| migration_flows | country | Displacement stock surge >20% YoY |
| ais_vessel | vessel → flagged_by → country | Fleet rerouting away from country ports |

**Entity link path:** `country <--located_in-- entity <--sanctioned_by-- program; country --bilateral_trade--> country; vessel --flagged_by--> country`
**Temporal window:** 14–90 days (cascade unfolds over weeks)
**Signal:** Country approaching a geopolitical/economic inflection — affects sovereign debt, commodity routes, companies with exposure
**Why nobody else sees this:** Requires GDELT + OFAC + Treasury TIC + UN Comtrade + UNHCR + AIS in one entity graph. Individual domain analysts see their slice; nobody sees the cascade.

#### Archetype 3: Supply Chain Disruption Propagation
**Sectors combined:** Physical world × Maritime × Energy × Corporate
**Why it matters:** Physical disruptions propagate through supply chains in predictable but slow sequences. The first signal comes from physical sensors (earthquake, weather, satellite); the market-relevant signal arrives days later as supply chains reroute.

| Signal Source | Entity | Observable |
|---------------|--------|------------|
| earthquake_proximity | facility/region | Seismic activity near critical infrastructure |
| weather_alerts | region/facility | Severe weather clustering near ports/fabs |
| satellite_activity | facility | Fire radiative power near industrial site; NDVI crop decline |
| ais_vessel | vessel → delivers_to → facility | Route deviation; port dwell time increase |
| supply_chain_monitor | sector/commodity | PPI acceleration by sector |
| power_grid | grid_zone → serves → facility | Demand collapse (plant shutdown) |
| transportation_throughput | border_port | Container volume MoM collapse |

**Entity link path:** `earthquake/weather --affects--> facility <--delivers_to-- vessel <--operates-- company; facility --located_in--> grid_zone`
**Temporal window:** 1–14 days (physical event → logistics impact → price impact)
**Signal:** Specific supply chain node disrupted → companies dependent on that node face margin squeeze, delay, or forced substitute sourcing
**Why nobody else sees this:** Requires USGS + NWS + Sentinel/VIIRS + AIS + BLS PPI + EIA + CBP in one graph. Quant funds use price; we detect the cause before the price moves.

#### Archetype 4: Crypto-to-Real-Economy Bridge
**Sectors combined:** DeFi × Prediction markets × Corporate filings × Geopolitics
**Why it matters:** Crypto-native actors (whales, protocols) increasingly intersect with real-economy events (regulation, M&A, elections). Wallet behavior on-chain can precede real-world event resolution.

| Signal Source | Entity | Observable |
|---------------|--------|------------|
| whale_alert | wallet | Large transfer timing; custody→exchange flow |
| defi_flows | protocol | TVL drain velocity pre-exploit; stablecoin mint/burn |
| polymarket_whales | trader/wallet | Whale accuracy trajectory; contrarian conviction bet |
| regulatory_gazette | agency → regulates → protocol | New crypto-related rulemaking |
| sanctions_monitor | wallet → owned_by → person/org | Sanctioned wallet activity |
| insider_filings | person → works_for → company → exposed_to → protocol | Insider trading at crypto-adjacent company |

**Entity link path:** `wallet --owned_by--> person/org --works_for--> company; wallet --interacts_with--> protocol <--regulates-- agency`
**Temporal window:** 1–7 days (crypto moves fast)
**Signal:** Informed crypto actor positioning ahead of real-world event (regulation, exploit, governance vote, corporate announcement)
**Why nobody else sees this:** Requires on-chain analytics + Polymarket API + SEC EDGAR + OFAC in one graph. Crypto-native firms miss the TradFi side; TradFi firms miss the on-chain side.

#### Archetype 5: Energy-to-Macro Leading Indicator
**Sectors combined:** Energy × Infrastructure × Labor × Fiscal
**Why it matters:** Real-time electricity demand is one of the few true real-time economic indicators. Grid demand deviations precede official economic statistics by weeks to months. Interconnection queue applications reveal capex intentions before earnings calls.

| Signal Source | Entity | Observable |
|---------------|--------|------------|
| power_grid | grid_zone | Demand-forecast deviation persistence |
| electricity_monitor | balancing_authority | Cross-region interchange flow reversal |
| interconnection_queue | company/project | MW application clustering by region; cancellation rate |
| building_permits | region | Permit-to-start ratio; commercial vs residential mix shift |
| job_postings | company → located_in → region | Hiring velocity correlated with grid demand |
| treasury_receipts | fiscal_category | Tax receipt category volatility (withholding = employment) |

**Entity link path:** `grid_zone --serves--> region <--located_in-- company; company --applied_for--> interconnection_queue; region --generates--> tax_receipts`
**Temporal window:** 7–60 days (leading indicator window)
**Signal:** Regional economic acceleration or contraction detectable through energy signature weeks before official data
**Why nobody else sees this:** Requires ISO/RTO data + FERC + Census + Indeed + Treasury Daily in one graph. Macro funds use survey data (lagging); we use physical consumption (leading).

#### Archetype 6: Regulatory Cascade → Affected Entity Mapping
**Sectors combined:** Regulatory × Lobbying × Government contracts × Corporate
**Why it matters:** Policy changes don't arrive suddenly — they progress through regulatory stages (proposed rule → comment period → final rule) while affected companies simultaneously adjust lobbying, contract positioning, and hiring.

| Signal Source | Entity | Observable |
|---------------|--------|------------|
| regulatory_gazette | agency, rule_id | New proposed/final rules by agency |
| lobbying | company/org → lobbies_on → issue_code | Spending acceleration on affected issue codes |
| gov_contracts | company | Contract wins clustering with regulatory agency |
| political_risk | candidate/PAC | Campaign spending correlated with regulatory outcome |
| foia_requests | entity, agency | Multi-agency FOIA clustering on one entity |
| job_postings | company | Compliance hiring surge (GRC, legal) |

**Entity link path:** `agency --proposes--> rule --affects--> sector <--operates_in-- company --lobbies--> issue_code; company --contracts_with--> agency`
**Temporal window:** 30–120 days (regulatory cycle)
**Signal:** Specific companies are positioning for an imminent regulatory change — their exposure (positive or negative) will resolve on the final rule date
**Why nobody else sees this:** Requires Federal Register + Senate LDA + USASpending + FEC + FOIA logs in one graph. Policy analysts read text; we detect the behavioral response to text.

#### Archetype 7: Health-to-Economic Cascade
**Sectors combined:** Health/disease × Pharma × IP × Supply chain × Labor
**Why it matters:** Disease events propagate through multiple economic channels: pharma pipeline, supply chain disruption, labor force impact, regional economic slowdown. COVID proved this at global scale; smaller, more frequent events (regional outbreaks, drug safety signals, clinical trial results) create entity-level micro-alpha.

| Signal Source | Entity | Observable |
|---------------|--------|------------|
| disease_surveillance | pathogen, region | Wastewater PCR spike; novel pathogen detection |
| drug_regulatory | drug → developed_by → company | FDA priority review; adverse event cluster |
| patent_filings | company | Patent filing on alternative treatment/vaccine platform |
| supply_chain_monitor | sector | Medical supply PPI acceleration |
| job_postings | company | Pharma hiring surge in specific therapeutic area |
| clinical_trials | drug, company | Phase transition acceleration or termination |

**Entity link path:** `pathogen --treated_by--> drug --developed_by--> company; pathogen --affects--> region --disrupts--> supply_chain`
**Temporal window:** 7–60 days (health data lags reality by ~7d; pharma response takes weeks)
**Signal:** Pharma company with pipeline exposure to an emerging pathogen or drug safety signal — valuation impact before market prices it
**Why nobody else sees this:** Requires CDC NWSS + FDA FAERS + USPTO + BLS + ClinicalTrials.gov in one graph. Health analysts don't watch patents; pharma analysts don't watch wastewater.

#### Archetype 8: Capital Structure Stress → Insolvency Detection
**Sectors combined:** Corporate credit × Filings × Maritime × Insider behavior
**Why it matters:** Corporate insolvency doesn't happen overnight. The distress sequence — creditor filings cluster, insider selling, short volume spike, vessel rerouting (for physical companies), hiring freeze — plays out across domains weeks before public bankruptcy.

| Signal Source | Entity | Observable |
|---------------|--------|------------|
| creditor_filings | company | UCC filing cluster; collateral pledge acceleration |
| bankruptcy_court | company | Chapter 11 filing; related creditor actions |
| finra_short_volume | company | Short-volume ratio persistent elevation |
| insider_filings | person → works_for → company | Insider selling cluster (smart money exiting) |
| ais_vessel | vessel → operated_by → company | Fleet activity decline (fewer voyages, longer dwell) |
| job_postings | company | Hiring velocity collapse; key function de-listings |
| lobbying | company | Lobbying spend collapse (cash conservation) |

**Entity link path:** `person --works_for--> company <--lien_filed_by-- creditor; company --operates--> vessel; company --spends_on--> lobbying`
**Temporal window:** 14–90 days (distress unfolds over weeks)
**Signal:** Company approaching insolvency — creditors, insiders, and operations all show stress before public disclosure
**Why nobody else sees this:** Requires UCC + PACER + FINRA + SEC + AIS + Indeed + Senate LDA in one graph. Credit analysts use financial statements (backward-looking); we detect the behavioral exhaust of distress (real-time).

#### Archetype 9: Trade War / Tariff Escalation Dynamics
**Sectors combined:** Trade × Geopolitics × Maritime × Corporate × Energy
**Why it matters:** Trade conflicts between countries manifest as a cascade of observable signals across multiple domains before becoming priced. Tariff announcements move markets, but the preparation for tariffs — rerouted trade flows, pre-shipment hoarding, lobbying spend spikes — is detectable.

| Signal Source | Entity | Observable |
|---------------|--------|------------|
| comtrade | country × country × commodity | Bilateral trade flow deviation from 12-month baseline |
| gdelt | country × country | Diplomatic tension acceleration per dyad |
| sanctions_monitor | entity → linked_to → country | Program scope expansion targeting one country |
| ais_vessel | vessel | Rerouting to bypass expected tariff origins |
| supply_chain_monitor | sector/commodity | PPI divergence in affected sector |
| lobbying | company/sector | Spending surge on trade-related issue codes |
| capital_flows | country | FDI flow reversal between affected countries |

**Entity link path:** `country_A --bilateral_trade--> country_B; company --imports_from--> country_A --conflict_with--> country_B; vessel --carries--> commodity`
**Temporal window:** 7–30 days (pre-announcement positioning)
**Signal:** Specific sectors/companies are about to be hit by tariff escalation — trade flows already adjusting before announcement
**Why nobody else sees this:** Requires UN Comtrade + GDELT + OFAC + AIS + BLS + LDA + Treasury in one graph. Trade analysts use monthly data; we detect daily behavioral shifts.

#### Archetype 10: Stealth Technology Deployment
**Sectors combined:** Tech infrastructure × IP × Energy × Labor × Corporate
**Why it matters:** Large technology deployments (data centers, cloud regions, AI training clusters) leave observable traces across infrastructure, energy, and corporate domains months before public announcement. The capex is hidden in aggregate; the entity-level traces are specific.

| Signal Source | Entity | Observable |
|---------------|--------|------------|
| cert_transparency | domain → operated_by → company | New subdomain burst in cloud/infra namespace |
| dns_monitor | domain | A-record changes to new IP ranges; NS migration |
| interconnection_queue | company/project | MW application clustering in specific region |
| power_grid | grid_zone | Persistent demand-forecast overshoot in specific zone |
| building_permits | region | Commercial permit surge with specific company as applicant |
| patent_filings | company | Patent velocity on specific CPC class (AI/ML, networking) |
| job_postings | company | Hiring surge at specific location for infra roles |
| academic_preprints | institution | Research output shift toward deployment topics |

**Entity link path:** `company --operates--> domain; company --applied_for--> interconnection_queue --located_in--> grid_zone; company --builds_in--> region`
**Temporal window:** 30–120 days (infra buildout is slow)
**Signal:** Specific company deploying major infrastructure — capex implications, competitive dynamics, real estate/energy impact in the region
**Why nobody else sees this:** Requires CT logs + FERC + ISO/RTO + Census + USPTO + Indeed + arXiv in one graph. Tech analysts follow earnings calls; we detect the physical footprint.

#### Summary: Cross-Sector Convergence Architecture Requirements

The 10 archetypes above share a common structure:

1. **Multiple independent tools** fire on entities that are connected through `entity_links`
2. **The time window is bounded** — signals must co-occur within an archetype-specific window (1–120 days)
3. **No single signal is sufficient** — the convergence creates the alpha, not the individual anomaly
4. **The entity link path is the detection mechanism** — given an elevated entity, traverse entity_links to check if linked entities are also elevated
5. **The pattern is learned, not hard-coded** — the 10 archetypes describe the STRUCTURE of what to detect, not fixed rules. The Hawkes multivariate cross-entity excitation + entity_links temporal join implements this generically.

**Architectural implication for Phase 20:** The entity scorer must, after scoring individual entities, perform a **convergence pass**: for each elevated entity, query entity_links to find linked entities, check their statuses, and produce a `ConvergenceCluster` when multiple linked entities in different domains co-fire within the temporal window. This is not a Phase 21 add-on — it is the core value proposition and must be designed into the scorer from day one.

**Implementation approach:**
- After individual entity scoring produces EntityAlerts, the scorer traverses entity_links for all elevated/critical entities
- For each elevated entity, query linked entities and check their alert status
- When 2+ linked entities from different source domains are elevated within the temporal window → produce a ConvergenceCluster
- The ConvergenceCluster captures: the trigger entity, linked entities, the link types, the temporal span, the contributing tools, and a convergence score (how many independent domains are co-firing)
- Hawkes multivariate extension provides the mathematical foundation: cross-entity excitation means activity at entity A increases expected intensity at linked entity B

**The entity scorer must be type-agnostic.** The same CUSUM/BOCPD/Hawkes/Event Study math applies to a person, company, wallet, vessel, country, or grid zone. What changes is the observation time series and the convergence patterns. The Hawkes multivariate extension models cross-entity excitation: an event at entity A (any type) can excite intensity at entity B (any type) if they share an entity_link.

---

## Surveillance Surface Audit: Entity Coverage

**~24 of 57+ tools currently write entity observations.** The remaining ~33 tools produce aggregate signals without entity resolution. The entity scorer works with whatever tools write to `entity_observations` — as more tools get L2 upgrades (per the GNN-guided expansion doctrine), the entity scorer automatically gains more signal without code changes.

**Entity types registered (5):** person, company, wallet, country, vessel

**Entity types that SHOULD exist (from tool audit):**
- **domain** — cert_transparency, dns_monitor already produce domain-level observations
- **protocol** — defi_flows produces protocol-level TVL data
- **grid_zone** — power_grid produces zone-level demand data
- **pathogen** — disease_surveillance produces pathogen-level concentration data
- **drug** — drug_regulatory produces per-drug approval events
- **facility/infrastructure** — earthquake_proximity, satellite_activity produce location-level data

These additional entity types are candidates for Phase 20's entity scorer OR future L2 upgrades. The scorer should be designed to handle arbitrary entity types, not just the current 5.

---

## Current Architecture

### Entity Data Flow (Audit Results)

**Entity types:** person, company, wallet, country, vessel

**Storage:** PipelineStore (SQLite) has four entity tables:
- `entities` — canonical entity records (entity_id, type, name, metadata)
- `entity_aliases` — source-specific ID mappings (enables cross-source resolution)
- `entity_observations` — per-entity timestamped observation payloads (indexed by entity_id, source_tool, observed_at)
- `entity_links` — typed cross-entity relationships (works_for, transacts_with, event_involves, etc.)

**Entity APIs available:**
- `register_entity()`, `add_entity_alias()`, `resolve_entity()`, `get_entity()`
- `query_entity_observations(entity_id, source_tool, since, until, depth_level, limit)`
- `link_entities()`, `query_entity_links()`, `query_co_occurrences()`
- `query_all_entities()`, `query_all_entity_links()`

**GNN inference DAG:** Builds HetTGN graph from entities + links → produces per-entity 128-dim embeddings + `id_map` (entity_id → embedding index).

### The Bottleneck: GNNFeatureBuilder

In `agent/features/gnn_builder.py`, the `build()` method:
1. Gets per-entity embeddings from GNN: shape `(n_entities, 128)` per type
2. Computes centroid per type: `centroid = emb.mean(dim=0)`
3. Computes mean deviation from centroid: `scalar` (one number per type)
4. Outputs 11 scalar features total (5 anomaly + 5 activity + 1 cross-entity)

**What is lost:** Per-entity anomaly scores, per-entity activity, per-entity stress contribution, individual embeddings, graph centrality.

### Downstream Consumers

- **Bayesian DAG:** 20 nodes (3 latent + 17 observed). GNN features feed obs nodes 10–20. All type-level aggregates.
- **Kalman filter:** obs_dim=17, state_dim=3. Hand-coded H matrix. GNN aggregates at indices 6–16.
- **BeliefState output:** `variable_name` only — no `entity_id` field. All beliefs are variable-level.

### Existing Capabilities (Available for Integration)

- **BOCPD:** Fully implemented, tested, but not integrated into pipeline for per-entity use.
- **Entity linking:** Phase 17 complete. 8 link types available. Graph structure ready.
- **Co-occurrence queries:** `query_co_occurrences()` can find temporally coincident entity events.

---

## Observations

### What Already Exists
- Robust entity storage infrastructure (Phase 17)
- Per-entity GNN embeddings (Phase 12, 14-15, 19)
- BOCPD changepoint detection (Phase 7c)
- ~24 tools writing entity-level observations across finance, geopolitics, maritime, crypto, health, infrastructure
- Entity link graph schema (8 link types)
- Convergence detection (Phase 7c) for macro-level

### What Is Missing
1. **Per-entity feature output:** GNNFeatureBuilder destroys entity resolution
2. **Per-entity baselines:** Only type centroid exists; no individual entity history tracking
3. **Per-entity alerting/scoring:** No mechanism to surface "entity X is anomalous"
4. **Entity-to-instrument mapping:** No link from entity anomaly to tradeable outcome
5. **Sequential per-entity monitoring:** CUSUM/BOCPD not applied per entity
6. **Event clustering detection:** No Hawkes-process-style burst detection per entity
7. **Entity-level belief output:** BeliefState has no entity_id field
8. **Transfer entropy between entities:** No information-flow measurement
9. **Cross-domain convergence detection:** No mechanism to detect multi-tool, multi-entity signal clusters (the L3 moat)
10. **Additional entity types:** domain, protocol, grid_zone, pathogen, drug — all have entity-level data but no entity registration

### Important Constraints
- Kalman filter requires fixed obs_dim — can't dynamically expand per entity
- DAG complexity explodes if per-entity nodes are added (CPD learning sparse)
- Number of entities varies over time as new ones are discovered
- Must preserve backward compatibility with existing aggregate beliefs

---

## Risks

### Technical Risks
- **Scalability:** Per-entity monitoring at O(N entities × T timesteps) — need to bound computation. Solution: only monitor entities with sufficient observation history (minimum 20 observations).
- **Sparsity:** Many entities have few observations. BOCPD and CUSUM need reasonable sample sizes. Solution: Use adaptive thresholds per entity; fall back to type-level baseline when entity-specific history is insufficient.
- **Kalman dimensionality:** Cannot expand obs_dim per entity. Solution: Keep macro Kalman unchanged; entity scoring is a parallel track.
- **False positives:** Entity-level anomaly without context is noise. Solution: Combine multiple evidence channels (CUSUM alert + BOCPD changepoint + observation recency + Hawkes intensity) into composite score.

### Testing Risks
- Per-entity scoring needs synthetic entities with known anomaly patterns for validation
- Edge cases: entity with 0 observations, entity with 1 observation, entity observed only by 1 tool

### Licensing/Reuse Risks
- PyOD: BSD-2-Clause — safe for commercial use
- All referenced algorithms are from public academic papers — no licensing concern

---

## Data Requirements

### Required Inputs
- Per-entity GNN embeddings (already produced by GNN inference DAG)
- Per-entity observation time series (already in `entity_observations` table)
- Entity link graph (already in `entity_links` table)
- Macro-level regime beliefs (already produced by world model update DAG)

### What Already Exists Locally
- 5 entity types with registration + aliasing (person, company, wallet, country, vessel)
- ~24 tools writing entity observations across all domains
- HetTGN producing 128-dim per-entity embeddings
- BOCPD implementation in `agent/quant/changepoint.py`
- Entity linking with 8 link types (works_for, transacts_with, event_involves, etc.)
- Co-occurrence query API for temporal correlation across entities

### What Still Needs to Be Added
- Per-entity CUSUM state tracking (new) — type-agnostic, works for any entity
- Per-entity rolling baseline computation (new) — per-entity history, not type centroid
- Hawkes intensity estimator (new) — models event clustering per entity AND cross-entity
- Entity anomaly scoring function (new) — domain-agnostic composite scorer
- Entity alert protocol (new data structure) — EntityAlert works for any entity type
- Cross-entity convergence detection — multi-tool signal cluster identification (L3)

---

## Math/Algorithm Survey

### Approach 1: Per-Entity CUSUM Monitoring (Recommended — Primary Alert)

**What it is:** Sequential cumulative sum statistic per entity, applied to the entity's GNN anomaly score time series.

**Formulation:**
$$S_{n+1}^{(e)} = \max\left(0, S_n^{(e)} + z_n^{(e)} - k\right)$$

where $z_n^{(e)}$ is entity $e$'s standardized anomaly score at time $n$, and $k$ is the allowance parameter (typically 0.5σ). Alert when $S^{(e)} > h$ (threshold, calibrated for desired ARL).

**Complexity:** $O(1)$ per entity per update. State: 1 scalar per entity.
**Trusted source:** Page (1954), Biometrika. Standard in SPC literature.
**Why preferred:** Lightest possible per-entity monitor. Can scale to thousands of entities. Detects persistent mean shifts with statistical guarantees (ARL control).

### Approach 2: Per-Entity BOCPD (Recommended — Changepoint Depth)

**What it is:** Apply existing BOCPD implementation to each entity's observation time series to detect structural breaks.

**Formulation:** Already implemented. Run-length posterior $P(r_t | x_{1:t})$ via message passing with NIG conjugate prior.

**Complexity:** $O(T)$ per entity per update (bounded if max run length capped).
**Trusted source:** Adams & MacKay (2007), "Bayesian Online Changepoint Detection." arXiv:0710.3742.
**Why preferred:** Richer than CUSUM — detects distributional changes, not just mean shifts. Already implemented and tested.
**When to use:** After CUSUM triggers an alert, run BOCPD on the entity's recent history for confirmation and characterization.

### Approach 3: Hawkes Process Entity Event Intensity (Recommended — Burst Detection)

**What it is:** Model each entity's event stream as a self-exciting point process. Estimate instantaneous intensity $\lambda_t^{(e)}$ to detect event clustering / bursts.

**Formulation:**
$$\lambda_t^{(e)} = \mu^{(e)} + \sum_{t_k^{(e)} < t} \alpha \cdot e^{-\beta(t - t_k^{(e)})}$$

where $\mu^{(e)}$ is baseline rate, $\alpha$ is excitation magnitude, $\beta$ is decay rate. Branching ratio $\alpha/\beta < 1$ for stationarity.

**Complexity:** $O(K)$ per update where $K$ is number of events in recent window.
**Trusted source:** Hawkes (1971), Biometrika; Hawkes (2018) review in Quantitative Finance.
**Why preferred:** Captures the intuitive notion that "an entity getting hit by multiple anomalous signals in rapid succession is more interesting than one isolated event" — whether that entity is a company with clustered insider filings, a country with escalating sanctions listings, a vessel with repeated route deviations, or a grid zone with persistent demand anomalies. The self-exciting property means recent events make future events more likely — exactly the pattern of information-driven activity. The multivariate extension is critical: events at entity A (any type) can excite intensity at entity B (any type) if they share an entity_link, enabling cross-domain convergence detection (the L3 moat).
**Implementation:** Use exponential kernel for computational efficiency. Estimate $\mu, \alpha, \beta$ per entity type, not per entity (too few events per individual entity for reliable MLE).

### Approach 4: Event Study Abnormal Score (Recommended — Entity Baseline)

**What it is:** Adapted Event Study methodology: measure each entity's current observation against its own historical baseline.

**Formulation:**
$$AS_t^{(e)} = \frac{x_t^{(e)} - \bar{x}_{baseline}^{(e)}}{\hat{\sigma}_{baseline}^{(e)}}$$

where baseline is the entity's own rolling history (estimation window of 30-90 days, excluding recent 5 days to avoid contamination).

**Complexity:** $O(W)$ per entity where $W$ is baseline window length.
**Trusted source:** MacKinlay (1997), J. Financial Economics.
**Why preferred:** Gives entity-specific standardization that works across ALL domains. A vessel that is normally on a fixed route deviating is more significant than a frequently-rerouted vessel deviating. A country with stable GDELT scores suddenly spiking is more significant than a conflict zone scoring high. An insider who never files suddenly filing is more significant than a frequent filer. The math is identical — only the observation source changes.

### Approach 5: Transfer Entropy (Deferred — Requires Data Maturity)

**What it is:** Measure directed information flow between entity pairs.

**Formulation:** $T_{X \to Y} = H(Y_t | Y_{t-1:t-L}) - H(Y_t | Y_{t-1:t-L}, X_{t-1:t-L})$

**Why deferred:** Requires long, dense observation histories for reliable estimation. Most entity pairs have sparse data. Better to implement after entity monitoring matures and generates sufficient history.
**Future value:** High. When data is sufficient, transfer entropy will identify "leading entities" — entities whose behavior predicts outcomes at other entities.

### Approach 6: Graph-Based Node Anomaly (Considered — Not Needed)

**What it is:** Use DOMINANT/CoLA-style graph autoencoders to score per-node anomaly.

**Why rejected for now:** We already have per-entity GNN embeddings from HetTGN. The scoring layer (distance from centroid, per-entity z-score) can be computed directly from existing embeddings without adding another GNN. DOMINANT/CoLA would be redundant with our existing HetTGN.
**When to revisit:** If the HetTGN embeddings prove insufficient for anomaly discrimination.

### Summary: Recommended Toolset

| Method | Role | Per-Entity State | Complexity | Phase |
|--------|------|-----------------|------------|-------|
| **CUSUM** | Primary alert trigger | 1 scalar | O(1) | 20 |
| **BOCPD** | Changepoint confirmation | run-length posterior | O(T) bounded | 20 |
| **Hawkes intensity** | Burst/clustering detection | event timestamps | O(K) | 20 |
| **Event Study AS** | Entity baseline comparison | rolling mean/std | O(W) | 20 |
| Transfer Entropy | Entity-to-entity flow | full history | O(T²) | Future |

**Why this toolset and not more:** Per the project principle "prefer the smallest high-signal toolset that preserves edge." These four methods cover different aspects of entity anomaly (persistent shift, structural break, event clustering, baseline deviation) with minimal overlap. Adding more methods increases computation and maintenance without proportional signal gain. Transfer entropy is deferred because it needs data maturity that doesn't exist yet.

---

## Architecture Proposal: Dual-Resolution Fusion

### Design Decision: Parallel Tracks, Not DAG Expansion

**Options considered:**
1. **Expand DAG with per-entity nodes** — Add obs.person_anomaly.{entity_id} nodes. Rejected: DAG complexity explodes, CPD learning becomes sparse, Kalman dimensionality breaks.
2. **Replace aggregate features with entity features** — Rejected: loses macro-level signal that is already working.
3. **Parallel entity scoring layer** (CHOSEN) — Keep existing macro pipeline untouched. Add a post-GNN entity scoring phase that operates alongside: GNN → entity scorer → entity alerts. The macro pipeline continues: GNN → aggregates → DAG → Kalman → macro beliefs.

**Why Option 3:**
- Zero risk to working macro pipeline
- Entity scoring is domain-specific logic that evolves faster than causal structure
- Decoupled testing — entity scorer can be validated independently
- Easy to iterate: can change scoring heuristics without touching DAG/Kalman

### Proposed Pipeline Flow

```
GNN Inference DAG
├─ Per-entity embeddings + id_map (person, company, wallet, country, vessel, ...)
│
├── [EXISTING] GNNFeatureBuilder → 11 scalar aggregates → DAG → Kalman → macro BeliefState
│
└── [NEW] EntityAnomalyScorer (type-agnostic — same math for any entity)
    ├─ Per-entity z-score (distance from type centroid, standardized per entity history)
    ├─ Per-entity CUSUM monitoring (persistent shift: insider buying, trade flow collapse, demand deviation)
    ├─ Per-entity BOCPD (structural break: route change, policy pivot, regime transition)
    ├─ Per-entity Hawkes intensity (event burst: filing cluster, sanctions listing wave, seismic swarm)
    ├─ Per-entity Event Study AS (baseline-relative: is THIS entity behaving abnormally for ITSELF?)
    ├─ Composite entity score → EntityAlert outputs (any entity type, any domain)
    │
    └── [NEW] Cross-Sector Convergence Pass
        ├─ For each elevated/critical entity → query entity_links for connected entities
        ├─ Check linked entities' alert statuses (from same scoring run)
        ├─ If 2+ linked entities from DIFFERENT source domains elevated within temporal window
        │   → produce ConvergenceCluster
        ├─ ConvergenceCluster captures: trigger entity, linked entities, link types,
        │   temporal span, contributing tools/domains, convergence score
        └─ Maps to named archetypes (corporate pre-announcement, sovereign stress, etc.)
            when pattern matches, but also detects NOVEL convergence not in any archetype
```

### New Data Structures

**EntityAlert** — per-entity scored output:
```python
@dataclass(frozen=True)
class EntityAlert:
    entity_id: str          # canonical entity ID
    entity_type: str        # person|company|wallet|country|vessel|domain|protocol|...
    entity_name: str        # human-readable name
    alert_time: float       # unix epoch
    
    # Scores (all z-scored for cross-entity comparability)
    anomaly_zscore: float       # GNN embedding deviation from type centroid
    cusum_statistic: float      # current CUSUM value (0 = baseline, >h = alert)
    bocpd_changepoint_prob: float  # probability of recent changepoint [0,1]
    hawkes_intensity: float     # current Hawkes process intensity
    event_study_car: float      # cumulative abnormal score
    
    # Composite
    composite_score: float      # weighted combination, calibrated [0,1]
    alert_level: str            # "normal" | "elevated" | "critical"
    
    # Context
    regime_context: str         # current macro regime (from macro beliefs)
    evidence_sources: list[str] # which tools contributed observations
    observation_count: int      # total observations for this entity
    metadata: dict | None
```

**ConvergenceCluster** — cross-sector entity cluster output:
```python
@dataclass(frozen=True)
class ConvergenceCluster:
    cluster_id: str              # unique cluster identifier
    trigger_entity_id: str       # the entity that initiated the convergence check
    trigger_entity_type: str     # type of trigger entity
    cluster_time: float          # unix epoch of cluster detection
    
    # Cluster members
    member_entities: list[dict]  # [{entity_id, entity_type, entity_name, alert_level, composite_score}]
    link_types: list[str]        # entity_link types connecting members (works_for, transacts_with, etc.)
    source_domains: list[str]    # unique data domains contributing (finance, geopolitics, maritime, etc.)
    evidence_tools: list[str]    # all tools contributing to member alerts
    
    # Cluster scores
    domain_diversity: int        # number of distinct source domains (higher = rarer = more valuable)
    temporal_span_hours: float   # time span between earliest and latest member alert
    convergence_score: float     # composite: f(domain_diversity, member scores, temporal tightness) [0,1]
    cluster_level: str           # "minor" (2 domains) | "significant" (3-4 domains) | "major" (5+ domains)
    
    # Archetype matching (optional — learned, not hard-coded)
    matched_archetype: str | None   # e.g. "corporate_pre_announcement", "sovereign_stress_cascade", None for novel
    archetype_confidence: float | None  # how well the cluster matches a known archetype [0,1]
    
    metadata: dict | None
```

**Key design choice:** `matched_archetype` is informational, not decisional. The convergence score is computed generically from domain diversity + temporal tightness + member scores. Archetype matching is a label, not a filter — novel cross-sector combinations that don't match any named archetype still produce clusters. Per the project principle: maximize learnable structure, minimize hand-coded logic.

### Files Affected (Preliminary)

| File | Change Type | Description |
|------|------------|-------------|
| `agent/fusion/__init__.py` | **NEW** | Fusion module init |
| `agent/fusion/entity_scorer.py` | **NEW** | EntityAnomalyScorer class — per-entity scoring pipeline |
| `agent/fusion/cusum.py` | **NEW** | CUSUM monitor implementation |
| `agent/fusion/hawkes.py` | **NEW** | Hawkes process intensity estimator |
| `agent/fusion/alert.py` | **NEW** | EntityAlert dataclass + alert level calibration |
| `agent/fusion/convergence.py` | **NEW** | ConvergenceCluster dataclass + convergence pass logic |
| `agent/fusion/entity_baseline.py` | **NEW** | Per-entity rolling baseline (Event Study adaptation) |
| `agent/pipeline/dags/entity_scoring.py` | **NEW** | Entity scoring DAG (runs after GNN inference) |
| `agent/features/gnn_builder.py` | **MODIFY** | Expose per-entity scores alongside aggregates |
| `agent/pipeline/store.py` | **MODIFY** | Add entity_alerts + convergence_clusters tables + store/query methods |
| `agent/models/belief.py` | **MODIFY** | Add optional entity_id to BeliefState (backward compatible) |

---

## Paradigm Revision: Self-Supervised Prediction Surprise (April 2026)

**Critical insight:** The HetTGN already has 3-component self-supervised training (obs_type prediction via CE, time_delta prediction via MSE, contrastive link loss via margin). This means the GNN is ALREADY building a world model of entity behavior. Instead of building hand-coded anomaly detectors (CUSUM, Hawkes, Event Study) whose output IS the anomaly signal, we should:

1. **Keep CUSUM/Hawkes/EventStudy as node feature enrichment** — they go INTO the GNN as additional per-entity features (12d → 39d), helping the GNN learn faster on sparse data (like positional encoding helps transformers).
2. **Extract the GNN's own prediction surprise as the PRIMARY anomaly signal** — obs_type surprise (-log P(actual)), temporal surprise (|dt_pred - dt_actual|), value surprise (new prediction head), memory drift (||m_t - m_{t-1}||).
3. **Convergence = correlated prediction surprise across graph neighborhoods** — no hand-coded graph traversal, no domain diversity counting, no temporal window. The GNN's HGT attention weights naturally propagate surprise across entity_links.
4. **New value prediction head** extends self-supervised training from "what type + when" to "what type + when + how much."
5. **The 10 named archetypes are permanently removed from code, tests, evaluation.** They remain as human context in this research doc only.

This is the LLM analogy: next-observation prediction IS TirraMind's "next token prediction." Anomaly = prediction surprise (perplexity). Convergence = correlated perplexity across graph neighborhoods (paragraph-level surprise).

**Literature foundations:**
- TGN (Rossi et al. 2020, arXiv:2006.10637) — temporal graph network with memory
- SL-GAD (Zheng et al. 2021, IEEE TKDE, arXiv:2108.09896) — generative + contrastive SSL for graph anomaly
- Graph SSL Survey (Liu et al. 2021, IEEE TKDE, arXiv:2103.00111) — taxonomy: generation, auxiliary-property, contrast, hybrid
- GraphMAE (Hou et al. 2022, KDD, arXiv:2205.10803) — masked feature reconstruction for graphs
- PyGOD (Liu et al. 2024, JMLR) — graph anomaly detection library; DOMINANT/CoLA/CONAD all use reconstruction error as anomaly signal

**Key references for our existing architecture:**
- Time2Vec (Kazemi et al. 2019) — learnable time encoding (already in HeteroMemory)
- HGT (Hu et al. 2020) — heterogeneous graph transformer attention (already in HetTGN)

---

## Implementation Intent

### Concepts Approved for Implementation
1. **Per-entity CUSUM monitoring** — lightweight, scalable, well-understood
2. **Per-entity BOCPD** — already implemented, just needs per-entity application
3. **Per-entity Hawkes intensity** — captures event-clustering dynamics
4. **Per-entity Event Study baseline** — entity-specific standardization
5. **EntityAlert protocol** — structured output for entity-level signals
6. **ConvergenceCluster protocol** — cross-sector entity convergence output
7. **Cross-sector convergence pass** — traverse entity_links after individual scoring to detect multi-domain co-firing
8. **Parallel scoring DAG** — minimal risk to existing pipeline
9. **Entity-level feature exposure** from GNNFeatureBuilder

### Concepts Rejected (This Phase)
- **Transfer Entropy:** Deferred to future phase — insufficient per-entity observation density
- **DOMINANT/CoLA graph anomaly:** Redundant with existing HetTGN embeddings
- **Per-entity DAG nodes:** Complexity explosion, sparse CPD learning
- **Dynamic Kalman obs_dim:** Breaks fixed-dimension state space
- **Hard-coded archetype rules:** Archetypes are labels for human interpretation, not detection rules. The convergence pass uses generic graph traversal + temporal window + domain diversity — it detects novel convergence patterns, not just the 10 named archetypes.

### Notes for the Spec
- Keep existing macro pipeline (aggregates → DAG → Kalman) 100% untouched
- Entity scoring is a NEW parallel track, not a modification of existing flow
- **Scorer must be type-agnostic:** the same code handles person, company, wallet, vessel, country, domain, protocol — entity type is a parameter, not a code branch
- Must handle entities with 0 observations gracefully (skip scoring, no crash)
- CUSUM threshold calibration: start with ARL₀ = 500 (false alarm every 500 updates); tune later
- Hawkes parameters ($\alpha, \beta$) estimated per entity TYPE, not per entity (data sparsity)
- Composite score: start with equal weights, move to learned weights in Phase 21 (RL)
- EntityAlert stored in new `entity_alerts` table with entity_id index
- Alert levels: "normal" < 0.3, "elevated" 0.3-0.7, "critical" > 0.7 (on composite score)
- **Design for future entity types:** domain, protocol, grid_zone, pathogen, drug, facility — these exist in tool data but lack entity registration today. The scorer's type-agnostic design means adding them is a data/registration change, not a code change
- **Cross-sector convergence is Phase 20 core, not deferred.** After individual entity scoring, the scorer performs a convergence pass: traverse entity_links for elevated entities, check linked entity statuses, produce ConvergenceCluster when 2+ linked entities from different data domains co-fire within a temporal window. This is the L3 moat — the entire value proposition.
- **Convergence score formula:** `convergence_score = σ(w_d · domain_diversity + w_t · temporal_tightness + w_s · mean_member_score)` where domain_diversity counts unique source domains, temporal_tightness = 1/(1 + span_hours/window), and mean_member_score is average composite_score of cluster members. Initial weights equal; learned in Phase 21.
- **Cluster levels:** minor (2 domains), significant (3-4 domains), major (5+ domains)
- **Archetype matching is optional labeling:** The 10 named archetypes (corporate pre-announcement, sovereign stress, supply chain disruption, etc.) are matched post-hoc using tool/entity-type pattern matching. Novel convergence patterns that don't match any archetype still produce clusters — the system is not limited to known patterns.

---

## Related

- [[signal_fusion_spec]]
- [[world_model_bridge]] — Phase 19 (GNN ↔ World Model integration, predecessor)
- [[world_model]] — Phase 14 (initial world model research)
- [[temporal_het_gnn]] — Phase 12 (HetTGN architecture)
- [[gnn_pattern_and_finetuning]] — Phase 14-15 (GNN training, per-entity embeddings)
- [[entity_linking_layer]] — Phase 17 (entity storage + linking)
- [[convergence_detection]] — Phase 7c (macro convergence, BOCPD implementation)
- [[signal_protocol_feature_engineering]] — Phase 8 (EngineeredFeature protocol)
- [[cross_entity_l3]] — L3 cross-entity research

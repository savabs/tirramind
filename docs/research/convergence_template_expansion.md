---
title: "Research: Convergence Template Expansion — Senior Quant Patterns"
tags:
  - doc/research
  - phase/7c
  - topic/convergence
---

# Research: Convergence Template Expansion — Senior Quant Patterns

## Objective
Design 10 new causal chain templates for the convergence detection system that are:
- **Non-obvious**: require connecting disparate data sources nobody else watches together
- **Cross-category rich**: span 3–5 taxonomy categories (cross-category coincidence is more informative)
- **Exploit our unique data surface**: satellite, DeFi, DNS, Wikipedia, AIS, lobbying, Form 144, etc.
- **Grounded in real causal economics**: not curve-fitted patterns but structural causal chains

## Current State
- **12 existing templates** in `agent/convergence/templates.py`
- **11 taxonomy categories** with 46 registered extractors (43 real + 3 stubs)
- Existing templates mostly follow conventional macro/geopolitical chains (supply disruption, monetary policy, credit stress, etc.)
- **Gap**: No templates exploit the unique cross-domain signals that are TirraMind's edge — DeFi × physical, behavioral × positioning divergence, satellite × regulatory, etc.

## Conceptual Foundations

### 1. Soros Reflexivity (Alchemy of Finance, 1987)
Markets are not merely passive reflectors of fundamentals — participants' biased perceptions *change* the fundamentals. Boom/bust cycles are reflexive: rising prices → increased lending → further price rises → until reality diverges too far from perception → sudden reversal. **Template implication**: Detect the divergence between "narrative signals" (Wikipedia, prediction markets) and "reality signals" (physical flow, insider actions). When they diverge, the reflexive cycle is near its turning point.

### 2. Minsky's Financial Instability Hypothesis (Stabilizing an Unstable Economy, 1986)
Stability breeds risk-taking: hedge → speculative → Ponzi financing phases. The "Minsky moment" is when leveraged positions can't service debt from cash flows. **Template implication**: Watch DeFi leverage + credit stress signals for the speculative→Ponzi transition. DeFi is transparent and 24/7 — it shows stress 7-14 days before TradFi because of settlement lag.

### 3. Financial Contagion (Forbes & Rigobon, 2002; Allen & Gale, 2000)
Crisis transmission through channels: trade links, financial links, information cascades. The key insight is that correlations *increase* during crises — independent systems become correlated. **Template implication**: Watch for signals that normally don't co-move suddenly activating together (exactly what convergence detection does, but templates should encode *which* co-movements indicate *which* type of crisis).

### 4. Baltic Dry Index as Leading Indicator
Physical shipping rates lead equity/macro by months because "people don't book freighters unless they have cargo to move." **Template implication**: Our AIS vessel data + transport throughput + commodity positioning can detect physical economy changes before they show up in macro data.

### 5. Carry Trade Dynamics (Rise of Carry, Lee et al., 2020)
Carry trades (borrow low-yield, invest high-yield) build up silently in calm periods, then unwind catastrophically. The yen carry trade collapse contributed to 2008 crisis. **Template implication**: Central bank rate changes → capital flow reversals → DeFi liquidations (modern carry unwind includes crypto) → commodity positioning unwind.

### 6. Dark Shipping / Sanctions Evasion Literature
Vessels turn off AIS transponders to circumvent sanctions ("dark shipping"). Simultaneously, sanctioned entities shift operations online (DNS changes, DeFi circumvention). This is observable across multiple otherwise-unrelated data streams. **Template implication**: Regulatory action → physical flow anomalies → digital infrastructure changes → crypto flows → geopolitical escalation.

## Signal Inventory by Category

For reference — these are the signal_id patterns available for template matching:

| Category | Signal Patterns | Source Tools |
|----------|----------------|--------------|
| physical_flow | `ais.*`, `transport.*`, `energy_supply.*`, `power_grid.freq*\|power_grid.load*` | ais_vessel_tracking, transport_throughput, energy_supply, power_grid |
| physical_disruption | `weather.*`, `earthquake.*`, `dns.*`, `internet.*`, `satellite.fire*\|satellite.events*`, `power_grid.outage*` | weather_alerts, earthquake_proximity, dns_monitor, internet_infrastructure, satellite_activity, power_grid |
| financial_stress | `sovereign_debt.*`, `creditor.*`, `bankruptcy.*`, `liquidity.*`, `defi.*`, `whale_alert.*` | sovereign_debt, creditor_filings, bankruptcy_court, liquidity_regime, defi_flows, whale_alert |
| monetary_policy | `central_bank.*`, `capital_flows.*` | central_bank_balance, capital_flows |
| regulatory_action | `sanctions.*`, `drug_regulatory.*`, `regulatory_gazette.*` | sanctions_monitor, drug_regulatory, regulatory_gazette |
| behavioral_intent | `patent.*`, `lobbying.*`, `wikipedia.*`, `cert_trans.*`, `jobs.*` | patent_filings, lobbying, wikipedia_pageviews, cert_transparency, job_postings |
| positioning | `cftc.*`, `finra.*`, `polymarket.*`, `polymarket.whale*`, `insider.*`, `form144.*` | cftc, finra_short_volume, polymarket, polymarket_whales, insider_filings, form144 |
| macro_momentum | `pmi.*`, `treasury.*`, `building_permits.*`, `consumer_sentiment.*`, `macro.*` | global_pmi, treasury_receipts, building_permits, consumer_sentiment, macro_data |
| biological | `disease.*`, `food_security.*` | disease_surveillance, food_security |
| geopolitical | `gdelt.*`, `political_risk.*` | gdelt, political_risk |
| supply_chain | `comtrade.*`, `supply_chain.*`, `satellite.vegetation*` | comtrade, supply_chain_prices, satellite_activity |

## Existing 12 Templates (for gap analysis)

1. **supply_chain_disruption**: physical_disruption → physical_flow → positioning → macro_momentum
2. **monetary_policy_shift**: monetary_policy → financial_stress → positioning → macro_momentum
3. **geopolitical_escalation**: geopolitical → regulatory_action → physical_flow → positioning
4. **health_crisis**: biological → physical_flow/disruption → behavioral_intent → macro_momentum
5. **agricultural_shock**: physical_disruption → biological → positioning → macro_momentum
6. **energy_crisis**: physical_flow/disruption → physical_flow → positioning → macro_momentum
7. **credit_stress_cascade**: financial_stress → financial_stress → positioning → macro_momentum
8. **tech_disruption**: behavioral_intent → behavioral_intent → supply_chain → positioning
9. **labor_market_shift**: behavioral_intent → macro_momentum → macro_momentum → positioning
10. **trade_war_escalation**: regulatory_action → geopolitical → physical_flow → positioning
11. **construction_cycle**: macro_momentum → behavioral_intent → financial_stress → macro_momentum
12. **digital_infrastructure_crisis**: physical_disruption → financial_stress → behavioral_intent → macro_momentum

### Gap Analysis
- **No template combines DeFi + physical world signals** (DeFi is only in credit_stress_cascade and digital_infrastructure_crisis, both financial-only chains)
- **No template uses satellite signals** (satellite was just converted from stub)
- **No template detects narrative-vs-reality divergence** (positioning diverging from physical evidence)
- **No template uses Form 144 or whale alerts as leading signals**
- **No template exploits sanctions evasion as a multi-domain observable**
- **No template captures capital flight through crypto channels**
- **No template uses lobbying as a leading signal for regulatory outcomes**
- **No template combines food security + satellite + trade data**
- **No template detects stealth accumulation patterns**
- **No template combines power grid + internet as infrastructure decay**

## 10 New Templates

### Template 13: `silent_nationalization`
**Thesis**: Before a country nationalizes resources or imposes export controls, detectable pre-signals appear across lobbying, satellite imagery, regulatory changes, and smart money exits. This chain is almost impossible to detect without our specific data surface.

**Causal chain**:
1. `lobbying.*` (behavioral_intent) — lobbying spend surges in target commodity sector
2. `satellite.fire*|satellite.events*` (physical_disruption) — unusual physical activity in producing region
3. `regulatory_gazette.*` (regulatory_action) — enabling legislation or executive orders published
4. `insider.*|form144.*` (positioning) — insider selling in exposed companies
5. `ais.*` (physical_flow) — shipping routes reroute away from affected ports

**Categories spanned**: behavioral_intent → physical_disruption → regulatory_action → positioning → physical_flow (5 categories)
**Within_days**: 0, 14, 21, 30, 45
**Why it's unique**: Nobody else watches lobbying + satellite + regulatory gazette + insider filings as a single chain. Each data source is cheap/free. Together they reveal state action before it's announced.

---

### Template 14: `defi_canary`
**Thesis**: DeFi operates 24/7 with transparent on-chain data. Stress in DeFi (liquidations, whale exits, TVL drops) precedes TradFi credit events by 7-14 days because TradFi has settlement lag, reporting delay, and market hours. The "canary" signal propagates through public attention and prediction markets before hitting traditional credit.

**Causal chain**:
1. `defi.*` (financial_stress) — DeFi protocol stress: liquidations, TVL drops, stablecoin depeg
2. `whale_alert.*` (financial_stress) — large crypto whale movements (exit signal)
3. `wikipedia.*` (behavioral_intent) — panic attention: Wikipedia views for crisis-related topics spike
4. `polymarket.*|polymarket.whale*` (positioning) — prediction market odds shift reflecting new information
5. `bankruptcy.*|creditor.*` (financial_stress) — TradFi credit stress materializes

**Categories spanned**: financial_stress → financial_stress → behavioral_intent → positioning → financial_stress (3 categories)
**Within_days**: 0, 3, 7, 14, 30
**Why it's unique**: Uses DeFi as a 24/7 transparent stress sensor that leads traditional markets. The Wikipedia attention signal confirms information is propagating to the public. Minsky-inspired: the Ponzi financing phase in DeFi is more visible than in TradFi.

---

### Template 15: `pandemic_physical_evidence`
**Thesis**: Disease surveillance may report elevated case counts, but the *physical evidence* of a pandemic's economic impact appears in satellite imagery (factory/port activity), transport throughput, and trade flows BEFORE it shows up in macro data. This differs from existing `health_crisis` template by centering on satellite physical evidence rather than just disease→transport.

**Causal chain**:
1. `disease.*` (biological) — disease surveillance signals elevate
2. `satellite.fire*|satellite.events*` (physical_disruption) — satellite shows unusual physical activity or disrupted facilities
3. `wikipedia.*` (behavioral_intent) — Wikipedia pageviews for disease/health topics surge (information cascade)
4. `transport.*` (physical_flow) — transport throughput measurably drops
5. `cftc.*` (positioning) — commodity speculative positioning shifts in response

**Categories spanned**: biological → physical_disruption → behavioral_intent → physical_flow → positioning (5 categories)
**Within_days**: 0, 7, 14, 21, 30
**Why it's unique**: The satellite physical evidence layer is what makes this distinct from the existing health_crisis template. Nobody else combines disease surveillance + satellite imagery + Wikipedia attention as a three-point confirmation of pandemic economic impact.

---

### Template 16: `capital_flight_crypto`
**Thesis**: When political instability threatens a country, capital flight now includes crypto channels. The chain: political risk → traditional capital outflows → DeFi stablecoin flows surge (modern flight capital) → sovereign debt comes under pressure → central bank intervenes. This is Soros reflexivity in action: capital flight → currency weakness → more capital flight.

**Causal chain**:
1. `political_risk.*` (geopolitical) — political instability measures elevate
2. `capital_flows.*` (monetary_policy) — traditional capital outflows detected
3. `defi.*` (financial_stress) — DeFi stablecoin minting/flows surge (crypto as flight vehicle)
4. `sovereign_debt.*` (financial_stress) — affected country's sovereign spreads widen
5. `central_bank.*` (monetary_policy) — central bank response (rate hike, reserve deployment)

**Categories spanned**: geopolitical → monetary_policy → financial_stress → financial_stress → monetary_policy (3 categories)
**Within_days**: 0, 7, 14, 21, 30
**Why it's unique**: Maps the modern capital flight pathway that includes crypto. Traditional models miss the DeFi channel entirely. The reflexive loop (political risk → outflows → currency weakness → more political risk) means early detection matters enormously.

---

### Template 17: `infrastructure_decay_cascade`
**Thesis**: Physical infrastructure deterioration is observable across multiple sensor networks simultaneously. Power grid frequency deviations + internet outages + building permits declining form a pattern that precedes economic decline in a region. This is a slow-burn pattern that's invisible to anyone not watching physical infrastructure data.

**Causal chain**:
1. `power_grid.*` (physical_flow) — grid frequency instability or load anomalies detected
2. `internet.*|dns.*` (physical_disruption) — internet infrastructure degradation
3. `building_permits.*` (macro_momentum) — construction/maintenance activity declining
4. `jobs.*` (behavioral_intent) — job postings decline in affected infrastructure sectors
5. `consumer_sentiment.*` (macro_momentum) — regional consumer confidence drops

**Categories spanned**: physical_flow → physical_disruption → macro_momentum → behavioral_intent → macro_momentum (4 categories)
**Within_days**: 0, 14, 30, 45, 60
**Why it's unique**: Nobody treats power grid + internet infrastructure as leading economic indicators in combination. The temporal structure (60 day window) captures the slow cascade from physical decay to economic impact. Free data sources (ISO/RTO grid data, internet monitoring) that nobody else combines.

---

### Template 18: `commodity_hoarding`
**Thesis**: When weather disrupts a producing region AND satellite vegetation data confirms crop damage, smart money begins hoarding before the shortage becomes public. The signal propagates: physical damage → vegetation confirmation → shipping reroutes → speculative positioning → food security alerts. This is a tighter, more specific version of agricultural_shock with satellite confirmation.

**Causal chain**:
1. `weather.*` (physical_disruption) — weather disruption in producing region
2. `satellite.vegetation*` (supply_chain) — satellite NDVI confirms vegetation/crop damage
3. `ais.*|transport.*` (physical_flow) — shipping routes change as supply chains reroute
4. `cftc.*` (positioning) — CFTC speculative long positioning surges in affected commodity
5. `food_security.*` (biological) — food security monitoring confirms shortage

**Categories spanned**: physical_disruption → supply_chain → physical_flow → positioning → biological (5 categories)
**Within_days**: 0, 7, 14, 21, 30
**Why it's unique**: The satellite vegetation confirmation layer is the key differentiator. Weather alerts are noisy (many don't cause real crop damage). The satellite NDVI data *confirms* actual vegetation stress, turning a weather alert into a verified supply signal. This is our edge - we watch the physical evidence rather than just the weather forecast.

---

### Template 19: `smart_money_divergence`
**Thesis**: Soros reflexivity predicts that boom phases have a characteristic divergence signature: retail attention (Wikipedia) + public confidence (prediction markets) rising WHILE smart money quietly exits (insider selling, short volume increasing, DeFi whale exits). The divergence IS the signal of an approaching Minsky moment. This template detects the gap between narrative and positioning.

**Causal chain**:
1. `wikipedia.*` (behavioral_intent) — retail attention/euphoria: Wikipedia pageviews for topic surge
2. `polymarket.*` (positioning) — prediction market shows bullish public sentiment
3. `form144.*|insider.*` (positioning) — BUT insiders begin selling (Form 144 filings appear)
4. `finra.*` (positioning) — short volume increases (informed traders positioning for reversal)
5. `defi.*|whale_alert.*` (financial_stress) — DeFi whale exits accelerate

**Categories spanned**: behavioral_intent → positioning → positioning → positioning → financial_stress (3 categories)
**Direction key**: Steps 1-2 are direction=+1 (bullish); steps 3-5 are direction=-1 (bearish). The DIVERGENCE is the signal.
**Within_days**: 0, 7, 14, 21, 30
**Why it's unique**: This is the only template that detects a *divergence* between signals rather than a unidirectional cascade. Retail euphoria + smart money exit = reflexive turning point. Nobody else has this combination of behavioral (Wikipedia), public prediction (Polymarket), insider (Form 144), institutional (FINRA shorts), and crypto (DeFi whales) in one template.

---

### Template 20: `sanctions_evasion_network`
**Thesis**: When sanctions are imposed, evasion is observable across multiple data streams simultaneously: ships go dark (AIS transponder off), DNS/cert changes appear on sanctioned entity domains, DeFi flows surge in circumvention patterns, and geopolitical event intensity increases. Each evasion signal is weak alone; together they confirm the scale and velocity of sanctions circumvention, which predicts the sanctions' economic effectiveness.

**Causal chain**:
1. `sanctions.*` (regulatory_action) — new sanctions regime imposed
2. `ais.*` (physical_flow) — vessel transponder gaps / dark shipping detected
3. `cert_trans.*|dns.*` (behavioral_intent) — DNS/SSL cert changes on sanctioned entity domains
4. `defi.*` (financial_stress) — DeFi flows through circumvention patterns
5. `gdelt.*` (geopolitical) — GDELT event intensity increases (diplomatic/conflict events)

**Categories spanned**: regulatory_action → physical_flow → behavioral_intent → financial_stress → geopolitical (5 categories)
**Within_days**: 0, 7, 14, 21, 30
**Why it's unique**: Sanctions evasion is observable in real-time through our data surface, but nobody else watches ALL these channels together. Dark shipping + DNS changes + DeFi circumvention + GDELT escalation as a combined signal is unique to our platform. The pattern predicts whether sanctions will be effective (lots of evasion = sanctions failing) or will cause real economic disruption.

---

### Template 21: `carry_trade_unwind`
**Thesis**: Carry trade unwinding follows a specific cascade: central bank rate surprise triggers capital flow reversals, which cascade through DeFi (modern carry includes crypto yield farming), sovereign debt stress, speculative positioning unwind, and finally macro impact. Based on the carry trade literature (Lee et al. 2020, Plantin & Shin 2010) and the 2008/2022 unwind events.

**Causal chain**:
1. `central_bank.*` (monetary_policy) — rate surprise or balance sheet change
2. `capital_flows.*` (monetary_policy) — capital flows reverse direction
3. `defi.*` (financial_stress) — DeFi yield farming unwind / liquidations
4. `sovereign_debt.*` (financial_stress) — sovereign spreads widen in affected countries
5. `cftc.*` (positioning) — CFTC positioning unwind in carry-sensitive commodities
6. `pmi.*` (macro_momentum) — real economy impact shows up in PMI

**Categories spanned**: monetary_policy → monetary_policy → financial_stress → financial_stress → positioning → macro_momentum (4 categories)
**Within_days**: 0, 7, 14, 21, 30, 45
**Why it's unique**: The DeFi layer is what makes this different from the existing monetary_policy_shift template. Modern carry trades include crypto yield farming, and the unwind cascade now flows through DeFi before hitting traditional markets. The 2022 Terra/Luna collapse demonstrated this exact chain. 6 steps make it a longer, more specific template.

---

### Template 22: `stealth_accumulation`
**Thesis**: Before a major regulatory catalyst (approval, ban, tariff), informed actors accumulate positions quietly. The pattern is detectable across behavioral signals that individually look like noise but together form a pattern: Wikipedia interest slowly rising (information gathering), lobbying spend increasing (influence campaign), patent filings appearing (IP positioning), CFTC positions building quietly, then the regulatory catalyst drops.

**Causal chain**:
1. `wikipedia.*` (behavioral_intent) — slow rise in Wikipedia pageviews for a topic (research phase)
2. `lobbying.*` (behavioral_intent) — lobbying expenditure increases in relevant sector
3. `patent.*` (behavioral_intent) — patent filings appear in related technology/sector
4. `cftc.*|insider.*` (positioning) — quiet position building in related instruments
5. `regulatory_gazette.*|drug_regulatory.*` (regulatory_action) — the regulatory catalyst drops

**Categories spanned**: behavioral_intent → behavioral_intent → behavioral_intent → positioning → regulatory_action (3 categories)
**Within_days**: 0, 30, 45, 60, 90
**Why it's unique**: This template has the longest temporal window (90 days) because stealth accumulation is slow by design. The key insight is that the *combination* of Wikipedia + lobbying + patents is the accumulation fingerprint — each is noise alone, but together they indicate informed positioning ahead of a regulatory event. The 90-day window also means this template won't fire on random co-occurrences.

## Risks & Edge Cases
- Templates with many steps (5-6) may be harder to match — `min_match` should be set to `len(steps) - 1` or even `len(steps) - 2` for 6-step templates
- Direction constraints on `smart_money_divergence` are critical — the template only works if early steps are +1 and late steps are -1
- Long temporal windows (stealth_accumulation at 90 days) increase false positive risk — compensated by requiring more steps to match
- Some signal_id patterns are broad (e.g., `defi.*` matches many sub-signals) — this is intentional for templates since we want any DeFi stress signal to count

## References
- Soros, G. (1987). *The Alchemy of Finance*. Reflexivity theory — boom/bust cycles driven by participant bias.
- Minsky, H. (1986). *Stabilizing an Unstable Economy*. Financial instability hypothesis — stability breeds risk-taking.
- Forbes, K. & Rigobon, R. (2002). "No Contagion, Only Interdependence." Financial contagion measurement.
- Lee, T., Lee, J., & Coldiron, K. (2020). *The Rise of Carry*. Carry trade dynamics and systemic risk.
- Plantin, G. & Shin, H.S. (2010). "Carry Trades and Speculative Dynamics." Mathematical model for carry trade unwinds.
- Allen, F. & Gale, D. (2000). "Financial Contagion." General equilibrium model of crisis transmission.
- NY Fed Yield Curve model — Term spread as recession predictor (Estrella & Mishkin, 1996).
- Baltic Dry Index literature — Physical shipping as leading economic indicator.

---

## Related

- [[convergence_template_expansion_spec|Spec: Convergence Template Expansion]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]

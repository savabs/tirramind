---
title: "Feature: Convergence Signal Expansion Priorities"
tags:
  - doc/research
  - phase/7c
  - topic/convergence
---

# Feature: Convergence Signal Expansion Priorities

## Goal

Identify which additional signal families are most worth adding to TirraMind's
convergence layer before broadening the schema further.

The question is not "can we add more fields?" but:

1. Which signals increase cross-category causal coverage the most?
2. Which signals arrive early enough to matter for prediction?
3. Which sources are free, stable, and commercially usable?
4. Which additions improve the world-model evidence surface rather than just
   adding more dashboard noise?

## Current Architecture

### Relevant local modules

- `agent/convergence/extractors.py`
- `agent/convergence/signals.py`
- `agent/convergence/templates.py`
- `agent/tools/consumer_sentiment.py`
- `agent/tools/food_security.py`
- `agent/tools/political_risk.py`
- `agent/tools/power_grid.py`
- `agent/tools/defi_flows.py`
- `[[7b-AM_consumer_sentiment]]`
- `[[7b-AI_internet_infrastructure]]`
- `[[7b-AO_supply_chain_monitor]]`

### Existing patterns to preserve

- Prefer signal families that bridge categories already used by templates.
- Prefer sources with structured machine-readable output over output-only tools.
- Prefer free/no-auth or low-friction sources with commercially safe use.
- Avoid field bloat in `ConvergenceSignal`; add richer upstream evidence first.

### Important current-state correction

Several formerly-missing convergence extractors are already implemented:

- `consumer_sentiment`
- `food_security`
- `political_risk`
- `supply_chain_prices`

By contrast, several high-potential surfaces are still weakly represented or
not represented at all in convergence:

- `internet_infrastructure` — stub extractor only
- `electricity_monitor` — stub extractor only
- `power_grid` — no convergence extractor
- `defi_flows` — no convergence extractor

This changes the priority: the best next work is not adding a fifth field to
already-emitted consumer or political signals. It is converting high-signal,
currently-unused surfaces into evidence.

## Search Log

### Local sources read

- `[[convergence_detection]]`
- `[[convergence_audit_pre_worldmodel]]`
- `[[7b-AM_consumer_sentiment]]`
- `[[7b-AI_internet_infrastructure]]`
- `[[7b-AO_supply_chain_monitor]]`
- `[[7b-N_fcc_spectrum]]`
- `agent/convergence/taxonomy.py`
- `agent/convergence/extractors.py`
- `agent/tools/consumer_sentiment.py`
- `agent/tools/food_security.py`
- `agent/tools/political_risk.py`
- `agent/tools/power_grid.py`
- `agent/tools/defi_flows.py`

### Documentation reviewed

- Eurostat `ei_bsco_m` dataset API
  - Open without auth
  - Monthly consumer-confidence data
  - Updated as recently as 2026-03-30
- World Bank indicator API for `AG.PRD.FOOD.XD`
  - Annual food production index
  - Broad global coverage, but low cadence
- OpenFEC developer docs
  - Nightly campaign-finance updates
  - `DEMO_KEY` possible, own key lifts limits to 1,000 calls/hour
- NYISO energy market and operational data portal
  - Public day-ahead and real-time pricing plus operational datasets
- DefiLlama API docs
  - Public coverage for TVL, stablecoins, DEX volume
  - Richer historical and inflow metrics partly move into paid tiers
- BLS API docs
  - Public multi-series POST API
  - Up to 50 series in one request
  - 20-year windows with registration key
- FRED API docs
  - Series observations and vintages available
  - Useful for survey-vs-release divergence work

## Evaluation Framework

Each candidate signal family was ranked on five dimensions:

1. **Cadence / latency** — how quickly the source reacts to reality
2. **Causal breadth** — how many downstream assets or templates it can touch
3. **Orthogonality** — whether it adds genuinely new evidence rather than
   duplicating an existing surface
4. **Operational safety** — free, stable, documented, commercially usable
5. **Implementation leverage** — how much value we get per unit engineering work

Scale: 1 (weak) to 5 (strong)

## Ranked Signal Families

| Rank | Signal family | Cadence | Breadth | Orthogonality | Safety | Leverage | Why it matters |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | Internet infrastructure outages and censorship | 5 | 4 | 5 | 4 | 5 | Country-level outage and censorship events are still mostly missing from convergence and are highly orthogonal to macro/positioning data. |
| 2 | Power and grid stress | 5 | 4 | 4 | 5 | 4 | Physical stress data is early, operational, and reacts before macro releases. |
| 3 | DeFi liquidity and rotation | 5 | 4 | 4 | 4 | 4 | Adds a 24/7 liquidity/risk surface not captured by legacy macro tools. |
| 4 | Supply-chain price pressure breadth | 3 | 5 | 4 | 5 | 4 | Upstream producer and import-price shocks transmit into inflation, margins, and rates. |
| 5 | Consumer expectation divergence | 3 | 4 | 3 | 5 | 3 | Valuable, but partially covered already; better to deepen breadth and gap signals rather than add many more raw fields. |
| 6 | Political-finance stress | 3 | 3 | 3 | 4 | 3 | Useful around election cycles, but narrower and more regime-specific than the top four. |
| 7 | Food-security stress | 1 | 4 | 4 | 5 | 2 | Important structurally, but annual cadence makes it weak for near-term convergence. |
| 8 | FCC / spectrum | 0 | 2 | 4 | 0 | 0 | Research shows the free programmatic surface is effectively unusable; do not prioritize. |

## Detailed Findings

### 1. Internet infrastructure is the best next addition

**Why it ranks first**

- It is still only a stub in convergence despite having strong prior research.
- It observes Layer 0 / Layer 1 digital disruption directly.
- It is orthogonal to existing macro, positioning, and survey data.
- It connects naturally to geopolitical, regulatory, and supply-chain templates.

**Evidence from docs**

- IODA provides country-level outage alerts, scored events, and 30-minute raw
  connectivity series.
- OONI provides censorship incidents and aggregated anomaly rates by country and
  test type with a commercially compatible CC BY license.

**Highest-value signals to add**

- `internet.country.outage_score`
- `internet.country.bgp_visibility_drop`
- `internet.country.censorship_rate`
- `internet.country.messaging_block_incidents`
- `internet.country.multi_source_disruption_breadth`

**Why these beat extra schema fields today**

These create a new causal pathway into geopolitical escalation,
physical-disruption proxies, and policy/regulatory chains. A new orthogonal
signal family is worth more than adding three extra metadata fields to an
already-classified convergence event.

### 2. Power and grid stress should be converted into convergence evidence

**Why it ranks second**

- NYISO data is public, operational, and high frequency.
- Power stress often appears before macro deterioration is visible in surveys.
- The current convergence layer has no real `power_grid` extractor.

**Highest-value signals**

- `power_grid.system.demand_forecast_gap_pct`
- `power_grid.system.da_rt_spread`
- `power_grid.system.fuel_mix_gas_share`
- `power_grid.zone.stress_breadth`
- `power_grid.zone.peak_load_anomaly`

**Stakes**

This is one of the cleanest physical-world proxies for industrial activity,
heat stress, congestion, and energy cost pressure. It is much earlier than
monthly macro releases.

### 3. DeFi flows add a missing 24/7 liquidity surface

**Why it ranks third**

- The tool exists, but convergence is not yet extracting evidence from it.
- Stablecoin supply, chain rotation, and DEX panic volume are fast-moving and
  globally informative.
- These signals matter both for crypto and for the broader global-liquidity /
  risk-on-risk-off state.

**Highest-value signals**

- `defi.stablecoin.total_supply_delta`
- `defi.chain.tvl_rotation_share`
- `defi.dex.volume_panic_ratio`
- `defi.protocol.tvl_drawdown_breadth`
- `defi.category.lending_stress`

**Stakes**

This is not just crypto color. Stablecoin supply and on-chain stress are often
faster liquidity state variables than conventional macro series.

### 4. Supply-chain price pressure remains high value, but as breadth

**Why it ranks fourth**

- It has strong cross-asset transmission into inflation, margins, and central
  bank response.
- The extractor exists, but the important opportunity is breadth and diffusion,
  not many additional raw fields.

**Highest-value signals**

- `supply_chain.breadth.accelerating_sectors`
- `supply_chain.import_domestic_spread`
- `supply_chain.ppi_cpi_passthrough_gap`
- `supply_chain.energy_input_pressure`
- `supply_chain.semiconductor_pressure`

**Stakes**

The current `supply_chain.pressure_index` is directionally right, but breadth
signals tell us whether pressure is concentrated or systemic. That distinction
matters for the world model.

### 5. Consumer sentiment should be deepened selectively, not exploded

**Why it ranks fifth**

- The extractor already covers headline EU/US/CPI gap signals.
- The highest remaining value is in breadth, divergence, and expectation gaps,
  not dozens of new survey line items.

**Highest-value additions**

- `consumer_sentiment.eu.breadth_negative`
- `consumer_sentiment.eu.major_purchases_diffusion`
- `consumer_sentiment.us.1y_5y_expectation_gap`
- `consumer_sentiment.transatlantic_divergence`
- `consumer_sentiment.reality_gap_persistence`

**Stakes**

Useful for recession and credibility regimes, but not as orthogonal as internet,
power, or DeFi.

### 6. Political risk is useful, but regime-bound

**Why it ranks sixth**

- OpenFEC is legitimate and updated nightly.
- However, the signal is US-election-centric and strongest in campaign windows.

**Highest-value additions**

- `political_risk.race_spend_concentration`
- `political_risk.oppose_support_imbalance`
- `political_risk.cash_burn_acceleration`
- `political_risk.policy_battle_breadth`

**Stakes**

Worth adding after the more universal surfaces above, especially if the world
model will include US regulatory-regime nodes.

### 7. Food security is strategic, not tactical

**Why it ranks low for near-term convergence**

- The World Bank food-production series is annual.
- It is excellent for slow structural state estimation, not daily or weekly
  convergence detection.

**Keep / don't overinvest**

- Keep the existing food-security signals for structural world-model priors.
- Do not prioritize adding many more annual-only food fields before faster
  signals are integrated.

### 8. FCC / spectrum should remain deprioritized

Research already showed the free programmatic surface is blocked, stale, or dead.
This is not a good use of engineering time right now.

## Recommended Build Order

### Tier 1 — highest-value next signal work

1. Convert `internet_infrastructure` from stub to real extractor-backed evidence
   with outage and censorship signals.
2. Add a real `power_grid` convergence extractor for demand, spread, and zone
   breadth signals.
3. Add a `defi_flows` convergence extractor focused on stablecoin supply,
   rotation, and DEX stress.

### Tier 2 — deepen already-good surfaces

4. Expand `supply_chain_prices` from single pressure metrics into breadth and
   pass-through signals.
5. Expand `consumer_sentiment` with diffusion and divergence signals, not raw
   field proliferation.
6. Expand `political_risk` with concentration and imbalance signals.

### Tier 3 — defer

7. Keep `food_security` mostly as a structural / slow-moving prior surface.
8. Keep FCC / spectrum skipped unless a materially better free source appears.

## Signal-Design Guidance

The next useful signals should prefer these shapes:

- **Breadth signals** — how many countries/zones/protocols/sectors are affected
- **Spread signals** — day-ahead vs real-time, import vs domestic, survey vs reality
- **Concentration signals** — whether risk is broad or concentrated in one node
- **Persistence / acceleration signals** — not just level, but worsening speed
- **Gap signals** — expectations vs outcomes, normal vs disrupted baseline

Avoid spending the next phase on adding many extra metadata fields to
`ConvergenceSignal`. Right now the bigger bottleneck is missing evidence
surfaces, not output-schema richness.

## Risks

- Overweighting slow annual signals will make convergence look academically rich
  but operationally late.
- Adding many low-level survey fields can create feature bloat without adding new
  causal information.
- Crypto and power signals need normalization discipline or they can dominate due
  to cadence rather than importance.
- Internet-disruption signals need strong geographic aggregation to avoid
  overfitting to one-country noise.

## Conclusion

The most useful next signals are the ones that add **new causal surfaces** to
convergence:

1. internet infrastructure
2. power/grid stress
3. DeFi liquidity

After that, deepen breadth and gap signals in supply-chain prices and consumer
sentiment. Do not prioritize FCC/spectrum. Do not spend the next step mainly on
adding more output fields.

---

## Related

- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]

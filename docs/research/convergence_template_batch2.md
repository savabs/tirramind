---
title: "Research: Convergence Template Batch 2 (Templates #23–#50)"
tags:
  - doc/research
  - layer/surveillance
  - phase/7c
  - topic/convergence
---

# Research: Convergence Template Batch 2 (Templates #23–#50)

## Goal
Expand from 22 → 50 templates, hard-capped to current surveillance surface (46 extractors, ~150+ signal_ids). Every template must be grounded in signals we actually observe.

## Current Architecture
- 22 templates in TEMPLATE_LIBRARY (12 original + 10 batch-1)
- 11 taxonomy categories, CausalTemplate with TemplateStep(category_pattern, signal_pattern, within_days, direction)
- match_template() handles temporal ordering, direction constraints, partial matching
- 187 passing tests across 2 test files

## Signal Inventory (by extractor → signal prefix)
| Extractor | Signal Prefixes | Category |
|-----------|----------------|----------|
| ais_vessel_tracking | ais.{zone}.vessel_count, ais.{zone}.tanker_ratio, ais.destination.{chokepoint} | physical_flow |
| weather_alerts | weather.us.alert_count, weather.global.fire_count_infra, weather.us.severe_alert_count | physical_disruption |
| sanctions_monitor | sanctions.global.recent_additions, sanctions.global.program_count | regulatory_action |
| cftc | cftc.{commodity}.mm_net_pct_oi | positioning |
| finra_short_volume | finra.{ticker}.short_ratio, finra.market.avg_short_ratio, finra.{ticker}.days_to_cover | positioning |
| disease_surveillance | disease.{name}.detection_rate, disease.who.outbreak_count | biological |
| earthquake_proximity | earthquake.{zone}.count, earthquake.global.near_infrastructure | physical_disruption |
| global_pmi | pmi.{country}.{mode}, pmi.{country}.{mode}.mom | macro_momentum |
| treasury_receipts | treasury.tga.daily_change_pct, treasury.us.net_flow_today | macro_momentum |
| job_postings | jobs.us.{suffix} | behavioral_intent |
| transport_throughput | transport.us.border_total, transport.us.volume_change | physical_flow |
| capital_flows | capital_flows.ust.coordinated_selling/buying, capital_flows.{country}.holdings_mom_pct, capital_flows.reserves.stress_count | monetary_policy |
| sovereign_debt | sovereign.us.curve_2s10s, sovereign.us.curve_3m10y, sovereign.{country}.spread_vs_de | financial_stress |
| creditor_filings | creditor.sec.filing_count, creditor.uk.red_flags, creditor.sec.cluster_count | financial_stress |
| bankruptcy_court | bankruptcy.{mode}.filing_count, bankruptcy.us.chapter_11, bankruptcy.sec.enforcement_count | financial_stress |
| liquidity_regime | liquidity.us.regime, liquidity.us.composite_zscore, liquidity.us.changepoint_count | financial_stress |
| central_bank_balance | cb.{code}.balance_wow_pct, cb.us.net_liquidity_usd, cb.global.policy_synchronized, cb.global.divergence_count | monetary_policy |
| drug_regulatory | fda.approvals.count, fda.adverse_events.seriousness_ratio, fda.labels.boxed_warning_count | regulatory_action |
| regulatory_gazette | regulatory.us.document_count, regulatory.us.significant_count | regulatory_action |
| building_permits | permits.us.{type}.mom_pct, permits.us.{type}.consecutive_declines | macro_momentum |
| patent_filings | patent.{cpc}.total_count, patent.{cpc}.yoy_growth | behavioral_intent |
| lobbying | lobbying.{slug}.spend_anomaly, lobbying.us.filing_count | behavioral_intent |
| wikipedia_pageviews | wiki.{slug}.spike_zscore, wiki.{slug}.views_zscore | behavioral_intent |
| cert_transparency | cert.{slug}.count, cert.{slug}.active_ratio | behavioral_intent |
| dns_monitor | dns.{slug}.change_count, dns.bulk.total_records | behavioral_intent |
| polymarket | polymarket.{slug}.probability, polymarket.{slug}.price_change_24h | positioning |
| polymarket_whales | polymarket_whales.avg_composite, polymarket_whales.market_concentration | positioning |
| insider_filings | insider.market.purchase_count, insider.{slug}.cluster | positioning |
| form144 | form144.market.filing_count, form144.{slug}.sell_cluster | positioning |
| gdelt | gdelt.global.avg_goldstein, gdelt.global.event_count, gdelt.global.material_conflict_ratio | geopolitical |
| whale_alert | crypto.btc.large_tx_count, crypto.btc.whale_volume | financial_stress |
| comtrade | comtrade.{slug}.{flow}_value_usd, comtrade.{commodity}.trade_volume | supply_chain |
| energy_supply | energy.{slug}.level, energy.{slug}.wow_change, energy.rig_count.total | physical_flow |
| supply_chain_prices | supply_chain.{slug}.mom_pct, supply_chain.pressure_index | supply_chain |
| macro_data | macro.{slug}.latest | macro_momentum |
| market_data | market.{slug}.return, market.{slug}.volume | positioning |
| satellite_activity | satellite.fire.*, satellite.vegetation.*, satellite.events.* | physical_disruption / supply_chain |
| internet_infrastructure | internet.outage.*, internet.censorship.*, internet.signals.*, internet.incidents.* | physical_disruption |
| consumer_sentiment | consumer_sentiment.eu.*, consumer_sentiment.us.*, consumer_sentiment.cpi.mom | macro_momentum |
| food_security | food_security.{country}.* | biological |
| political_risk | political_risk.ie_total_spend, political_risk.oppose_ratio, political_risk.target.* | geopolitical |
| power_grid | power_grid.demand.*, power_grid.fuel.*, power_grid.pricing.*, power_grid.forecast.* | physical_flow |
| defi_flows | defi.tvl.*, defi.stablecoin.*, defi.dex.*, defi.chain.* | financial_stress |

## Gap Analysis vs Existing 22 Templates

### Uncovered macro themes
1. **Currency/BOP crisis** — no template for EM currency crisis, dollar squeeze
2. **Sovereign debt spiral** — distinct from credit_stress_cascade (that's corporate credit)
3. **Fiscal dominance** — government spending overriding monetary policy (2020s-relevant)
4. **Real estate bubble** — permits + credit + sentiment → bust
5. **Inflation persistence** — wage-price spiral signals
6. **Deflation trap** — opposite: demand collapse spiral
7. **Chokepoint disruption** — specific shipping chokepoint (Suez/Panama/Hormuz)
8. **Dark fleet expansion** — shadow fleet growing under sanctions
9. **Liquidity freeze** — sudden stop in market liquidity
10. **Digital bank run** — crypto + tradfi bank run
11. **Financial contagion cross-sector** — one sector failure spreads
12. **Climate → insurance → financial** — physical event cascade
13. **Water stress → food crisis** — drought → crop → food security
14. **Stablecoin depeg** — stablecoin-specific cascade
15. **Drug safety crisis** — FDA adverse events → company collapse
16. **Election positioning** — political cycle → market positioning
17. **Critical mineral bottleneck** — rare earth / lithium supply crunch
18. **Internet censorship escalation** — censorship → capital flight
19. **Bond-equity divergence** — cross-market signal divergence
20. **Supply chain decoupling** — friend-shoring / reshoring signals

### Conceptual foundations (web research)
- **Currency crisis (Krugman 1979, Obstfeld 1986)**: First/second/third generation models. Speculative attacks on pegs, self-fulfilling prophecies, twin crises (banking + currency). Key signals: capital flow reversal, FX reserve depletion, sovereign spreads.
- **Sovereign default (Reinhart & Rogoff "This Time Is Different")**: Over-indebtedness, capital flow reversal, rising interest rates. Debt-to-GDP + spread widening as leading indicators.
- **Fiscal dominance (Sargent & Wallace)**: Government deficits force monetary accommodation → inflation. COVID-era example: $5-6T stimulus + near-zero rates. Signals: treasury flows, CB balance, inflation expectations.
- **Real estate bubble**: Building permits, price-to-income, credit growth → bust → banking stress. Leading indicators: permit momentum, credit filings.
- **Wage-price spiral**: Inflation expectations → labor demand → supply chain pressure → PMI input costs. Self-reinforcing loop.

## 28 New Template Designs

### Cluster: Currency/FX (3)
23. currency_crisis_em — CB action → capital outflow → spreads → turmoil → positioning
24. dollar_squeeze — CB tightening → sovereign stress → capital flight → DeFi stress → PMI
25. twin_deficit_crisis — Treasury receipts fall → capital outflow → spreads → political risk → positioning

### Cluster: Sovereign/Fiscal (2)
26. sovereign_debt_spiral — Spreads spike → capital flight → creditor stress → turmoil → sentiment collapse
27. fiscal_dominance — CB action → treasury flows → sovereign stress → inflation expectations → bond positioning

### Cluster: Real Estate (2)
28. real_estate_bubble — Permits surge → CB policy → lobbying → credit stress → sentiment drop
29. construction_bust_banking — Permits collapse → jobs drop → bankruptcy → liquidity stress → PMI drop

### Cluster: Inflation/Deflation (2)
30. inflation_persistence — Inflation expectations → labor demand → supply pressure → PMI up → CB response
31. deflation_trap — Sentiment collapse → PMI drop → jobs drop → bankruptcy → CB response

### Cluster: Shipping/Chokepoint (3)
32. chokepoint_disruption — Geopolitical event → AIS rerouting → supply chain pressure → energy → positioning
33. dark_fleet_expansion — Sanctions → AIS anomaly → GDELT → positioning → energy
34. shipping_regime_change — AIS/transport changes → trade flows → PMI → positioning → Wikipedia attention

### Cluster: Liquidity/Banking (3)
35. liquidity_freeze — Liquidity regime change → sovereign stress → CB response → shorts spike → PMI
36. bank_run_digital — DeFi stress → Wikipedia panic → creditor/bankruptcy → liquidity → Polymarket
37. contagion_cascade — Bankruptcy filing → sovereign stress → DeFi stress → capital flight → sentiment

### Cluster: Climate/Environment (2)
38. climate_insurance_cascade — Weather event → satellite fire → creditor stress → permits drop → sentiment
39. water_stress_food_crisis — Drought → vegetation anomaly → food security → trade flows → GDELT

### Cluster: Crypto/Digital (2)
40. stablecoin_depeg — Stablecoin stress → DEX panic → crypto whale → Wikipedia → Polymarket
41. crypto_energy_nexus — Power grid stress → DeFi stress → energy supply → regulatory → Wikipedia

### Cluster: Pharma (2)
42. drug_safety_crisis — FDA adverse events → Wikipedia → insider selling → creditor filings → shorts
43. pharma_pipeline_collapse — FDA rejection → insider selling → patent drop → shorts → bankruptcy

### Cluster: Political (2)
44. election_positioning — Political risk → lobbying → Polymarket → regulatory → sentiment
45. regime_change_market — GDELT/political risk → Wikipedia → capital flows → sanctions/regulation → positioning

### Cluster: Materials/Trade (2)
46. critical_mineral_bottleneck — Sanctions/regulation → trade flows → patents → AIS → positioning
47. supply_chain_decoupling — Sanctions/regulation → trade flows → lobbying/patents → AIS/transport → permits

### Cluster: Cross-Market (2)
48. bond_equity_divergence — Sovereign stress → liquidity → PMI drop → positioning → sentiment
49. commodity_demand_collapse — Energy/transport drop → AIS → PMI → supply chain pressure → positioning

### Cluster: Internet/Censorship (1)
50. internet_censorship_escalation — Censorship anomaly → DNS/cert changes → GDELT → capital flight → DeFi

## Risks
- 50 templates creates a larger multiple-testing penalty in FDR control (manageable since FDR scales logarithmically, not linearly)
- Some templates share sub-chains (e.g., capital_flows → sovereign_debt appears in 5+ templates). This is intentional — same mechanism, different trigger conditions.
- Long temporal windows (45-90 days) in some templates mean slower detection but catch gradual regime changes

## Reuse Constraints
All conceptual frameworks used here (Krugman, Reinhart/Rogoff, Sargent/Wallace, Minsky) are well-established economic theory. Our implementation uses only our own signal patterns and matching logic. No external code ported.

---

## Related

- [[convergence_template_batch2_spec|Spec: Convergence Template Batch2]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_audit_pre_worldmodel]]

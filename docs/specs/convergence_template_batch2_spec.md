---
title: "Spec: Convergence Template Batch 2 (#23–#50)"
tags:
  - doc/spec
  - phase/7c
  - topic/convergence
---

# Spec: Convergence Template Batch 2 (#23–#50)

## Goal
Add 28 CausalTemplate entries to TEMPLATE_LIBRARY, reaching exactly 50 templates total.

## Files Affected
- `agent/convergence/templates.py` — append 28 CausalTemplate entries, update header count
- `tests/test_convergence_templates.py` — update hardcoded counts (22→50)
- `tests/test_convergence_template_expansion.py` — update library count assertion (22→50)
- `tests/test_convergence_template_batch2.py` — new test file for 28 templates

## Implementation Steps

### Step 1: Add 28 templates to templates.py

All 5-step templates use min_match=0 (effective=3). All step-0 have within_days=0.
Step within_days values are monotonically non-decreasing within each template.

#### #23 currency_crisis_em (5 steps)
"Emerging market currency crisis: central bank intervention triggers capital flight, sovereign stress, geopolitical turmoil, and speculative positioning."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | monetary_policy | `central_bank\.\|capital_flows\.` | 0 | None |
| 1 | monetary_policy | `capital_flows\.` | 7 | -1 |
| 2 | financial_stress | `sovereign_debt\.\|sovereign\.` | 14 | +1 |
| 3 | geopolitical | `gdelt\.\|political_risk\.` | 21 | +1 |
| 4 | positioning | `cftc\.\|finra\.` | 30 | -1 |

#### #24 dollar_squeeze (5 steps)
"Dollar funding squeeze propagates from monetary tightening through sovereign stress, capital reversal, DeFi liquidations, and macro contraction."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | monetary_policy | `central_bank\.\|rate_monitor\.` | 0 | None |
| 1 | financial_stress | `sovereign_debt\.\|sovereign\.` | 7 | +1 |
| 2 | monetary_policy | `capital_flows\.` | 14 | -1 |
| 3 | financial_stress | `defi\.` | 21 | +1 |
| 4 | macro_momentum | `pmi\.` | 30 | -1 |

#### #25 twin_deficit_crisis (5 steps)
"Budget and current account deficits converge into crisis: falling tax receipts, capital outflow, sovereign stress, political pressure, speculative positioning."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | macro_momentum | `treasury\.` | 0 | -1 |
| 1 | monetary_policy | `capital_flows\.` | 7 | -1 |
| 2 | financial_stress | `sovereign_debt\.\|sovereign\.` | 14 | +1 |
| 3 | geopolitical | `political_risk\.` | 21 | +1 |
| 4 | positioning | `cftc\.\|finra\.` | 30 | +1 |

#### #26 sovereign_debt_spiral (5 steps)
"Sovereign debt spiral: spreads spike, capital flees, creditors file, political turmoil erupts, consumer sentiment collapses."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | financial_stress | `sovereign_debt\.\|sovereign\.` | 0 | +1 |
| 1 | monetary_policy | `capital_flows\.` | 7 | -1 |
| 2 | financial_stress | `creditor\.\|bankruptcy\.` | 14 | +1 |
| 3 | geopolitical | `gdelt\.\|political_risk\.` | 21 | +1 |
| 4 | macro_momentum | `consumer_sentiment\.` | 30 | -1 |

#### #27 fiscal_dominance (5 steps)
"Government fiscal pressures override monetary policy: central bank acts, treasury flows shift, sovereign stress rises, inflation expectations embed, bond positioning adjusts."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | monetary_policy | `central_bank\.` | 0 | None |
| 1 | macro_momentum | `treasury\.` | 7 | +1 |
| 2 | financial_stress | `sovereign_debt\.\|sovereign\.` | 14 | +1 |
| 3 | macro_momentum | `consumer_sentiment\.` | 21 | +1 |
| 4 | positioning | `cftc\.` | 30 | -1 |

#### #28 real_estate_bubble (5 steps)
"Housing bubble formation: building permits surge, monetary policy accommodates, real estate lobbying intensifies, credit stress emerges, consumer confidence erodes."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | macro_momentum | `permits\.\|building_permits\.` | 0 | +1 |
| 1 | monetary_policy | `central_bank\.\|rate_monitor\.` | 14 | None |
| 2 | behavioral_intent | `lobbying\.` | 21 | +1 |
| 3 | financial_stress | `creditor\.\|bankruptcy\.` | 30 | +1 |
| 4 | macro_momentum | `consumer_sentiment\.` | 45 | -1 |

#### #29 construction_bust_banking (5 steps)
"Construction cycle collapse cascades into banking: permits fall, construction jobs vanish, bankruptcies spike, liquidity dries up, PMI contracts."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | macro_momentum | `permits\.\|building_permits\.` | 0 | -1 |
| 1 | behavioral_intent | `jobs\.` | 14 | -1 |
| 2 | financial_stress | `bankruptcy\.\|creditor\.` | 21 | +1 |
| 3 | financial_stress | `liquidity\.` | 30 | +1 |
| 4 | macro_momentum | `pmi\.` | 45 | -1 |

#### #30 inflation_persistence (5 steps)
"Wage-price spiral builds: inflation expectations rise, labor demand surges, supply chain pressure increases, PMI input costs climb, central bank forced to respond."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | macro_momentum | `consumer_sentiment\.` | 0 | +1 |
| 1 | behavioral_intent | `jobs\.` | 7 | +1 |
| 2 | supply_chain | `supply_chain\.` | 14 | +1 |
| 3 | macro_momentum | `pmi\.` | 21 | +1 |
| 4 | monetary_policy | `central_bank\.` | 30 | None |

#### #31 deflation_trap (5 steps)
"Deflationary spiral: consumer confidence collapses, PMI contracts, job postings evaporate, bankruptcies mount, central bank forced to respond."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | macro_momentum | `consumer_sentiment\.` | 0 | -1 |
| 1 | macro_momentum | `pmi\.` | 7 | -1 |
| 2 | behavioral_intent | `jobs\.` | 14 | -1 |
| 3 | financial_stress | `bankruptcy\.` | 21 | +1 |
| 4 | monetary_policy | `central_bank\.` | 30 | None |

#### #32 chokepoint_disruption (5 steps)
"Maritime chokepoint disruption: geopolitical event triggers AIS rerouting, supply chain stress, energy price spikes, and commodity positioning."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | geopolitical | `gdelt\.\|political_risk\.` | 0 | +1 |
| 1 | physical_flow | `ais\.` | 7 | None |
| 2 | supply_chain | `supply_chain\.` | 14 | +1 |
| 3 | physical_flow | `energy_supply\.\|energy\.` | 21 | +1 |
| 4 | positioning | `cftc\.` | 30 | +1 |

#### #33 dark_fleet_expansion (5 steps)
"Sanctions-driven shadow fleet: new sanctions prompt dark shipping activity, geopolitical coverage, speculative positioning, and energy market disruption."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | regulatory_action | `sanctions\.` | 0 | +1 |
| 1 | physical_flow | `ais\.` | 7 | None |
| 2 | geopolitical | `gdelt\.` | 14 | +1 |
| 3 | positioning | `cftc\.` | 21 | +1 |
| 4 | physical_flow | `energy_supply\.\|energy\.` | 30 | None |

#### #34 shipping_regime_change (5 steps)
"Shipping rate regime change: vessel and transport patterns shift, trade flows adjust, PMI reflects, positioning moves, public attention follows."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | physical_flow | `ais\.\|transport\.` | 0 | None |
| 1 | supply_chain | `supply_chain\.\|comtrade\.` | 7 | None |
| 2 | macro_momentum | `pmi\.` | 14 | None |
| 3 | positioning | `cftc\.` | 21 | None |
| 4 | behavioral_intent | `wikipedia\.\|wiki\.` | 30 | +1 |

#### #35 liquidity_freeze (5 steps)
"Sudden liquidity freeze: regime change detected, sovereign spreads widen, central bank responds, short interest spikes, PMI contracts."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | financial_stress | `liquidity\.` | 0 | +1 |
| 1 | financial_stress | `sovereign_debt\.\|sovereign\.` | 3 | +1 |
| 2 | monetary_policy | `central_bank\.` | 7 | None |
| 3 | positioning | `finra\.` | 14 | +1 |
| 4 | macro_momentum | `pmi\.` | 30 | -1 |

#### #36 bank_run_digital (5 steps)
"Digital bank run: DeFi stress triggers Wikipedia panic search, creditor filings emerge, liquidity freezes, prediction markets reprice."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | financial_stress | `defi\.` | 0 | +1 |
| 1 | behavioral_intent | `wikipedia\.\|wiki\.` | 3 | +1 |
| 2 | financial_stress | `creditor\.\|bankruptcy\.` | 7 | +1 |
| 3 | financial_stress | `liquidity\.` | 14 | +1 |
| 4 | positioning | `polymarket\.` | 21 | None |

#### #37 contagion_cascade (5 steps)
"Cross-sector financial contagion: corporate distress spreads to sovereign bonds, crypto, capital flows, and consumer sentiment."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | financial_stress | `bankruptcy\.\|creditor\.` | 0 | +1 |
| 1 | financial_stress | `sovereign_debt\.\|sovereign\.` | 7 | +1 |
| 2 | financial_stress | `defi\.` | 14 | +1 |
| 3 | monetary_policy | `capital_flows\.` | 21 | -1 |
| 4 | macro_momentum | `consumer_sentiment\.` | 30 | -1 |

#### #38 climate_insurance_cascade (5 steps)
"Climate event cascades through insurance and construction sectors: physical event triggers satellite confirmation, creditor stress, construction pullback, sentiment drop."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | physical_disruption | `weather\.\|earthquake\.` | 0 | +1 |
| 1 | physical_disruption | `satellite\.fire` | 7 | +1 |
| 2 | financial_stress | `creditor\.\|bankruptcy\.` | 14 | +1 |
| 3 | macro_momentum | `permits\.\|building_permits\.` | 30 | -1 |
| 4 | macro_momentum | `consumer_sentiment\.` | 45 | -1 |

#### #39 water_stress_food_crisis (5 steps)
"Water stress to food crisis: drought severity rises, satellite shows crop stress, food security alerts fire, trade flows shift, geopolitical tensions escalate."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | physical_disruption | `weather\.` | 0 | +1 |
| 1 | supply_chain | `satellite\.vegetation` | 14 | -1 |
| 2 | biological | `food_security\.` | 21 | +1 |
| 3 | physical_flow\|supply_chain | `ais\.\|comtrade\.` | 30 | None |
| 4 | geopolitical | `gdelt\.` | 45 | +1 |

#### #40 stablecoin_depeg (5 steps)
"Stablecoin depeg cascade: stablecoin metrics stress, DEX panic volume spikes, crypto whale movements, Wikipedia panic search, prediction markets reprice."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | financial_stress | `defi\.stablecoin` | 0 | +1 |
| 1 | financial_stress | `defi\.dex` | 3 | +1 |
| 2 | financial_stress | `crypto\.\|whale_alert\.` | 7 | +1 |
| 3 | behavioral_intent | `wikipedia\.\|wiki\.` | 14 | +1 |
| 4 | positioning | `polymarket\.` | 21 | None |

#### #41 crypto_energy_nexus (5 steps)
"Crypto mining strains energy grid, triggering DeFi stress, energy supply disruption, regulatory response, and public attention."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | physical_flow | `power_grid\.` | 0 | +1 |
| 1 | financial_stress | `defi\.` | 7 | +1 |
| 2 | physical_flow | `energy_supply\.\|energy\.` | 14 | +1 |
| 3 | regulatory_action | `regulatory_gazette\.\|regulatory\.` | 21 | None |
| 4 | behavioral_intent | `wikipedia\.\|wiki\.` | 30 | +1 |

#### #42 drug_safety_crisis (5 steps)
"Drug safety crisis: FDA adverse events spike, Wikipedia attention rises, insiders sell, creditor filings emerge, short sellers pile on."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | regulatory_action | `fda\.\|drug_regulatory\.` | 0 | +1 |
| 1 | behavioral_intent | `wikipedia\.\|wiki\.` | 7 | +1 |
| 2 | positioning | `form144\.\|insider\.` | 14 | -1 |
| 3 | financial_stress | `creditor\.` | 21 | +1 |
| 4 | positioning | `finra\.` | 30 | +1 |

#### #43 pharma_pipeline_collapse (5 steps)
"Pharma pipeline failure: drug rejection triggers insider selling, patent activity drops, shorts accumulate, bankruptcy risk rises."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | regulatory_action | `fda\.\|drug_regulatory\.` | 0 | -1 |
| 1 | positioning | `form144\.\|insider\.` | 7 | -1 |
| 2 | behavioral_intent | `patent\.` | 14 | -1 |
| 3 | positioning | `finra\.` | 21 | +1 |
| 4 | financial_stress | `bankruptcy\.` | 30 | +1 |

#### #44 election_positioning (5 steps)
"Election cycle market positioning: political risk rises, lobbying intensifies, prediction markets shift, regulatory activity changes, consumer sentiment adjusts."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | geopolitical | `political_risk\.` | 0 | +1 |
| 1 | behavioral_intent | `lobbying\.` | 14 | +1 |
| 2 | positioning | `polymarket\.` | 21 | None |
| 3 | regulatory_action | `regulatory_gazette\.\|regulatory\.` | 30 | None |
| 4 | macro_momentum | `consumer_sentiment\.` | 45 | None |

#### #45 regime_change_market (5 steps)
"Political regime change triggers market reorientation: geopolitical event detected, Wikipedia attention surges, capital flows adjust, regulatory landscape shifts, positioning realigns."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | geopolitical | `gdelt\.\|political_risk\.` | 0 | +1 |
| 1 | behavioral_intent | `wikipedia\.\|wiki\.` | 7 | +1 |
| 2 | monetary_policy | `capital_flows\.` | 14 | None |
| 3 | regulatory_action | `sanctions\.\|regulatory_gazette\.` | 21 | None |
| 4 | positioning | `cftc\.\|insider\.` | 30 | None |

#### #46 critical_mineral_bottleneck (5 steps)
"Critical mineral supply crunch: regulatory/sanctions action, trade flow disruption, patent race intensifies, shipping adjusts, speculative positioning builds."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | regulatory_action | `sanctions\.\|regulatory_gazette\.` | 0 | +1 |
| 1 | supply_chain | `comtrade\.\|supply_chain\.` | 7 | None |
| 2 | behavioral_intent | `patent\.` | 14 | +1 |
| 3 | physical_flow | `ais\.` | 21 | None |
| 4 | positioning | `cftc\.` | 30 | +1 |

#### #47 supply_chain_decoupling (5 steps, 45d window)
"Friend-shoring and supply chain decoupling: regulatory triggers, trade flows shift, lobbying/patent activity surges, shipping reroutes, construction permits rise."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | regulatory_action | `sanctions\.\|regulatory_gazette\.` | 0 | +1 |
| 1 | supply_chain | `comtrade\.\|supply_chain\.` | 14 | None |
| 2 | behavioral_intent | `lobbying\.\|patent\.` | 21 | +1 |
| 3 | physical_flow | `ais\.\|transport\.` | 30 | None |
| 4 | macro_momentum | `permits\.\|building_permits\.` | 45 | +1 |

#### #48 bond_equity_divergence (5 steps)
"Bond market signals recession while equities lag: sovereign stress, liquidity tightens, PMI contracts, positioning shifts, consumer sentiment confirms."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | financial_stress | `sovereign_debt\.\|sovereign\.` | 0 | +1 |
| 1 | financial_stress | `liquidity\.` | 7 | +1 |
| 2 | macro_momentum | `pmi\.` | 14 | -1 |
| 3 | positioning | `cftc\.` | 21 | -1 |
| 4 | macro_momentum | `consumer_sentiment\.` | 30 | -1 |

#### #49 commodity_demand_collapse (5 steps)
"Physical demand collapse signals: energy/transport volumes drop, AIS confirms shipping decline, PMI contracts, supply chain pressure falls, positioning adjusts."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | physical_flow | `energy_supply\.\|energy\.\|transport\.` | 0 | -1 |
| 1 | physical_flow | `ais\.` | 7 | None |
| 2 | macro_momentum | `pmi\.` | 14 | -1 |
| 3 | supply_chain | `supply_chain\.` | 21 | -1 |
| 4 | positioning | `cftc\.` | 30 | -1 |

#### #50 internet_censorship_escalation (5 steps)
"Internet censorship escalates: censorship anomaly detected, DNS/cert infrastructure changes, geopolitical coverage spikes, capital flees, crypto activity surges."
| Step | category_pattern | signal_pattern | within_days | direction |
|------|-----------------|---------------|-------------|-----------|
| 0 | physical_disruption | `internet\.censorship\|internet\.outage` | 0 | +1 |
| 1 | behavioral_intent | `dns\.\|cert\.` | 7 | None |
| 2 | geopolitical | `gdelt\.` | 14 | +1 |
| 3 | monetary_policy | `capital_flows\.` | 21 | -1 |
| 4 | financial_stress | `defi\.` | 30 | +1 |

### Step 2: Update template count in header comment
Change "22 templates" → "50 templates"

### Step 3: Update test files
- `tests/test_convergence_templates.py`: 22→50 counts
- `tests/test_convergence_template_expansion.py`: 22→50 count assertion

### Step 4: Write batch-2 test suite
Create `tests/test_convergence_template_batch2.py` covering:
- All 28 new templates present
- 50-template total count
- Name uniqueness across all 50
- Structural property tests (categories, regex, direction, within_days)
- Synthetic matching for each new template
- Edge cases

## Edge Cases
- Templates with identical trigger categories but different signal patterns (e.g., currency_crisis_em vs dollar_squeeze both start with monetary_policy but different regexes)
- Water_stress_food_crisis step 3 has pipe-separated category "physical_flow|supply_chain"
- Stablecoin_depeg has 3-day window on step 1 (fast cascade)
- Pharma_pipeline_collapse step 0 has direction=-1 (rejection, not approval)

## Testing Plan
1. Structural validation: all 28 templates load, correct step counts, valid categories/regexes
2. Full match: synthetic evidence perfectly matching each template
3. Direction constraints: pharma_pipeline_collapse step 0 direction=-1, drug_safety_crisis step 2 direction=-1
4. Cross-template regression: all original 22 templates unchanged
5. Edge cases: empty cliques, wrong categories, temporal violations

---

## Related

- [[convergence_template_batch2|Research: Convergence Template Batch2]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_template_expansion]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_audit_pre_worldmodel]]

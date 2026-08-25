---
title: "Spec: Convergence Template Expansion — 10 Senior Quant Patterns"
tags:
  - doc/spec
  - phase/7c
  - topic/convergence
---

# Spec: Convergence Template Expansion — 10 Senior Quant Patterns

## Goal
Add 10 new causal chain templates (#13–#22) to TEMPLATE_LIBRARY in `agent/convergence/templates.py`. Each template encodes a non-obvious cross-category causal chain grounded in financial economics literature. No new extractors or tools — only template definitions.

## Research
`[[convergence_template_expansion]]`

## Files Affected
| File | Action |
|------|--------|
| `agent/convergence/templates.py` | Add 10 CausalTemplate entries to TEMPLATE_LIBRARY |
| `tests/test_convergence_templates.py` | Create comprehensive edge-case test suite |
| `[[convergence_template_expansion]]` | Task tracking |

## Implementation Steps

### Step 1: Add 10 templates to TEMPLATE_LIBRARY
Append after template #12 in `templates.py`. Each template is a `CausalTemplate` with:
- `name`: snake_case identifier
- `description`: one-sentence human explanation
- `steps`: tuple of `TemplateStep` entries (category_pattern, signal_pattern, within_days, direction)
- `min_match`: 0 (auto-compute) for 4-step templates, explicit for 5–6 step templates

Template definitions (from research doc):

**13. silent_nationalization** (5 steps, min_match=3)
- Step 0: behavioral_intent, `lobbying\\.`, 0, +1
- Step 1: physical_disruption, `satellite\\.`, 14, +1
- Step 2: regulatory_action, `regulatory_gazette\\.`, 21, None
- Step 3: positioning, `insider\\.|form144\\.`, 30, -1
- Step 4: physical_flow, `ais\\.|transport\\.`, 45, None

**14. defi_canary** (5 steps, min_match=3)
- Step 0: financial_stress, `defi\\.`, 0, +1
- Step 1: financial_stress, `whale_alert\\.`, 3, None
- Step 2: behavioral_intent, `wikipedia\\.`, 7, +1
- Step 3: positioning, `polymarket\\.|polymarket\\.whale`, 14, None
- Step 4: financial_stress, `bankruptcy\\.|creditor\\.`, 30, +1

**15. pandemic_physical_evidence** (5 steps, min_match=3)
- Step 0: biological, `disease\\.`, 0, +1
- Step 1: physical_disruption, `satellite\\.`, 7, +1
- Step 2: behavioral_intent, `wikipedia\\.`, 14, +1
- Step 3: physical_flow, `transport\\.`, 21, -1
- Step 4: positioning, `cftc\\.`, 30, None

**16. capital_flight_crypto** (5 steps, min_match=3)
- Step 0: geopolitical, `political_risk\\.`, 0, +1
- Step 1: monetary_policy, `capital_flows\\.`, 7, -1
- Step 2: financial_stress, `defi\\.`, 14, +1
- Step 3: financial_stress, `sovereign_debt\\.`, 21, +1
- Step 4: monetary_policy, `central_bank\\.`, 30, None

**17. infrastructure_decay_cascade** (5 steps, min_match=3)
- Step 0: physical_flow, `power_grid\\.`, 0, +1
- Step 1: physical_disruption, `internet\\.|dns\\.`, 14, +1
- Step 2: macro_momentum, `building_permits\\.`, 30, -1
- Step 3: behavioral_intent, `jobs\\.`, 45, -1
- Step 4: macro_momentum, `consumer_sentiment\\.`, 60, -1

**18. commodity_hoarding** (5 steps, min_match=3)
- Step 0: physical_disruption, `weather\\.`, 0, +1
- Step 1: supply_chain, `satellite\\.vegetation`, 7, -1
- Step 2: physical_flow, `ais\\.|transport\\.`, 14, None
- Step 3: positioning, `cftc\\.`, 21, +1
- Step 4: biological, `food_security\\.`, 30, +1

**19. smart_money_divergence** (5 steps, min_match=3)
- Step 0: behavioral_intent, `wikipedia\\.`, 0, +1
- Step 1: positioning, `polymarket\\.`, 7, +1
- Step 2: positioning, `form144\\.|insider\\.`, 14, -1
- Step 3: positioning, `finra\\.`, 21, +1
- Step 4: financial_stress, `defi\\.|whale_alert\\.`, 30, +1

**20. sanctions_evasion_network** (5 steps, min_match=3)
- Step 0: regulatory_action, `sanctions\\.`, 0, +1
- Step 1: physical_flow, `ais\\.`, 7, None
- Step 2: behavioral_intent|physical_disruption, `cert_trans\\.|dns\\.`, 14, None
- Step 3: financial_stress, `defi\\.`, 21, +1
- Step 4: geopolitical, `gdelt\\.`, 30, +1

**21. carry_trade_unwind** (6 steps, min_match=4)
- Step 0: monetary_policy, `central_bank\\.|rate_monitor\\.`, 0, None
- Step 1: monetary_policy, `capital_flows\\.`, 7, -1
- Step 2: financial_stress, `defi\\.`, 14, +1
- Step 3: financial_stress, `sovereign_debt\\.`, 21, +1
- Step 4: positioning, `cftc\\.`, 30, -1
- Step 5: macro_momentum, `pmi\\.`, 45, -1

**22. stealth_accumulation** (5 steps, min_match=3)
- Step 0: behavioral_intent, `wikipedia\\.`, 0, +1
- Step 1: behavioral_intent, `lobbying\\.`, 30, +1
- Step 2: behavioral_intent, `patent\\.`, 45, None
- Step 3: positioning, `cftc\\.|insider\\.`, 60, +1
- Step 4: regulatory_action, `regulatory_gazette\\.|drug_regulatory\\.`, 90, None

### Step 2: Write edge-case test suite
Create `tests/test_convergence_templates.py` covering:
- All 10 new templates exist in TEMPLATE_LIBRARY
- Template name uniqueness across all 22 templates
- min_match values are correct (3 for 5-step, 4 for 6-step)
- Step regex patterns compile without error
- Category patterns contain only valid categories
- within_days is monotonically non-decreasing within each template
- Step 0 always has within_days=0
- Direction values are valid (+1, -1, or None)
- Template matching: synthetic evidence that should match each template
- Template matching: evidence that should NOT match (wrong categories, wrong order)
- match_score calculation for partial matches
- Temporal ordering validation
- No duplicate template names between existing 12 and new 10
- Empty clique / empty evidence returns 0 score

## Edge Cases
- 6-step template (carry_trade_unwind) needs min_match=4 explicitly
- smart_money_divergence has opposing directions in steps — ensure direction matching works
- stealth_accumulation has 90-day window — verify long temporal windows work correctly
- sanctions_evasion_network step 2 has pipe-separated category_pattern — ensure matches_category handles it

## Testing Plan
- Unit tests for all template structural properties
- Functional tests with synthetic evidence for each template
- Regression: existing 12 templates unchanged
- Match scoring: verify partial match scoring is correct

---

## Related

- [[convergence_template_expansion|Research: Convergence Template Expansion]]
- [[convergence_detection]]
- [[convergence_backtest]]
- [[convergence_signal_expansion]]
- [[convergence_napm_refresh]]
- [[convergence_backtest_fast_mode]]
- [[convergence_backtest_score_cache]]
- [[convergence_template_batch2]]
- [[convergence_audit_pre_worldmodel]]

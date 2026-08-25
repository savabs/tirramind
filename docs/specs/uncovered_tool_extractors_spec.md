---
title: "Spec: Uncovered Tool Extractors"
tags:
  - doc/spec
---

# Spec: Uncovered Tool Extractors

## Goal
Add convergence extractors for 3 existing data tools (labor_disruptions, gov_contracts, academic_preprints), bringing the total from 46 to 49 registered extractors.

## Files Affected
- `agent/convergence/extractors.py` — append 3 extractor functions + register calls
- `tests/test_convergence_extractors_batch2.py` — new test file for all 3 extractors

## Implementation Steps

### Step 1: labor_disruptions extractor

Extract from `data["signals"]` (overview mode) or single-series mode.

**Signals produced:**

| signal_id | category | source field | direction | confidence | ttl |
|-----------|----------|-------------|-----------|------------|-----|
| strike.us.workers_involved | behavioral_intent | signals.workers.latest_value (overview) or signals.latest_value (work_stoppages) | +1 if trend in {ESCALATING, RISING}, -1 if DECLINING, 0 otherwise | 0.75 | 2,592,000 (30d) |
| strike.us.idle_days | macro_momentum | signals.idle_days.latest_value (overview) or signals.latest_value (idle_days) | same trend logic | 0.70 | 2,592,000 |
| strike.us.intensity | macro_momentum | signals.intensity_ratio (overview only) | +1 if > 1.5, -1 if < 0.5, 0 otherwise | 0.65 | 2,592,000 |
| strike.us.consecutive_months | behavioral_intent | signals.consecutive_active_months (overview only) | +1 if >= 3, 0 if 0, else +1 with lower val | 0.70 | 2,592,000 |

**Defensive handling:** If `data` is not dict or `signals` key missing, return []. If overview mode, extract from nested `workers`/`idle_days` sub-dicts. If single-series mode (has `label` key), extract only the matching signal.

### Step 2: gov_contracts extractor

Extract from `data["awards"]` list + aggregate signals.

**Signals produced:**

| signal_id | category | source field | direction | confidence | ttl |
|-----------|----------|-------------|-----------|------------|-----|
| gov_contract.{region}.award_count | regulatory_action | data.count | +1 always (awards = activity) | 0.65 | 21,600 (6h) |
| gov_contract.{region}.total_value | regulatory_action | sum(a["amount_usd"] for a in awards) | +1 if > 0 | 0.70 | 21,600 |
| gov_contract.{region}.defense_share | geopolitical | fraction of awards where agency contains defense/DoD/MoD keywords | +1 if defense_share > 0.3, 0 otherwise | 0.75 | 21,600 |

Region is `us` (default) or `uk` (if `data.get("region") == "uk"`).

**Defensive handling:** If no awards list or empty, return []. Skip signals with zero/None values.

### Step 3: academic_preprints extractor

Extract from papers mode (`data["papers"]`) or trials mode (`data["studies"]`).

**Signals produced:**

| signal_id | category | source field | direction | confidence | ttl |
|-----------|----------|-------------|-----------|------------|-----|
| trials.active_count | biological | len(studies where status in {Recruiting, Active, Enrolling}) | +1 (activity) | 0.60 | 86,400 (1d) |
| trials.completed_count | regulatory_action | len(studies where status == Completed) | +1 (completion signal) | 0.75 | 86,400 |
| trials.industry_ratio | behavioral_intent | fraction where sponsor_class == INDUSTRY | +1 if > 0.5 | 0.60 | 86,400 |
| arxiv.volume | behavioral_intent | data.count or data.total_results | +1 (research activity) | 0.50 | 86,400 |

**Defensive handling:** Detect mode from keys: `"studies"` → trials, `"papers"` → arxiv. If neither, return [].

### Step 4: Write test suite

File: `tests/test_convergence_extractors_batch2.py`

Cover per extractor:
1. Valid data → correct signal_ids, categories, directions, confidence, ttl
2. Empty/None data → []
3. Malformed data (wrong types, missing keys) → [] (no crash)
4. Overview vs single-series mode (labor_disruptions only)
5. US vs UK region (gov_contracts only)
6. Papers vs trials mode (academic_preprints only)
7. Direction logic edge cases (trend values, threshold boundaries)
8. Defense share threshold boundary (gov_contracts)

## Edge Cases
- labor_disruptions with "INSUFFICIENT_DATA" trend → direction 0
- gov_contracts with all zero-amount awards → total_value signal skipped
- academic_preprints with studies missing "status" field → skip that study
- academic_preprints with empty sponsor_class → skip industry_ratio

## Testing Plan
All tests use synthetic data dicts (no HTTP calls). Verify:
- 49 total extractors registered
- Each new extractor produces expected Evidence list from well-formed data
- Each new extractor returns [] from malformed data
- Signal IDs match documented patterns
- Categories match taxonomy

---

## Related

- [[uncovered_tool_extractors|Research: Uncovered Tool Extractors]]

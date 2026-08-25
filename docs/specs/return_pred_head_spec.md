---
title: "Spec: Return Supervision & Graph Builder Fixes (A1/A2/A3)"
tags:
  - doc/spec
  - phase/47
  - topic/world-model
  - layer/feature-engineering
  - layer/world-model
  - status/active
---

# Spec: Return Supervision & Graph Builder Fixes

## Goal

Increase IC from -0.033 (noise) toward IC > 0.03 by (1) retroactively
removing low-tension GDELT noise from the DB, (2) switching the return
supervision signal from daily log_return to 21-day forward returns, and
(3) fixing a silent key-name bug that caused obs_type_dist enrichment
features to be all-zeros.

## Research Reference

[[architecture_stress_test]] — full failure mode analysis, ordered fix plan,
leakage audit results.

## Files Affected

| File | Change |
|------|--------|
| `.tirra_pipeline/pipeline.db` | A1: DELETE geopolitical_event with goldstein >= -5.0 (DONE) |
| `agent/models/gnn/trainer.py` | A2: add `_build_forward_return_lookup()` + wire into return loss |
| `agent/models/gnn/graph_builder.py` | A3: fix `obs_type` → `observation_type` key |

## Implementation Steps

### Step A1: DB cleanup (DONE ✅)
- Backed up to `.tirra_pipeline/pipeline.db.bak_20260512`
- Deleted 811,020 low-tension GDELT events (goldstein >= -5.0)
- Result: 1,150,184 → 339,164 total obs; 901,704 → 90,684 geopolitical_event

### Step A2: 21-day forward return supervision (trainer.py)

**Problem:** Current target = `log_return` (daily return stored in obs). Daily
returns have near-zero predictive signal at cross-sectional level (efficient
market at 1-day horizon). 21-day returns have higher persistence and IC.

**Fix:**
1. Add `forward_return_horizon: int = 21` to `TrainerConfig`.
2. Add `use_forward_returns: bool = True` to `TrainerConfig`.
3. Add `_build_forward_return_lookup(obs, horizon_days)` standalone function:
   - Groups instrument_daily obs by entity_id
   - Sorts each group by observed_at
   - For each obs_i, finds the obs closest to obs_i.observed_at + horizon_days*86400*7/5
     (21 trading days ≈ 29.4 calendar days; search in ±5-day window)
   - Computes `(close_later - close_now) / abs(close_now)` (normalised)
   - NaN/Inf entries excluded; returns `{(entity_id, int(observed_at)) -> fwd_return}`
4. In `train()`, after prefetch, compute lookup and store as `self._forward_returns`.
5. In the return loss loop, look up `(entity_id, int(observed_at))`:
   - If found AND `use_forward_returns=True`, use as target
   - Otherwise fall back to daily `log_return` (backward compatibility)

**Math justification:**
- Spearman IC for daily cross-sectional log_returns is typically < 0.01 (noise)
- 21-day holding period aligns with monthly rebalancing (GNN windows = 1 calendar month)
- ListNet minimises KL(p_target || p_pred) on the cross-section → directly optimises IC
- Forward return is the CORRECT oracle: we want the model to learn "which instruments
  will outperform in the next 21 days", not "what was yesterday's return"

**Key constraint:** Must not cause temporal leakage. The lookup is keyed by
`observed_at` of the CURRENT obs. The forward return target uses the close price
21 trading days AFTER the current obs timestamp. During inference (backtest), we
don't use the lookup — only the embedding. No leakage.

### Step A3: Fix obs_type key in graph_builder.py

**Problem:** `_compute_distributional_features()` calls `o.get("obs_type", "")`
but the store's `_entity_obs_row_to_dict()` returns `observation_type` (from DB
column). The key mismatch means `obs_type_counts` always sees `""` as the type,
and `obs_type_dist` features (dims 9–44 of ENRICHMENT_DIM) are always zero.

**Fix:** Change `o.get("obs_type", "")` → `o.get("observation_type", "") or o.get("obs_type", "")`
at the single location in `_compute_distributional_features`.

**Impact:** Corrects 35 of 55 ENRICHMENT_DIM features. The obs_type_dist dims
will now show real signal: instrument nodes → `instrument_daily` dominant,
country nodes → `geopolitical_event` dominant (now much smaller after A1),
etc. These features feed directly into node feature tensors used by HGTConv.

## Edge Cases

- `close == 0`: guard with `abs(close_now) < 1e-8` check → skip, no target.
- No later obs within search window: target not found → fall back to log_return.
- `log_return` also absent: skip obs for this return loss step.
- NaN/Inf in forward return: existing `torch.isfinite` guard handles it.

## Testing Plan

After code changes, before Kaggle upload:
```bash
python3.11 -m pytest tests/unit/test_trainer.py -x -q  # existing suite
python3.11 -c "
from agent.models.gnn.trainer import _build_forward_return_lookup
# Smoke test: two obs 21 trading days apart
obs = [
    {'entity_id': 'inst_1', 'observation_type': 'instrument_daily',
     'observed_at': 1000000.0, 'value': {'close': 100.0, 'log_return': 0.01}},
    {'entity_id': 'inst_1', 'observation_type': 'instrument_daily',
     'observed_at': 1000000.0 + 21*7/5*86400, 'value': {'close': 110.0, 'log_return': 0.02}},
]
lookup = _build_forward_return_lookup(obs, horizon_days=21)
assert len(lookup) >= 1, f'Expected at least 1 entry, got {lookup}'
val = list(lookup.values())[0]
assert abs(val - 0.1) < 0.01, f'Expected ~0.10 forward return, got {val}'
print('A2 smoke test PASS')
"
python3.11 -c "
from agent.models.gnn.graph_builder import _compute_distributional_features
obs = [
    {'entity_id': 'e1', 'observation_type': 'instrument_daily',
     'source_tool': 'yfinance', 'value': {'close': 100.0}},
]
r = _compute_distributional_features(obs)
assert r['obs_type_dist']['instrument_daily'] > 0, 'A3 fix failed — obs_type_dist still zero'
print('A3 smoke test PASS')
"
```

## Related

- [[architecture_stress_test]] — research note and failure mode analysis
- [[quant_training_ground]] — roadmap owner

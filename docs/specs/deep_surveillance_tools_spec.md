---
title: "Spec: Deep Surveillance Tools — Phase 10a"
tags:
  - doc/spec
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Spec: Deep Surveillance Tools — Phase 10a (Depth Evaluation Framework + Entity Registry)

## Goal

Build the foundational infrastructure for entity-level surveillance and depth measurement. After this phase, the system can:

1. Register entities across data sources with deterministic ID matching.
2. Record entity-level observations tagged with source tool and depth level.
3. Measure whether deeper data adds predictive signal via conditional MI and KL divergence.
4. Prove the loop end-to-end with `insider_filings` as the first wired tool.

This spec covers the L2/L3 foundations only. A future L4 layer should sit on top of linked-entity patterns and emit inferred latent-state or motif signals rather than raw observations.

## Files Affected

| File | Action |
|------|--------|
| `agent/pipeline/store.py` | **Modify** — add entity tables + depth_evaluations to `_SCHEMA_SQL`, add CRUD methods |
| `agent/pipeline/entity.py` | **Create** — entity name normalization, canonical ID generation, SEC seed loader |
| `agent/pipeline/depth_eval.py` | **Create** — MI computation (KSG), KL divergence measurement, depth evaluation runner |
| `tests/test_entity_store.py` | **Create** — entity registry CRUD + edge cases |
| `tests/test_entity_normalization.py` | **Create** — name normalization + SEC seed loader tests |
| `tests/test_depth_eval.py` | **Create** — MI/KL computation + depth evaluation tests |

## Implementation Steps

### Step 10a.1: Add entity tables to PipelineStore schema

Add 3 new tables to `_SCHEMA_SQL` in `agent/pipeline/store.py`:
- `entities` — canonical entity registry (entity_id, entity_type, canonical_name, created_at, metadata_json)
- `entity_aliases` — cross-source ID mappings (source, external_id → entity_id, with confidence)
- `entity_observations` — timestamped entity data points with depth_level

Add indexes: `idx_entity_obs_lookup` on `(entity_id, source_tool, observed_at)`, `idx_entity_aliases_lookup` on `(source, external_id)`.

Add store/query methods:
- `register_entity(entity_type, canonical_name, metadata) → entity_id`
- `add_entity_alias(entity_id, source, external_id, confidence)`
- `resolve_entity(source, external_id) → entity_id | None`
- `store_entity_observation(entity_id, source_tool, observed_at, observation_type, value, depth_level, metadata)`
- `query_entity_observations(entity_id, source_tool, since, until, depth_level, limit)`

### Step 10a.2: Add depth_evaluations table + methods

Add `depth_evaluations` table to `_SCHEMA_SQL`:
- `(id, tool_name, depth_level, evaluated_at, target_variable, mi_gain, kl_divergence, sharpe_delta, sample_size, metadata_json)`

Add store/query methods:
- `store_depth_evaluation(tool_name, depth_level, target_variable, mi_gain, kl_divergence, sharpe_delta, sample_size, metadata)`
- `query_depth_evaluations(tool_name, depth_level, target_variable, limit)`

### Step 10a.3: Entity name normalization utilities

Create `agent/pipeline/entity.py` with:
- `normalize_company_name(name) → str` — lowercase, strip suffixes (Inc., Corp., Ltd., LLC, LP, Co., etc.), collapse whitespace
- `entity_id_from_key(entity_type, key) → str` — deterministic SHA-256 hash of `f"{entity_type}:{key}"`, truncated to 16 hex chars
- Type alias `EntityType = Literal["company", "person", "vessel", "wallet", "country", "organization"]`

### Step 10a.4: SEC company_tickers seed loader

In `agent/pipeline/entity.py`:
- `load_sec_company_tickers(store: PipelineStore, json_path: str | None = None) → int` — downloads SEC `company_tickers.json` (if no path given, uses bundled fallback), registers each as entity_type="company", adds aliases (source="sec_cik", external_id=CIK) and (source="ticker", external_id=ticker). Returns count loaded.
- Uses `normalize_company_name()` for canonical_name.
- Idempotent — uses `INSERT OR IGNORE` semantics via `resolve_entity` check.

### Step 10a.5: MI computation module (KSG estimator)

Create `agent/pipeline/depth_eval.py` with:
- `compute_conditional_mi(observations_new, observations_existing, targets) → float` — Conditional MI of new depth observations given existing ones, against target variable. Uses `sklearn.feature_selection.mutual_info_regression` (KSG internally) for continuous targets, `mutual_info_classif` for discrete.
- Input: numpy arrays. Output: scalar MI in nats.
- Handle edge cases: insufficient samples (<30) returns NaN, constant columns return 0.0.

### Step 10a.6: KL divergence measurement

In `agent/pipeline/depth_eval.py`:
- `compute_kl_divergence(prior_probs, posterior_probs) → float` — KL(posterior || prior) for discrete distributions (dict[str, float]). scipy.stats.entropy handles this.
- `measure_belief_shift(store, variable_name, before_version, after_version) → float | None` — Load two belief snapshots from the store, compute KL divergence between their probability distributions. Returns None if either belief not found or not discrete.

### Step 10a.7: Wire insider_filings to entity_observations — test full loop

Modify the research doc's integration note (no code file change yet — this is a spec anchor for Phase 10b step 8). At the end of Phase 10a, write a standalone integration test that:
1. Creates a PipelineStore in `:memory:`
2. Seeds SEC entities
3. Simulates insider_filings observations at L1 and L2
4. Computes MI gain from L2 vs L1
5. Records depth_evaluation result
6. Asserts the loop produces sensible numbers

## Edge Cases

1. **Duplicate entities** — same CIK registered twice should be idempotent (resolve first).
2. **Missing canonical name** — blank or whitespace-only name should raise ValueError.
3. **Alias conflict** — two entities claiming the same (source, external_id) should keep the first (UNIQUE constraint).
4. **Schema migration** — existing databases without entity tables should auto-add them on `_init_schema()` (handled by `CREATE TABLE IF NOT EXISTS`).
5. **Empty observations for MI** — <30 samples returns NaN, does not crash.
6. **Constant-valued observations** — MI should be 0.0 (no information).
7. **Mismatched array lengths** — observations vs targets arrays with different lengths should raise ValueError.
8. **Non-finite values** — NaN/Inf in observations should be filtered before MI computation.
9. **KL divergence with zero probabilities** — use scipy.stats.entropy which handles this via convention (0 * log(0/q) = 0).
10. **Unicode entity names** — normalization should handle non-ASCII company names.

## Testing Plan

### Unit Tests (per step)
- **10a.1:** CRUD for entities, aliases, observations. Test resolve_entity, query_entity_observations with filters.
- **10a.2:** Store/query depth_evaluations. Test nullable sharpe_delta.
- **10a.3:** normalize_company_name edge cases (suffixes, whitespace, unicode). entity_id_from_key determinism.
- **10a.4:** SEC seed loader — mock the JSON, verify entity + alias count. Idempotency test (load twice, count unchanged).
- **10a.5:** MI computation — known-distribution tests (independent → MI≈0, correlated → MI>0). Edge cases per above.
- **10a.6:** KL divergence — known distributions (identical → 0, shifted → positive). measure_belief_shift with real store.
- **10a.7:** Integration test — full loop from entity seed → observation → MI → depth_evaluation record.

### Edge Case Suite (mandatory post-implementation)
Full edge case suite covering all items in Edge Cases section above.

## Future Extension — L4 Latent Structure Layer

Once at least two L3 workflows are live, add a follow-on spec for L4 with these constraints:
- Inputs must be linked-entity structures or graph features, not raw tool payloads.
- Outputs must be explicit latent variables such as motif labels, hazard scores, or posterior state probabilities.
- Evaluation must use incremental value over L3, analogous to the current L2-vs-L1 MI tests.
- Storage should likely be a dedicated table for latent-state observations or motif observations rather than overloading `entity_observations`.

Candidate first L4 designs:
- Distress preannouncement state from insider filings + creditor overlap + short pressure
- Sanctions-evasion coordination state from sanctions + AIS + commodity positioning
- Industrial ramp state from electricity + vessel traffic + hiring + contracts

## Related

- [[deep_surveillance_tools]]
- [[deep_surveillance]]
- [[project_memory]]
- [[world_model_spec]]
- [[signal_protocol_feature_engineering_spec]]

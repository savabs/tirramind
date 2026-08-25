---
title: "Spec: Phase 42 — Entity Diversity Expansion"
tags:
  - doc/spec
  - phase/42
  - topic/gnn
  - topic/pipeline
  - topic/diversity
  - layer/surveillance
  - layer/world-model
---

# Spec: Phase 42 — Entity Diversity Expansion

## Goal

Eight new tools wired into `daily_collection`. Entity-type observation entropy lifted from ~0.17 nats to ≥ 1.0 nats. All 6 entity types have ≥ 100 observations. GNN retrained cleanly against the diversified graph.

## Files Affected

- `agent/pipeline/dags/daily_collection.py` — add 8 new parallel nodes.
- `tests/test_pipeline_registry.py` — update node count, expected set, add per-node config tests.
- `scripts/audit_observation_diversity.py` — **NEW**, a small post-collection diagnostic that reports entity-type observation distribution + Shannon entropy. Used in acceptance verification.
- `[[phase42_entity_diversity_expansion]]` — live task tracking.
- `[[chat_checkpoint_2026-04-21_phase42_complete]]` — handoff artifact.

## Implementation Steps

### 42.1 — Research
Complete. See [[phase42_entity_diversity_expansion]].

### 42.2 — DAG wiring
Edit `agent/pipeline/dags/daily_collection.py`. In `build_daily_collection_dag`, insert 8 new parallel nodes after `fetch_whale_alert` and before `fetch_macro`:

```python
dag.add("fetch_insider_filings",    operator="insider_filings",
        table_name="insider_filings",
        params={"days_back": 30, "min_cluster_size": 3},
        timeout=120, retries=2)

dag.add("fetch_central_bank_balance", operator="central_bank_balance",
        table_name="central_bank_balance",
        params={"mode": "balance_sheets", "period": "1y"},
        timeout=120, retries=2)

dag.add("fetch_sovereign_debt_us",  operator="sovereign_debt",
        table_name="sovereign_debt",
        params={"mode": "us_yields"},
        timeout=120, retries=2)

dag.add("fetch_sovereign_debt_eu",  operator="sovereign_debt",
        table_name="sovereign_debt",
        params={"mode": "eu_yields"},
        timeout=120, retries=2)

dag.add("fetch_global_pmi",         operator="global_pmi",
        table_name="global_pmi",
        params={"mode": "cli"},
        timeout=120, retries=2)

dag.add("fetch_capital_flows",      operator="capital_flows",
        table_name="capital_flows",
        params={"mode": "holdings"},
        timeout=120, retries=2)

dag.add("fetch_defi_flows",         operator="defi_flows",
        table_name="defi_flows",
        params={"mode": "tvl", "limit": 20},
        timeout=120, retries=2)

dag.add("fetch_wikipedia_pageviews", operator="wikipedia_pageviews",
        table_name="wikipedia_pageviews",
        params={"mode": "spike", "days_back": 30,
                "z_threshold": 2.0, "limit": 50},
        timeout=120, retries=2)

dag.add("fetch_lobbying",           operator="lobbying",
        table_name="lobbying",
        params={"mode": "spending"},
        timeout=120, retries=2)
```

Total new nodes: **9** (two `sovereign_debt` modes count separately). DAG grows from 9 → 18 nodes.

### 42.3 — Test updates
Edit `tests/test_pipeline_registry.py`:

- `test_node_count`: 9 → 18.
- `test_expected_node_ids`: add the 9 new ids.
- `test_single_parallel_layer`: `len(layers[0]) == 18`, `len(dag.roots()) == 18`.
- Add one compact per-node config test per new operator (9 total), asserting `operator`, `params`, `timeout`, `retries`.

### 42.4 — Audit script
Create `scripts/audit_observation_diversity.py`. Computes and prints:

- Observation counts per `source_tool`.
- Observation counts per `entity_type` (via the observation→entity join).
- Shannon entropy of the entity-type distribution.
- Link-type counts + density (`links / entities`).
- A single-line pass/fail verdict vs the acceptance criteria.

### 42.5 — Live collection + retrain
Execute in order:

```bash
python scripts/run_collection.py --dag daily_collection
python scripts/audit_observation_diversity.py
# if entropy ≥ 1.0 and all types ≥ 100:
python scripts/run_collection.py --dag feature_generation
python scripts/retrain_gnn.py --db-path .tirra_pipeline/pipeline.db \
    --epochs 5 --auto-tune --since 2023-01-01 --window-size 172800 --backup
python scripts/run_backtest.py
```

### 42.6 — Checkpoint + archive
- Update task steps to `[x]`.
- Move `[[phase42_entity_diversity_expansion]]` → `tasks/done/`.
- Write `[[chat_checkpoint_2026-04-21_phase42_complete]]` with final metrics.

## Edge Cases

1. **Missing FRED key** — central_bank_balance / sovereign_debt / capital_flows return structured errors; DAG run proceeds, other nodes still write data. The audit script flags missing tools explicitly.
2. **Wikipedia rate limit** — tool already handles 429s via its cache layer; timeout of 120s is conservative.
3. **OECD API unreachable** — global_pmi uses fallback cached data if available; on cold start it may return zero rows. Audit script will report "0 rows"; that's a warning not a failure for Phase 42 acceptance (we require non-zero for 7/8 tools — OECD is the known-fragile one).
4. **Entity resolution collision** — if two tools write to the same company (e.g., `insider_filings` + `lobbying` both touching AAPL), the `upsert_entity` path should idempotently merge. Verify via audit: `entities` row count for type=company should equal the union, not the sum.
5. **GNN graph_builder OOM** — larger graph could push memory. Current graph was 71K observations. Expect 75-100K after one run. Still well within RAM.

## Testing Plan

Tier 1 — DAG structure (fast, offline):
- `pytest tests/test_pipeline_registry.py -v` must pass fully.

Tier 2 — Live smoke (slow, network):
- `python scripts/run_collection.py --dag daily_collection` must complete without a full-DAG abort.
- `python scripts/audit_observation_diversity.py` reports entropy ≥ 1.0, every type ≥ 100 obs.

Tier 3 — GNN regression:
- Retrain completes in < 20 minutes.
- Loss monotonic, effective weights in [0.05, 20].
- Test top-1 ≥ 70%, time_delta MAE ≤ 300s (relaxed from Phase 41's 60s because we've 10x-diversified the prediction space).

Tier 4 — Backtest regression:
- All 3 baselines (EqualWeight, BuyHold SPY, 60/40) produce Sharpe ratios consistent with Phase 41 profile (0.9-1.0 range). No NaN/infinite values.

## Math/Algorithm Note

Shannon entropy of the observation distribution is our diversity metric:
$$H(\mathbf{p}) = -\sum_{k=1}^{K} p_k \log p_k$$

with $p_k$ the fraction of observations writing to entity type $k$. We use natural log (nats). For $K=6$ types, $H_{\max} = \ln 6 \approx 1.79$.

**Phase 42 passes when $H \geq 1.0$ nats.** That corresponds to an effective type count of $e^{1.0} \approx 2.7$ types — roughly equivalent to the distribution being dominated by ~3 types rather than 1.

No smoothing, no Laplace correction — raw fraction. Any type with 0 observations contributes $0 \log 0 = 0$ by convention, which is exactly what we want (the type is "invisible" to the distribution).

## Related

- [[phase42_entity_diversity_expansion]]
- [[phase41_model_refresh_hardening_spec]]
- [[chat_checkpoint_2026-04-21_phase41_complete]]
- [[project_memory]]

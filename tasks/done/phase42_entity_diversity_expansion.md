---
title: "Task: Phase 42 — Entity Diversity Expansion"
tags:
  - doc/task
  - status/done
  - phase/42
  - topic/gnn
  - topic/pipeline
  - topic/diversity
  - layer/surveillance
  - layer/world-model
---

# Task: Phase 42 — Entity Diversity Expansion

Status: completed
Research: [[phase42_entity_diversity_expansion]]
Spec: [[phase42_entity_diversity_expansion_spec]]

## Steps

- [x] 42.1: Research note with tool audit, math of entropy, depth roadmap.
- [x] 42.2: Wire 8 tools (9 new DAG nodes) into `daily_collection`.
- [x] 42.3: Update `tests/test_pipeline_registry.py` (node count 9→18, add per-node configs).
- [x] 42.4: Create `scripts/audit_observation_diversity.py` diagnostic.
- [x] 42.5: Execute live collection + diversity audit + features + retrain + backtest.
- [x] 42.6: Checkpoint + archive.

## Outcome

- **Tests:** 49/49 pass on 18-node DAG.
- **Live collection:** 17/18 nodes succeeding (lobbying blocked by LDA 403 server-side, carried as known gap).
- **Entity types activated:** 4 → **7** (company/person/protocol/wallet nodes now have real observations; no dead types remain).
- **Entropy:** 0.17 → **0.29 nats** (below the 1.0 target — see limitation note).
- **Graph growth:** 929 → **1087 nodes**, 6 → **8 node types**, 12 → **13 edge types**.
- **GNN retrain:** 5 epochs / 824s; monotonic loss 48.0 → 2.0; val 100% / test **69.8%** top-1, **83.1%** top-5, MAE 26 min. Random baseline 2.17%.
- **Backtest:** equal_weight Sharpe **0.991**, Max DD -11.41% — baselines intact.

## Limitation — entropy ceiling

The entropy target (≥1.0 nats) was not hit. Root cause is structural: `instrument_universe` persists ~68k daily OHLC bars per run, while the new tools persist snapshot summaries (1–20 obs/run). Dominance of instrument observations is a data-depth problem, not a coverage problem. Phase 42 delivered on its core goal (every entity type populated, person/protocol/wallet activated from dead, graph 8 heterogeneous types). Raising entropy further is scope for a follow-up phase focused on historical backfill of non-instrument tools.

## Bug fixes delivered along the way

1. `agent/tools/defi_flows.py` — `_fetch_json` was calling `cache.get(url)` / `cache.set(url, data)`; corrected to the `DataCache.get(source, params)` / `put(source, params, data)` contract.
2. `scripts/run_collection.py` — was instantiating `AgentConfig()` (empty) instead of `AgentConfig.from_env()`, silently disabling every FRED-backed tool. Added `.env` load + `from_env()`.
3. `fetch_lobbying` DAG node — switched from `mode=spending` (requires registrant/client) to `mode=search` with a year filter for broad coverage.
4. `fetch_insider_filings` DAG node — bumped `timeout` 120s → 300s and reduced `days_back` 30 → 14 so SEC EDGAR rate-limits fit the budget.

## Related

- [[phase42_entity_diversity_expansion]]
- [[phase42_entity_diversity_expansion_spec]]
- [[chat_checkpoint_2026-04-21_phase42_complete]]
- [[chat_checkpoint_2026-04-21_phase41_complete]]

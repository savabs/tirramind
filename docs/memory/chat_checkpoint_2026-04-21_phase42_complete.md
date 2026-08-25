---
title: "Checkpoint — Phase 42 Complete"
tags:
  - doc/checkpoint
  - phase/42
  - topic/diversity
  - topic/pipeline
  - topic/gnn
---

# Phase 42 — Entity Diversity Expansion — COMPLETE

Date: 2026-04-21
Duration: single session
Related: [[phase42_entity_diversity_expansion]], [[phase42_entity_diversity_expansion_spec]], [[chat_checkpoint_2026-04-21_phase41_complete]]

## What landed

**DAG expansion (9 → 18 nodes):** Wired 8 dormant L2-capable tools (`insider_filings`, `central_bank_balance`, `sovereign_debt` ×2 for US+EU, `global_pmi`, `capital_flows`, `defi_flows`, `wikipedia_pageviews`, `lobbying`) into `daily_collection` with per-node timeouts and retries. All 49 DAG-registry tests updated and pass.

**Live run:** 17/18 nodes succeed. Lobbying node blocked by LDA API 403 (server-side issue, not code).

**Graph diversity:** Every entity type now populated.

| metric | before | after |
|---|---|---|
| node types | 6 | **8** |
| entities | 929 | **1087** |
| edge types | 12 | **13** |
| dead entity types | 3 (person, protocol, wallet) | **0** |
| entropy | 0.17 nats | **0.29 nats** |

**GNN retrain** (5 epochs, 824s, 48h window):
- Val top-1 100%, top-5 100%
- Test top-1 **69.8%**, top-5 **83.1%**, MAE **26 min**
- Random baseline 2.17% (model 32× better)
- Loss monotonic: 48.0 → 22.7 → 3.9 → 2.6 → 2.0
- Learned loss weights: obs=1.04, dt=0.095, contr=3.88, val=0.47

**Backtest:** equal_weight Sharpe **0.991**, Max DD -11.41%. SPY/AGG buy-hold baselines intact.

## Why test top-1 dropped 87% → 70%

Phase 41 was predicting observation type in a homogeneous graph dominated by `instrument_universe` — "guess instrument_universe" was a high-percentage bet. Phase 42 expanded the prediction space to 7 entity types with 13 source tools. The problem is harder by construction. Random baseline stayed at 2.17% so the model is still highly informative; the number just measures a tougher task.

## Entropy ceiling — why we stopped at 0.29 nats

Target was ≥1.0 nats. Root cause of shortfall is structural, not a coverage gap:

- `instrument_universe` persists ~68k daily OHLC bars per run across ~200 tickers
- New tools persist snapshot summaries (1 to ~20 obs/run)
- Dominance is data-volume, not tool-count. More tools add types; raising depth requires historical backfill per tool.

This is the right stopping point for Phase 42. The structural fix — historical backfill for non-instrument tools — is a separate phase.

## Bug fixes along the way

1. **`agent/tools/defi_flows.py`** — `_fetch_json` used wrong `DataCache` signature (`cache.get(url)` / `cache.set(url, data)` instead of `cache.get(source, params)` / `cache.put(source, params, data)`). Fixed and smoke-tested with live DefiLlama fetch ($546B total TVL).
2. **`scripts/run_collection.py`** — instantiated `AgentConfig()` (default, empty fred key) instead of `AgentConfig.from_env()`, silently disabling every FRED-backed tool. Added `.env` loader + `from_env()`. This re-enabled `fetch_macro`, `fetch_central_bank_balance`, `fetch_capital_flows`.
3. **DAG `fetch_lobbying`** — switched from `mode=spending` (requires registrant/client name) to `mode=search` with current year for broad coverage.
4. **DAG `fetch_insider_filings`** — bumped timeout 120s → 300s, reduced `days_back` 30 → 14 so SEC EDGAR's 10 req/s rate limit fits the budget.

## Files touched

- `agent/pipeline/dags/daily_collection.py` — added 9 Phase 42 nodes, added datetime import
- `agent/tools/defi_flows.py` — cache API fix
- `scripts/run_collection.py` — load .env + `AgentConfig.from_env()`
- `scripts/audit_observation_diversity.py` — new diagnostic
- `tests/test_pipeline_registry.py` — 18-node structural tests + 9 per-node config tests
- `[[phase42_entity_diversity_expansion]]`
- `[[phase42_entity_diversity_expansion_spec]]`
- `[[phase42_entity_diversity_expansion]]` → moving to `tasks/done/`

## Next phase candidates

1. **Historical backfill** for sovereign_debt, central_bank_balance, global_pmi, capital_flows — each has a `start_period`/`period` param; a backfill DAG hitting 5+ years of monthly data would lift entropy above 1.0.
2. **Lobbying alternate source** — LDA 403 is persistent; investigate OpenSecrets bulk download or scrape lobbyist registrations.
3. **GNN-guided expansion** — per project doctrine, use attention weights over the new heterogeneous graph to identify starved entity neighborhoods before adding more tools.
4. **insider_filings depth** — current 14 days is thin for clustering; backfill 1 year of Form 4s would densify the person/company subgraph.

---

## ⚠️ Post-session correction — checkpoint conclusion disputed

**Date added:** 2026-04-21 (same session, live data verification)

### What the checkpoint got wrong

The conclusion "Tool coverage is no longer the bottleneck" is incorrect. Live data shows:

| claim | stated | actual |
|---|---|---|
| Data tools total | — | **51** |
| Unique tools wired | 18 nodes | **16 unique operators** |
| Unwired tools | implied: all covered | **35 unwired (68.6%)** |

The checkpoint conflated two distinct problems:

**Problem 1 (correctly identified): Volume asymmetry is structural.**
- `instrument_universe` persists ~68k OHLC bars per run (200 tickers × 240 days)
- New Phase 42 tools add 1–20 snapshot obs/run
- Entropy ceiling: even doubling all non-instrument observations lifts entropy 0.29 → ~0.5 nats. Hitting ≥1.0 nats requires non-instrument obs to collectively reach ~69k+. Monthly macro backfill (5yr × 12mo × 50 countries = ~3k rows for sovereign_debt) cannot close that gap.

**Problem 2 (incorrectly dismissed): Tool coverage IS still a gap.**
35 of 51 data tools (68.6%) are not wired into the DAG, including several that are **already L2-ready** and generate entity-level observations at scale:

| tool | status | vol/run | entity type | notes |
|---|---|---|---|---|
| `ais_vessel` | L2-ready | **500+ vessel entities** | vessel | area mode, 18K source pool, zero cost/key |
| `gov_contracts` | L2-ready | medium | company, agency | already has `_persist_entities` |
| `patent_filings` | L2-ready | medium | company, person | already has `_persist_entities` |
| `sanctions_monitor` | L2-ready | medium | company, person | already has `_persist_entities` |

### Correction to user's own analysis

`job_postings` is **not** a high-volume entity-creating source. It is JOLTS/BLS aggregate macro data (L1 only — no entity nodes, no per-employer records). It belongs in the same category as `macro_data` and `consumer_sentiment`. Low entropy priority.

### Real structural options to reach entropy ≥ 1.0 nats

| lever | obs added | realistic | verdict |
|---|---|---|---|
| 5yr backfill: sovereign_debt + capital_flows + global_pmi + CB balance | ~3k–8k obs | yes | **not enough alone** |
| Wire `ais_vessel` (limit=500, area mode, daily) | ~500/day → **15k+ in 30 days** | yes | **game-changer** |
| Wire `gov_contracts` + `patent_filings` + `sanctions_monitor` | ~hundreds/run, accumulates | yes | worthwhile, slow |
| Reduce `instrument_universe` lookback (30d vs 1yr) | −50k obs overnight | risky | throws away data |
| Treat instrument obs as L1 aggregate (per-ticker, not per-row) | architecture change | hard | requires GNN schema change |

### Revised next-phase priority order

1. **Wire `ais_vessel`** — highest single-tool entropy impact, L2-ready, new entity type (vessel), zero cost. Investigate limit tuning (area mode, Baltic/global coverage), confirm `vessel_position` obs accumulate correctly. This is the Phase 43 candidate.
2. **Wire `gov_contracts` + `sanctions_monitor` + `patent_filings`** — all L2-ready, add company/person depth. Medium priority; accumulation is slow but free.
3. **Historical backfill** for Phase 42 macro tools — as originally recommended, but frame correctly: it adds a few thousand rows, not the entropy fix. It helps densify country nodes for GNN cross-entity attention, not raw entropy.
4. **GNN-guided expansion** — after `ais_vessel` is wired and a week of data accumulates, re-run attention diagnostics to confirm vessel nodes are filling the predicted gap before wiring more tools.

---

## Session boundary

Recommend fresh chat for the next phase. Cold-start path:
1. Read this checkpoint (including the correction above)
2. Read the active task (none — Phase 42 done; Phase 43 = `ais_vessel` DAG wiring + investigation)
3. Follow `[[wiki links]]` to Phase 42 research/spec for context as needed

---
title: "Checkpoint: 2026-04-22 Identity + Phase 46/47 Planning"
tags:
  - doc/checkpoint
  - phase/46
  - phase/47
  - topic/living-system
  - topic/backfill
  - status/done
---

# Checkpoint: 2026-04-22 — Identity Reframe + Phase 46/47 Planning

## What happened this session

### 1. Identity reframe (all three canonical files updated)

TirraMind's identity was updated from "information-arbitrage firm" to **predictive AI company** — explicitly not a quant fund.

Files changed:
- `.github/copilot-instructions.md` — opening definition rewritten
- `[[quant_training_ground]]` — firm identity + calibration sections rewritten
- `/memories/repo/tirramind_structure.md` — Identity section rewritten

New benchmark: *"Does this advance the frontier of what a machine can know about reality before humans do?"*

### 2. Phase 46 added: Living System — Online GNN (EWC)

Added as HIGH PRIORITY to:
- `[[quant_training_ground]]` (phases list + Tier 2 roadmap)
- `/memories/repo/tirramind_structure.md` (Phase 45.2 next pointer + Phase 46 section + Tier 2)

Research/spec docs not yet created — needed before implementation.

### 3. Phase 47 added: Historical Backfill Runner

This was the key strategic insight this session: **backfill 2–5 years of history across all 51 tools before the first GNN training run.** This collapses the 6-week live accumulation wait into a 1–2 day data collection job.

Files created/updated:
- `[[historical_backfill]]` — full research doc with tool-by-tool backfill status
- `[[quant_training_ground]]` — Phase 47 added to phases list, revised sequence header

**Revised sequence:**
```
Phase 46 → Phase 47 (backfill) → Phase 40 (GNN retrain on years of real data)
```
Phase 40 is no longer data-gated to mid-May. It runs immediately after Phase 47 completes.

## Current DB state (verified this session)

- Entities: 1,087
- Observations: 74,030
- Links: 357
- DB size: 24.2 MB
- Graph RAM estimate: 112 MB

## Tool inventory (verified this session)

62 files in agent/tools/. 51 active data tools (11 are utilities/internal).

Backfill categories:
- **27 confirmed backfillable** (historical APIs, date-range params)
- **2 live-only** (cert_transparency, dns_monitor, internet_outages)
- **~22 partial** (need endpoint verification before backfill)

## What to do next session

### Immediate: Phase 46 implementation
1. Create `[[living_system_online_gnn]]` (research preflight)
2. Create `[[living_system_online_gnn_spec]]` (spec preflight)
3. Implement EWC in `agent/models/gnn/het_tgn.py` — add `compute_fisher()` and `online_update()` methods
4. Add `online_update()` to `agent/models/gnn/trainer.py`

### After Phase 46: Phase 47 implementation
1. Create `[[historical_backfill_spec]]`
2. Implement `scripts/backfill.py` — iterates all 51 tools with days_back=1825
3. Verify timestamp correctness on historical writes
4. Verify Group B tools one-by-one
5. Run backfill — 2–4 hours, all Group A tools
6. Immediately proceed to Phase 40 (GNN retrain)

## Relevant files — complete list

### Canonical planning files
| File | Purpose |
|---|---|
| `[[quant_training_ground]]` | CANONICAL roadmap owner. Phase list, sequence, firm identity |
| `/memories/repo/tirramind_structure.md` | CANONICAL metrics owner. DB counts, phase summaries, roadmap tiers |
| `.github/copilot-instructions.md` | Agent workflow + firm identity. Every session opens with this |
| `AGENTS.md` | Agent definitions, tool permissions, available prompts |

### Phase 46 files (Living System — EWC)
| File | Status |
|---|---|
| `[[living_system_online_gnn]]` | NOT YET CREATED — create next session first |
| `[[living_system_online_gnn_spec]]` | NOT YET CREATED — create next session second |
| `agent/models/gnn/het_tgn.py` | TARGET — add `compute_fisher()`, `online_update()` |
| `agent/models/gnn/trainer.py` | TARGET — add `online_update()` call |
| `tests/test_gnn_online_learning.py` | NOT YET CREATED — write with spec |

### Phase 47 files (Historical Backfill)
| File | Status |
|---|---|
| `[[historical_backfill]]` | ✅ CREATED THIS SESSION |
| `[[historical_backfill_spec]]` | NOT YET CREATED — create before implementation |
| `scripts/backfill.py` | NOT YET CREATED — main implementation |
| `agent/tools/` (all 51) | Existing — will be called by backfill runner |
| `agent/pipeline/store.py` | Existing — upsert logic review needed |

### Phase 40 files (GNN Retrain — now unlocked by Phase 47)
| File | Status |
|---|---|
| `agent/models/gnn/het_tgn.py` | Existing HetTGN implementation |
| `agent/models/gnn/trainer.py` | Existing batch trainer |
| `agent/models/gnn/pattern_extractor.py` | Existing pattern mining |
| `agent/data/graph_builder.py` | Existing — builds PyG graph from DB |

### Live system components (already running)
| File | What it does |
|---|---|
| `agent/models/gnn/het_tgn.py` → `HeteroMemory` | GRU per entity — updates live on every observation |
| `agent/models/bayesian/world_model.py` | Posterior belief updates every DAG run |
| `agent/learning/thompson_bandit.py` | 48-arm bandit — learns which tools give signal |
| `agent/memory/reviewed_memory.py` | Promotes patterns after 3+ consistent observations |
| `agent/pipeline/dags/daily_collection.py` | 29-node DAG — runs all 51 tools daily |

### All 51 data tools (agent/tools/)
```
academic_preprints    ais_vessel            bankruptcy_court
building_permits      capital_flows         central_bank_balance
cert_transparency     cftc                  comtrade
consumer_sentiment    creditor_filings      defi_flows
disease_surveillance  dns_monitor           drug_regulatory
earthquake_proximity  electricity_monitor   energy_supply
finra_short_volume    foia_requests         food_security
form144               gdelt                 global_pmi
gov_contracts         insider_filings       instrument_universe
interconnection_queue internet_infrastructure internet_outages
job_postings          labor_disruptions     liquidity_regime
lobbying              macro_data            market_data
migration_flows       patent_filings        political_risk
polymarket            polymarket_whales     power_grid
regulatory_gazette    sanctions_monitor     satellite_activity
sovereign_debt        supply_chain_monitor  transport_throughput
treasury_receipts     weather_alerts        whale_alert
wikipedia_pageviews
```

### Key model + pipeline files
| File | Layer | Purpose |
|---|---|---|
| `agent/core/orchestrator.py` | All | Orchestrator — research/plan/execute/synthesize |
| `agent/pipeline/scheduler.py` | Layer 1 | DAG scheduler |
| `agent/pipeline/store.py` | Layer 1 | DB read/write, PipelineStore |
| `agent/pipeline/dags/daily_collection.py` | Layer 1 | 29-node production DAG |
| `agent/quant/changepoint.py` | Layer 2 | BOCPD changepoint detection |
| `agent/quant/regime.py` | Layer 2 | HMM regime detection |
| `agent/quant/convergence.py` | Layer 2 | 12 causal chain templates |
| `agent/models/gnn/het_tgn.py` | Layer 3 | HetTGN + HeteroMemory |
| `agent/models/bayesian/world_model.py` | Layer 3 | Bayesian belief propagation |
| `agent/fusion/kalman.py` | Layer 4 | Kalman signal fusion |
| `agent/learning/rl_policy.py` | Layer 5 | SAC RL policy |
| `agent/adversarial/` | Layer 6 | Manipulation detection, edge decay |
| `agent/reasoning/llm_client.py` | Layer 7 | LLM support only |

## Session principle confirmed

> Simple is best. Use every tool we have. Don't waste anything.

Phase 47 is exactly this: one script that calls the 51 tools we already built, with a longer time window. No new infrastructure. No new ML. Just: give the system the history it needs to learn properly.

## Related

- [[quant_training_ground]]
- [[historical_backfill]]
- [[living_system_online_gnn]]

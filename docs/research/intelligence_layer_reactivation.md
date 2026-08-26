---
title: Research — Intelligence Layer Reactivation
tags:
  - doc/research
  - topic/pipeline
  - topic/gnn
  - status/active
---

# Research — Why Layers 2–6 Produce Nothing

*Investigated: 2026-08-26. All numbers below are direct measurements against
`.tirra_pipeline/pipeline.db` (138 MB) and `.tirra_pipeline/gnn_model.pt`, not
inference from code reading.*

---

## 1. Observed state

`daily_collection` works. Everything downstream of it does not.

| table | rows | layer |
|---|---:|---|
| `entity_observations` | 365,739 | L1/L3 ✅ |
| `entity_links` | 16,870 | L3 ✅ |
| `entities` | 5,628 | L3 ✅ |
| `features` | 93 | L2 ⚠️ stale (last write 2026-04-21) |
| `signals` | **0** | L2/L4 ❌ |
| `beliefs` | **0** | L3 ❌ |
| `entity_alerts` | **0** | L3 ❌ |
| `convergence_clusters` | **0** | L4 ❌ |
| `rl_transitions` | **0** | L5 ❌ |
| `portfolio_weights` | **0** | L5 ❌ |
| `paper_trade_pnl` | **0** | L5 ❌ |
| `depth_evaluations` | **0** | orphaned |

---

## 2. Root cause A — the DAG chain is never scheduled

All 11 DAGs declare a cron schedule forming a deliberate nightly chain:

```
18:00  daily_collection
18:30  convergence_detection, gnn_inference
18:45  entity_scoring
19:00  feature_generation
19:15  adversarial_scan
19:30  world_model_update, rl_training
19:45  inference
```

`PipelineScheduler.start()` (`agent/pipeline/scheduler.py:95`) is what arms those
cron triggers. It is called in exactly one place — `agent/cli.py:449` — and no
production entry point reaches it.

The only two production entry points both hardcode a single DAG:

- `scripts/tirra_engine.py:83` → `scheduler.trigger("daily_collection")`
- `deploy/systemd/tirra-collect.service` → `run_scheduled.sh collect` → same

`dag_runs` confirms this empirically. In the entire life of the DB:

| DAG | runs ever | last |
|---|---:|---|
| `daily_collection` | 8 | 2026-08-26 (failed) |
| `feature_generation` | 3 | 2026-04-21 |
| `gnn_inference` | 2 | 2026-04-19 (failed) |
| *other 8* | **0** | never |

Every run's `trigger` column reads `manual`. Nothing has ever fired on a schedule.

---

## 3. Root cause B — three-way schema drift

Measured across the live DB, the code constants, and the trained weights:

| dimension | live DB | code | trained model |
|---|---:|---:|---:|
| entity types | **12** | **11** | 12 |
| observation types | 38 present (4 unknown to code) | 48 | 48 |
| instrument feature dim | **49** | 49 | **23** |

### 3a. Instrument dim 23 → 49 (hard crash)

`GraphBuilder.build()` emits `instrument` as `93 × 49`.
`type_projections.instrument.weight` in the checkpoint is `(64, 23)`.

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (93x49 and 23x64)
```

Checkpoint archaeology shows the dim grew in steps and was never retrained after
the last one:

| checkpoint | date | in_ch[instrument] | weight | nodes |
|---|---|---:|---|---:|
| `gnn_model_live.pt` | 04-19 | 14 | (64,14) | 119 |
| `gnn_model_pre_phase40.pt` | 04-28 | 14 | (128,14) | 2451 |
| `gnn_model_h_g_current.pt` | 05-06 | 14 | (128,14) | 2502 |
| `gnn_model_phase50.pt` | 05-24 | **23** | (128,23) | 2688 |
| `gnn_model.pt` (active) | — | 49 *(metadata)* | **(64,23)** *(weights)* | 5628 |

`instrument` dim decomposes as `BASE_FEAT_DIM + PRICE + MICROSTRUCTURE + M15`
where `BASE_FEAT_DIM = len(ENTITY_TYPES) + 3`. The instrument-only extras grew
from 9 to 35 dims; base stayed 14. Hence 23 → 49.

This is CLAUDE.md §5 verbatim: *"If you change feature extraction, you're
invalidating learned weights."* No code change can bridge 23 → 49 without
discarding the learned projection anyway — **retraining is the only real fix.**

### 3b. `maritime_area` silently scored as `cftc_contract`

`graph_builder.py:533-536`:

```python
type_idx = _ENTITY_TYPE_TO_IDX.get(entity_type)
if type_idx is None:
    log.warning("Unknown entity type %r — defaulting to index 0", entity_type)
    type_idx = 0
```

`maritime_area` exists in the DB (1 entity) but not in `ENTITY_TYPES`, so it is
one-hot encoded as index 0 = `cftc_contract`. A warning is logged and the run
continues with wrong data. Silent corruption, not a crash.

### 3c. Four observation types unknown to code

Present in `entity_observations`, absent from `OBSERVATION_TYPES`:
`area_daily_activity`, `baltic_activity_proxy`, `futures_positioning_derived`,
`petroleum_inventory`.

Note `futures_positioning_derived` alone accounts for 5,080 observations.

---

## 4. Root cause C — silent failure in `inference`

Empirical run of all 8 never-executed DAGs against the real DB:

| DAG | run status | node statuses | rows written |
|---|---|---|---:|
| `world_model_update` | completed | completed | **23 beliefs** ✅ |
| `feature_generation` | completed | completed | **17 features** ✅ |
| `entity_scoring` | **failed** | `score_entities` failed | 0 |
| `inference` | **completed** | all 4 `completed` | **0** ⚠️ |
| `gnn_inference` | completed | completed | 0 ⚠️ |
| `convergence_detection` | completed | completed | 0 |
| `rl_training` | completed | completed | 0 |
| `adversarial_scan` | completed | completed | 0 |

`inference` is the dangerous one: every node reports `completed`, the DAG reports
success, and nothing is written — the GNN forward pass threw internally and was
swallowed as non-fatal:

```
GNN inference failed: mat1 and mat2 shapes cannot be multiplied (93x49 and 23x64)
```

In production this reads green forever while producing nothing. Same failure
class as the executor logging gap fixed on 2026-08-25.

`entity_scoring` fails loudly with `index 69 is out of bounds for dimension 1
with size 69` — an off-by-one consistent with obs-type cardinality drift.

**Good news:** `world_model_update` and `feature_generation` work correctly on
first contact with real data. The pipeline is not rotten — it is unscheduled and
schema-drifted.

---

## 5. Root cause D — checkpoints are mutated in place

`gnn_inference` overwrites `.tirra_pipeline/gnn_model.pt` in place on every run
(confirmed: running the DAG changed mtime and size, and rewrote `in_channels`
metadata to 49 while leaving the 23-dim weights untouched — producing a
checkpoint whose declared metadata contradicts its own weights).

This violates CLAUDE.md §5 *"Checkpoints are immutable once created. Create a new
one for each major run."*

---

## 6. Orphaned code

`run_depth_evaluation` (`agent/pipeline/depth_eval.py:206`) is fully implemented
and tested, but its only callers repo-wide are `tests/test_depth_eval.py` and
`tests/test_insider_filings_mi.py`. It is wired into no DAG, so scheduling alone
will never populate `depth_evaluations`.

---

## 7. Design space for the fix

**Orchestration.** Three options considered:
1. Long-running `scheduler.start()` daemon (APScheduler in-process).
2. One systemd timer per DAG mirroring each cron string.
3. A single "run the chain in dependency order" entry point + one timer.

(3) is preferred for the bootstrap case: the chain has cold-start dependencies
(`rl_training` needs alerts+beliefs; `inference` needs a SAC checkpoint that only
`rl_training` produces; `rl_transitions` only materialises on the *second*
consecutive `inference` run). A wall-clock cron cannot express "after upstream
actually succeeded". (1) is preferred for steady state. They compose: ship (3)
as `run_scheduled.sh chain`, keep (1) available via the CLI.

**Schema sync.** Adding `maritime_area` shifts `BASE_FEAT_DIM` 14 → 15 and thus
every node feature dim (instrument 49 → 50). This is only safe *because* we are
retraining — doing it without a retrain would break the checkpoint a second way.
Order matters: **sync constants first, then retrain**, so training sees the true
schema.

**Prevention.** The class of bug here — schema drift silently invalidating a
checkpoint — has no guard today. A startup assertion comparing checkpoint
`in_channels` against live `GraphBuilder` output would have caught this in April.
This belongs in LESSONS.md as a new entry.

---

## 8. Open items not addressed here

- `convergence_detection`, `rl_training`, `adversarial_scan` run clean but write
  nothing. Needs separate investigation — may be correct (genuinely nothing to
  emit yet) or may be a second silent-failure class.
- `depth_evaluations` needs a DAG home.
- 3 tools still need free API keys (NASA FIRMS, EIA ×2).
- 3 vendor APIs changed (LDA 403, USPTO 301, FEC validation).

---
title: Checkpoint — Intelligence Layer Reactivation
tags:
  - doc/memory
  - topic/pipeline
  - topic/gnn
  - status/active
---

# Checkpoint — 2026-08-26 — Intelligence Layer Reactivation

Research: `docs/research/intelligence_layer_reactivation.md`
Spec: `docs/specs/intelligence_layer_reactivation_spec.md`
Task: `tasks/active/intelligence_layer_reactivation.md`
New fuckup entry: **LESSONS.md F-12**

---

## What was learned

The intelligence layers were not broken — they had **never been invoked**, and
four independent bugs sat on top of that. All four were invisible: each one
either logged a warning nobody read or reported success while producing nothing.

### 1. Orchestration gap
All 11 DAGs declare cron schedules forming a nightly chain (18:00 collect →
19:45 inference). Those only fire under a long-running
`PipelineScheduler.start()` process, and **nothing in production ever started
one** — both entry points hardcode `trigger("daily_collection")`.

Verified against `dag_runs`: **8 of 11 DAGs had zero runs, ever.** Every run's
trigger read `manual`.

### 2. `ENRICHMENT_DIM` hardcoded → buffer overflow
Pinned at `55` (= 9 + 46 obs types) while the writer indexed
`offset + 9 + ot_idx` over the live `OBSERVATION_TYPES`. At 48 types with
`BASE_FEAT_DIM=14`: tensor `14+55=69` wide, `ot_idx=46` → index 69.

That is the `entity_scoring` crash, arithmetic-exact. **For instrument nodes the
same overflow instead ran into the price-feature block — silent corruption, not
a crash**, depending only on node type.

### 3. `inference` reported success while writing nothing
All three operators caught every exception and *returned* `{"status": "error"}`.
`DAGExecutor` fails a node only when its operator **raises** — a returned dict
always records as `completed`. A real shape error produced a fully green DAG
with 0 rows, indefinitely.

### 4. 60s node timeout killing every model node
`Node.timeout` defaults to 60 — right for one HTTP fetch in `daily_collection`,
hopeless for a 5.6k-node graph build + GNN pass. `gnn_inference` died at
**69.6s**, cascading into `sac_inference`/`emit_portfolio` being skipped. So
`portfolio_weights` was empty for a reason unrelated to the model. Had we only
fixed the schema and retrained, this would have looked like a training failure.

### 5. Schema drift (underneath all of it)
Three registries disagreed, with nothing comparing them:

| | live DB | code | trained weights |
|---|---:|---:|---:|
| entity types | 12 | 11 | 12 |
| observation types | 38 present, 4 unknown | 48 | 48 |
| instrument features | 49 | 49 | **23** |

`maritime_area` was in the DB but not `ENTITY_TYPES`, so it was one-hot encoded
as index 0 = `cftc_contract` — trained and scored as the wrong entity kind for
months. **A test asserted this** (`assert features[0, 0] == 1.0`), so the suite
was green over the corruption — same pattern as the DataCache tests fixed
2026-08-25.

---

## Results

Tables that had **never** held a row:

| table | before | after |
|---|---:|---:|
| `signals` | 0 | **23** |
| `beliefs` | 0 | **46** |
| `features` | 93 (stale since 2026-04-21) | **127** |

DAGs now completing: `convergence_detection`, `feature_generation`,
`adversarial_scan`, `world_model_update`, `rl_training`, `gnn_inference`,
`inference`.

---

## Known issues / next steps

**Only remaining blocker is the retrain (Phase D).** `entity_scoring` is the one
genuine failure left — pure schema drift:

| parameter | trained | current |
|---|---:|---:|
| `type_projections.instrument` | 23 | 50 |
| `type_projections.*` (11 others) | 14 | 15 |
| `obs_type_head` | 46 | 52 |
| `hgt_layers.{0,1}.{k,v}_rel` | 34 | 40 |

Cold-start chain, in order:

```
retrain GNN ──▶ entity_scoring works ──▶ entity_alerts populate
            ──▶ rl_training produces SAC checkpoint
            ──▶ inference writes portfolio_weights + paper_trade_pnl
            ──▶ 2nd consecutive inference run ──▶ rl_transitions
```

`inference` currently completes 4/4 writing nothing, and that is **correct**:
`rl_policy_checkpoints` is 0, so it takes the intended `skipped` path. Do not
"fix" this before the retrain.

Not addressed (tracked in the task file):
- `convergence_detection` / `rl_training` / `adversarial_scan` complete but write
  nothing — may be correct (nothing to emit yet) or a second silent-failure class.
- `depth_evaluations` is unreachable: `run_depth_evaluation` has no DAG home,
  only test callers.
- 3 tools need free API keys (NASA FIRMS, EIA ×2); 3 vendor APIs changed
  (LDA 403, USPTO 301, FEC validation).

---

## How to resume

```bash
# See the plan without running anything
.venv/bin/python scripts/run_chain.py --dry-run

# Downstream only (skips the slow 40-source collection)
./scripts/run_scheduled.sh chain --skip-collection

# Just the model DAGs
.venv/bin/python scripts/run_chain.py --only gnn_inference,entity_scoring,inference
```

Before retraining, re-read **LESSONS.md F-01/F-02/F-03/F-04 and the new F-12**,
then follow spec Step 7: print the active return-head branch at start (F-02),
assert `torch.std(emb, dim=0).mean() > 0.1` (F-01), confirm time-ordered eval
splits (F-04), confirm history arrays align on resume (F-03).

Checkpoints are now immutable — `agent/models/gnn/checkpoint_store.py` writes
versioned artifacts under `.tirra_pipeline/checkpoints/` and repoints
`gnn_model.pt`. Never write through an existing checkpoint.

---

## Session note

A diagnostic probe run during this session executed `gnn_inference`, which at the
time still overwrote `.tirra_pipeline/gnn_model.pt` in place — the May 25
checkpoint was lost. It was already unusable (23-dim weights vs 49-dim
features), `gnn_model_phase50.pt` (05-24) survives untouched, and the post-probe
file was preserved. This is precisely the immutability bug that Phase C then
fixed.

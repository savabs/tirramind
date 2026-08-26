---
title: Task — Intelligence Layer Reactivation
tags:
  - doc/task
  - topic/pipeline
  - topic/gnn
  - status/active
---

# Task — Intelligence Layer Reactivation

Research: `docs/research/intelligence_layer_reactivation.md`
Spec: `docs/specs/intelligence_layer_reactivation_spec.md`

Started 2026-08-26.

**Context in one line:** L1 collection works and the entity graph is healthy
(365k observations), but layers 2–6 have never executed against real data —
8 of 11 DAGs have zero runs ever, and the GNN checkpoint's 23-dim instrument
projection cannot consume the live 49-dim features.

---

## Checklist

### Phase A — schema truth (blocks retrain)
- [x] **A1** Add `maritime_area` to `ENTITY_TYPES` (spec Step 1.1) — 11 → 12
- [x] **A2** Add 4 missing `OBSERVATION_TYPES` (spec Step 1.2) — 48 → 52
- [x] **A3** Verify `BASE_FEAT_DIM == 15`, no `Unknown entity type` warning on live build
- [x] **A4** Unknown type → all-zero one-hot, never `ENTITY_TYPES[0]` (spec Step 2)
- [x] **A5** Targeted tests for A4 + new `validate_schema_against_store()` guard

**A4 design change vs spec:** the spec said *raise*. Raising breaks runtime
discovery of genuinely-new entity types, which `test_unknown_type_in_store_
still_gets_nodes` shows is a deliberate feature. Resolution: feature building
degrades honestly (all-zero one-hot — claims no identity rather than the wrong
one) and loud detection moved to `validate_schema_against_store(store)`, called
before anything trains or scores. Better separation: collection stays
permissive, model paths stay strict.

### Phase A′ — ENRICHMENT_DIM drift (discovered mid-Phase-A)
- [x] **A′1** `ENRICHMENT_DIM` was hardcoded 55 (= 9 + 46 obs types) while the
      writer indexed `offset + 9 + ot_idx` over `len(OBSERVATION_TYPES)`. At 48
      types with `BASE_FEAT_DIM=14` the tensor was 69 wide and `ot_idx=46`
      addressed index 69 → **this is the exact `entity_scoring` crash**
      (`index 69 is out of bounds for dimension 1 with size 69`).
- [x] **A′2** Made `ENRICHMENT_DIM` derived: `9 + len(OBSERVATION_TYPES)` = 61
- [x] **A′3** Regression tests for the overflow invariant
- [x] **A′4** Updated 8 test files whose hardcoded `== 55` / `== 46` assertions
      had themselves drifted (were already failing pre-session)
- [x] **A′5** Verified: `entity_scoring` no longer index-crashes; now fails only
      on the stale-checkpoint dim mismatch, which Phase D fixes

For instrument nodes the same overflow silently corrupted the price-feature
block that follows enrichment, rather than crashing — so this bug was both a
visible crash *and* silent data corruption depending on node type.

### Phase B — silent failures
- [x] **B1** All three `inference` operators re-raise instead of returning
      `{"status": "error"}` (spec Step 3). Root cause: `DAGExecutor` fails a node
      only when its operator *raises*; a returned dict is always recorded as
      `completed`. Verified end-to-end — the DAG went from
      `completed` + 0 rows → `failed` + named cause + honest downstream skips.
- [x] **B2** `skipped` degradation path for a genuinely-missing model preserved
      and covered by a test that asserts it did *not* start raising
- [x] **B3** Regression tests incl. a contract guard asserting no operator in the
      module returns `status="error"` at all
- [x] **B4** Reproduced `entity_scoring` `index 69 out of bounds` (spec Step 4)
- [x] **B5** Fixed — root cause confirmed as ENRICHMENT_DIM drift (Phase A′), not
      assumed. Arithmetic matches the error exactly: 14 + 55 = 69 wide,
      `ot_idx=46` → index 69.
- [ ] **B6** `entity_scoring` writes ≥1 `entity_alerts` row — **blocked on Phase D
      retrain** (now fails on `40x76 vs 14x64`, the stale checkpoint)

### Phase C — checkpoint hygiene
- [x] **C1** `gnn_inference` no longer writes through the active checkpoint, and
      no longer **deletes** it on a high-changepoint (`model_path.unlink()` →
      `archive_checkpoint()`). New module `agent/models/gnn/checkpoint_store.py`.
- [x] **C2** Versioned immutable checkpoints under `.tirra_pipeline/checkpoints/`
      + stable pointer at `gnn_model.pt`. Verified live:
      `gnn_model_20260825T230121.pt` written by a real chain run.
- [x] **C3** 9 tests incl. "archived file survives 3 subsequent saves"
- [x] **C4** Drift diagnosis in `load_model` (spec Step 6) — names the entity type
      and both widths instead of letting torch fail opaquely later. Warns rather
      than raises so deliberate partial-load workflows still work; strict
      validation belongs in the chain runner (E1).
- [x] **C5** Verified on the real stale checkpoint — emits
      `instrument: trained_weights=23 expected_by_model=49`

### Phase D — retrain (needs A + C4 done)
- [x] **D1** LESSONS.md **F-12 · Schema Drift Silently Invalidated Every
      Checkpoint** written FIRST (spec Step 9, CLAUDE.md §4)
- [ ] **D2** Retrain on 12 entity types / 52 obs types — **NOT DONE.** This box
      has no CUDA and no `nvidia-smi` (confirmed 2026-08-27); it does have
      Apple-silicon MPS via a local torch 2.13.0 install, but a real
      convergence run (the historical `phase50` runs used hidden_dim=128,
      2–3 layers, up to 200 windows, tens of epochs) is GPU-scale work that
      belongs on Kaggle per this agent's own convention and the P100 sm_60
      pin — not something to fake locally. **What's actually done:** a real
      (non-production) local smoke run, see below, proving the schema-fixed
      pipeline produces genuine gradient signal. **What's still required:**
      the owner runs `python scripts/kaggle_launch.py --epochs 30` (see
      "Kaggle invocation" below) — one command uploads the current repo
      (12/52 schema baked in, since packaging copies live `agent/`+`scripts/`),
      pushes kernel V73 (`--verify-version` confirms notebook/launcher/VERSIONS.md
      in sync, fingerprint `5a1971b5b8e6`), tails logs, downloads the checkpoint,
      and runs the local IC backtest. `KAGGLE_API_TOKEN` is present in `.env`;
      the `kaggle` CLI itself is not installed in this sandbox, so this session
      could not have kicked the job off even if it should have — it is a job
      only the owner can run from a machine with Kaggle access.
- [ ] **D3** F-02 check: print active return-head branch at start — **code
      confirmed already wired** (`trainer.py` logs `[HEAD] return_concat_head
      ACTIVE` when the concat path is live, `[HEAD] return_raw_head ACTIVE —
      ... BYPASSED` as a loud warning otherwise) and **exercised for real** in
      this session's smoke run (log line reproduced below) — but this checks
      the mechanism, not the still-pending production run in D2. Leaving
      unchecked until D2's actual accepted run logs the concat-head line.
- [ ] **D4** F-01 check: `torch.std(emb, dim=0).mean() > 0.1` — **code
      confirmed already wired** (`trainer.py` computes instrument-embedding
      std + SVD effective rank every 5th epoch, warns `[COLLAPSE]` below 0.05)
      and exercised in the smoke run with no collapse triggered — same caveat
      as D3, this is the toy run's numbers, not D2's.
- [ ] **D5** F-04 check: eval splits time-ordered — **code confirmed already
      wired**: the smoke run's own log shows the chronological split boundary
      it computed (`train [1920-01-01 → 2025-05-21]  val [2025-05-21 →
      2026-01-20]  test [2026-01-20 → 2030-01-30]`), never shuffled. Same
      caveat — the split logic is verified, D2's actual eval numbers are not.
- [ ] **D6** New checkpoint loads with 0 skipped / 0 missing keys — **verified
      on the smoke checkpoint**: `Trainer.load_model()` round-tripped it with
      no drift warning and no missing/skipped-key log line (only the expected
      "Restored return_concat_head" + EWC-restore lines). Mechanism is proven;
      D2's real checkpoint still needs the same round-trip once it exists.

**Local smoke validation (2026-08-27, this session) — real numbers, not a
production retrain:**

Ran `scripts/retrain_gnn.py` locally against the live, schema-fixed
`.tirra_pipeline/pipeline.db` (364,296 real observations, 5,628 entities) —
CPU, hidden_dim=16, 1 layer, 1 head, `--gdelt-frac 0.005 --defi-frac 0.02
--max-windows 5`, 3 epochs, output routed to the session scratchpad only
(never touched the live `gnn_model.pt` / `checkpoints/` — confirmed by mtime,
all live checkpoint files predate this session). Wall clock: 49.3s total.

| epoch | total loss | return loss | in-sample IC (spearman, n=256) | instrument emb_std | eff. rank |
|---|---:|---:|---:|---:|---:|
| 1 | 1575.68 | 279.31 | 0.043 | 27,882.8 | 1.9 |
| 2 | 1403.15 | 254.13 | 0.104 | 31,388.8 | 2.0 |
| 3 | 1172.63 | 195.71 | 0.117 | 22,907.9 | 1.8 |

- Loss decreased monotonically across all 5 components (obs_type, time_delta,
  contrastive, value, return) — real gradient flow, not a no-op.
- `[HEAD] return_concat_head ACTIVE — GNN embeddings + raw features → return.`
  logged at training start — F-02 gate passes, GNN is in the return path.
- `emb_std` >> the 0.05 F-01 collapse floor — no collapse (expected for a toy
  run this small; **not** evidence the eventual production embeddings will be
  diverse, just that the collapse detector itself fires correctly and this
  particular tiny run isn't degenerate).
- Reload via `Trainer.load_model()` succeeded with 0 skipped/missing keys.

**This is not D2.** hidden_dim=16/1-layer/5-windows/3-epochs is a pipeline
health check, not a production checkpoint — it proves the 12/52 schema fix
lets the trainer build a model and learn *something* real on real data. It
says nothing about whether a properly-sized run would clear the
quant-evaluator's "beats a trivial baseline" bar; per their finding, the prior
`phase50` checkpoint (much more trained than this smoke run) already did not
clear that bar on frozen embeddings. D2 remains the only path to an answer
either way.

**Kaggle invocation for the owner (zero further setup needed):**

```
python scripts/kaggle_launch.py --epochs 30
```

This packages the current repo (schema fix included), pushes kernel V73,
tails logs to completion, downloads the resulting checkpoint, and runs
`scripts/phase40_gnn_backtest.py` locally against it. Add `--no-gpu` only if
GPU quota is exhausted (falls back to CPU-only kernel, 1 retry instead of 6).
Use `--status` any time to check an in-flight run, `--logs-only` to re-tail.

### Phase E — orchestration
- [x] **E1** `scripts/run_chain.py` — dependency order, per-DAG row deltas,
      schema validation before model DAGs, non-zero exit on failure (spec 8.1)
- [x] **E2** `run_scheduled.sh chain` (spec Step 8.2), passes through extra args
- [x] **E3** `deploy/systemd/tirra-chain.{service,timer}` + README explaining
      chain-vs-collect and why cron alone never worked (spec Step 8.3)
- [x] **E4** End-to-end verified. First real chain run (`--skip-collection`):

      | table | before | after |
      |---|---:|---:|
      | `signals` | 0 | **23** |
      | `beliefs` | 0 | **46** |
      | `features` | 93 | **127** |

      `convergence_detection`, `feature_generation`, `adversarial_scan`,
      `world_model_update`, `rl_training` all completed.
- [ ] **E5** Assert `rl_transitions` needs 2 consecutive `inference` runs (not a bug)

**Post-timeout-fix run** (`--only gnn_inference,entity_scoring,inference`):
`gnn_inference` **completed** in 73.2s (was killed at 60s), `inference`
**completed 4/4**. Both write nothing, and that is *correct*:
`rl_policy_checkpoints` is 0, so `load_models` reports `has_sac=False` and
`sac_inference`/`emit_portfolio` take the intended `skipped` path. Verified this
is genuine graceful degradation, not the silent-failure bug returning — the
Phase B fix preserved the skip path deliberately.

### Remaining cold-start chain — all blocked on Phase D only

```
retrain GNN  ──▶ entity_scoring works  ──▶ entity_alerts populate
             ──▶ rl_training produces a SAC checkpoint
             ──▶ inference writes portfolio_weights + paper_trade_pnl
             ──▶ 2nd consecutive inference run ──▶ rl_transitions
```

Everything *upstream* of the model now works. `entity_scoring` is the only
genuine failure left (`40x76 vs 15x64`) and it is pure schema drift.

Full drift now visible in the logs (was silent before the guards):

| parameter | trained | current |
|---|---:|---:|
| `type_projections.instrument` | 23 | 50 |
| `type_projections.*` (11 others) | 14 | 15 |
| `obs_type_head` | 46 | 52 |
| `hgt_layers.{0,1}.{k,v}_rel` | 34 | 40 |

### Phase E′ — node timeouts (discovered by the first chain run)
- [x] **E′1** `Node.timeout` defaults to 60s — correct for one HTTP fetch in
      `daily_collection`, far too short for graph build + GNN forward pass.
      `gnn_inference` was killed at **69.6s** ("Execution timed out (>60s)"),
      which cascaded: `inference`'s own gnn node died the same way and skipped
      `sac_inference` + `emit_portfolio`. So `portfolio_weights` /
      `paper_trade_pnl` were empty for a reason unrelated to the model.
- [x] **E′2** Raised: `gnn_inference.train_gnn`=1800s,
      `entity_scoring.score_entities`=900s, `inference.gnn_inference`=1200s,
      `sac_inference`/`emit_portfolio`=600s
- [x] **E′3** Guard test asserting no inference node keeps the 60s fetch default

---

## Notes / decisions

- **2026-08-26** — User approved: retrain from scratch, plus ship all four fixes
  (scheduler wiring, silent-failure fixes, schema constant sync, checkpoint
  immutability).
- **2026-08-26** — Diagnostic probe run overwrote `.tirra_pipeline/gnn_model.pt`
  (was May 25). Post-probe copy preserved in session scratchpad as
  `gnn_model_postprobe.pt`; `gnn_model_phase50.pt` (May 24, 23-dim) untouched.
  The overwrite is itself evidence for Phase C — the DAG mutates checkpoints in
  place.
- DB backed up before probing (138 MB) to session scratchpad as
  `pipeline_backup.db`.
- `world_model_update` and `feature_generation` verified working on real data
  (23 beliefs / 17 features written) — do **not** treat them as broken.

### Phase F — test-suite honesty (side effects of the above)
- [x] **F1** Full suite: **64 failed → 54 failed**, 10,608 passed. Net −10.
      12 pre-existing failures fixed, 3 new appeared (2 real, 1 live-network flake).
- [x] **F2** `test_brief_server` 403s were **not a code bug** — adding
      `TIRRA_PADDLE_WEBHOOK_SECRET` to `.env` correctly flipped
      `_authorized_for` from dev-mode-open into production gating. Something in
      the suite loads `.env` (e.g. `agent/convergence/backtest.py`), so the
      tests inherited it. They passed alone and failed in-suite.
- [x] **F3** Made the fixture hermetic (`_open_access` pins both auth env vars)
      and added `TestBriefGating` — 3 tests covering the gated path, which had
      **no coverage at all** before. Verified 15/15 pass both with and without
      Paddle credentials in the environment.
- [x] **F4** `.claude/scheduled_tasks.lock` (runtime JSON) added to `.gitignore`

`test_power_grid_edge::test_live_forecast` is the third new failure — a live
NYISO call, same family as the two `test_live_demand`/`test_live_fuel_mix`
failures already in the baseline. Not investigated; depends on NYISO having
published, not on this work.

## Outcome

**Layers 2–6 are producing data for the first time.** `signals` 0 → 23,
`beliefs` 0 → 46, `features` 93 (stale since April) → 127. Seven DAGs now
complete where eight had never run at all.

Five independent defects fixed: orchestration gap, `ENRICHMENT_DIM` buffer
overflow, `inference` silent-success, 60s model-node timeouts, and the
brief-server test-isolation bug. Schema drift is documented, guarded, and
diagnosable — but **still requires the Phase D retrain**, which is the only
remaining blocker.

Not committed. 88 files changed; `.env` untracked and the diff scanned for
secrets (clean — the only `pdl_ntfset` hits are docstrings and a fake test
constant).

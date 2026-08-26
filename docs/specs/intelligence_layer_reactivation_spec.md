---
title: Spec — Intelligence Layer Reactivation
tags:
  - doc/spec
  - topic/pipeline
  - topic/gnn
  - status/active
---

# Spec — Intelligence Layer Reactivation

Research: `docs/research/intelligence_layer_reactivation.md`
Task: `tasks/active/intelligence_layer_reactivation.md`

Ordered, atomic, independently verifiable steps. **Order is load-bearing**:
schema constants must be correct *before* retraining, and retraining must
complete *before* the model-dependent DAGs are scheduled.

---

## Step 1 — Sync entity/observation type constants

**File:** `agent/models/gnn/graph_builder.py`

1.1 Add `"maritime_area"` to `ENTITY_TYPES`, preserving alphabetical order
    (between `instrument` and `organization`). Length 11 → 12.

1.2 Add to `OBSERVATION_TYPES`, preserving alphabetical order:
    `area_daily_activity`, `baltic_activity_proxy`,
    `futures_positioning_derived`, `petroleum_inventory`. Length 48 → 52.

**Why ordering matters:** `_ENTITY_TYPE_TO_IDX` / `_OBS_TYPE_TO_IDX` derive
one-hot positions from list index. Any insertion shifts every subsequent index,
which is exactly why this step must precede retraining.

**Verify:**
- `len(ENTITY_TYPES) == 12`, `len(OBSERVATION_TYPES) == 52`
- `BASE_FEAT_DIM == 15`
- `set(db_entity_types) - set(ENTITY_TYPES) == set()`
- `set(db_obs_types) - set(OBSERVATION_TYPES) == set()`
- Building the live graph emits **no** `Unknown entity type` warning.

---

## Step 2 — Fail loudly on unknown entity type

**File:** `agent/models/gnn/graph_builder.py:533-536`

Replace the silent `type_idx = 0` fallback. An unknown entity type must not be
silently one-hot encoded as `ENTITY_TYPES[0]` (`cftc_contract`).

Behaviour: raise a clear error naming the unknown type and pointing at
`ENTITY_TYPES`, **unless** an explicit opt-out flag is set for
forward-compatibility during collection of a brand-new type.

**Verify:** targeted test — build features for a fabricated entity type and
assert it raises rather than returning a `cftc_contract` one-hot.

---

## Step 3 — Make `inference` fail loudly

**File:** `agent/pipeline/dags/inference.py`

The `gnn_inference` node catches the forward-pass exception, logs
`GNN inference failed: ...`, and returns a result that downstream nodes treat as
success. All four nodes then report `completed` while writing zero rows.

Change: a GNN forward-pass failure must mark the node **failed**, not
`completed`. Preserve the *intentional* graceful-degradation path documented in
the module docstring (missing model → `status="skipped"`), which is a different
condition from "the model is present but threw".

Distinguish:
- model file / SAC checkpoint absent → `skipped` (intended, keep)
- model present but forward pass raised → `failed` (currently wrong)

**Verify:** targeted regression test — a DAG run whose GNN raises must produce
`run.status == "failed"`, not `completed`. This is the test that would have
caught the current bug.

---

## Step 4 — Fix `entity_scoring` index crash

**File:** `agent/pipeline/dags/entity_scoring.py` (+ scorer internals)

`index 69 is out of bounds for dimension 1 with size 69` — classic off-by-one
against an obs-type-cardinality-derived tensor width. Reproduce in isolation
first; do not guess. May resolve as a side effect of Step 1 (cardinality 48 →
52); if so, confirm *why* rather than assuming.

**Verify:** `entity_scoring` DAG runs to `completed` and writes ≥1 row to
`entity_alerts`.

---

## Step 5 — Checkpoint immutability

**Files:** `agent/models/gnn/trainer.py` (save path), `agent/pipeline/dags/gnn_inference.py`

`gnn_inference` currently overwrites `.tirra_pipeline/gnn_model.pt` in place.

Change: write a new versioned checkpoint (timestamped or run-id suffixed) and
update a stable pointer (symlink or a small `current.json`) rather than
clobbering the artifact. Existing checkpoints must never be written through.

**Verify:** run `gnn_inference` twice; assert the original file's mtime+hash are
unchanged and two distinct new checkpoints exist.

---

## Step 6 — Checkpoint/schema compatibility guard

**File:** `agent/models/gnn/trainer.py` (`load_model`)

Add an explicit compatibility assertion at load: compare checkpoint
`in_channels` against what the live `GraphBuilder` currently produces. On
mismatch, raise with both dicts in the message.

This is the guard that would have caught the 23 → 49 drift in April instead of
today. Must be defeasible (explicit flag) for deliberate partial-load workflows.

**Verify:** loading today's stale checkpoint against the live graph raises a
clear, actionable error naming `instrument: checkpoint=23 live=50`.

---

## Step 7 — Retrain the GNN

**Precondition:** Steps 1–2, 6 complete and verified. Training must see the
corrected 12-entity-type / 52-obs-type schema.

Per CLAUDE.md §4 and LESSONS.md, before accepting the run:
- Print the active return-head branch at training start (F-02).
- Verify embedding diversity: `torch.std(emb, dim=0).mean() > 0.1` (F-01).
- Confirm eval splits are time-ordered, not random (F-04).
- Confirm history arrays are length-aligned on any resume (F-03).

Write the resulting checkpoint as a **new** immutable artifact.

**Verify:** `Trainer.load_model(new_ckpt)` against the live graph passes the
Step 6 guard with zero skipped keys and zero missing keys.

---

## Step 8 — Wire the DAG chain

**Files:** `scripts/run_scheduled.sh`, `deploy/systemd/`, new chain entry point

8.1 Add a chain runner that executes DAGs in **dependency order**, not wall-clock
    order, stopping (or reporting) on upstream failure:
    `daily_collection → convergence_detection → gnn_inference → entity_scoring
     → feature_generation → adversarial_scan → world_model_update → rl_training
     → inference`

8.2 Add `run_scheduled.sh chain` invoking it synchronously (same
    fire-and-forget-safety reasoning as the existing `collect` mode).

8.3 Add a systemd service+timer for the chain; keep `tirra-collect` for the
    collection-only cadence.

**Verify:** one `chain` invocation, from a DB with empty intelligence tables,
produces non-zero rows in `beliefs` **and** `features`. Note `rl_transitions`
requires two consecutive `inference` runs by design — assert that explicitly
rather than treating 0 as failure on run 1.

---

## Step 9 — LESSONS.md entry

Add a new fuckup entry (symptom → root cause → fix → prevention rule) for
schema-drift-invalidates-checkpoint, citing the Step 6 guard as the prevention
rule. Per CLAUDE.md §4, this entry is written **before** the retrain is merged.

---

## Out of scope (tracked, not fixed here)

- `convergence_detection` / `rl_training` / `adversarial_scan` writing nothing.
- `depth_evaluations` having no DAG home.
- Missing API keys (NASA FIRMS, EIA ×2); changed vendor APIs (LDA, USPTO, FEC).

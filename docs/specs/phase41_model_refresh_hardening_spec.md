---
title: "Spec: Phase 41 — Model Refresh Hardening"
tags:
  - doc/spec
  - phase/41
  - topic/gnn
  - topic/backtest
  - topic/pipeline
  - layer/world-model
  - layer/learning
  - layer/surveillance
---

# Spec: Phase 41 — Model Refresh Hardening

## Goal

Bring the live GNN checkpoint back to a healthy state, harden the training loop so auto-tuned multi-task loss can no longer diverge, regenerate downstream artifacts (features + backtest) from the healthy model, and widen observation diversity by scheduling one new L2 tool into `daily_collection`.

Research: [[phase41_model_refresh_hardening]].

## Files Affected

- [agent/models/gnn/trainer.py](agent/models/gnn/trainer.py)
- [tests/test_trainer.py](tests/test_trainer.py)
- [.tirra_pipeline/gnn_model.pt](.tirra_pipeline/gnn_model.pt) (and siblings — file-level swap only)
- [agent/pipeline/dags/daily_collection.py](agent/pipeline/dags/daily_collection.py)
- One existing DAG test file (add an assertion for the new node)

## Implementation Steps

### 41.1 Clamp Kendall log-variance in the trainer

In `TrainerConfig`, add two fields with safe defaults:

```python
log_var_min: float = -3.0
log_var_max: float = 3.0
```

In `Trainer._log_vars`, compose the total loss using **clamped** copies of the log-variance parameters. Use `torch.clamp` in a fresh tensor (not in-place) so that gradients still flow for values inside the interval but saturate at the bounds, and so that the stored `Parameter` tensors are not mutated mid-training.

In `effective_loss_weights()`, return `math.exp(-clamped)` using the same bounds, so that serialized / reported weights always match what the training step actually used.

Optimizer registration is unchanged (the raw parameters still require grads; the clamp is applied in the forward path).

### 41.2 Tests for the clamp

Add targeted tests in [tests/test_trainer.py](tests/test_trainer.py):

- `test_auto_tune_log_var_clamp_bounds`: manually set each `_log_vars` parameter to a value well outside `[log_var_min, log_var_max]`, call `effective_loss_weights()`, and assert all four effective weights lie in `[exp(-log_var_max), exp(-log_var_min)]`.
- `test_auto_tune_loss_total_finite_on_zero_component`: construct a trainer, set all component losses to zero tensors, run one loss-composition step with an extreme log-variance, assert that the total loss is finite and bounded by `4 * log_var_max`.
- `test_fixed_weights_unaffected`: with `auto_tune_loss_weights=False`, assert `effective_loss_weights()` returns the config's fixed weights regardless of clamp fields.

### 41.3 Restore a healthy checkpoint as the live model

File-level swap, no code changes:

1. Move the current broken `gnn_model.pt` to `gnn_model_broken_10ep.pt` (preserve for forensics).
2. Copy `gnn_model_live.pt` (known-good 5-epoch model from 2026-04-19) to `gnn_model.pt`.
3. Verify the file is loadable and architecture matches the current pipeline (`entity_types`, `num_nodes`) by running a light smoke script or `scripts/retrain_gnn.py --skip-eval --epochs 0` if supported; otherwise, do a direct `torch.load` smoke check through a small helper.

If architecture has drifted (e.g. `num_nodes` mismatch after new entities), fall back to step 41.4 directly.

### 41.4 Fresh retrain with the clamp + regen features + rerun backtest

Run the retrain script with the settings that produced the healthy Run 1:

```bash
python scripts/retrain_gnn.py \
  --db-path .tirra_pipeline/pipeline.db \
  --epochs 5 --auto-tune --since 2023-01-01 \
  --window-size 172800 --backup
```

Acceptance criteria:

- Total loss decreases monotonically across epochs (no negative total loss).
- Final effective weights all lie within `[exp(-3), exp(3)] ≈ [0.05, 20]`.
- Test `obs_type` top-1 accuracy ≥ 50% (random baseline is ~14%).
- Test `time_delta` MAE ≤ 60 seconds.

If criteria pass, regenerate features:

```bash
python scripts/run_collection.py --dag feature_generation
```

Then rerun the walk-forward backtest (same entrypoint as Phase 40 step 40.4) and record the three baseline strategies (EqualWeight, BuyHold SPY, BuyHold SPY+AGG) for comparison to the Phase 40 numbers.

### 41.5 Wire `whale_alert` into `daily_collection`

Add a node to `build_daily_collection_dag` in [agent/pipeline/dags/daily_collection.py](agent/pipeline/dags/daily_collection.py), between `fetch_polymarket` and `fetch_macro`:

```python
dag.add(
    "fetch_whale_alert",
    operator="whale_alert",
    table_name="whale_alert",
    params={"mode": "confirmed", "min_btc": 10.0, "limit": 100},
    timeout=60,
    retries=2,
)
```

Use `confirmed` mode (latest confirmed block) rather than `mempool` to maximise persisted observations per daily run. Add a test ensuring the new node is present and that the DAG validates.

### 41.6 Update task and checkpoint

Mark 41.1–41.5 complete in the task file, and update the project memory with the final model metrics and the newly scheduled tool.

## Edge Cases

- `log_var_min >= log_var_max`: reject in `TrainerConfig.__post_init__` or clamp defensively. Cheapest is documented contract + a test that asserts the defaults are well-ordered.
- Clamp applied *only* when `auto_tune_loss_weights=True`. The fixed-weight path must be unaffected.
- Model swap (41.3) must not silently accept a checkpoint whose `num_nodes` differs from the current graph. The smoke check must surface such a mismatch.
- `whale_alert` node must gracefully degrade when `blockchain.info` is unreachable — the tool already returns a failed `ToolResult` rather than raising, so the DAG's `retries=2` plus `failure cascading = False` (default) is sufficient.
- Retrain may be infeasible in the current environment if the SQLite DB is locked by a running scheduler. Guard by checking that no APScheduler process is running.

## Testing Plan

- Unit: three new trainer tests in 41.2. Full `pytest tests/test_trainer.py` must remain green.
- Unit: one new DAG test asserting `fetch_whale_alert` exists with the right operator and params.
- Integration (manual / scripted): 41.4 acceptance criteria above. Record loss curve and backtest table in the checkpoint.

## Related

- [[phase41_model_refresh_hardening]]
- [[phase41_model_refresh_hardening]] task: [[phase41_model_refresh_hardening]]
- [[real_data_model_refresh_spec]]
- [[temporal_het_gnn]]
- [[whale_alert_l2]]
- [[chat_checkpoint_2026-04-20_phase40_full]]

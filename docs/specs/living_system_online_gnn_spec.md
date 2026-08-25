---
title: "Spec: Phase 46 — Living System: Online GNN with EWC"
tags:
  - doc/spec
  - phase/46
  - topic/living-system
  - layer/world-model
---

# Spec: Phase 46 — Living System: Online GNN with EWC Continuous Learning

## Goal

Make HetTGN weights evolve continuously between full retrains by adding Elastic Weight
Consolidation (EWC) online updates. After each batch of ≥100 new observations (collected
since the last update), run 1 gradient step on those new events penalised by EWC. GNN
weights stay current without catastrophic forgetting.

Math: $\mathcal{L} = \mathcal{L}_\text{new} + \lambda \sum_i F_i(\theta_i - \theta_i^*)^2$

## Files Affected

| File | Action |
|------|--------|
| `agent/models/gnn/ewc.py` | CREATE |
| `agent/models/gnn/trainer.py` | MODIFY |
| `agent/pipeline/dags/gnn_inference.py` | MODIFY |
| `tests/test_ewc.py` | CREATE |

## Implementation Steps

### 46.1 — Create `agent/models/gnn/ewc.py`

Create a self-contained EWC module. No dependencies outside torch and the existing trainer.

```python
# agent/models/gnn/ewc.py

@dataclass
class EWCState:
    fisher: dict[str, Tensor]    # Fisher diagonal — {param_name: tensor}
    anchor: dict[str, Tensor]    # Model weights at last full retrain
    lambda_: float = 1000.0      # EWC regularisation strength
    last_update_ts: float = 0.0  # Unix timestamp of last online update
    obs_count_at_update: int = 0 # Observation count when last update ran
```

Functions:
- `compute_fisher(model, loss_fn, data_iter) -> dict[str, Tensor]`
  - Run one forward+backward pass per batch, accumulate squared gradients
  - Return {name: mean_squared_grad} for all named parameters
  - Zero gradients before and after
- `ewc_penalty(model, state) -> Tensor`
  - Return scalar: `lambda_ * sum(F_i * (theta_i - theta_i*)^2)` over all params
  - Must handle shape mismatches gracefully (return 0.0 if shapes differ — schema change guard)

**Exit condition:** `python -c "from agent.models.gnn.ewc import EWCState, compute_fisher, ewc_penalty; print('ok')"` exits 0.

---

### 46.2 — Extend `TrainerConfig` in `trainer.py`

Add two new fields (with defaults so existing code is unaffected):

```python
ewc_lambda: float = 1000.0
online_batch_threshold: int = 100
```

**Exit condition:** `TrainerConfig()` instantiates without error. Existing config dicts that omit these keys still work.

---

### 46.3 — Extend `Trainer.train()` to compute Fisher + anchor after full retrain

After the final training epoch completes, before `return`:

```python
# Compute and store EWC state
fisher = compute_fisher(self._model, loss_fn=self._compute_loss, data_iter=train_events)
self._ewc_state = EWCState(
    fisher=fisher,
    anchor={n: p.data.clone() for n, p in self._model.named_parameters()},
    lambda_=self.config.ewc_lambda,
    last_update_ts=time.time(),
    obs_count_at_update=len(all_events),
)
```

`self._compute_loss` is an internal helper that runs one forward pass and returns the scalar loss.
It already exists in the training loop — extract it as a named method or lambda.

**Exit condition:** After `trainer.train()`, `trainer._ewc_state` is not None.

---

### 46.4 — Extend `save_model` / `load_model` to persist EWC state

In `save_model`, add to the checkpoint dict (only if `_ewc_state` is not None):

```python
if self._ewc_state is not None:
    checkpoint["ewc_fisher"] = {k: v.cpu() for k, v in self._ewc_state.fisher.items()}
    checkpoint["ewc_anchor"] = {k: v.cpu() for k, v in self._ewc_state.anchor.items()}
    checkpoint["ewc_lambda"] = self._ewc_state.lambda_
    checkpoint["ewc_last_update_ts"] = self._ewc_state.last_update_ts
    checkpoint["ewc_obs_count_at_update"] = self._ewc_state.obs_count_at_update
```

In `load_model`, reconstruct EWC state if fields present:

```python
if "ewc_fisher" in checkpoint:
    trainer._ewc_state = EWCState(
        fisher=checkpoint["ewc_fisher"],
        anchor=checkpoint["ewc_anchor"],
        lambda_=checkpoint.get("ewc_lambda", 1000.0),
        last_update_ts=checkpoint.get("ewc_last_update_ts", 0.0),
        obs_count_at_update=checkpoint.get("ewc_obs_count_at_update", 0),
    )
```

Old checkpoints without EWC fields load cleanly (`_ewc_state = None`).

**Exit condition:** Round-trip: `save_model` → `load_model` → `trainer._ewc_state` is not None and Fisher shapes match.

---

### 46.5 — Add `Trainer.online_update(new_events)` method

```python
def online_update(self, new_events: list[dict]) -> dict:
    """Run 1 EWC-regularised gradient step on new_events.

    Args:
        new_events: list of raw event dicts (same format as training data).

    Returns:
        dict with keys: loss_new, loss_ewc, loss_total, n_events.

    Raises:
        RuntimeError: if model or EWC state not initialised.
    """
```

Logic:
1. Guard: if `_model is None` or `_ewc_state is None`, raise `RuntimeError`
2. Build a mini-batch HeteroData from `new_events` (reuse `GraphBuilder`)
3. Forward pass → compute `L_new`
4. Compute `L_ewc = ewc_penalty(self._model, self._ewc_state)`
5. `L_total = L_new + L_ewc`
6. `optimizer.zero_grad(); L_total.backward(); clip_grad_norm_(..., 1.0); optimizer.step()`
7. Update `_ewc_state.last_update_ts = time.time()` and `obs_count_at_update`
8. Return metrics dict

**Exit condition:** `trainer.online_update([])` raises `RuntimeError("No events")`. With ≥1 event, returns dict with expected keys and `loss_total >= 0`.

---

### 46.6 — Wire into `run_gnn_inference` in `gnn_inference.py`

After `trainer.train()` completes (or after `load_model` on a run that skips training):

```python
# Query observations newer than last online update
last_ts = getattr(trainer, '_ewc_state', None)
last_ts = last_ts.last_update_ts if last_ts else 0.0
new_events = store.query_observations_since(last_ts)  # new store method — see note

if len(new_events) >= min(cfg.online_batch_threshold, 50):
    update_result = trainer.online_update(new_events)
    log.info("EWC online update: %s", update_result)
    trainer.save_model(model_path)   # persist updated weights + new EWC timestamp
```

Note on `store.query_observations_since(ts)`: `PipelineStore` already has `query_observations`
with various filters. Check if a `since_ts` filter exists; if not, add it as a one-liner filter
on the existing query (single-line addition, not a new method).

**Exit condition:** After a DAG run that finds ≥100 new obs, model file is updated and `ewc_last_update_ts` in the saved checkpoint is > the prior value.

---

## Edge Cases

1. **No EWC state on first run** — `_ewc_state is None` before first full retrain. `online_update` must raise cleanly, not silently no-op.
2. **Schema change** — new entity type added means model rebuilt with new params. Fisher shapes no longer match anchor. `ewc_penalty` returns 0.0 and logs a warning. Fisher recomputed after next full retrain.
3. **Threshold not reached** — `len(new_events) < threshold` → skip online update, no file write.
4. **Empty graph** — `online_update` with 0 events → raise `RuntimeError("No events for online update")`.
5. **Checkpoint without EWC** — old checkpoint loaded → `_ewc_state = None` → no EWC penalty → normal training continues, EWC populated after next full retrain.

## Testing Plan

`tests/test_ewc.py` — all tests use synthetic data (no real DB):

| Test | What it proves |
|------|----------------|
| `test_ewc_state_creation` | `EWCState` instantiates with correct defaults |
| `test_compute_fisher_returns_dict` | `compute_fisher` returns dict with same keys as `model.named_parameters()` |
| `test_ewc_penalty_zero_when_no_drift` | penalty = 0 when `theta == theta_anchor` |
| `test_ewc_penalty_positive_when_drift` | penalty > 0 after one gradient step without anchor update |
| `test_ewc_penalty_shape_mismatch_returns_zero` | wrong Fisher shape → 0.0, no crash |
| `test_trainer_has_ewc_state_after_train` | `_ewc_state` not None after `trainer.train()` |
| `test_save_load_roundtrip_with_ewc` | Fisher + anchor survive save/load, shapes identical |
| `test_save_load_roundtrip_without_ewc` | Old checkpoint (no EWC fields) loads cleanly, `_ewc_state = None` |
| `test_online_update_reduces_loss` | 1 gradient step on new events returns valid loss dict |
| `test_online_update_no_model_raises` | `RuntimeError` when model not built |
| `test_online_update_no_ewc_state_raises` | `RuntimeError` when EWC state not computed |
| `test_online_update_empty_events_raises` | `RuntimeError` on empty event list |

## Related

- [[living_system_online_gnn]]
- [[quant_training_ground]]

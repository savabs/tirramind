---
title: "Spec: Phase 41b — GNN Signal Extraction (ListNet + Auto-Tune)"
tags:
  - doc/spec
  - phase/41b
  - topic/world-model
  - layer/feature-engineering
  - status/active
---

# Spec: Phase 41b — GNN Signal Extraction

## Goal

Move IC from -0.033 (noise) to >0.03 (t>2.0) by fixing the two confirmed root causes in the GNN
training loop: wrong loss function for cross-sectional ranking (RC2) and return gradient starvation
relative to other losses (RC1). A third fix (GDELT volume reduction) is already active via
`gdelt_subsample_frac=0.05` in the CLI default.

Research: [[phase41b_gnn_signal_extraction]]

## What's Already Implemented (do NOT re-implement)

| Fix | Status | Where |
|---|---|---|
| Homoscedastic uncertainty weighting (Kendall 2018) | ✅ DONE | `TrainerConfig.auto_tune_loss_weights`, `_log_vars` in `Trainer` |
| GDELT volume subsampling | ✅ DONE | `--gdelt-frac 0.05` CLI default in `retrain_gnn.py` |
| `return_pred_head` MLP on instrument nodes | ✅ DONE | `het_tgn.py` line 423, `trainer.py` line 1361 |

## What's Missing (this spec)

1. **ListNet ranking loss** — return head still uses `F.huber_loss`; need ListNet cross-entropy on ranks
2. **Minimum-2-instrument guard** — ListNet requires ≥2 ranked items; Huber guard (`any()`) must become `sum() >= 2` when ListNet is active
3. **`--listnet` CLI flag** in `retrain_gnn.py`
4. **North Star propagation diagnostic** — `scripts/phase41b_propagation_diagnostic.py`

## Files Affected

| File | Change |
|---|---|
| `agent/models/gnn/trainer.py` | Add `_listnet_loss()` helper + 2 `TrainerConfig` fields + replace loss call |
| `scripts/retrain_gnn.py` | Add `--listnet` flag |
| `scripts/phase41b_propagation_diagnostic.py` | New file — Granger causality test |

## Implementation Steps

### 41b.1 — Add `_listnet_loss` helper to trainer.py

Add this module-level function immediately before the `SyntheticGraphGenerator` class (or in a
logical position near the top after imports):

```python
def _listnet_loss(scores: torch.Tensor, targets: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """ListNet top-1 approximation (Cao et al. 2007, ICML).

    Minimises KL(p_target || p_pred) where p = softmax(x / tau).
    Directly optimises cross-sectional rank ordering (IC metric).

    Args:
        scores: Model predicted scores, shape (N,).  N must be >= 2.
        targets: Realised returns (or any continuous ranking target), shape (N,).
        tau: Softmax temperature. 1.0 = standard; lower = harder ranking.

    Returns:
        Scalar loss tensor (non-negative).
    """
    p_target = F.softmax(targets / tau, dim=0)
    log_p_pred = F.log_softmax(scores / tau, dim=0)
    return -(p_target * log_p_pred).sum()
```

**Test:** `_listnet_loss(torch.tensor([1.0, -1.0]), torch.tensor([1.0, -1.0]))` should return near 0
(perfect ranking). `_listnet_loss(torch.tensor([-1.0, 1.0]), torch.tensor([1.0, -1.0]))` should
return a positive value (wrong ranking).

### 41b.2 — Add two fields to `TrainerConfig`

Add after the `return_weight` docstring block (before `gdelt_subsample_frac`):

```python
use_listnet_return_loss: bool = False
"""When True, replace Huber return loss with ListNet cross-entropy ranking loss
(Cao et al. 2007, ICML). Requires >= 2 instrument observations per window;
windows with < 2 finite-return instruments are skipped for the return loss.
Directly optimises cross-sectional IC (Spearman rank correlation)."""

listnet_temperature: float = 1.0
"""Softmax temperature tau for ListNet. Higher = softer target distribution
(less peaked on best instrument). 1.0 is standard. Lower values approach
hard argmax ranking."""
```

### 41b.3 — Replace Huber return loss call with conditional ListNet

In `trainer.py`, find the block ending with:
```python
        if _finite_mask.any():
            _ret_pred = model.return_pred_head(_ret_emb_t).squeeze(-1)
            ret_loss = F.huber_loss(
                _ret_pred[_finite_mask], _ret_tgt_t[_finite_mask]
            )
```

Replace with:
```python
        _n_valid = int(_finite_mask.sum().item())
        _min_required = 2 if cfg.use_listnet_return_loss else 1
        if _n_valid >= _min_required:
            _ret_pred = model.return_pred_head(_ret_emb_t).squeeze(-1)
            if cfg.use_listnet_return_loss:
                ret_loss = _listnet_loss(
                    _ret_pred[_finite_mask],
                    _ret_tgt_t[_finite_mask],
                    tau=cfg.listnet_temperature,
                )
            else:
                ret_loss = F.huber_loss(
                    _ret_pred[_finite_mask], _ret_tgt_t[_finite_mask]
                )
```

### 41b.4 — Add `--listnet` flag to retrain_gnn.py

After the `--auto-tune` argument block, add:

```python
parser.add_argument(
    "--listnet",
    action="store_true",
    help="Use ListNet ranking loss for return head instead of Huber (Phase 41b).",
)
```

Pass to config:
```python
config = TrainerConfig(
    ...
    use_listnet_return_loss=args.listnet,
    ...
)
```

### 41b.5 — Build `scripts/phase41b_propagation_diagnostic.py`

Granger causality test: does embedding[entity_upstream, T-N] → embedding[instrument, T]?

For each instrument + upstream entity type pair, test across lags N = [1, 7, 14, 21] days.
Reports: which entity types Granger-cause which instrument embeddings, at which lags.

Validates whether the GNN perceptual layer encodes real pre-emergence causal information.

### 41b.6 — Kaggle retrain

Upload updated code, retrain from epoch 20 to epoch 30 with:
```
python scripts/retrain_gnn.py \
  --auto-tune --listnet \
  --gdelt-frac 0.05 \
  --epochs 30 --resume 20 \
  --backup
```

### 41b.7 — Re-run IC diagnostic

```
python scripts/phase40_gnn_backtest.py
```

Exit condition: Mean IC > 0.03 and t-stat > 2.0 for GNN-ReturnHead strategy.
If not met: run `scripts/phase41b_propagation_diagnostic.py` to diagnose.

## Edge Cases

- Windows with 0 or 1 instrument in `next_obs` — ListNet skips silently (ret_loss stays 0.0)
- All instruments have identical returns in a window — softmax is uniform → loss = -log(1/N), which is fine
- NaN/Inf targets — already handled by `_finite_mask`; counts toward `_n_valid` check
- `tau` approaching 0 — numerical instability; clamp `tau >= 0.01` if ever configuring below 0.1

## Testing Plan

1. Unit test `_listnet_loss`: perfect ranking → loss ≈ 0, inverted ranking → positive loss
2. Unit test minimum-2 guard: single instrument window → ret_loss stays 0.0
3. Integration test: TrainerConfig with `use_listnet_return_loss=True` runs 1 epoch without error
4. Check that `use_listnet_return_loss=False` (default) produces identical results to current code

## Related

- [[phase41b_gnn_signal_extraction]] — research doc
- [[phase41b_gnn_signal_extraction]] — task file (tasks/active/)
- [[quant_training_ground]] — roadmap owner

"""
TirraMind — Elastic Weight Consolidation (EWC) for Continuous GNN Learning
(Phase 46)

Provides:
    EWCState        — Dataclass holding Fisher diagonal, anchor weights,
                      regularisation strength, and bookkeeping timestamps.
    compute_fisher  — Estimate the Fisher Information diagonal from a
                      callable loss function by accumulating squared gradients.
    ewc_penalty     — Compute the EWC regularisation term:
                          lambda * sum_i F_i * (theta_i - theta_i*)^2

Design principle:
    This module is deliberately decoupled from GraphBuilder, PipelineStore,
    and the training loop. It operates only on:
        - torch.nn.Module (the model)
        - Callable[[], Tensor] (a loss closure owned by the Trainer)
        - EWCState (the anchor + Fisher diagonal)

    The Trainer is responsible for constructing the loss closure; this module
    is responsible for the mathematics.

References:
    Kirkpatrick et al. 2017 — "Overcoming Catastrophic Forgetting in Neural
        Networks" arXiv:1612.00796 — canonical EWC paper.
    Schwarz et al. 2018 — "Progress & Compress" arXiv:1805.06370
        (Online EWC — potential Phase 49 upgrade).
    Spec step: 46.1
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# EWCState
# ═══════════════════════════════════════════════════════════════


@dataclass
class EWCState:
    """Persistent EWC state produced after a full GNN retrain.

    Attributes:
        fisher: Fisher Information diagonal approximation.
                Maps parameter name → tensor of same shape as the parameter.
                Each value F_i ≈ E[(d log p / d theta_i)^2].
        anchor: Model parameter values at the time of the last full retrain.
                Maps parameter name → cloned tensor (detached, CPU).
        lambda_: EWC regularisation strength. Higher = more conservative
                 (less forgetting, less plasticity). Default 1000.0 is a
                 reasonable starting point per Kirkpatrick et al. 2017.
        last_update_ts: Unix timestamp of the most recent online update
                        (or full retrain if no online updates have run yet).
        obs_count_at_update: Observation count in the store when the last
                             online update (or full retrain) ran. Used to
                             detect whether enough new observations have
                             accumulated to justify the next update.
    """

    fisher: dict[str, Tensor]
    anchor: dict[str, Tensor]
    lambda_: float = 1000.0
    last_update_ts: float = field(default_factory=time.time)
    obs_count_at_update: int = 0


# ═══════════════════════════════════════════════════════════════
# compute_fisher
# ═══════════════════════════════════════════════════════════════


def compute_fisher(
    model: nn.Module,
    loss_fn: Callable[[], Tensor],
    n_samples: int = 1,
) -> dict[str, Tensor]:
    """Approximate the Fisher Information diagonal by accumulating squared gradients.

    The Fisher diagonal F_i is estimated as:
        F_i ≈ (1/N) * sum_{n=1}^{N} (d L_n / d theta_i)^2

    where each L_n is computed by calling ``loss_fn()`` once and performing
    a backward pass.  All parameters that do not receive a gradient (e.g.
    frozen layers or parameters not in the computational graph for a given
    batch) receive F_i = 0.

    Args:
        model:    The neural network whose Fisher diagonal we estimate.
        loss_fn:  A zero-argument callable that builds a mini-batch internally
                  and returns a scalar loss Tensor with ``requires_grad=True``.
                  The Trainer owns data construction; this module owns the math.
        n_samples: Number of forward/backward passes to average.  1 is
                   sufficient for the standard EWC approximation and keeps
                   overhead minimal.

    Returns:
        Dict mapping parameter name → Fisher diagonal tensor (same shape as
        the parameter, values ≥ 0, detached and on CPU).

    Raises:
        ValueError: If ``n_samples < 1``.
        RuntimeError: If ``loss_fn`` returns a non-scalar or non-differentiable
                      tensor.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be ≥ 1, got {n_samples}")

    # Accumulator: sum of squared gradients per named parameter
    fisher_accum: dict[str, Tensor] = {
        name: torch.zeros_like(param, dtype=torch.float32)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    model.train()

    for sample_idx in range(n_samples):
        model.zero_grad()

        loss = loss_fn()

        if not isinstance(loss, Tensor):
            raise RuntimeError(f"loss_fn() must return a Tensor, got {type(loss).__name__}")
        if loss.dim() != 0:
            raise RuntimeError(f"loss_fn() must return a scalar (0-dim) Tensor, got shape {tuple(loss.shape)}")
        if not loss.requires_grad:
            log.warning(
                "compute_fisher: loss_fn() returned a tensor with requires_grad=False "
                "on sample %d — gradients will be zero for this sample.",
                sample_idx,
            )
            continue

        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and name in fisher_accum:
                if param.grad is not None:
                    fisher_accum[name] += param.grad.detach().float() ** 2
                # If param.grad is None, F_i stays 0 for this sample

    model.zero_grad()

    # Average and move to CPU
    n = float(n_samples)
    fisher: dict[str, Tensor] = {name: (accum / n).cpu() for name, accum in fisher_accum.items()}

    n_params = sum(t.numel() for t in fisher.values())
    nonzero = sum((t > 0).sum().item() for t in fisher.values())
    log.info(
        "compute_fisher: %d params, %d non-zero Fisher entries (%.1f%%)",
        n_params,
        nonzero,
        100.0 * nonzero / max(n_params, 1),
    )

    return fisher


# ═══════════════════════════════════════════════════════════════
# ewc_penalty
# ═══════════════════════════════════════════════════════════════


def ewc_penalty(model: nn.Module, state: EWCState) -> Tensor:
    """Compute the EWC regularisation penalty.

    Implements:
        L_ewc = lambda * sum_i F_i * (theta_i - theta_i*)^2

    where theta_i* is the anchor (weights at last full retrain) and F_i is
    the Fisher diagonal for parameter i.

    Shape mismatches between the current model parameters and the stored
    Fisher/anchor are handled gracefully: mismatched parameters are skipped
    and a warning is logged. This occurs when the model architecture changes
    (e.g. new entity types added) — in that case the EWC penalty is partial
    or zero until the next full retrain recomputes the Fisher.

    Args:
        model: The current (possibly partially updated) HetTGN.
        state: EWCState produced by the last full retrain.

    Returns:
        Scalar Tensor: the EWC penalty term (≥ 0). Returns a zero tensor
        (with grad_fn if model params require_grad) if no parameter shapes
        match.
    """
    penalty = torch.tensor(0.0)
    n_matched = 0
    n_skipped = 0

    for name, param in model.named_parameters():
        if name not in state.fisher or name not in state.anchor:
            continue

        fisher_i = state.fisher[name].to(param.device)
        anchor_i = state.anchor[name].to(param.device)

        if fisher_i.shape != param.shape or anchor_i.shape != param.shape:
            log.warning(
                "ewc_penalty: shape mismatch for '%s' — model=%s, fisher=%s, anchor=%s. Skipping.",
                name,
                tuple(param.shape),
                tuple(fisher_i.shape),
                tuple(anchor_i.shape),
            )
            n_skipped += 1
            continue

        diff = param - anchor_i
        penalty = penalty + (fisher_i * diff.pow(2)).sum()
        n_matched += 1

    if n_skipped > 0:
        log.warning(
            "ewc_penalty: skipped %d / %d parameters due to shape mismatch. "
            "EWC will be recomputed after next full retrain.",
            n_skipped,
            n_matched + n_skipped,
        )

    return state.lambda_ * penalty

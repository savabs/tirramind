"""
TirraMind — Path Signature Encoder (Idea 2)

Computes depth-3 truncated path signatures for entity event streams using
Chen's recursion on piecewise-linear interpolation.  No external dependencies
— pure PyTorch, fully differentiable, GPU-compatible.

Theory
------
The path signature S(X)^≤3 for a d-channel path X: [0,T] → R^d is defined
via iterated Riemann-Stieltjes integrals:

    Level 1 (d dims):     S^1_i     = ∫ dX_i
    Level 2 (d² dims):    S^2_{ij}  = ∫∫ dX_i ⊗ dX_j
    Level 3 (d³ dims):    S^3_{ijk} = ∫∫∫ dX_i ⊗ dX_j ⊗ dX_k

Total dims: d + d² + d³

For a piecewise-linear path, Chen's identity gives an exact recursion per
segment.  With increment a = X_k - X_{k-1}:

    S1_new[i]     = S1[i] + a[i]
    S2_new[i,j]   = S2[i,j] + S1[i]·a[j] + a[i]·a[j]/2
    S3_new[i,j,k] = S3[i,j,k] + S2[i,j]·a[k]
                               + S1[i]·a[j]·a[k]/2
                               + a[i]·a[j]·a[k]/6

Why this replaces hand-crafted per-source extractors
-----------------------------------------------------
The signature is provably universal for continuous paths (Lyons & McLeod 2022,
Theorem 3.1): any continuous function of the path can be approximated to
arbitrary precision by a linear function of its signature.  This means one
pipeline handles AIS trajectories, price time series, filing cadences, and
vessel routing patterns with identical treatment — no bespoke feature code.

References
----------
    Chen, K.-T. (1954) "Iterated path integrals." Bull. AMS 83(5):831–879.
    Lyons, T. (1998) "Differential equations driven by rough signals."
        Rev. Mat. Iberoam. 14(2):215–310.
    Kidger & Lyons (2021) "Signatory: Differentiable Computations of the
        Signature and Log-signature." arXiv:2001.00706.
    Lyons & McLeod (2022) "Signature Methods in Machine Learning."
        arXiv:2206.14674.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Path channels: (normalized_time, normalized_value, type_encoding)
PATH_CHANNELS: int = 3
SIGNATURE_DEPTH: int = 3
# d + d^2 + d^3  =  3 + 9 + 27 = 39
SIGNATURE_DIM: int = sum(PATH_CHANNELS**d for d in range(1, SIGNATURE_DEPTH + 1))
# Projected output dimension (PathSignatureEncoder default)
SIGNATURE_OUT_DIM: int = 16


# ═══════════════════════════════════════════════════════════════════════════════
# Core: Chen's recursion
# ═══════════════════════════════════════════════════════════════════════════════


def compute_path_signature(
    path: torch.Tensor,
    depth: int = SIGNATURE_DEPTH,
) -> torch.Tensor:
    """Compute the truncated signature of a piecewise-linear path.

    Uses Chen's identity for exact computation on each linear segment.

    Args:
        path:  (seq_len, channels) unbatched  OR
               (batch, seq_len, channels) batched.
               Must have seq_len ≥ 1.
        depth: Truncation depth in {1, 2, 3}. Default 3.

    Returns:
        Signature tensor of shape (sig_dim,) or (batch, sig_dim) where
        sig_dim = channels + channels² + channels³  (for depth=3).
        Returns zeros for paths with seq_len < 2 (no increments).

    Raises:
        ValueError: if depth not in {1, 2, 3}.
    """
    if depth not in (1, 2, 3):
        raise ValueError(f"depth must be 1, 2, or 3; got {depth}")

    squeeze = path.dim() == 2
    if squeeze:
        path = path.unsqueeze(0)  # → (1, seq_len, d)

    B, T, d = path.shape
    dev = path.device
    dtype = path.dtype

    sig_dim = sum(d**k for k in range(1, depth + 1))

    if T < 2:
        out = torch.zeros(B, sig_dim, device=dev, dtype=dtype)
        return out.squeeze(0) if squeeze else out

    # Increments: (B, T-1, d)
    inc = path[:, 1:, :] - path[:, :-1, :]

    S1 = torch.zeros(B, d, device=dev, dtype=dtype)
    if depth >= 2:
        S2 = torch.zeros(B, d, d, device=dev, dtype=dtype)
    if depth >= 3:
        S3 = torch.zeros(B, d, d, d, device=dev, dtype=dtype)

    for t in range(T - 1):
        a = inc[:, t, :]  # (B, d)

        if depth >= 3:
            # S3_new[i,j,k] = S3[i,j,k] + S2[i,j]*a[k]
            #                + S1[i]*a[j]*a[k]/2
            #                + a[i]*a[j]*a[k]/6
            a_outer2 = torch.einsum("bi,bj->bij", a, a)  # (B,d,d)
            S3 = (
                S3
                + torch.einsum("bij,bk->bijk", S2, a)
                + torch.einsum("bi,bjk->bijk", S1, a_outer2) * 0.5
                + torch.einsum("bij,bk->bijk", a_outer2, a) * (1.0 / 6.0)
            )

        if depth >= 2:
            # S2_new[i,j] = S2[i,j] + S1[i]*a[j] + a[i]*a[j]/2
            S2 = (
                S2
                + torch.einsum("bi,bj->bij", S1, a)
                + torch.einsum("bi,bj->bij", a, a) * 0.5
            )

        S1 = S1 + a

    parts: list[torch.Tensor] = [S1.reshape(B, -1)]
    if depth >= 2:
        parts.append(S2.reshape(B, -1))
    if depth >= 3:
        parts.append(S3.reshape(B, -1))

    sig = torch.cat(parts, dim=-1)  # (B, sig_dim)
    return sig.squeeze(0) if squeeze else sig


# ═══════════════════════════════════════════════════════════════════════════════
# Path construction from observation dicts
# ═══════════════════════════════════════════════════════════════════════════════

_VALUE_KEYS = (
    "usd_amount",
    "btc_amount",
    "value",
    "estimated_value",
    "goldstein_scale",
    "num_articles",
)


def entity_observations_to_path(
    observations: list[dict[str, Any]],
    max_seq_len: int = 64,
) -> torch.Tensor:
    """Build a (seq_len, PATH_CHANNELS) path tensor from observation dicts.

    Path channels:
        0  normalized time       t / T ∈ [0, 1]
        1  normalized value      tanh(v / (|v|+1)) ∈ (-1, 1)
        2  observation type      type_idx / n_types ∈ [0, 1)

    Observations are sorted by timestamp, then truncated to the most
    recent ``max_seq_len`` points.

    Args:
        observations: List of observation dicts (entity_id, observed_at,
                      value, observation_type / obs_type).
        max_seq_len:  Maximum path length.  Longer streams are truncated.

    Returns:
        (seq_len, PATH_CHANNELS) float32 tensor.  seq_len ∈ [1, max_seq_len].
    """
    if not observations:
        return torch.zeros(1, PATH_CHANNELS)

    # Lazy import avoids circular dependency at module load time
    from agent.models.gnn.graph_builder import _OBS_TYPE_TO_IDX  # noqa: PLC0415

    n_types = max(len(_OBS_TYPE_TO_IDX), 1)

    sorted_obs = sorted(observations, key=lambda o: float(o.get("observed_at", 0.0)))
    sorted_obs = sorted_obs[-max_seq_len:]

    n = len(sorted_obs)
    path = torch.zeros(n, PATH_CHANNELS)

    t0 = float(sorted_obs[0].get("observed_at", 0.0))
    t_end = float(sorted_obs[-1].get("observed_at", 0.0))
    t_span = max(t_end - t0, 1.0)

    for i, obs in enumerate(sorted_obs):
        t = float(obs.get("observed_at", 0.0))
        path[i, 0] = (t - t0) / t_span

        # Observation value — try standard keys in priority order
        v = obs.get("value", {})
        val = 0.0
        if isinstance(v, dict):
            for k in _VALUE_KEYS:
                if k in v:
                    try:
                        raw = float(v[k])
                        if math.isfinite(raw):
                            # tanh-normalise: maps any finite value to (-1,1)
                            val = math.tanh(raw / (abs(raw) + 1.0))
                    except (TypeError, ValueError):
                        pass
                    break
        path[i, 1] = val

        # Observation type — normalised integer index
        ot = obs.get("observation_type", "") or obs.get("obs_type", "")
        path[i, 2] = float(_OBS_TYPE_TO_IDX.get(ot, 0)) / n_types

    return path


def compute_entity_signature(
    observations: list[dict[str, Any]],
    max_seq_len: int = 64,
    depth: int = SIGNATURE_DEPTH,
) -> torch.Tensor:
    """One-call helper: observations → (SIGNATURE_DIM,) signature vector.

    Suitable for use inside ``_build_node_features()`` without importing the
    full encoder class.

    Returns:
        (sig_dim,) float32 tensor where sig_dim = PATH_CHANNELS^1 + ... +
        PATH_CHANNELS^depth.  Zero-vector when observations is empty.
    """
    path = entity_observations_to_path(observations, max_seq_len=max_seq_len)
    return compute_path_signature(path, depth=depth)


# ═══════════════════════════════════════════════════════════════════════════════
# PathSignatureEncoder — learnable module (for future use inside HetTGN)
# ═══════════════════════════════════════════════════════════════════════════════


class PathSignatureEncoder(nn.Module):
    """Learnable path signature encoder: event stream → fixed-dim feature vector.

    Wraps ``compute_path_signature`` with a learned two-layer projection.
    Use this module when signatures should be fine-tuned end-to-end inside
    the GNN.  For static (pre-computed) features, use
    ``compute_entity_signature`` directly.

    Parameters
    ----------
    output_dim : int
        Output feature dimension per entity.  Default ``SIGNATURE_OUT_DIM`` (16).
    max_seq_len : int
        Path truncation length.
    depth : int
        Signature truncation depth in {1, 2, 3}.

    Reference:
        Lyons & McLeod (2022) §4 — signature + linear head as universal
        approximator for path-dependent functionals.
    """

    def __init__(
        self,
        output_dim: int = SIGNATURE_OUT_DIM,
        max_seq_len: int = 64,
        depth: int = SIGNATURE_DEPTH,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.depth = depth
        raw_dim = sum(PATH_CHANNELS**d for d in range(1, depth + 1))
        self.projection = nn.Sequential(
            nn.Linear(raw_dim, raw_dim * 2),
            nn.ReLU(),
            nn.Linear(raw_dim * 2, output_dim),
        )

    def forward(self, path: torch.Tensor) -> torch.Tensor:
        """Project the raw signature to ``output_dim``.

        Args:
            path: (seq_len, PATH_CHANNELS) or (batch, seq_len, PATH_CHANNELS).

        Returns:
            (output_dim,) or (batch, output_dim).
        """
        raw = compute_path_signature(path, depth=self.depth)
        return self.projection(raw)

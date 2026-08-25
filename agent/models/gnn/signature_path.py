"""
TirraMind — Path Signature Builder (M1 Component)

Computes prefix log-signatures of event message paths as features for the
control path Z(t) in the ContinuousWorldModel.

For each event k in a sequence of n events, this module returns the
log-signature of the path from event 0 to event k.  These "prefix
signatures" capture the cumulative shape, area, and curvature of the
path up to each knot point.

Why signatures?
    Two entity histories with the same arrival/departure timestamps but
    different intermediate patterns have DIFFERENT signatures.  A vessel
    that oscillates (suspicious loitering) vs. one on a monotone transit
    produces a different Lévy area even if the endpoints are identical.
    Standard RNN encodings cannot distinguish these; signatures can.

Libraries
---------
    Primary: iisignature (C extension, ~100x faster than pure Python).
             pip install iisignature
    Fallback: manual depth-2 implementation using Chen's identity.
              Depth-2 log-signature for d=4 projection: 10 features.

References
----------
    Lyons, T. (1998). "Differential Equations Driven by Rough Signals."
        Rev. Mat. Iberoamericana 14(2).  Foundational rough paths theory.
    Chevyrev, I., Kormilitzin, A. (2016). "A Primer on the Signature
        Method in Machine Learning."  arXiv:1603.03788.
    Morrill et al. (2021). "Neural Rough Differential Equations for Long
        Time Series."  ICML 2021.  Log-signatures inside Neural CDEs.
    iisignature: https://github.com/bottler/iisignature
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

try:
    import iisignature as _iisig

    _IISIG_AVAILABLE = True
    log.debug("iisignature available — using depth-3 log-signatures.")
except ImportError:
    _iisig = None  # type: ignore[assignment]
    _IISIG_AVAILABLE = False
    log.warning(
        "iisignature not installed — SignaturePathBuilder falling back to "
        "manual depth-2 log-signatures (10 features for d=4 projection).  "
        "Install with: pip install iisignature"
    )

_PROJ_DIM: int = 4
_SIG_DEPTH: int = 3
_FALLBACK_DEPTH2_DIM: int = _PROJ_DIM + (_PROJ_DIM * (_PROJ_DIM - 1)) // 2  # 4+6=10


def _logsig_dim_iisig(proj_dim: int, depth: int) -> int:
    """Return the log-signature output dimension via iisignature."""
    s = _iisig.prepare(proj_dim, depth)  # type: ignore[union-attr]
    return int(_iisig.logsiglength(proj_dim, depth))  # type: ignore[union-attr]


def _depth2_logsig_incremental(
    increments: torch.Tensor,
) -> torch.Tensor:
    """Manual depth-2 log-signature computed incrementally via Chen's identity.

    For a d-dimensional path of increments [dx_0, ..., dx_{n-1}]:

        Level-1 prefix at step k:  S1_k = sum_{i=0}^{k} dx_i          (d)
        Level-2 Lévy area at step k:
            A^{ij}_k = sum_{0 <= p < q <= k} (dx^i_p * dx^j_q
                                              - dx^j_p * dx^i_q) / 2   (d(d-1)/2)

    The Lévy area is the antisymmetric part of the depth-2 signature.
    It encodes the "looping" or "rotating" character of the path.

    Chen's identity (incremental update):
        S1_{k+1} = S1_k + dx_{k+1}
        A^{ij}_{k+1} = A^{ij}_k + (S1_k^i * dx_{k+1}^j
                                    - S1_k^j * dx_{k+1}^i) / 2

    Args:
        increments: (n, d) tensor of path increments.

    Returns:
        (n, d + d*(d-1)/2) prefix log-signatures, one per step.
    """
    n, d = increments.shape
    n_pairs = d * (d - 1) // 2
    device = increments.device
    dtype = increments.dtype

    S1 = torch.zeros(d, device=device, dtype=dtype)
    A = torch.zeros(n_pairs, device=device, dtype=dtype)

    results: list[torch.Tensor] = []

    # Pre-compute pair indices (i < j)
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]

    for k in range(n):
        dx = increments[k]  # (d,)
        # Update Lévy area before updating S1
        for p_idx, (i, j) in enumerate(pairs):
            A[p_idx] = A[p_idx] + (S1[i] * dx[j] - S1[j] * dx[i]) * 0.5
        S1 = S1 + dx
        results.append(torch.cat([S1.clone(), A.clone()]))

    return torch.stack(results, dim=0)  # (n, d + n_pairs)


# ═══════════════════════════════════════════════════════════════════════════
# SignaturePathBuilder
# ═══════════════════════════════════════════════════════════════════════════


class SignaturePathBuilder(nn.Module):
    """Compute prefix log-signatures of entity event message paths.

    Each event's message is projected to R^{proj_dim} via a learned linear
    layer, then the log-signature of the growing path is computed at each
    event.  The output is a (n_events, sig_dim) tensor of prefix signatures.

    This tensor is concatenated with the Hawkes hidden states to form the
    control path knots for ContinuousWorldModel.

    Parameters
    ----------
    message_dim : int
        Dimension of incoming event messages (= HeteroMemory.memory_dim).
    proj_dim : int
        Projection dimension before signature.  Default 4.
        Log-sig output: iisig.logsiglength(proj_dim, depth) or fallback.
    depth : int
        Log-signature depth.  Default 3 (requires iisignature).
        Falls back to depth-2 manual if iisignature unavailable.
    """

    def __init__(
        self,
        message_dim: int,
        proj_dim: int = _PROJ_DIM,
        depth: int = _SIG_DEPTH,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.depth = depth

        self.msg_proj = nn.Linear(message_dim, proj_dim, bias=False)

        if _IISIG_AVAILABLE:
            self._sig_s = _iisig.prepare(proj_dim, depth)  # type: ignore[union-attr]
            self.sig_dim: int = int(
                _iisig.logsiglength(proj_dim, depth)  # type: ignore[union-attr]
            )
            self._use_iisig = True
        else:
            self.sig_dim = _FALLBACK_DEPTH2_DIM
            self._use_iisig = False

        log.debug(
            "SignaturePathBuilder: proj_dim=%d depth=%d sig_dim=%d use_iisig=%s",
            proj_dim,
            depth,
            self.sig_dim,
            self._use_iisig,
        )

    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        """Compute prefix log-signatures for a sequence of event messages.

        Args:
            messages: (n, message_dim) — event messages in temporal order.

        Returns:
            (n, sig_dim) prefix log-signatures, one row per event.
            Returns zeros if n == 0.
        """
        n = messages.shape[0]
        device = messages.device

        if n == 0:
            return torch.zeros(0, self.sig_dim, device=device)

        # Project to lower-dimensional path
        proj = self.msg_proj(messages)  # (n, proj_dim)

        if self._use_iisig:
            return self._compute_iisig(proj)
        else:
            return self._compute_fallback(proj)

    def _compute_iisig(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute depth-3 prefix log-signatures via iisignature.

        For each prefix [proj[0], ..., proj[k]], compute the log-signature
        of the piecewise-linear path through those points.
        """
        n = proj.shape[0]
        results: list[torch.Tensor] = []

        path_np = proj.detach().cpu().numpy()

        for k in range(n):
            # Path from step 0 to step k (k+1 points)
            prefix = path_np[: k + 1]
            if prefix.shape[0] < 2:
                # Single point — log-sig is zero
                ls = torch.zeros(self.sig_dim, device=proj.device)
            else:
                import numpy as np

                ls_np = _iisig.logsig(prefix, self._sig_s)  # type: ignore[union-attr]
                ls = torch.tensor(ls_np, dtype=proj.dtype, device=proj.device)
            results.append(ls)

        return torch.stack(results, dim=0)  # (n, sig_dim)

    def _compute_fallback(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute depth-2 prefix log-signatures via Chen's identity (pure PyTorch)."""
        if proj.shape[0] == 1:
            # Single event: level-1 = proj[0], Lévy area = 0
            d = proj.shape[1]
            n_pairs = d * (d - 1) // 2
            return torch.cat(
                [proj[0], torch.zeros(n_pairs, device=proj.device)]
            ).unsqueeze(0)

        # Compute increments: dx[k] = proj[k] - proj[k-1], dx[0] = proj[0]
        increments = torch.zeros_like(proj)
        increments[0] = proj[0]
        increments[1:] = proj[1:] - proj[:-1]

        return _depth2_logsig_incremental(increments)


def build_control_knots(
    messages: torch.Tensor,
    time_feats: torch.Tensor,
    sig_builder: SignaturePathBuilder | None,
) -> torch.Tensor:
    """Assemble control path knots from time features, messages, signatures.

    Args:
        messages:   (n, msg_dim) — projected messages (already in ctrl_msg_dim).
        time_feats: (n, time_dim) — Time2Vec-encoded deltas.
        sig_builder: Optional signature builder; adds sig_dim features if provided.

    Returns:
        (n, d_z) control path knots where
        d_z = time_dim + msg_dim [+ sig_dim].
    """
    parts = [time_feats, messages]

    if sig_builder is not None:
        sigs = sig_builder(messages)  # (n, sig_dim)
        parts.append(sigs)

    return torch.cat(parts, dim=-1)  # (n, d_z)


def compute_d_z(
    ctrl_time_dim: int,
    ctrl_msg_dim: int,
    sig_builder: SignaturePathBuilder | None = None,
) -> int:
    """Compute control path dimension d_z from config."""
    d_z = ctrl_time_dim + ctrl_msg_dim
    if sig_builder is not None:
        d_z += sig_builder.sig_dim
    return d_z

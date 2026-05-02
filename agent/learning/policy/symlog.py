"""TirraMind — Symlog / Symexp Transforms

DreamerV3 (Hafner et al. 2023, arXiv:2301.04104, §3.2) normalization
for reward and value targets.  Symlog compresses magnitude while
preserving sign, which is critical in finance where regime shifts
can change reward scale by orders of magnitude.

Mathematical properties proven in tests:
    1. symexp(symlog(x)) = x              (exact inverse)
    2. symlog(0) = 0                       (origin-preserving)
    3. sign(symlog(x)) = sign(x)           (sign-preserving)
    4. |symlog(x)| ≤ |x|  for |x| ≥ 0     (compressive)
    5. symlog is monotonically increasing   (order-preserving)
    6. Gradients exist everywhere           (C¹ smooth)

Definition:
    symlog(x) = sign(x) · ln(|x| + 1)
    symexp(x) = sign(x) · (exp(|x|) − 1)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# ── Torch versions (differentiable) ───────────────────────────


def symlog(x: Tensor) -> Tensor:
    """Symmetric logarithm: sign(x) · ln(|x| + 1)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    """Inverse of symlog: sign(x) · (exp(|x|) − 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# ── Numpy versions (for scoring / non-differentiable paths) ──


def symlog_np(x: np.ndarray) -> np.ndarray:
    """Numpy symlog: sign(x) · ln(|x| + 1)."""
    return np.sign(x) * np.log1p(np.abs(x))


def symexp_np(x: np.ndarray) -> np.ndarray:
    """Numpy symexp: sign(x) · (exp(|x|) − 1)."""
    return np.sign(x) * (np.expm1(np.abs(x)))

"""
TirraMind — Detector Threshold Optimizer (Learning Layer)

GP-Bayesian optimization of detector hyper-parameters —
CUSUM (k, h), Hawkes (μ, α, β), convergence (z, p, fdr_q).

Each detector type has a named ParamSpace.  The optimizer runs monthly:
  1. Suggest N parameter settings from the GP posterior.
  2. Evaluate each against retroactive ground truth (alert F1, precision, etc.).
  3. Record trials and pick the best.

Spec reference: [[learned_vs_handcoded_architecture_spec]], Change 7.

Math: Same GP + EI framework as BayesianParamOptimizer (see param_optimizer.py).
The only twist is that each detector type has its own independent optimizer
instance so their trial histories don't contaminate each other.

Design decision — separate optimizers per detector:
    CUSUM and Hawkes have fundamentally different objective surfaces (CUSUM
    is a cumulative sum with step-change detection semantics; Hawkes is a
    self-exciting point process).  Sharing a single GP across detector types
    would conflate these surfaces.  N independent optimizers with N small
    GPs is cleaner and more stable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from agent.learning.param_optimizer import BayesianParamOptimizer, ParamSpace

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detector parameter spaces
# ---------------------------------------------------------------------------

DETECTOR_SPACES: dict[str, ParamSpace] = {
    "cusum": ParamSpace(
        names=["k", "h"],
        bounds=[
            (0.1, 2.0),  # k: allowance (shift threshold in std units)
            (2.0, 10.0),  # h: decision interval (higher = fewer alerts)
        ],
    ),
    "hawkes": ParamSpace(
        names=["mu", "alpha", "beta"],
        bounds=[
            (0.01, 0.5),  # μ: baseline intensity
            (0.1, 0.95),  # α: excitation (must be < β for sub-critical)
            (0.5, 3.0),  # β: decay rate
        ],
    ),
    "convergence": ParamSpace(
        names=["z_threshold", "p_threshold", "fdr_q"],
        bounds=[
            (1.0, 4.0),  # z: anomaly z-score
            (0.001, 0.20),  # p: pairwise significance level
            (0.01, 0.20),  # fdr_q: Benjamini-Hochberg target
        ],
    ),
}


# ---------------------------------------------------------------------------
# Threshold Optimizer
# ---------------------------------------------------------------------------


class ThresholdOptimizer:
    """Multi-detector Bayesian optimizer.

    Wraps one BayesianParamOptimizer per detector type, each with its own
    parameter space and trial history.

    Usage::

        opt = ThresholdOptimizer(persist_dir=Path("detectors/"))
        params = opt.suggest("cusum")          # → {"k": 0.6, "h": 4.2}
        opt.record("cusum", params, f1=0.72)   # record trial
        best = opt.current_best("cusum")       # → {"k": ..., "h": ...}

    """

    def __init__(
        self,
        persist_dir: Path | None = None,
        *,
        n_random: int = 5,
        seed: int | None = None,
    ) -> None:
        self._persist_dir = persist_dir
        self._optimizers: dict[str, BayesianParamOptimizer] = {}

        for name, space in DETECTOR_SPACES.items():
            pp = persist_dir / f"{name}_bo.json" if persist_dir else None
            self._optimizers[name] = BayesianParamOptimizer(
                space,
                persist_path=pp,
                n_random=n_random,
                seed=seed,
            )

    @property
    def detector_names(self) -> list[str]:
        return list(self._optimizers.keys())

    def n_trials(self, detector_name: str) -> int:
        return self._get(detector_name).n_trials

    def suggest(self, detector_name: str) -> dict[str, float]:
        """Suggest the next parameter vector for the given detector."""
        params = self._get(detector_name).suggest()
        # Enforce Hawkes sub-criticality: α/β < 1
        if detector_name == "hawkes" and params["alpha"] >= params["beta"]:
            params["alpha"] = params["beta"] * 0.95
        return params

    def record(
        self,
        detector_name: str,
        params: dict[str, float],
        objective: float,
        metadata: dict | None = None,
    ) -> None:
        """Record a completed evaluation for the given detector.

        Args:
            detector_name: One of "cusum", "hawkes", "convergence".
            params: The parameter dict (as returned by suggest()).
            objective: Scalar metric — higher is better (e.g., F1 score).
            metadata: Optional extra info (date range, n_events, etc.).
        """
        self._get(detector_name).record(params, objective, metadata)

    def current_best(self, detector_name: str) -> dict[str, float] | None:
        """Return the best-performing params for the given detector."""
        return self._get(detector_name).best_params()

    def _get(self, name: str) -> BayesianParamOptimizer:
        if name not in self._optimizers:
            raise ValueError(
                f"Unknown detector '{name}'. "
                f"Available: {list(self._optimizers.keys())}"
            )
        return self._optimizers[name]

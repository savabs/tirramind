"""TirraMind — Crowding Risk Estimator

Quantifies the concentration / crowding danger implied by convergence
clusters and current portfolio positions.

Mathematical formulation
------------------------
For convergence cluster C with member entities {e_1, …, e_k}:

    crowd(C) = |C| / mean_cluster_size  ×  ρ_intra(C)

where ρ_intra is ``ConvergenceCluster.correlated_surprise_score`` (mean
pairwise cosine similarity of surprise vectors).

Per-entity unwind risk:

    unwind(e, C) = crowd(C) × w_e / (liq_e + ε)

where w_e is the absolute position weight and liq_e is the rolling mean
volume for entity e.

Design:
    The crowding score is a *diagnostic* for the RL policy — it does not
    directly block trades.  The reward function applies a penalty
    proportional to severity.

Conceptual source: Khandani & Lo (2011) "What Happened to the Quants
in August 2007?" (cross-strategy crowding and simultaneous unwinding).

References:
    - Spec step 22a.4
    - Research: docs/research/adversarial.md §Crowding Risk
"""

from __future__ import annotations

import numpy as np

from agent.adversarial.config import CrowdingConfig
from agent.adversarial.flags import AdversarialFlag
from agent.fusion.convergence import ConvergenceCluster


class CrowdingEstimator:
    """Estimate crowding risk from convergence clusters and portfolio state."""

    def __init__(self, config: CrowdingConfig | None = None) -> None:
        self._cfg = config or CrowdingConfig()

    def assess(
        self,
        clusters: list[ConvergenceCluster],
        position_weights: dict[str, float],
        volume_history: dict[str, np.ndarray],
        *,
        timestamp: float | None = None,
    ) -> list[AdversarialFlag]:
        """Produce crowding-risk flags for all qualifying clusters.

        Parameters
        ----------
        clusters : current convergence clusters from ConvergenceDetector.
        position_weights : {entity_id → signed weight}.  Entities not
            found here are treated as zero-weight.
        volume_history : {entity_id → 1-D volume array}.  Only the last
            ``CrowdingConfig.volume_lookback`` values are used for the
            liquidity proxy.  Entities not found here use the global mean.
        timestamp : optional flag timestamp.

        Returns
        -------
        List of ``AdversarialFlag`` (may be empty).
        """
        if not clusters:
            return []

        # Mean cluster size for normalisation
        sizes = [len(c.member_alerts) for c in clusters]
        mean_size = float(np.mean(sizes)) if sizes else 1.0

        flags: list[AdversarialFlag] = []
        for cluster in clusters:
            n_members = len(cluster.member_alerts)
            if n_members < self._cfg.cluster_size_threshold:
                continue

            crowd_score = self.cluster_crowding_score(cluster, mean_size)

            # Per-entity unwind risk
            for alert in cluster.member_alerts:
                eid = alert.entity_id
                w = abs(position_weights.get(eid, 0.0))
                if w == 0.0:
                    continue

                liq = self._liquidity_proxy(eid, volume_history)
                eps = 1e-10
                unwind = crowd_score * w / (liq + eps)

                severity = float(min(unwind, 1.0))
                if severity < 0.01:
                    continue

                flags.append(
                    AdversarialFlag(
                        flag_type="crowding_risk",
                        severity=severity,
                        confidence=min(crowd_score, 1.0),
                        entity_id=eid,
                        signal_name=None,
                        evidence={
                            "crowd_score": crowd_score,
                            "unwind_risk": float(unwind),
                            "position_weight": w,
                            "liquidity_proxy": liq,
                            "cluster_id": cluster.cluster_id,
                            "cluster_size": n_members,
                        },
                        timestamp=timestamp if timestamp is not None else 0.0,
                    )
                )
        return flags

    def cluster_crowding_score(
        self,
        cluster: ConvergenceCluster,
        mean_cluster_size: float = 1.0,
    ) -> float:
        """Compute crowding score for a single cluster.

        crowd(C) = (|C| / mean_size) × ρ_intra(C)
        """
        n = len(cluster.member_alerts)
        rho = cluster.correlated_surprise_score
        denom = max(mean_cluster_size, 1.0)
        return (n / denom) * rho

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _liquidity_proxy(
        self,
        entity_id: str,
        volume_history: dict[str, np.ndarray],
    ) -> float:
        """Rolling mean volume as liquidity proxy."""
        hist = volume_history.get(entity_id)
        if hist is None or len(hist) == 0:
            return 1.0  # default: assume unit liquidity
        arr = np.asarray(hist, dtype=np.float64).ravel()
        lookback = self._cfg.volume_lookback
        window = arr[-lookback:] if len(arr) >= lookback else arr
        mean_vol = float(np.mean(window))
        return max(mean_vol, 0.0)

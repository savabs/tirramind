"""
TirraMind — Edge Confidence Tracker

Tracks BIC-δ edge contribution scores on rolling time windows and uses
hysteresis-based decision logic to suggest structural changes to the
causal DAG.

Math — Edge Contribution Score (BIC-δ):
    For each directed edge (parent → child), the marginal contribution is:
        Δ = local_score(child, parents_with_edge) - local_score(child, parents_without_edge)
    where local_score is pgmpy's decomposable BIC score (log-likelihood minus
    complexity penalty).  Positive Δ means the edge improves fit; negative
    means it hurts.

    We compute Δ on multiple rolling windows (e.g., 30d, 60d, 90d) and derive:
        confidence = sigmoid(mean(Δs))     — high when edge consistently helps
        stability  = 1 - std(Δs)/(|mean(Δs)| + ε)  — high when consistent

    Hysteresis prevents flip-flopping: edges are only added/removed after
    crossing the threshold for N consecutive evaluations.

Trusted source: Schwarz 1978 (BIC), Chickering 2002 ("Optimal Structure
Identification with Greedy Search") for decomposable scoring.  Hysteresis
is repo-specific engineering for noisy financial data.

Spec: docs/specs/tier7_self_modifying_structure_spec.md (steps 13.1–13.2)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EdgeConfidence:
    """Confidence and stability metrics for a single directed edge."""

    parent: str
    child: str
    confidence: float  # sigmoid(mean(BIC-δ)), in [0, 1]
    stability: float  # 1 - std/|mean|, in (-inf, 1], clipped to [0, 1]
    deltas: tuple[float, ...]  # raw BIC-δ per window
    n_windows: int


@dataclass
class EdgeSuggestion:
    """Structural changes suggested by the confidence tracker."""

    edges_to_add: list[tuple[str, str]] = field(default_factory=list)
    edges_to_remove: list[tuple[str, str]] = field(default_factory=list)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


class EdgeConfidenceTracker:
    """Track and evaluate edge contributions to causal DAG quality.

    Parameters
    ----------
    node_names : sequence of str
        Names of nodes that participate in structure learning (typically
        observed nodes only — regime/latent nodes are excluded).
    windows_days : tuple of int
        Rolling window sizes in days for BIC-δ evaluation.
    day_seconds : float
        Seconds per day (for timestamp→day conversion).
    """

    _DAY_SECONDS = 86400.0

    def __init__(
        self,
        node_names: list[str] | tuple[str, ...],
        windows_days: tuple[int, ...] = (30, 60, 90),
    ) -> None:
        self._node_names = tuple(sorted(node_names))
        self._windows_days = windows_days
        # Consecutive evaluation counter for hysteresis
        # Positive = consecutive above add_threshold
        # Negative = consecutive below remove_threshold
        self._consecutive: dict[tuple[str, str], int] = {}

    @property
    def node_names(self) -> tuple[str, ...]:
        return self._node_names

    @property
    def windows_days(self) -> tuple[int, ...]:
        return self._windows_days

    def compute_edge_contributions(
        self,
        edges: list[tuple[str, str]],
        feature_df: pd.DataFrame,
    ) -> dict[tuple[str, str], float]:
        """Compute BIC-δ for each edge on the given data.

        Parameters
        ----------
        edges : list of (parent, child) tuples
        feature_df : DataFrame with discretized node values as columns

        Returns
        -------
        dict mapping (parent, child) → BIC-δ (positive = edge helps)
        """
        from pgmpy.structure_score import BIC

        if feature_df.empty or len(feature_df) < 5:
            return {}

        scorer = BIC(feature_df)
        contributions: dict[tuple[str, str], float] = {}

        for parent, child in edges:
            if child not in feature_df.columns or parent not in feature_df.columns:
                continue

            # Get current parents from edges list
            current_parents = tuple(sorted(p for p, c in edges if c == child and p in feature_df.columns))
            parents_without = tuple(p for p in current_parents if p != parent)

            try:
                score_with = scorer.local_score(child, current_parents)
                score_without = scorer.local_score(child, parents_without)
                contributions[(parent, child)] = score_with - score_without
            except Exception as exc:
                log.debug("BIC-δ failed for %s → %s: %s", parent, child, exc)

        return contributions

    def evaluate(
        self,
        edges: list[tuple[str, str]],
        windowed_dataframes: list[pd.DataFrame],
    ) -> dict[tuple[str, str], EdgeConfidence]:
        """Evaluate edge confidence across multiple rolling windows.

        Parameters
        ----------
        edges : current edges to evaluate
        windowed_dataframes : list of DataFrames, one per window in
            self._windows_days order.  Each contains discretized features
            for that window period.

        Returns
        -------
        dict mapping (parent, child) → EdgeConfidence
        """
        if not windowed_dataframes:
            return {}

        # Collect deltas per edge across windows
        edge_deltas: dict[tuple[str, str], list[float]] = {}
        for df in windowed_dataframes:
            contribs = self.compute_edge_contributions(edges, df)
            for edge, delta in contribs.items():
                edge_deltas.setdefault(edge, []).append(delta)

        result: dict[tuple[str, str], EdgeConfidence] = {}
        for edge, deltas in edge_deltas.items():
            if not deltas:
                continue
            mean_d = sum(deltas) / len(deltas)
            if len(deltas) > 1:
                variance = sum((d - mean_d) ** 2 for d in deltas) / (len(deltas) - 1)
                std_d = math.sqrt(variance)
            else:
                std_d = 0.0

            confidence = _sigmoid(mean_d)
            # Stability: 1 when std is 0 relative to |mean|
            stability = max(0.0, min(1.0, 1.0 - std_d / (abs(mean_d) + 1e-8)))

            result[edge] = EdgeConfidence(
                parent=edge[0],
                child=edge[1],
                confidence=confidence,
                stability=stability,
                deltas=tuple(deltas),
                n_windows=len(deltas),
            )

        return result

    def suggest_changes(
        self,
        edge_confidences: dict[tuple[str, str], EdgeConfidence],
        current_edges: set[tuple[str, str]],
        candidate_additions: list[tuple[str, str]] | None = None,
        *,
        add_threshold: float = 0.7,
        remove_threshold: float = 0.3,
        stability_min: float = 0.5,
        consecutive_required: int = 2,
        protected_edges: set[tuple[str, str]] | None = None,
    ) -> EdgeSuggestion:
        """Suggest edge additions/removals based on confidence with hysteresis.

        Parameters
        ----------
        edge_confidences : output of evaluate()
        current_edges : set of current (parent, child) edges
        candidate_additions : edges not in current_edges to consider adding.
            If None, only evaluates removal of current edges.
        add_threshold : confidence above this → candidate for addition
        remove_threshold : confidence below this → candidate for removal
        stability_min : minimum stability required for any change
        consecutive_required : how many consecutive evaluations an edge
            must cross threshold before change is applied
        protected_edges : edges that cannot be removed (e.g., expert regime→observed)

        Returns
        -------
        EdgeSuggestion with edges_to_add and edges_to_remove
        """
        protected = protected_edges or set()
        suggestion = EdgeSuggestion()

        # --- Evaluate removal of current edges ---
        for edge in current_edges:
            if edge in protected:
                continue
            conf = edge_confidences.get(edge)
            if conf is None:
                continue

            if conf.confidence < remove_threshold and conf.stability >= stability_min:
                # Increment removal counter (negative direction)
                prev = self._consecutive.get(edge, 0)
                self._consecutive[edge] = min(prev, 0) - 1
            else:
                # Reset removal counter (but keep addition counter if positive)
                prev = self._consecutive.get(edge, 0)
                if prev < 0:
                    self._consecutive[edge] = 0

            if self._consecutive.get(edge, 0) <= -consecutive_required:
                suggestion.edges_to_remove.append(edge)
                log.info(
                    "Suggest remove %s → %s (conf=%.3f, stab=%.3f, consec=%d)",
                    edge[0],
                    edge[1],
                    conf.confidence,
                    conf.stability,
                    self._consecutive[edge],
                )

        # --- Evaluate addition of candidate edges ---
        if candidate_additions:
            for edge in candidate_additions:
                if edge in current_edges:
                    continue
                conf = edge_confidences.get(edge)
                if conf is None:
                    continue

                if conf.confidence > add_threshold and conf.stability >= stability_min:
                    prev = self._consecutive.get(edge, 0)
                    self._consecutive[edge] = max(prev, 0) + 1
                else:
                    prev = self._consecutive.get(edge, 0)
                    if prev > 0:
                        self._consecutive[edge] = 0

                if self._consecutive.get(edge, 0) >= consecutive_required:
                    suggestion.edges_to_add.append(edge)
                    log.info(
                        "Suggest add %s → %s (conf=%.3f, stab=%.3f, consec=%d)",
                        edge[0],
                        edge[1],
                        conf.confidence,
                        conf.stability,
                        self._consecutive[edge],
                    )

        return suggestion

    def reset_consecutive(self, edge: tuple[str, str]) -> None:
        """Reset the consecutive counter for an edge (after applying a change)."""
        self._consecutive.pop(edge, None)

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker state for persistence."""
        return {
            "node_names": list(self._node_names),
            "windows_days": list(self._windows_days),
            "consecutive": {f"{p}|{c}": count for (p, c), count in self._consecutive.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdgeConfidenceTracker:
        """Restore tracker from serialized state."""
        tracker = cls(
            node_names=data["node_names"],
            windows_days=tuple(data.get("windows_days", (30, 60, 90))),
        )
        for key, count in data.get("consecutive", {}).items():
            parts = key.split("|", 1)
            if len(parts) == 2:
                tracker._consecutive[(parts[0], parts[1])] = count
        return tracker

"""
TirraMind — Per-Entity Rolling Baseline (Event Study Adaptation)

Per-entity rolling mean/std for standardized abnormal scoring, adapted from
the financial Event Study methodology.

Role: **node feature enrichment** — abnormal scores feed into the GNN
as input features. They are NOT the anomaly output.

Estimation window: most recent ``window`` observations, excluding the most
recent ``gap`` observations (to avoid contamination from the event itself).

    abnormal_score = (current - μ_est) / σ_est

Requires at least ``min_obs`` observations in the estimation window for a
valid baseline.

Reference:
    MacKinlay, A. C. (1997). "Event studies in economics and finance."
        Journal of Economic Literature, 35(1), 13-39.
"""

from __future__ import annotations

import math
from collections import deque


class EntityBaseline:
    """Per-entity rolling baseline for standardized abnormal scoring.

    Thread-safety: NOT thread-safe.
    """

    __slots__ = ("_window", "_gap", "_min_obs", "_history")

    def __init__(self, window: int = 30, gap: int = 5, min_obs: int = 10) -> None:
        if window < 1:
            raise ValueError(f"Window must be >= 1, got {window}")
        if gap < 0:
            raise ValueError(f"Gap must be >= 0, got {gap}")
        if min_obs < 2:
            raise ValueError(f"min_obs must be >= 2 (need variance), got {min_obs}")
        if min_obs > window:
            raise ValueError(f"min_obs ({min_obs}) > window ({window})")
        self._window = window
        self._gap = gap
        self._min_obs = min_obs
        # entity_id → deque of recent observations (most recent at right)
        self._history: dict[str, deque[float]] = {}

    def add_observation(self, entity_id: str, value: float) -> None:
        """Record an observation for baseline computation.

        NaN/inf values are silently skipped.
        """
        if not math.isfinite(value):
            return
        if entity_id not in self._history:
            self._history[entity_id] = deque(maxlen=self._window + self._gap)
        self._history[entity_id].append(value)

    def abnormal_score(self, entity_id: str, current_value: float) -> float | None:
        """Compute standardized abnormal score.

        Returns None if insufficient history for a valid baseline.

        Args:
            entity_id: entity to score.
            current_value: the observation to compare against baseline.

        Returns:
            (current - μ_est) / σ_est, or None if < min_obs in estimation window.
        """
        est = self._estimation_window(entity_id)
        if est is None:
            return None
        mu, sigma = est
        if sigma == 0:
            return 0.0  # constant baseline → no abnormality measurable
        return (current_value - mu) / sigma

    def cumulative_abnormal_score(
        self, entity_id: str, values: list[float]
    ) -> float | None:
        """Compute CAR (cumulative abnormal score) over an event window.

        Returns None if insufficient history for baseline.
        """
        if not values:
            return 0.0
        est = self._estimation_window(entity_id)
        if est is None:
            return None
        mu, sigma = est
        if sigma == 0:
            return 0.0
        return sum((v - mu) / sigma for v in values)

    def observation_count(self, entity_id: str) -> int:
        """Number of observations stored for this entity."""
        if entity_id not in self._history:
            return 0
        return len(self._history[entity_id])

    def _estimation_window(self, entity_id: str) -> tuple[float, float] | None:
        """Extract (mean, std) from estimation window.

        Returns None if fewer than min_obs observations in the window.
        """
        if entity_id not in self._history:
            return None
        hist = self._history[entity_id]
        n_total = len(hist)
        # Estimation window: all except the most recent `gap` observations
        n_available = n_total - self._gap
        if n_available < self._min_obs:
            return None
        # Take up to `window` observations from the estimation region
        est_end = n_total - self._gap
        est_start = max(0, est_end - self._window)
        est_values = list(hist)[est_start:est_end]
        n = len(est_values)
        if n < self._min_obs:
            return None
        mu = sum(est_values) / n
        variance = sum((x - mu) ** 2 for x in est_values) / (
            n - 1
        )  # Bessel's correction
        sigma = math.sqrt(variance)
        return mu, sigma

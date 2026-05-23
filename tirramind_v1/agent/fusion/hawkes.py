"""
TirraMind — Hawkes Process Intensity Estimator

Per-entity self-exciting point process for event burst detection.

Role: **node feature enrichment** — Hawkes intensity feeds into the GNN
as an input feature. It is NOT the anomaly output.

Exponential kernel with O(1) recursive update:
    λ(t) = μ + Σ_{t_k < t} α · exp(-β · (t - t_k))

If A_n = Σ_{k=1..n} exp(-β(t_n - t_k)), then:
    A_{n+1} = exp(-β(t_{n+1} - t_n)) · (A_n + 1)

so each update is O(1) regardless of event history length.

Default parameters:
    μ = 0.1  (low background rate)
    α = 0.5  (excitation magnitude per event)
    β = 1.0  (decay rate; branching ratio α/β = 0.5, sub-critical)

Reference:
    Hawkes, A. G. (1971). "Spectra of some self-exciting and mutually exciting
        point processes." Biometrika, 58(1), 83-90.
"""

from __future__ import annotations

import math


class HawkesIntensity:
    """Per-entity exponential-kernel Hawkes process intensity estimator.

    Thread-safety: NOT thread-safe.
    """

    __slots__ = ("_mu", "_alpha", "_beta", "_entity_state")

    def __init__(self, mu: float = 0.1, alpha: float = 0.5, beta: float = 1.0) -> None:
        if mu < 0:
            raise ValueError(f"Baseline mu must be >= 0, got {mu}")
        if alpha < 0:
            raise ValueError(f"Excitation alpha must be >= 0, got {alpha}")
        if beta <= 0:
            raise ValueError(f"Decay beta must be > 0, got {beta}")
        if alpha / beta >= 1.0:
            raise ValueError(
                f"Branching ratio alpha/beta must be < 1 for sub-critical process, "
                f"got {alpha}/{beta} = {alpha / beta:.3f}"
            )
        self._mu = mu
        self._alpha = alpha
        self._beta = beta
        # entity_id → (last_event_time, accumulated_kernel_sum A_n)
        self._entity_state: dict[str, tuple[float, float]] = {}

    def update(self, entity_id: str, event_time: float) -> float:
        """Record new event for entity, return current intensity at event_time.

        Events MUST be monotonically non-decreasing in time per entity.

        Args:
            entity_id: unique entity identifier.
            event_time: unix timestamp of the event.

        Returns:
            Intensity λ(event_time) after incorporating this event.

        Raises:
            ValueError: if event_time < previous event time for this entity.
        """
        if entity_id in self._entity_state:
            last_t, a_prev = self._entity_state[entity_id]
            if event_time < last_t:
                raise ValueError(
                    f"Event time {event_time} < previous {last_t} for {entity_id}. "
                    "Events must be monotonically non-decreasing."
                )
            dt = event_time - last_t
            # Recursive update: A_{n+1} = exp(-β·dt) · (A_n + 1)
            decay = math.exp(-self._beta * dt) if dt < 700 / self._beta else 0.0
            a_new = decay * (a_prev + 1.0)
        else:
            a_new = 0.0  # first event: no self-excitation yet

        self._entity_state[entity_id] = (event_time, a_new)
        # λ(t) = μ + α · A_n  (A_n already includes the current event's contribution)
        return self._mu + self._alpha * a_new

    def intensity_at(self, entity_id: str, query_time: float) -> float:
        """Compute intensity at arbitrary time without recording an event.

        Args:
            entity_id: unique entity identifier.
            query_time: unix timestamp to query.

        Returns:
            Intensity λ(query_time). Returns μ if entity has no events.
        """
        if entity_id not in self._entity_state:
            return self._mu

        last_t, a_last = self._entity_state[entity_id]
        if query_time < last_t:
            raise ValueError(f"Query time {query_time} < last event {last_t} for {entity_id}.")
        dt = query_time - last_t
        decay = math.exp(-self._beta * dt) if dt < 700 / self._beta else 0.0
        # At query time, the last event's self-excitation is (A_last + 1) decayed
        a_query = decay * (a_last + 1.0)
        return self._mu + self._alpha * a_query

    def get_state(self, entity_id: str) -> tuple[float, float] | None:
        """Return (last_event_time, kernel_sum) or None if entity unseen."""
        return self._entity_state.get(entity_id)

    def get_all_entities(self) -> list[str]:
        """List all entity IDs with recorded events."""
        return list(self._entity_state.keys())

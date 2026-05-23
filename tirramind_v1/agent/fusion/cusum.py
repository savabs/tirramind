"""
TirraMind — CUSUM Sequential Monitor

Per-entity cumulative sum (CUSUM) control chart for detecting persistent
mean shifts in observation anomaly z-scores.

Role: **node feature enrichment** — CUSUM state feeds into the GNN as an
additional input feature. It is NOT the anomaly output.

The one-sided upper CUSUM is used:
    S_{n+1} = max(0, S_n + z_n - k)
    Alert when S > h.

Parameters:
    k (allowance/slack): half the shift to detect, in σ units.
        Default 0.5 → optimal for detecting 1σ shifts (SPC convention).
    h (decision threshold): controls ARL₀ (average run length under H₀).
        Default 5.0 → ARL₀ ≈ 465 under normality (Hawkins & Olwell 1998, Table 4.2).

Reference:
    Page, E. S. (1954). "Continuous inspection schemes." Biometrika, 41(1-2), 100-115.
    Hawkins, D. M. & Olwell, D. H. (1998). Cumulative Sum Charts and Charting
        for Quality Improvement. Springer.
"""

from __future__ import annotations


class CUSUMMonitor:
    """Per-entity one-sided upper CUSUM for detecting persistent positive mean shifts.

    Thread-safety: NOT thread-safe. Caller must synchronize if shared across threads.
    """

    __slots__ = ("_k", "_h", "_cap", "_states")

    def __init__(self, k: float = 0.5, h: float = 5.0, cap_multiplier: float = 10.0) -> None:
        if k < 0:
            raise ValueError(f"Allowance k must be >= 0, got {k}")
        if h <= 0:
            raise ValueError(f"Threshold h must be > 0, got {h}")
        self._k = k
        self._h = h
        self._cap = h * cap_multiplier  # prevent float overflow on sustained shifts
        self._states: dict[str, float] = {}

    def update(self, entity_id: str, z_score: float) -> tuple[float, bool]:
        """Update CUSUM for one entity.

        Args:
            entity_id: unique entity identifier.
            z_score: standardized observation value.

        Returns:
            (cusum_value, alert_triggered) where alert_triggered = cusum > h.
        """
        prev = self._states.get(entity_id, 0.0)
        new = max(0.0, prev + z_score - self._k)
        new = min(new, self._cap)  # cap to prevent overflow
        self._states[entity_id] = new
        return new, new > self._h

    def reset(self, entity_id: str) -> None:
        """Reset CUSUM to 0 after acknowledged alert."""
        self._states[entity_id] = 0.0

    def get_state(self, entity_id: str) -> float:
        """Current CUSUM value for entity (0.0 if never seen)."""
        return self._states.get(entity_id, 0.0)

    def get_all_states(self) -> dict[str, float]:
        """Snapshot of all entity CUSUM states."""
        return dict(self._states)

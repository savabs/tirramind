"""Atomic signal computation for convergence detection.

Converts aligned evidence streams into standardized anomaly scores:
rolling z-score, empirical percentile, anomaly flag, and direction.

Each signal stream maintains independent rolling statistics.  Direction
normalization ensures positive z-scores always mean "stress / expansion"
regardless of the raw value convention of the source tool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from agent.convergence.evidence import Evidence
from agent.convergence.taxonomy import SignalMeta

log = logging.getLogger(__name__)

# Guard threshold: treat σ below this as zero to avoid div-by-zero
_EPSILON = 1e-10


# ── AtomicSignalResult ─────────────────────────────────────────


@dataclass
class AtomicSignalResult:
    """Output of per-signal anomaly computation.

    Attributes
    ----------
    signal_id : str
        Unique signal identifier.
    timestamp : float
        Unix epoch of the latest observation used.
    raw_value : float
        Direction-normalized value (after flip_sign).
    z_score : float
        Standard deviations from rolling mean.
    percentile : float
        Empirical rank percentile in [0, 1].
    is_anomaly : bool
        True if z-score or percentile exceeds thresholds.
    direction : int
        +1 (stress / expansion) or -1 (relief / contraction).
    """

    signal_id: str
    timestamp: float
    raw_value: float
    z_score: float
    percentile: float
    is_anomaly: bool
    direction: int


# ── Rolling statistics ─────────────────────────────────────────


class RollingStats:
    """Maintains a rolling window of observations for z-score and percentile.

    Parameters
    ----------
    window : int
        Maximum number of recent observations to keep (default 52,
        roughly 1 year of weekly data).
    """

    def __init__(self, window: int = 52) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = window
        self._values: list[float] = []

    # ── Ingestion ──────────────────────────────────────────────

    def update(self, values: np.ndarray) -> None:
        """Replace the internal buffer with non-NaN values from *values*.

        Keeps the last ``window`` valid observations.  This is a full
        refresh, not incremental — call with the complete sorted history.
        """
        arr = np.asarray(values, dtype=np.float64).ravel()
        clean = [float(v) for v in arr if not np.isnan(v)]
        if len(clean) > self._window:
            clean = clean[-self._window :]
        self._values = clean

    # ── Properties ─────────────────────────────────────────────

    @property
    def window(self) -> int:
        return self._window

    @property
    def mean(self) -> float:
        if not self._values:
            return float("nan")
        return float(np.mean(self._values))

    @property
    def std(self) -> float:
        """Sample standard deviation (ddof=1).  Returns 0.0 if n < 2."""
        if len(self._values) < 2:
            return 0.0
        return float(np.std(self._values, ddof=1))

    @property
    def n_observations(self) -> int:
        return len(self._values)

    # ── Scoring ────────────────────────────────────────────────

    def z_score(self, value: float) -> float:
        """Compute ``(value - mean) / std``.  Returns 0.0 if σ < ε."""
        s = self.std
        if s < _EPSILON:
            return 0.0
        return (value - self.mean) / s

    def percentile(self, value: float) -> float:
        """Empirical percentile: fraction of stored values ≤ *value*.

        Returns 0.5 (neutral) when the buffer is empty.
        """
        if not self._values:
            return 0.5
        n = len(self._values)
        rank = sum(1 for v in self._values if v <= value)
        return rank / n

    def __repr__(self) -> str:
        return f"RollingStats(window={self._window}, " f"n={self.n_observations})"


# ── Anomaly detection ──────────────────────────────────────────


def compute_anomaly(
    z: float,
    pct: float,
    z_threshold: float = 2.0,
    pct_lo: float = 0.05,
    pct_hi: float = 0.95,
) -> bool:
    """Return True if the observation is anomalous.

    An observation is anomalous when *any* of these hold:

    - ``|z| > z_threshold``
    - ``pct < pct_lo``
    - ``pct > pct_hi``
    """
    return abs(z) > z_threshold or pct < pct_lo or pct > pct_hi


# ── Direction normalization ────────────────────────────────────


def normalize_direction(value: float, flip_sign: bool) -> float:
    """Apply sign-flip so positive always means stress / expansion.

    Parameters
    ----------
    value : float
        Raw observation value.
    flip_sign : bool
        If True, negate *value* (source convention is opposite ours).
    """
    return -value if flip_sign else value


# ── SignalStream (per-signal state) ────────────────────────────


class SignalStream:
    """Accumulates evidence for a single signal and computes atomic scores.

    Parameters
    ----------
    signal_id : str
        Unique signal identifier.
    meta : SignalMeta
        Metadata (frequency, flip_sign, min_observations, …).
    window : int
        Rolling window size for RollingStats.
    """

    def __init__(
        self,
        signal_id: str,
        meta: SignalMeta,
        window: int = 52,
    ) -> None:
        self._signal_id = signal_id
        self._meta = meta
        self._obs: dict[float, float] = {}  # timestamp → normalised value
        self._window = window

    # ── Ingestion ──────────────────────────────────────────────

    def ingest(self, evidence_list: list[Evidence]) -> None:
        """Accumulate evidence observations.

        - Direction is normalized via ``flip_sign``.
        - Duplicate timestamps keep the latest value (last-write-wins).
        - Internal state is sorted by timestamp after each call.
        """
        for ev in evidence_list:
            val = normalize_direction(ev.value, self._meta.flip_sign)
            self._obs[ev.timestamp] = val

    # ── Computation ────────────────────────────────────────────

    def compute(self, as_of: float) -> AtomicSignalResult | None:
        """Compute the atomic signal result as of a given time.

        Only observations with ``timestamp <= as_of`` are used.
        Returns ``None`` if fewer than ``meta.min_observations``
        non-NaN values are available.
        """
        valid = [(t, v) for t, v in sorted(self._obs.items()) if t <= as_of]
        if not valid:
            return None

        values = np.array([v for _, v in valid], dtype=np.float64)
        stats = RollingStats(window=self._window)
        stats.update(values)

        if stats.n_observations < self._meta.min_observations:
            return None

        latest_ts, latest_val = valid[-1]

        # Cannot score a NaN latest value
        if np.isnan(latest_val):
            return None

        z = stats.z_score(latest_val)
        pct = stats.percentile(latest_val)

        return AtomicSignalResult(
            signal_id=self._signal_id,
            timestamp=latest_ts,
            raw_value=latest_val,
            z_score=z,
            percentile=pct,
            is_anomaly=compute_anomaly(z, pct),
            direction=1 if z >= 0 else -1,
        )

    # ── History access ─────────────────────────────────────────

    def history(self) -> np.ndarray:
        """Return direction-normalized values sorted by timestamp."""
        items = sorted(self._obs.items())
        if not items:
            return np.array([], dtype=np.float64)
        return np.array([v for _, v in items], dtype=np.float64)

    def __repr__(self) -> str:
        return f"SignalStream({self._signal_id!r}, " f"n_obs={len(self._obs)})"

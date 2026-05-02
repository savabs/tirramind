"""TirraMind — Signal Evaluator (Change 15)

Evaluates the predictive signal content of a discovered data source by
computing mutual information between the source's numeric columns and
existing features in the pipeline store.

**Method:** k-NN mutual information estimator (Kraskov, Stögbauer,
Grassberger 2004), accessed via ``sklearn.feature_selection.mutual_info_regression``.
MI is non-parametric, handles nonlinear relationships, and works with
small samples (≥50 aligned points).

Reference: Spec step 15.3 in [[tier8_autonomous_discovery_spec]].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from agent.discovery.source_scout import DataSourceCandidate
    from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

_DEFAULT_MIN_SAMPLES = 50
_DEFAULT_MI_THRESHOLD = 0.05


@dataclass
class SignalReport:
    """Result of signal evaluation for a data source candidate."""

    max_mi: float = 0.0
    mean_mi: float = 0.0
    best_pair: tuple[str, str] = ("", "")  # (source_column, feature_name)
    n_aligned: int = 0
    passes_threshold: bool = False
    details: dict[str, float] = field(default_factory=dict)


class SignalEvaluator:
    """Evaluate predictive signal in a data source candidate.

    Parameters
    ----------
    store : PipelineStore
        Source of existing feature time series for MI comparison.
    min_samples : int
        Minimum aligned data points for a reliable MI estimate.
    mi_threshold : float
        Minimum max MI for a source to be considered valuable.
    """

    def __init__(
        self,
        store: PipelineStore,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
        mi_threshold: float = _DEFAULT_MI_THRESHOLD,
    ) -> None:
        self._store = store
        self._min_samples = min_samples
        self._threshold = mi_threshold

    def evaluate(self, candidate: DataSourceCandidate) -> SignalReport:
        """Compute mutual information between candidate data and existing features.

        Returns a ``SignalReport`` summarising the MI scores.
        """
        if candidate.probe_sample is None:
            return SignalReport()

        # Extract numeric columns from probe sample
        source_series = self._extract_numeric_series(candidate.probe_sample)
        if not source_series:
            log.info("No numeric columns found in %s", candidate.name)
            return SignalReport()

        # Get existing feature time series from store
        feature_series = self._get_feature_series()
        if not feature_series:
            log.info("No existing features in store for MI comparison")
            return SignalReport()

        # Compute MI for all pairs
        mi_scores: dict[str, float] = {}
        best_mi = 0.0
        best_pair = ("", "")
        best_n = 0

        for src_col, src_values in source_series.items():
            for feat_name, feat_values in feature_series.items():
                # Align by index (simple positional alignment for probe data)
                n = min(len(src_values), len(feat_values))
                if n < self._min_samples:
                    continue

                x = np.asarray(src_values[:n], dtype=np.float64).reshape(-1, 1)
                y = np.asarray(feat_values[:n], dtype=np.float64)

                # Skip if constant
                if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                    continue

                mi = self._compute_mi(x, y)
                pair_key = f"{src_col}|{feat_name}"
                mi_scores[pair_key] = mi

                if mi > best_mi:
                    best_mi = mi
                    best_pair = (src_col, feat_name)
                    best_n = n

        if not mi_scores:
            return SignalReport()

        mean_mi = float(np.mean(list(mi_scores.values())))

        return SignalReport(
            max_mi=best_mi,
            mean_mi=mean_mi,
            best_pair=best_pair,
            n_aligned=best_n,
            passes_threshold=best_mi > self._threshold,
            details=mi_scores,
        )

    def _compute_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information using KSG estimator.

        Falls back to a simple correlation-based proxy if sklearn is
        unavailable.
        """
        try:
            from sklearn.feature_selection import mutual_info_regression

            mi = mutual_info_regression(x, y, n_neighbors=3, random_state=42)
            return float(mi[0])
        except ImportError:
            # Fallback: |correlation| as a poor man's dependence measure
            corr = float(np.corrcoef(x.ravel(), y)[0, 1])
            return abs(corr) if not np.isnan(corr) else 0.0

    def _extract_numeric_series(
        self,
        probe_sample: dict | list,
    ) -> dict[str, list[float]]:
        """Extract numeric time series from probe sample data."""
        rows: list[dict[str, Any]] = []

        if isinstance(probe_sample, list):
            rows = [r for r in probe_sample if isinstance(r, dict)]
        elif isinstance(probe_sample, dict):
            # Try common structures: {"data": [...]} or {"results": [...]}
            for key in ("data", "results", "records", "rows", "items"):
                if key in probe_sample and isinstance(probe_sample[key], list):
                    rows = [r for r in probe_sample[key] if isinstance(r, dict)]
                    break
            if not rows and all(isinstance(v, (int, float)) for v in probe_sample.values()):
                # Single-row dict of numbers
                rows = [probe_sample]

        if not rows:
            return {}

        # Identify numeric columns
        series: dict[str, list[float]] = {}
        for row in rows:
            for key, val in row.items():
                if isinstance(val, (int, float)):
                    series.setdefault(key, []).append(float(val))
                elif isinstance(val, str):
                    try:
                        series.setdefault(key, []).append(float(val))
                    except (ValueError, TypeError):
                        pass

        # Filter: need minimum length
        return {k: v for k, v in series.items() if len(v) >= self._min_samples}

    def _get_feature_series(self) -> dict[str, list[float]]:
        """Load recent feature values from store as time series."""
        series: dict[str, list[float]] = {}
        try:
            conn = self._store._get_conn()  # noqa: SLF001
            rows = conn.execute(
                "SELECT feature_name, value FROM features WHERE value IS NOT NULL ORDER BY effective_at DESC LIMIT 500"
            ).fetchall()
            for r in rows:
                fname = r[0]
                val = r[1]
                if val is not None:
                    series.setdefault(fname, []).append(float(val))
        except Exception:
            log.warning("Failed to load feature series from store", exc_info=True)

        # Reverse to chronological order
        return {k: list(reversed(v)) for k, v in series.items() if len(v) >= self._min_samples}

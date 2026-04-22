"""Feature builders — deterministic transformers from pipeline state to EngineeredFeature.

Each builder reads from ``PipelineStore`` and produces a list of
``EngineeredFeature`` records.  Builders are pure functions of
(store state, as_of) — no side effects, no LLM calls, deterministic.

Builders
--------
- ``ConvergenceFeatureBuilder``: convergence-derived aggregate features
  (stress_breadth, stress_intensity, regime_persistence).
- ``MacroStateFeatureBuilder``: continuous macro features
  (rate_momentum, yield_curve_slope, liquidity_pressure).

References:
    - Spec: docs/specs/signal_protocol_feature_engineering_spec.md (steps 3-4)
    - Research: docs/research/signal_protocol_feature_engineering.md (8.3, 8.4)
"""

from __future__ import annotations

import abc
import json
import logging
import math
from typing import Any

import numpy as np

from agent.features.protocol import EngineeredFeature
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# 7 days in seconds
_7D = 7 * 86_400
# 30 days in seconds
_30D = 30 * 86_400
# 90 days in seconds (baseline window for z-scores)
_90D = 90 * 86_400


# ── Abstract base ──────────────────────────────────────────────


class FeatureBuilder(abc.ABC):
    """Base class for all feature builders.

    Subclasses implement ``name`` and ``build()``.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable builder name used in EngineeredFeature.builder."""

    @abc.abstractmethod
    def build(
        self,
        store: PipelineStore,
        as_of: float,
    ) -> list[EngineeredFeature]:
        """Produce engineered features from current pipeline state.

        Parameters
        ----------
        store : PipelineStore
            Data source — read-only access.
        as_of : float
            Unix epoch reference time.  All lookbacks are relative to this.

        Returns
        -------
        list[EngineeredFeature]
            Zero or more valid features.  May include explicit missingness.
        """


# ── Convergence-Derived Feature Builder ────────────────────────


class ConvergenceFeatureBuilder(FeatureBuilder):
    """Aggregate convergence state into model-ready features.

    Reads convergence signals from the ``signals`` table (names matching
    ``convergence.*``) within a 7-day lookback window and produces:

    1. **convergence.stress_breadth.7d** — count of distinct active
       convergence signals.  Unit: ``count``.
    2. **convergence.stress_intensity.7d** — max boosted score across
       active events.  Unit: ``probability`` (0-1 score).
    3. **convergence.regime_persistence.7d** — max persistence_days
       across active events.  Unit: ``duration_days``.

    All three are complementary dimensions of convergence state.
    """

    VERSION = 1

    @property
    def name(self) -> str:
        return "ConvergenceFeatureBuilder"

    def build(
        self,
        store: PipelineStore,
        as_of: float,
    ) -> list[EngineeredFeature]:
        since = as_of - _7D

        # Query all convergence signals in the window.
        # PipelineStore.query_signals only filters on one signal_name,
        # so we do a direct query for the "convergence." prefix.
        rows = self._query_convergence_signals(store, since, as_of)

        if not rows:
            # Check if pipeline has any data at all — if yes, emit
            # zero-valued features (data observed, no convergence detected)
            # rather than None (no data available).
            has_data = bool(
                store.query_data("cftc", since=since, limit=1)
                or store.query_data("gdelt", since=since, limit=1)
                or store.query_data("polymarket", since=since, limit=1)
            )
            if has_data:
                return self._zero_features(as_of)
            return []

        # ── stress_breadth: count of distinct signal names ─────
        distinct_names = {r["signal_name"] for r in rows}
        breadth = float(len(distinct_names))

        # ── stress_intensity: max value (boosted_score) ────────
        intensity = max(r["value"] for r in rows)

        # ── regime_persistence: max persistence_days from metadata
        max_persistence = 0.0
        for r in rows:
            meta = r.get("metadata")
            if isinstance(meta, dict):
                pd_val = meta.get("persistence_days", 0)
                if isinstance(pd_val, (int, float)) and not math.isnan(pd_val):
                    max_persistence = max(max_persistence, float(pd_val))

        return [
            EngineeredFeature(
                feature_name="convergence.stress_breadth.7d",
                version=self.VERSION,
                effective_at=as_of,
                computed_at=as_of,
                horizon="7d",
                value=breadth,
                quality=1.0,
                source_signals=tuple(sorted(distinct_names)),
                builder=self.name,
                unit="count",
            ),
            EngineeredFeature(
                feature_name="convergence.stress_intensity.7d",
                version=self.VERSION,
                effective_at=as_of,
                computed_at=as_of,
                horizon="7d",
                value=intensity,
                quality=1.0,
                source_signals=tuple(sorted(distinct_names)),
                builder=self.name,
                unit="probability",
            ),
            EngineeredFeature(
                feature_name="convergence.regime_persistence.7d",
                version=self.VERSION,
                effective_at=as_of,
                computed_at=as_of,
                horizon="7d",
                value=max_persistence,
                quality=1.0,
                source_signals=tuple(sorted(distinct_names)),
                builder=self.name,
                unit="duration_days",
            ),
        ]

    # ── Internal helpers ───────────────────────────────────────

    def _query_convergence_signals(
        self,
        store: PipelineStore,
        since: float,
        until: float,
    ) -> list[dict[str, Any]]:
        """Query signals table for convergence signals in [since, until].

        Uses a direct SQL query because ``query_signals`` only supports
        filtering by exact signal name, not prefix matching.
        """
        conn = store._get_conn()
        rows = conn.execute(
            "SELECT * FROM signals "
            "WHERE signal_name LIKE 'convergence.%' "
            "AND computed_at >= ? AND computed_at <= ? "
            "ORDER BY computed_at DESC",
            (since, until),
        ).fetchall()
        return [store._signal_row_to_dict(r) for r in rows]

    def _missing_features(
        self,
        as_of: float,
        reason: str,
    ) -> list[EngineeredFeature]:
        """Emit explicit missingness for all three convergence features."""
        names = [
            ("convergence.stress_breadth.7d", "count"),
            ("convergence.stress_intensity.7d", "probability"),
            ("convergence.regime_persistence.7d", "duration_days"),
        ]
        return [
            EngineeredFeature(
                feature_name=fname,
                version=self.VERSION,
                effective_at=as_of,
                computed_at=as_of,
                horizon="7d",
                value=None,
                quality=0.0,
                missing_reason=reason,
                source_signals=("convergence",),
                builder=self.name,
                unit=unit,
            )
            for fname, unit in names
        ]

    def _zero_features(self, as_of: float) -> list[EngineeredFeature]:
        """Emit zero-valued features: data observed but no convergence detected."""
        names = [
            ("convergence.stress_breadth.7d", "count"),
            ("convergence.stress_intensity.7d", "probability"),
            ("convergence.regime_persistence.7d", "duration_days"),
        ]
        return [
            EngineeredFeature(
                feature_name=fname,
                version=self.VERSION,
                effective_at=as_of,
                computed_at=as_of,
                horizon="7d",
                value=0.0,
                quality=1.0,
                source_signals=("convergence",),
                builder=self.name,
                unit=unit,
            )
            for fname, unit in names
        ]


# ── Macro Continuous-State Feature Builder ─────────────────────


class MacroStateFeatureBuilder(FeatureBuilder):
    """Continuous macro features from FRED series stored in pipeline_data.

    Reads ``source="macro_data"`` rows and produces:

    1. **macro.rate_momentum.30d** — 30-day rate-of-change of the Federal
       Funds Rate (DFF series).  Unit: ``bps`` (basis points).
    2. **macro.yield_curve_slope.spot** — current 10Y-2Y Treasury spread
       (GS10 − GS2).  Unit: ``bps``.
    3. **macro.liquidity_pressure.30d** — z-score of 30-day change in
       Fed balance sheet (WALCL).  Negative = tightening.  Unit: ``z_score``.

    Design notes:
        - Reads from ``pipeline_data`` with ``source="macro_data"``.
        - Stored data shape: ``{series_id: [{date, value}, ...]}``
        - 90-day baseline window for z-score computation.
        - Deterministic, LLM-free.
    """

    VERSION = 1

    # FRED series we need
    SERIES_DFF = "DFF"  # Federal funds rate
    SERIES_GS10 = "GS10"  # 10-year Treasury yield
    SERIES_GS2 = "GS2"  # 2-year Treasury yield
    SERIES_WALCL = "WALCL"  # Fed balance sheet total assets

    @property
    def name(self) -> str:
        return "MacroStateFeatureBuilder"

    def build(
        self,
        store: PipelineStore,
        as_of: float,
    ) -> list[EngineeredFeature]:
        # Load all macro_data rows from last 90 days
        rows = store.query_data("macro_data", since=as_of - _90D, limit=500)

        if not rows:
            # No macro data available (e.g. FRED API key not configured).
            # Return empty rather than emitting missing-valued features.
            return []

        # Extract series from all rows (merge across fetches)
        series = self._extract_series(rows)

        features: list[EngineeredFeature] = []
        features.append(self._build_rate_momentum(series, as_of))
        features.append(self._build_yield_curve_slope(series, as_of))
        features.append(self._build_liquidity_pressure(series, as_of))
        return features

    # ── Individual feature builders ────────────────────────────

    def _build_rate_momentum(
        self,
        series: dict[str, list[tuple[str, float]]],
        as_of: float,
    ) -> EngineeredFeature:
        """30-day rate-of-change of DFF in basis points."""
        dff = series.get(self.SERIES_DFF)
        if not dff or len(dff) < 2:
            return self._missing(
                "macro.rate_momentum.30d",
                "30d",
                "bps",
                as_of,
                "insufficient_dff_history",
            )

        # Sort by date descending, take latest and 30d-ago
        sorted_dff = sorted(dff, key=lambda x: x[0], reverse=True)
        latest_val = sorted_dff[0][1]

        # Find the observation closest to 30 days ago
        target_date = self._date_offset(sorted_dff[0][0], -30)
        old_val = self._closest_value(sorted_dff, target_date)

        if old_val is None:
            return self._missing(
                "macro.rate_momentum.30d",
                "30d",
                "bps",
                as_of,
                "insufficient_dff_history",
            )

        # Rate-of-change in basis points (1% = 100 bps)
        momentum_bps = (latest_val - old_val) * 100.0

        return EngineeredFeature(
            feature_name="macro.rate_momentum.30d",
            version=self.VERSION,
            effective_at=as_of,
            computed_at=as_of,
            horizon="30d",
            value=round(momentum_bps, 2),
            quality=1.0 if len(sorted_dff) >= 20 else 0.7,
            source_signals=(f"macro.{self.SERIES_DFF}",),
            builder=self.name,
            unit="bps",
        )

    def _build_yield_curve_slope(
        self,
        series: dict[str, list[tuple[str, float]]],
        as_of: float,
    ) -> EngineeredFeature:
        """10Y−2Y spread in basis points."""
        gs10 = series.get(self.SERIES_GS10)
        gs2 = series.get(self.SERIES_GS2)

        if not gs10 or not gs2:
            return self._missing(
                "macro.yield_curve_slope.spot",
                "spot",
                "bps",
                as_of,
                "missing_treasury_yields",
            )

        # Latest values for each
        latest_10y = sorted(gs10, key=lambda x: x[0], reverse=True)[0][1]
        latest_2y = sorted(gs2, key=lambda x: x[0], reverse=True)[0][1]

        spread_bps = (latest_10y - latest_2y) * 100.0

        return EngineeredFeature(
            feature_name="macro.yield_curve_slope.spot",
            version=self.VERSION,
            effective_at=as_of,
            computed_at=as_of,
            horizon="spot",
            value=round(spread_bps, 2),
            quality=1.0,
            source_signals=(
                f"macro.{self.SERIES_GS10}",
                f"macro.{self.SERIES_GS2}",
            ),
            builder=self.name,
            unit="bps",
        )

    def _build_liquidity_pressure(
        self,
        series: dict[str, list[tuple[str, float]]],
        as_of: float,
    ) -> EngineeredFeature:
        """Z-score of 30-day change in Fed balance sheet (WALCL)."""
        walcl = series.get(self.SERIES_WALCL)
        if not walcl or len(walcl) < 5:
            return self._missing(
                "macro.liquidity_pressure.30d",
                "30d",
                "z_score",
                as_of,
                "insufficient_walcl_history",
            )

        # Sort by date ascending for diff computation
        sorted_walcl = sorted(walcl, key=lambda x: x[0])
        values = np.array([v for _, v in sorted_walcl], dtype=np.float64)

        # 30-day (roughly 4-weekly) differences
        # WALCL is weekly so ~4 obs per 30 days
        step = max(1, min(4, len(values) - 1))
        diffs = values[step:] - values[:-step]

        if len(diffs) < 2:
            return self._missing(
                "macro.liquidity_pressure.30d",
                "30d",
                "z_score",
                as_of,
                "insufficient_walcl_history",
            )

        latest_diff = float(diffs[-1])
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1))

        if std_diff < 1e-10:
            z = 0.0
        else:
            z = (latest_diff - mean_diff) / std_diff

        # Clamp to reasonable range to avoid extreme outliers
        z = max(-10.0, min(10.0, z))

        quality = 1.0 if len(diffs) >= 8 else 0.6

        return EngineeredFeature(
            feature_name="macro.liquidity_pressure.30d",
            version=self.VERSION,
            effective_at=as_of,
            computed_at=as_of,
            horizon="30d",
            value=round(z, 4),
            quality=quality,
            source_signals=(f"macro.{self.SERIES_WALCL}",),
            builder=self.name,
            unit="z_score",
        )

    # ── Data extraction helpers ────────────────────────────────

    @staticmethod
    def _extract_series(
        rows: list[dict[str, Any]],
    ) -> dict[str, list[tuple[str, float]]]:
        """Extract ``{series_id: [(date_str, value), ...]}`` from pipeline rows.

        Merges across multiple fetch rows — the same series may appear in
        multiple pipeline_data rows from different fetch times. Deduplicates
        by date to avoid double-counting.
        """
        by_series: dict[str, dict[str, float]] = {}

        for row in rows:
            data = row.get("data", {})
            if not isinstance(data, dict):
                continue
            for series_id, observations in data.items():
                if not isinstance(observations, list):
                    continue
                if series_id not in by_series:
                    by_series[series_id] = {}
                for obs in observations:
                    if not isinstance(obs, dict):
                        continue
                    date = obs.get("date")
                    val_raw = obs.get("value")
                    if date is None or val_raw is None:
                        continue
                    # Skip "." sentinel values from FRED
                    if val_raw == ".":
                        continue
                    try:
                        val = float(val_raw)
                    except (ValueError, TypeError):
                        continue
                    if math.isnan(val) or math.isinf(val):
                        continue
                    by_series[series_id][str(date)] = val

        return {
            sid: sorted(vals.items(), key=lambda x: x[0])
            for sid, vals in by_series.items()
        }

    @staticmethod
    def _date_offset(date_str: str, days: int) -> str:
        """Shift an ISO date string by `days` (naive string arithmetic).

        Handles YYYY-MM-DD format.  Not timezone aware — sufficient for
        daily-granularity FRED comparisons.
        """
        from datetime import datetime, timedelta

        try:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            shifted = dt + timedelta(days=days)
            return shifted.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return date_str

    @staticmethod
    def _closest_value(
        sorted_desc: list[tuple[str, float]],
        target_date: str,
    ) -> float | None:
        """Find the observation closest to `target_date`.

        ``sorted_desc`` is sorted by date descending.
        Returns None if no observations within 10 days of target.
        """
        best: tuple[int, float] | None = None  # (abs_days_away, value)
        for date_str, val in sorted_desc:
            try:
                from datetime import datetime

                d = datetime.strptime(date_str[:10], "%Y-%m-%d")
                t = datetime.strptime(target_date[:10], "%Y-%m-%d")
                delta = abs((d - t).days)
            except (ValueError, TypeError):
                continue
            if best is None or delta < best[0]:
                best = (delta, val)
        if best is None or best[0] > 10:
            return None
        return best[1]

    def _missing(
        self,
        feature_name: str,
        horizon: str,
        unit: str,
        as_of: float,
        reason: str,
    ) -> EngineeredFeature:
        """Emit a single missing feature."""
        return EngineeredFeature(
            feature_name=feature_name,
            version=self.VERSION,
            effective_at=as_of,
            computed_at=as_of,
            horizon=horizon,
            value=None,
            quality=0.0,
            missing_reason=reason,
            source_signals=("macro_data",),
            builder=self.name,
            unit=unit,
        )

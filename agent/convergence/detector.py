"""Top-level convergence detector — orchestrates a full detection cycle.

Wires together every convergence sub-module into a single ``detect()``
call:

1. Load recent pipeline data from the store.
2. Extract evidence via per-tool extractors.
3. Build per-signal streams and compute atomic anomaly scores.
4. Smart pair selection (cross-category + within-category filtering).
5. Align pairs and compute pairwise coincidence scores.
6. Apply FDR controls (BH, graph → cliques, Fisher, persistence,
   cross-category).
7. Match cliques against causal templates.
8. Return :class:`ConvergenceSignal` objects ready for emission.

This module is deterministic and LLM-free (Pipeline Layer contract).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from agent.convergence.alignment import FREQUENCY_TO_GRID, TimeGrid, align_pair
from agent.convergence.atomic_signals import AtomicSignalResult, SignalStream
from agent.convergence.coincidence import (
    CoincidenceResult,
    combined_coincidence_score,
)
from agent.convergence.evidence import Evidence
from agent.convergence.extractors import extract_evidence, registered_tools
from agent.convergence.fdr import apply_all_controls
from agent.convergence.graph import ConvergenceClique
from agent.convergence.taxonomy import CATEGORIES, SignalMeta, SignalRegistry
from agent.convergence.templates import (
    TemplateMatchResult,
    best_match,
    match_all_templates,
)
from agent.pipeline.store import PipelineStore

log = logging.getLogger(__name__)

# Seconds per day.
_DAY = 86_400


# ── Configuration ──────────────────────────────────────────────


@dataclass
class ConvergenceDetectorConfig:
    """Tuning knobs for the convergence detector.

    All defaults match the spec.  Override individual fields as needed.
    """

    z_threshold: float = 2.0
    """Anomaly z-score threshold for atomic signals."""

    p_threshold: float = 0.05
    """Pairwise significance level before FDR correction."""

    fdr_q: float = 0.05
    """Benjamini-Hochberg FDR target."""

    min_clique_size: int = 3
    """Minimum signals per convergence clique."""

    min_categories: int = 2
    """Minimum distinct taxonomy categories in a clique."""

    min_persistence: int = 2
    """Consecutive detection cycles before emission."""

    corr_window: int = 20
    """Rolling correlation window (pairwise scoring)."""

    baseline_window: int = 100
    """Baseline window for correlation z-score."""

    template_boost: float = 0.5
    """Score boost factor for template-matched cliques:
    ``score × (1 + template_boost × template_match)``."""

    lookback_days: int = 365
    """How far back to read pipeline data (days)."""

    max_pairs: int = 500
    """Safety cap on pairs scored per cycle."""


# ── Detector ───────────────────────────────────────────────────


class ConvergenceDetector:
    """Orchestrates one full convergence detection cycle.

    Parameters
    ----------
    store : PipelineStore
        Pipeline data store to read tool outputs and write signals.
    signal_registry : SignalRegistry
        Registry of known signal metadata.
    config : ConvergenceDetectorConfig | None
        Configuration overrides.  Uses defaults if None.
    """

    def __init__(
        self,
        store: PipelineStore,
        signal_registry: SignalRegistry,
        config: ConvergenceDetectorConfig | None = None,
    ) -> None:
        self._store = store
        self._registry = signal_registry
        self._config = config or ConvergenceDetectorConfig()
        self._persistence_history: dict[tuple[str, ...], int] = {}

    # ── Public API ─────────────────────────────────────────────

    def detect(self, as_of: float | None = None) -> list[DetectionResult]:
        """Run one full detection cycle.

        Parameters
        ----------
        as_of : float | None
            Reference time (unix epoch).  Defaults to now.

        Returns
        -------
        list[DetectionResult]
            Convergence events that survived all controls.
        """
        if as_of is None:
            as_of = time.time()

        cfg = self._config
        since = as_of - cfg.lookback_days * _DAY

        # ── Step 1: Load pipeline data ─────────────────────────
        all_evidence = self._load_evidence(since, as_of)
        if not all_evidence:
            log.info("No evidence loaded — nothing to detect.")
            return []

        # ── Step 2: Group evidence by signal_id ────────────────
        by_signal: dict[str, list[Evidence]] = {}
        for ev in all_evidence:
            by_signal.setdefault(ev.signal_id, []).append(ev)

        # ── Step 3: Build streams + compute atomic scores ──────
        streams: dict[str, SignalStream] = {}
        atomic_results: dict[str, AtomicSignalResult] = {}

        for sig_id, ev_list in by_signal.items():
            meta = self._registry.get(sig_id)
            if meta is None:
                continue
            stream = SignalStream(sig_id, meta)
            stream.ingest(ev_list)
            result = stream.compute(as_of)
            if result is not None:
                streams[sig_id] = stream
                atomic_results[sig_id] = result

        if len(atomic_results) < 2:
            log.info(
                "Only %d signal(s) computed — need ≥ 2 for pairwise scoring.",
                len(atomic_results),
            )
            return []

        # ── Step 4: Smart pair selection ───────────────────────
        pairs = self._select_pairs(atomic_results)
        log.info(
            "Selected %d pairs from %d active signals.",
            len(pairs),
            len(atomic_results),
        )

        if not pairs:
            return []

        # ── Step 5: Align + score pairs ────────────────────────
        scores: dict[tuple[str, str], CoincidenceResult] = {}
        pairs_p: dict[tuple[str, str], float] = {}

        for sig_a, sig_b in pairs:
            meta_a = self._registry.get(sig_a)
            meta_b = self._registry.get(sig_b)
            if meta_a is None or meta_b is None:
                continue

            ev_a = by_signal.get(sig_a, [])
            ev_b = by_signal.get(sig_b, [])

            timestamps, vals_a, vals_b = align_pair(ev_a, ev_b, meta_a, meta_b)
            if len(timestamps) < 3:
                continue

            # Build z-scored versions
            z_a = self._z_score_array(vals_a)
            z_b = self._z_score_array(vals_b)

            result = combined_coincidence_score(vals_a, vals_b, z_a, z_b)
            key = (sig_a, sig_b)
            scores[key] = result
            pairs_p[key] = result.p_value

        if not scores:
            log.info("No valid pairwise scores computed.")
            return []

        # ── Step 6: FDR controls → cliques ─────────────────────
        categories = {
            sig_id: (
                self._registry.get(sig_id).category
                if self._registry.get(sig_id)
                else "unknown"
            )
            for sig_id in atomic_results
        }

        cliques = apply_all_controls(
            pairs_p=pairs_p,
            scores=scores,
            categories=categories,
            history=self._persistence_history,
            q=cfg.fdr_q,
            min_persist=cfg.min_persistence,
            min_cats=cfg.min_categories,
            min_clique_size=cfg.min_clique_size,
        )

        if not cliques:
            log.info("No convergences survived FDR controls.")
            return []

        # ── Step 7: Template matching ──────────────────────────
        results: list[DetectionResult] = []
        for clique in cliques:
            # Gather evidence timeline for this clique's signals
            clique_evidence = []
            for sig_id in clique.signals:
                clique_evidence.extend(by_signal.get(sig_id, []))
            clique_evidence.sort(key=lambda e: e.timestamp)

            tmpl_result = best_match(clique, clique_evidence)
            event_type = tmpl_result.template_name if tmpl_result else "unknown_pattern"
            template_score = tmpl_result.match_score if tmpl_result else 0.0
            lead_signal = tmpl_result.lead_signal if tmpl_result else None
            lag_signals = tmpl_result.lag_signals if tmpl_result else []

            # Boost score with template match
            boosted_score = clique.score * (1.0 + cfg.template_boost * template_score)
            # Clamp to [0, 1]
            boosted_score = min(1.0, max(0.0, boosted_score))

            # Aggregate direction from pairwise coincidence scores
            direction = self._aggregate_direction(clique, scores)

            results.append(
                DetectionResult(
                    clique=clique,
                    event_type=event_type,
                    template_match=template_score,
                    boosted_score=boosted_score,
                    lead_signal=lead_signal,
                    lag_signals=lag_signals,
                    template_result=tmpl_result,
                    direction=direction,
                )
            )

        log.info("Detected %d convergence event(s).", len(results))
        return results

    @property
    def persistence_history(self) -> dict[tuple[str, ...], int]:
        """Current persistence state (for inspection/testing)."""
        return dict(self._persistence_history)

    # ── Internal helpers ───────────────────────────────────────

    @staticmethod
    def _aggregate_direction(
        clique: ConvergenceClique,
        scores: dict[tuple[str, str], CoincidenceResult],
    ) -> int:
        """Compute net direction from pairwise coincidence scores.

        Weighted vote: each pair contributes its direction × score.
        Returns +1 (stress), −1 (relief), or 0 (ambiguous).
        """
        sig_set = set(clique.signals)
        weighted_sum = 0.0
        weight_total = 0.0
        for (a, b), result in scores.items():
            if a in sig_set and b in sig_set:
                w = max(result.score, 0.0)
                weighted_sum += result.direction * w
                weight_total += w
        if weight_total < 1e-10:
            return 1  # default stress
        if weighted_sum > 0:
            return 1
        elif weighted_sum < 0:
            return -1
        return 0

    def _load_evidence(self, since: float, until: float) -> list[Evidence]:
        """Load pipeline data rows and extract evidence."""
        all_evidence: list[Evidence] = []
        tools = registered_tools()

        for tool_name in tools:
            rows = self._store.query_data(
                tool_name, since=since, until=until, limit=1000
            )
            for row in rows:
                data = row.get("data")
                if data is None:
                    continue
                evidence = extract_evidence(tool_name, data)
                all_evidence.extend(evidence)

        log.debug(
            "Loaded %d evidence items from %d tools.",
            len(all_evidence),
            len(tools),
        )
        return all_evidence

    def _select_pairs(
        self,
        atomic_results: dict[str, AtomicSignalResult],
    ) -> list[tuple[str, str]]:
        """Smart pair selection to avoid O(n²) blowup.

        Strategy:
        1. Cross-category: for each pair of categories, pick the
           signal with the highest |z-score| from each category →
           one pair per category combination (~55 pairs for 11 cats).
        2. Within-category: include pairs where both signals are
           anomalous (|z| > threshold) since those are the only
           interesting within-category pairs.
        3. Cap at max_pairs for safety.
        """
        cfg = self._config

        # Group signals by category
        by_cat: dict[str, list[AtomicSignalResult]] = {}
        for result in atomic_results.values():
            meta = self._registry.get(result.signal_id)
            if meta is None:
                continue
            by_cat.setdefault(meta.category, []).append(result)

        # Sort each category by |z_score| descending (best signal first)
        for cat in by_cat:
            by_cat[cat].sort(key=lambda r: abs(r.z_score), reverse=True)

        pairs: set[tuple[str, str]] = set()

        # 1. Cross-category: top signal from each category pair
        cat_list = sorted(by_cat.keys())
        for i, cat_a in enumerate(cat_list):
            for cat_b in cat_list[i + 1 :]:
                sigs_a = by_cat[cat_a]
                sigs_b = by_cat[cat_b]
                if not sigs_a or not sigs_b:
                    continue
                # Top signal from each category
                pair = tuple(sorted([sigs_a[0].signal_id, sigs_b[0].signal_id]))
                pairs.add(pair)  # type: ignore[arg-type]

                # Also add second-best if available (more coverage)
                for sa in sigs_a[:2]:
                    for sb in sigs_b[:2]:
                        if sa.signal_id != sb.signal_id:
                            pair = tuple(sorted([sa.signal_id, sb.signal_id]))
                            pairs.add(pair)  # type: ignore[arg-type]

        # 2. Within-category: anomalous pairs only
        for cat, sigs in by_cat.items():
            anomalous = [s for s in sigs if s.is_anomaly]
            for a, b in combinations(anomalous, 2):
                pair = tuple(sorted([a.signal_id, b.signal_id]))
                pairs.add(pair)  # type: ignore[arg-type]

        # 3. Safety cap
        pair_list = sorted(pairs)
        if len(pair_list) > cfg.max_pairs:
            log.warning(
                "Pair count %d exceeds max_pairs=%d — truncating.",
                len(pair_list),
                cfg.max_pairs,
            )
            pair_list = pair_list[: cfg.max_pairs]

        return pair_list  # type: ignore[return-value]

    @staticmethod
    def _z_score_array(values: np.ndarray) -> np.ndarray:
        """Z-score an aligned array (NaN-aware)."""
        arr = np.asarray(values, dtype=np.float64)
        mask = ~np.isnan(arr)
        if mask.sum() < 2:
            return np.zeros_like(arr)
        mu = float(np.nanmean(arr))
        sigma = float(np.nanstd(arr, ddof=1))
        if sigma < 1e-10:
            return np.zeros_like(arr)
        result = (arr - mu) / sigma
        result[~mask] = np.nan
        return result


# ── DetectionResult ────────────────────────────────────────────


@dataclass
class DetectionResult:
    """Output of a single convergence detection, pre-emission.

    Carries the raw clique plus template-matching metadata so that
    the signal emission layer can build :class:`ConvergenceSignal`.
    """

    clique: ConvergenceClique
    event_type: str
    template_match: float
    boosted_score: float
    lead_signal: str | None
    lag_signals: list[str] = field(default_factory=list)
    template_result: TemplateMatchResult | None = None
    direction: int = 1

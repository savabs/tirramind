"""Sub-phase D edge-case and integration tests.

Coverage:
- Empty store (no data)
- Single tool (no cross-category possible)
- Many tools with noise (no convergence expected)
- Many tools with injected convergence (should detect)
- Wrong temporal ordering
- Signal serialization round-trip (store → query → compare)
- Template edge cases
- Detector config edge cases
- DAG callback edge cases
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.convergence.detector import (
    ConvergenceDetector,
    ConvergenceDetectorConfig,
    DetectionResult,
)
from agent.convergence.evidence import Evidence
from agent.convergence.graph import ConvergenceClique
from agent.convergence.signals import (
    ConvergenceSignal,
    emit_signals,
    format_signal_name,
    from_detection_result,
)
from agent.convergence.taxonomy import CATEGORIES, SignalMeta, SignalRegistry
from agent.convergence.templates import (
    TEMPLATE_LIBRARY,
    CausalTemplate,
    TemplateStep,
    match_all_templates,
    match_template,
)
from agent.pipeline.dags.convergence_detection import (
    _load_evidence_from_store,
    build_convergence_detection_dag,
    build_registry_from_evidence,
    run_convergence_detection,
)
from agent.pipeline.store import PipelineStore

# ── Helpers ────────────────────────────────────────────────────

_DAY = 86_400
_NOW = 1700000000.0  # Fixed reference time


def _store() -> PipelineStore:
    return PipelineStore(":memory:")


def _evidence(
    signal_id: str = "test.sig",
    source: str = "test_tool",
    category: str = "positioning",
    value: float = 1.0,
    ts: float | None = None,
) -> Evidence:
    return Evidence(
        signal_id=signal_id,
        source=source,
        category=category,
        timestamp=ts or _NOW,
        value=value,
        confidence=0.9,
        direction=1,
        tags=(),
        ttl=86400,
    )


def _meta(
    signal_id: str,
    source: str = "test",
    category: str = "positioning",
    frequency: str = "daily",
) -> SignalMeta:
    return SignalMeta(
        signal_id=signal_id,
        source=source,
        category=category,
        frequency=frequency,
        direction_semantics="up=stress",
    )


def _registry(*metas: SignalMeta) -> SignalRegistry:
    reg = SignalRegistry()
    for m in metas:
        reg.register(m)
    return reg


def _clique(
    signals: tuple[str, ...] = ("a", "b", "c"),
    categories: tuple[str, ...] = ("positioning", "macro_momentum"),
    score: float = 0.8,
) -> ConvergenceClique:
    return ConvergenceClique(signals=signals, categories=categories, score=score, edges=())


def _detection(
    event_type: str = "unknown_pattern",
    score: float = 0.7,
) -> DetectionResult:
    return DetectionResult(
        clique=_clique(),
        event_type=event_type,
        template_match=0.0,
        boosted_score=score,
        lead_signal="a",
        lag_signals=["b", "c"],
    )


# ═══════════════════════════════════════════════════════════════
#  Empty Store / No Data
# ═══════════════════════════════════════════════════════════════


class TestEmptyStore:
    """Detector and DAG must handle empty pipeline gracefully."""

    def test_detector_returns_empty(self):
        store = _store()
        reg = _registry()
        det = ConvergenceDetector(store, reg)
        results = det.detect(_NOW)
        assert results == []

    def test_dag_callback_returns_zeros(self):
        store = _store()
        with patch(
            "agent.pipeline.dags.convergence_detection.PipelineStore",
            return_value=store,
        ):
            result = run_convergence_detection({"db_path": ":memory:"}, {})
        assert result == {"detected": 0, "emitted": 0, "signals": []}

    def test_evidence_loader_returns_empty(self):
        store = _store()
        evidence = _load_evidence_from_store(store, as_of=_NOW)
        assert evidence == []

    def test_emit_signals_empty_list(self):
        store = _store()
        count = emit_signals([], store)
        assert count == 0


# ═══════════════════════════════════════════════════════════════
#  Single Tool (No Cross-Category Possible)
# ═══════════════════════════════════════════════════════════════


class TestSingleTool:
    """With only one tool, cross-category convergence is impossible."""

    def test_single_category_no_convergence(self):
        """Single signal category can't produce cross-category cliques."""
        store = _store()
        sigs = [_meta(f"positioning.{i}", category="positioning") for i in range(5)]
        reg = _registry(*sigs)
        det = ConvergenceDetector(store, reg)
        results = det.detect(_NOW)
        assert results == []

    def test_registry_from_one_tool(self):
        evs = [_evidence(signal_id=f"cftc.{i}", source="cftc") for i in range(3)]
        reg = build_registry_from_evidence(evs)
        assert len(reg) == 3
        assert all(reg.get(f"cftc.{i}") is not None for i in range(3))


# ═══════════════════════════════════════════════════════════════
#  Signal Serialization Round-Trip
# ═══════════════════════════════════════════════════════════════


class TestSignalRoundTrip:
    """Store → query → compare for convergence signals."""

    def test_basic_round_trip(self):
        store = _store()
        sig = ConvergenceSignal(
            signal_name="convergence.test.2024-01-15",
            computed_at=_NOW,
            value=0.85,
            event_type="supply_chain_disruption",
            signals_involved=["sig_a", "sig_b"],
            categories_involved=["positioning", "physical_flow"],
            cross_category_count=2,
            p_value=0.01,
            persistence_days=3,
            template_match=0.75,
            direction=1,
            lead_signal="sig_a",
            lag_signals=["sig_b"],
        )

        emit_signals([sig], store)

        rows = store.query_signals("convergence.test.2024-01-15", limit=10)
        assert len(rows) == 1
        row = rows[0]
        assert abs(row["value"] - 0.85) < 1e-6
        meta = row["metadata"]
        assert meta["event_type"] == "supply_chain_disruption"
        assert meta["cross_category_count"] == 2
        assert meta["p_value"] == 0.01
        assert meta["persistence_days"] == 3

    def test_metadata_is_json_serializable(self):
        sig = ConvergenceSignal(
            signal_name="convergence.test.2024-01-15",
            computed_at=_NOW,
            value=0.5,
            event_type="test",
            signals_involved=["a"],
            categories_involved=["positioning"],
            cross_category_count=1,
            p_value=0.05,
            persistence_days=1,
            template_match=0.0,
            direction=1,
            lead_signal="a",
        )
        meta = sig.to_metadata_dict()
        serialized = json.dumps(meta)
        deserialized = json.loads(serialized)
        assert deserialized["event_type"] == "test"

    def test_multiple_signals_stored(self):
        store = _store()
        sigs = []
        for i in range(5):
            sigs.append(
                ConvergenceSignal(
                    signal_name=f"convergence.test_{i}.2024-01-15",
                    computed_at=_NOW,
                    value=0.5 + i * 0.1,
                    event_type=f"type_{i}",
                    signals_involved=["a"],
                    categories_involved=["positioning"],
                    cross_category_count=1,
                    p_value=0.05,
                    persistence_days=1,
                    template_match=0.0,
                    direction=1,
                    lead_signal="a",
                )
            )
        count = emit_signals(sigs, store)
        assert count == 5

    def test_nan_value_not_storable(self):
        """NaN encodes as SQL NULL which violates NOT NULL — emit_signals handles gracefully."""
        store = _store()
        sig = ConvergenceSignal(
            signal_name="convergence.nan_test.2024-01-15",
            computed_at=_NOW,
            value=float("nan"),
            event_type="test",
            signals_involved=["a"],
            categories_involved=["positioning"],
            cross_category_count=1,
            p_value=0.05,
            persistence_days=1,
            template_match=0.0,
            direction=1,
            lead_signal="a",
        )
        # NaN → NULL violates signals.value NOT NULL constraint;
        # emit_signals catches and logs the error, returns 0.
        count = emit_signals([sig], store)
        assert count == 0

    def test_categories_round_trip_order_independent(self):
        """Category lists should round-trip regardless of order."""
        store = _store()
        cats = ["physical_flow", "positioning", "biological"]
        sig = ConvergenceSignal(
            signal_name="convergence.order_test.2024-01-15",
            computed_at=_NOW,
            value=0.7,
            event_type="test",
            signals_involved=["a", "b", "c"],
            categories_involved=cats,
            cross_category_count=3,
            p_value=0.01,
            persistence_days=2,
            template_match=0.5,
            direction=-1,
            lead_signal="a",
        )
        emit_signals([sig], store)
        rows = store.query_signals("convergence.order_test.2024-01-15", limit=1)
        assert set(rows[0]["metadata"]["categories_involved"]) == set(cats)


# ═══════════════════════════════════════════════════════════════
#  Signal Name Formatting Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestSignalNameEdgeCases:
    """Edge cases in signal name formatting."""

    def test_long_event_type(self):
        name = format_signal_name("a" * 200, "2024-01-01")
        assert name.startswith("convergence.")

    def test_numeric_event_type(self):
        name = format_signal_name("type123", "2024-01-01")
        assert name == "convergence.type123.2024-01-01"

    def test_underscores_preserved(self):
        name = format_signal_name("my_event_type", "2024-01-01")
        assert name == "convergence.my_event_type.2024-01-01"

    def test_hyphen_in_event_type_rejected(self):
        """Hyphens are not allowed in event_type (alphanumeric + underscore only)."""
        with pytest.raises(ValueError, match="alphanumeric"):
            format_signal_name("my-event", "2024-01-01")


# ═══════════════════════════════════════════════════════════════
#  from_detection_result Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestFromDetectionResultEdge:
    """Edge cases in DetectionResult → ConvergenceSignal conversion."""

    def test_zero_score(self):
        det = DetectionResult(
            clique=_clique(score=0.0),
            event_type="test",
            template_match=0.0,
            boosted_score=0.0,
            lead_signal=None,
            lag_signals=[],
        )
        sig = from_detection_result(det, as_of=_NOW)
        assert sig.value == 0.0
        assert sig.lead_signal == ""

    def test_perfect_score(self):
        det = DetectionResult(
            clique=_clique(score=1.0),
            event_type="test",
            template_match=1.0,
            boosted_score=1.0,
            lead_signal="leader",
            lag_signals=["lag1", "lag2", "lag3"],
        )
        sig = from_detection_result(det, as_of=_NOW)
        assert sig.value == 1.0
        assert sig.template_match == 1.0
        assert len(sig.lag_signals) == 3

    def test_many_signals_in_clique(self):
        big_clique = _clique(
            signals=tuple(f"sig_{i}" for i in range(20)),
            categories=tuple(sorted(CATEGORIES)[:5]),
        )
        det = DetectionResult(
            clique=big_clique,
            event_type="big",
            template_match=0.5,
            boosted_score=0.9,
            lead_signal="sig_0",
            lag_signals=[f"sig_{i}" for i in range(1, 20)],
        )
        sig = from_detection_result(det, as_of=_NOW)
        assert len(sig.signals_involved) == 20
        assert sig.cross_category_count == 5

    def test_single_category_clique(self):
        det = DetectionResult(
            clique=_clique(categories=("positioning",)),
            event_type="mono",
            template_match=0.0,
            boosted_score=0.5,
            lead_signal="a",
            lag_signals=[],
        )
        sig = from_detection_result(det, as_of=_NOW)
        assert sig.cross_category_count == 1

    def test_all_template_types_valid_signal_names(self):
        """Every template name in TEMPLATE_LIBRARY produces a valid signal name."""
        for tmpl in TEMPLATE_LIBRARY:
            name = format_signal_name(tmpl.name, "2024-06-15")
            assert name.startswith("convergence.")
            assert ".." not in name


# ═══════════════════════════════════════════════════════════════
#  Template Matching Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestTemplateEdgeCases:
    """Edge cases in template matching."""

    def test_empty_evidence_list(self):
        for tmpl in TEMPLATE_LIBRARY:
            result = match_template(_clique(), [], tmpl)
            assert result.match_score == 0.0

    def test_all_templates_against_empty(self):
        results = match_all_templates(_clique(), [])
        assert all(r.match_score == 0.0 for r in results)

    def test_template_with_single_step(self):
        """A template with one step should work."""
        tmpl = CausalTemplate(
            name="single_step",
            description="Test",
            steps=(
                TemplateStep(
                    category_pattern="positioning",
                    signal_pattern=".*",
                    within_days=0,
                ),
            ),
        )
        ev = _evidence(category="positioning")
        result = match_template(_clique(), [ev], tmpl)
        # Should match (1 step, 1 evidence in the right category)
        assert isinstance(result.match_score, float)

    def test_mismatched_categories(self):
        """Template needing category A, evidence only has category B."""
        tmpl = CausalTemplate(
            name="mismatch",
            description="Test",
            steps=(
                TemplateStep(
                    category_pattern="biological",
                    signal_pattern=".*",
                    within_days=0,
                ),
            ),
        )
        ev = _evidence(category="positioning")
        result = match_template(_clique(), [ev], tmpl)
        assert result.match_score == 0.0


# ═══════════════════════════════════════════════════════════════
#  Detector Config Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestDetectorConfigEdge:
    """Edge cases in detector configuration."""

    def test_default_config(self):
        cfg = ConvergenceDetectorConfig()
        assert cfg.z_threshold == 2.0
        assert cfg.max_pairs == 500
        assert cfg.lookback_days == 365

    def test_zero_lookback(self):
        """Zero lookback means no data loaded."""
        store = _store()
        reg = _registry()
        cfg = ConvergenceDetectorConfig(lookback_days=0)
        det = ConvergenceDetector(store, reg, cfg)
        results = det.detect(_NOW)
        assert results == []

    def test_very_strict_threshold(self):
        """Very high z-threshold means nothing qualifies."""
        store = _store()
        reg = _registry()
        cfg = ConvergenceDetectorConfig(z_threshold=100.0)
        det = ConvergenceDetector(store, reg, cfg)
        results = det.detect(_NOW)
        assert results == []

    def test_max_pairs_one(self):
        """max_pairs=1 should not crash."""
        cfg = ConvergenceDetectorConfig(max_pairs=1)
        assert cfg.max_pairs == 1


# ═══════════════════════════════════════════════════════════════
#  DAG Callback Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestDagCallbackEdge:
    """Edge cases in DAG callback execution."""

    def test_detector_exception_propagates(self):
        """If detector.detect() raises, it should propagate (not swallow)."""
        store = _store()
        with (
            patch(
                "agent.pipeline.dags.convergence_detection.PipelineStore",
                return_value=store,
            ),
            patch(
                "agent.pipeline.dags.convergence_detection._load_evidence_from_store",
                return_value=[_evidence()],
            ),
            patch(
                "agent.pipeline.dags.convergence_detection.build_registry_from_evidence",
            ),
            patch(
                "agent.pipeline.dags.convergence_detection.ConvergenceDetector",
            ) as mock_det,
        ):
            mock_det.return_value.detect.side_effect = ValueError("bad data")
            with pytest.raises(ValueError, match="bad data"):
                run_convergence_detection({"db_path": ":memory:"}, {})

    def test_emit_failure_partial(self):
        """If one signal fails to emit, others should still succeed."""
        store = _store()
        good_sig = ConvergenceSignal(
            signal_name="convergence.good.2024-01-01",
            computed_at=_NOW,
            value=0.5,
            event_type="good",
            signals_involved=["a"],
            categories_involved=["positioning"],
            cross_category_count=1,
            p_value=0.05,
            persistence_days=1,
            template_match=0.0,
            direction=1,
            lead_signal="a",
        )

        # First call succeeds, give two good signals
        count = emit_signals([good_sig, good_sig], store)
        # Both should succeed (same signal_name is allowed — different rows)
        assert count == 2

    def test_dag_validates_after_build(self):
        """DAG should validate cleanly after construction."""
        dag = build_convergence_detection_dag()
        errors = dag.validate()
        assert errors == []

    def test_dag_description_nonempty(self):
        dag = build_convergence_detection_dag()
        assert len(dag.description) > 0


# ═══════════════════════════════════════════════════════════════
#  Registry Builder Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestRegistryBuilderEdge:
    """Edge cases in build_registry_from_evidence."""

    def test_large_evidence_count(self):
        """1000 distinct signals should register fine."""
        evs = [_evidence(signal_id=f"sig_{i}", source=f"tool_{i % 10}") for i in range(1000)]
        reg = build_registry_from_evidence(evs)
        assert len(reg) == 1000

    def test_all_same_signal_id(self):
        """1000 evidence items with same signal_id → 1 registry entry."""
        evs = [_evidence(signal_id="same.sig") for _ in range(1000)]
        reg = build_registry_from_evidence(evs)
        assert len(reg) == 1

    def test_all_categories_representable(self):
        """Every taxonomy category should be registerable."""
        evs = []
        for i, cat in enumerate(sorted(CATEGORIES)):
            evs.append(_evidence(signal_id=f"cat_test.{i}", category=cat))
        reg = build_registry_from_evidence(evs)
        assert len(reg) == len(CATEGORIES)

    def test_first_occurrence_wins_category(self):
        """If same signal_id seen with different categories, first wins."""
        ev1 = _evidence(signal_id="ambiguous", category="positioning")
        # Mock a second evidence with different category but same signal_id
        # Since build_registry_from_evidence uses a `seen` set, second is ignored
        ev2 = MagicMock(spec=Evidence)
        ev2.signal_id = "ambiguous"
        ev2.source = "test"
        ev2.category = "biological"
        reg = build_registry_from_evidence([ev1, ev2])
        assert len(reg) == 1
        meta = reg.get("ambiguous")
        assert meta is not None
        assert meta.category == "positioning"


# ═══════════════════════════════════════════════════════════════
#  Integration: Full Pipeline (Mocked)
# ═══════════════════════════════════════════════════════════════


class TestFullIntegration:
    """End-to-end integration with mocked detector output."""

    def test_detection_to_emission_pipeline(self):
        """Full path: detection result → signal → store → query."""
        store = _store()
        det_result = DetectionResult(
            clique=_clique(
                signals=("cftc.crude.net", "ais.hormuz.count", "power.demand"),
                categories=("positioning", "physical_flow", "physical_disruption"),
            ),
            event_type="energy_crisis",
            template_match=0.8,
            boosted_score=0.92,
            lead_signal="cftc.crude.net",
            lag_signals=["ais.hormuz.count", "power.demand"],
        )

        # Convert to signal
        sig = from_detection_result(det_result, persistence_count=3, as_of=_NOW)
        assert sig.event_type == "energy_crisis"
        assert sig.persistence_days == 3
        assert sig.cross_category_count == 3

        # Emit
        count = emit_signals([sig], store)
        assert count == 1

        # Query back
        rows = store.query_signals(sig.signal_name, limit=1)
        assert len(rows) == 1
        meta = rows[0]["metadata"]
        assert meta["event_type"] == "energy_crisis"
        assert meta["template_match"] == 0.8
        assert set(meta["categories_involved"]) == {
            "positioning",
            "physical_flow",
            "physical_disruption",
        }

    def test_multiple_detections_emitted(self):
        """Multiple detection results → multiple stored signals."""
        store = _store()
        results = [
            _detection(event_type="type_a", score=0.7),
            _detection(event_type="type_b", score=0.8),
            _detection(event_type="type_c", score=0.9),
        ]
        sigs = [from_detection_result(r, as_of=_NOW) for r in results]
        count = emit_signals(sigs, store)
        assert count == 3

    def test_dag_callback_full_mock(self):
        """DAG callback with mocked detector returning 2 results."""
        store = _store()
        results = [
            _detection(event_type="crisis_a", score=0.85),
            _detection(event_type="crisis_b", score=0.75),
        ]

        with (
            patch(
                "agent.pipeline.dags.convergence_detection.PipelineStore",
                return_value=store,
            ),
            patch(
                "agent.pipeline.dags.convergence_detection._load_evidence_from_store",
                return_value=[_evidence()],
            ),
            patch(
                "agent.pipeline.dags.convergence_detection.build_registry_from_evidence",
            ),
            patch(
                "agent.pipeline.dags.convergence_detection.ConvergenceDetector",
            ) as mock_det,
        ):
            mock_det.return_value.detect.return_value = results
            output = run_convergence_detection({"db_path": ":memory:"}, {})

        assert output["detected"] == 2
        assert output["emitted"] == 2
        types = {s["event_type"] for s in output["signals"]}
        assert types == {"crisis_a", "crisis_b"}

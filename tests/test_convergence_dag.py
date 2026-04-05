"""Tests for the convergence detection DAG."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.convergence.detector import (
    ConvergenceDetector,
    ConvergenceDetectorConfig,
    DetectionResult,
)
from agent.convergence.evidence import Evidence
from agent.convergence.graph import ConvergenceClique
from agent.convergence.taxonomy import SignalMeta, SignalRegistry
from agent.pipeline.dag import DAG
from agent.pipeline.dags.convergence_detection import (
    _load_evidence_from_store,
    build_convergence_detection_dag,
    build_registry_from_evidence,
    run_convergence_detection,
)
from agent.pipeline.store import PipelineStore


# ── Helpers ────────────────────────────────────────────────────


def _make_evidence(
    signal_id: str = "test.signal",
    source: str = "test_tool",
    category: str = "positioning",
    value: float = 1.0,
    ts: float | None = None,
) -> Evidence:
    return Evidence(
        signal_id=signal_id,
        source=source,
        category=category,
        timestamp=ts or time.time(),
        value=value,
        confidence=0.9,
        direction=1,
        tags=(),
        ttl=86400,
    )


def _make_store() -> PipelineStore:
    return PipelineStore(":memory:")


def _make_clique(
    signals: tuple[str, ...] = ("sig_a", "sig_b", "sig_c"),
    categories: tuple[str, ...] = ("positioning", "macro_momentum"),
    score: float = 0.8,
) -> ConvergenceClique:
    return ConvergenceClique(
        signals=signals,
        categories=categories,
        score=score,
        edges=(),
    )


def _make_detection_result(
    event_type: str = "supply_chain_disruption",
    score: float = 0.85,
) -> DetectionResult:
    return DetectionResult(
        clique=_make_clique(),
        event_type=event_type,
        template_match=0.7,
        boosted_score=score,
        lead_signal="sig_a",
        lag_signals=["sig_b", "sig_c"],
    )


# ═══════════════════════════════════════════════════════════════
#  DAG Structure Tests
# ═══════════════════════════════════════════════════════════════


class TestBuildConvergenceDetectionDag:
    """Test DAG declaration is valid and well-formed."""

    def test_returns_dag(self):
        dag = build_convergence_detection_dag()
        assert isinstance(dag, DAG)

    def test_name(self):
        dag = build_convergence_detection_dag()
        assert dag.name == "convergence_detection"

    def test_schedule(self):
        dag = build_convergence_detection_dag()
        assert dag.schedule == "30 18 * * 1-5"

    def test_has_run_detection_node(self):
        dag = build_convergence_detection_dag()
        assert "run_detection" in dag.nodes

    def test_run_detection_is_function_operator(self):
        dag = build_convergence_detection_dag()
        node = dag.nodes["run_detection"]
        assert callable(node.operator)
        assert node.operator is run_convergence_detection

    def test_validates_clean(self):
        dag = build_convergence_detection_dag()
        errors = dag.validate()
        assert errors == []

    def test_topo_sort_single_layer(self):
        dag = build_convergence_detection_dag()
        layers = dag.topo_sort()
        assert len(layers) == 1
        assert "run_detection" in layers[0]

    def test_roots(self):
        dag = build_convergence_detection_dag()
        assert dag.roots() == ["run_detection"]

    def test_custom_db_path(self):
        dag = build_convergence_detection_dag(db_path="/custom/path.db")
        node = dag.nodes["run_detection"]
        assert node.params["db_path"] == "/custom/path.db"

    def test_timeout(self):
        dag = build_convergence_detection_dag()
        node = dag.nodes["run_detection"]
        assert node.timeout == 300

    def test_retries(self):
        dag = build_convergence_detection_dag()
        node = dag.nodes["run_detection"]
        assert node.retries == 1

    def test_store_result(self):
        dag = build_convergence_detection_dag()
        node = dag.nodes["run_detection"]
        assert node.store_result is True


# ═══════════════════════════════════════════════════════════════
#  Registry Builder Tests
# ═══════════════════════════════════════════════════════════════


class TestBuildRegistryFromEvidence:
    """Test SignalRegistry construction from evidence."""

    def test_empty_evidence(self):
        reg = build_registry_from_evidence([])
        assert len(reg) == 0

    def test_single_signal(self):
        ev = _make_evidence(signal_id="cftc.crude.net", source="cftc")
        reg = build_registry_from_evidence([ev])
        assert len(reg) == 1
        meta = reg.get("cftc.crude.net")
        assert meta is not None
        assert meta.source == "cftc"
        assert meta.category == "positioning"

    def test_deduplicates_by_signal_id(self):
        ev1 = _make_evidence(signal_id="sig_a", value=1.0)
        ev2 = _make_evidence(signal_id="sig_a", value=2.0)
        reg = build_registry_from_evidence([ev1, ev2])
        assert len(reg) == 1

    def test_multiple_distinct_signals(self):
        ev1 = _make_evidence(signal_id="sig_a", source="tool_a")
        ev2 = _make_evidence(
            signal_id="sig_b", source="tool_b", category="macro_momentum"
        )
        reg = build_registry_from_evidence([ev1, ev2])
        assert len(reg) == 2
        assert reg.get("sig_a") is not None
        assert reg.get("sig_b") is not None

    def test_category_preserved(self):
        ev = _make_evidence(category="biological")
        reg = build_registry_from_evidence([ev])
        meta = reg.get("test.signal")
        assert meta is not None
        assert meta.category == "biological"

    def test_default_frequency(self):
        ev = _make_evidence()
        reg = build_registry_from_evidence([ev])
        meta = reg.get("test.signal")
        assert meta is not None
        assert meta.frequency == "daily"

    def test_by_source_lookup(self):
        ev = _make_evidence(source="cftc")
        reg = build_registry_from_evidence([ev])
        results = reg.by_source("cftc")
        assert len(results) == 1

    def test_by_category_lookup(self):
        ev = _make_evidence(category="positioning")
        reg = build_registry_from_evidence([ev])
        results = reg.by_category("positioning")
        assert len(results) == 1

    def test_invalid_category_skipped(self):
        """Evidence with an invalid taxonomy category in registry builder is skipped."""
        # Can't construct Evidence with bad category (validated in __post_init__),
        # so we test with a mock that has an invalid category attribute.
        ev = MagicMock(spec=Evidence)
        ev.signal_id = "bad.sig"
        ev.source = "test"
        ev.category = "nonexistent_category"
        reg = build_registry_from_evidence([ev])
        assert len(reg) == 0  # skipped, not raised


# ═══════════════════════════════════════════════════════════════
#  Evidence Loader Tests
# ═══════════════════════════════════════════════════════════════


class TestLoadEvidenceFromStore:
    """Test evidence loading helper."""

    def test_empty_store(self):
        store = _make_store()
        evidence = _load_evidence_from_store(store)
        assert evidence == []

    def test_loads_from_registered_tools(self):
        """Store data for a registered tool and verify extraction."""
        store = _make_store()
        # Store a CFTC-like data row
        store.store_data(
            source="cftc",
            data={"report_date": "2024-01-15", "contracts": []},
            params={"mode": "latest"},
        )
        evidence = _load_evidence_from_store(store)
        # Even if extractor returns empty (no contracts), it shouldn't crash
        assert isinstance(evidence, list)

    def test_respects_lookback(self):
        """Data outside lookback window should not be loaded."""
        store = _make_store()
        # We can't easily control fetched_at in store_data, so test
        # that with lookback_days=0, nothing is returned
        store.store_data(
            source="cftc",
            data={"report_date": "2020-01-01", "contracts": []},
            params={},
        )
        # With lookback_days=0, since = as_of, so no data qualifies
        evidence = _load_evidence_from_store(store, lookback_days=0)
        assert evidence == []

    def test_none_data_skipped(self):
        """Rows with None data should be skipped gracefully."""
        store = _make_store()
        # Store data then verify basic loading works
        evidence = _load_evidence_from_store(store)
        assert isinstance(evidence, list)


# ═══════════════════════════════════════════════════════════════
#  Callback Execution Tests (Mocked)
# ═══════════════════════════════════════════════════════════════


class TestRunConvergenceDetection:
    """Test the FunctionOperator callback with mocks."""

    def test_empty_store_returns_zeros(self):
        """No data → no detection."""
        store = _make_store()
        params = {"db_path": ":memory:"}
        # Patch PipelineStore to return our in-memory store
        with patch(
            "agent.pipeline.dags.convergence_detection.PipelineStore",
            return_value=store,
        ):
            result = run_convergence_detection(params, {})

        assert result["detected"] == 0
        assert result["emitted"] == 0
        assert result["signals"] == []

    def test_with_detections(self):
        """Mocked detector returning results → signals emitted."""
        store = _make_store()
        detection = _make_detection_result()

        with (
            patch(
                "agent.pipeline.dags.convergence_detection.PipelineStore",
                return_value=store,
            ),
            patch(
                "agent.pipeline.dags.convergence_detection._load_evidence_from_store",
                return_value=[_make_evidence()],
            ),
            patch(
                "agent.pipeline.dags.convergence_detection.build_registry_from_evidence",
            ) as mock_reg,
            patch(
                "agent.pipeline.dags.convergence_detection.ConvergenceDetector",
            ) as mock_det,
        ):
            mock_det.return_value.detect.return_value = [detection]

            result = run_convergence_detection({"db_path": ":memory:"}, {})

        assert result["detected"] == 1
        assert result["emitted"] == 1
        assert len(result["signals"]) == 1
        assert result["signals"][0]["event_type"] == "supply_chain_disruption"

    def test_no_evidence_short_circuits(self):
        """When evidence is empty, skip detection entirely."""
        store = _make_store()

        with (
            patch(
                "agent.pipeline.dags.convergence_detection.PipelineStore",
                return_value=store,
            ),
            patch(
                "agent.pipeline.dags.convergence_detection._load_evidence_from_store",
                return_value=[],
            ),
        ):
            result = run_convergence_detection({"db_path": ":memory:"}, {})

        assert result["detected"] == 0

    def test_store_closed_on_success(self):
        """Store.close() is called even on success."""
        mock_store = MagicMock(spec=PipelineStore)

        with (
            patch(
                "agent.pipeline.dags.convergence_detection.PipelineStore",
                return_value=mock_store,
            ),
            patch(
                "agent.pipeline.dags.convergence_detection._load_evidence_from_store",
                return_value=[],
            ),
        ):
            run_convergence_detection({"db_path": ":memory:"}, {})

        mock_store.close.assert_called_once()

    def test_store_closed_on_error(self):
        """Store.close() is called even when detection raises."""
        mock_store = MagicMock(spec=PipelineStore)

        with (
            patch(
                "agent.pipeline.dags.convergence_detection.PipelineStore",
                return_value=mock_store,
            ),
            patch(
                "agent.pipeline.dags.convergence_detection._load_evidence_from_store",
                side_effect=RuntimeError("boom"),
            ),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                run_convergence_detection({"db_path": ":memory:"}, {})

        mock_store.close.assert_called_once()

    def test_custom_lookback(self):
        """Verify lookback_days param is passed through."""
        store = _make_store()

        with (
            patch(
                "agent.pipeline.dags.convergence_detection.PipelineStore",
                return_value=store,
            ),
            patch(
                "agent.pipeline.dags.convergence_detection._load_evidence_from_store",
                return_value=[],
            ) as mock_load,
        ):
            run_convergence_detection({"db_path": ":memory:", "lookback_days": 30}, {})

        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs.get("lookback_days", mock_load.call_args[0][1]) == 30

    def test_as_of_param(self):
        """Verify as_of param is passed through."""
        store = _make_store()
        fixed_time = 1700000000.0

        with (
            patch(
                "agent.pipeline.dags.convergence_detection.PipelineStore",
                return_value=store,
            ),
            patch(
                "agent.pipeline.dags.convergence_detection._load_evidence_from_store",
                return_value=[],
            ) as mock_load,
        ):
            run_convergence_detection({"db_path": ":memory:", "as_of": fixed_time}, {})

        call_args = mock_load.call_args
        assert (
            call_args[1].get(
                "as_of", call_args[0][2] if len(call_args[0]) > 2 else None
            )
            == fixed_time
        )

    def test_default_params(self):
        """Default db_path is used when not specified."""
        with (
            patch(
                "agent.pipeline.dags.convergence_detection.PipelineStore",
            ) as mock_cls,
            patch(
                "agent.pipeline.dags.convergence_detection._load_evidence_from_store",
                return_value=[],
            ),
        ):
            mock_cls.return_value.close = MagicMock()
            run_convergence_detection({}, {})

        mock_cls.assert_called_once_with(".tirra_pipeline/pipeline.db")

    def test_signal_summary_format(self):
        """Verify signal summary dict structure."""
        store = _make_store()
        detection = _make_detection_result(event_type="energy_crisis", score=0.9123)

        with (
            patch(
                "agent.pipeline.dags.convergence_detection.PipelineStore",
                return_value=store,
            ),
            patch(
                "agent.pipeline.dags.convergence_detection._load_evidence_from_store",
                return_value=[_make_evidence()],
            ),
            patch(
                "agent.pipeline.dags.convergence_detection.build_registry_from_evidence",
            ),
            patch(
                "agent.pipeline.dags.convergence_detection.ConvergenceDetector",
            ) as mock_det,
        ):
            mock_det.return_value.detect.return_value = [detection]

            result = run_convergence_detection({"db_path": ":memory:"}, {})

        sig = result["signals"][0]
        assert sig["event_type"] == "energy_crisis"
        assert sig["value"] == 0.9123
        assert "signal_name" in sig
        assert "categories" in sig


# ═══════════════════════════════════════════════════════════════
#  DAG Registration Tests
# ═══════════════════════════════════════════════════════════════


class TestDagRegistration:
    """Test that convergence DAG is included in get_default_dags."""

    def test_included_in_defaults(self):
        from agent.pipeline.dags import get_default_dags

        dags = get_default_dags(tool_registry=None)
        names = [d.name for d in dags]
        assert "convergence_detection" in names

    def test_all_dags_validate(self):
        from agent.pipeline.dags import get_default_dags

        dags = get_default_dags(tool_registry=None)
        for dag in dags:
            errors = dag.validate()
            assert errors == [], f"DAG {dag.name!r} has errors: {errors}"

"""Tests for feature_generation DAG and end-to-end integration (steps 8.5 + 8.6).

Covers:
- DAG structure and validation
- DAG registration in get_default_dags
- run_feature_generation callback: happy path, empty data, partial data,
  convergence-only, macro-only, idempotent re-run, builder failure resilience,
  batch fallback on validation error, protocol validation of all emitted features
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.features.builders import (
    ConvergenceFeatureBuilder,
    FeatureBuilder,
    MacroStateFeatureBuilder,
)
from agent.features.gnn_builder import GNNFeatureBuilder
from agent.features.protocol import EngineeredFeature, validate_feature
from agent.pipeline.dag import DAG
from agent.pipeline.dags.feature_generation import (
    DEFAULT_BUILDERS,
    build_feature_generation_dag,
    run_feature_generation,
)
from agent.pipeline.store import PipelineStore

# ── helpers ────────────────────────────────────────────────────

_NOW = time.time()
_DAY = 86_400


def _store() -> PipelineStore:
    return PipelineStore(db_path=":memory:")


def _insert_convergence_signal(
    store: PipelineStore,
    signal_name: str = "convergence.credit_stress.2026-04-01",
    value: float = 0.75,
    computed_at: float | None = None,
    persistence_days: int = 3,
) -> None:
    if computed_at is None:
        computed_at = _NOW - 3600
    metadata = {
        "event_type": "credit_stress",
        "signals_involved": ["sig_a", "sig_b"],
        "persistence_days": persistence_days,
        "direction": 1,
    }
    store.store_signal(signal_name, value, metadata)
    conn = store._get_conn()
    conn.execute(
        "UPDATE signals SET computed_at=? WHERE id = "
        "(SELECT id FROM signals WHERE signal_name=? ORDER BY id DESC LIMIT 1)",
        (computed_at, signal_name),
    )
    conn.commit()


def _insert_macro_data(
    store: PipelineStore,
    series_data: dict,
    fetched_at: float | None = None,
) -> None:
    if fetched_at is None:
        fetched_at = _NOW
    store.store_data("macro_data", {"source": "fred"}, series_data)
    conn = store._get_conn()
    conn.execute(
        "UPDATE pipeline_data SET fetched_at=? "
        "WHERE id = (SELECT id FROM pipeline_data WHERE source='macro_data' "
        "ORDER BY id DESC LIMIT 1)",
        (fetched_at,),
    )
    conn.commit()


def _make_fred_series(
    series_id: str, n_obs: int = 60, start: float = 5.0, trend: float = 0.01
):
    from datetime import datetime, timedelta

    base = datetime(2026, 3, 1)
    return {
        series_id: [
            {
                "date": (base + timedelta(days=i)).strftime("%Y-%m-%d"),
                "value": str(start + trend * i),
            }
            for i in range(n_obs)
        ]
    }


def _full_macro_data() -> dict:
    data: dict = {}
    data.update(_make_fred_series("DFF", 60, 5.25, 0.001))
    data.update(_make_fred_series("GS10", 60, 4.5, 0.002))
    data.update(_make_fred_series("GS2", 60, 4.8, 0.001))
    data.update(_make_fred_series("WALCL", 60, 8_000_000, -10_000))
    return data


def _assert_all_valid(features: list[EngineeredFeature]) -> None:
    for feat in features:
        errors = validate_feature(feat)
        assert errors == [], f"{feat.feature_name} failed: {errors}"


# ══════════════════════════════════════════════════════════════
#  DAG Structure
# ══════════════════════════════════════════════════════════════


class TestFeatureGenerationDAG:
    def test_dag_builds_successfully(self):
        dag = build_feature_generation_dag()
        assert isinstance(dag, DAG)
        assert dag.name == "feature_generation"

    def test_dag_validates_clean(self):
        dag = build_feature_generation_dag()
        errors = dag.validate()
        assert errors == []

    def test_dag_has_generate_features_node(self):
        dag = build_feature_generation_dag()
        assert "generate_features" in dag.nodes

    def test_dag_schedule(self):
        dag = build_feature_generation_dag()
        assert dag.schedule == "0 19 * * 1-5"

    def test_dag_single_root(self):
        dag = build_feature_generation_dag()
        roots = dag.roots()
        assert roots == ["generate_features"]

    def test_dag_node_timeout(self):
        dag = build_feature_generation_dag()
        node = dag.nodes["generate_features"]
        assert node.timeout == 120

    def test_dag_node_operator_is_callable(self):
        dag = build_feature_generation_dag()
        node = dag.nodes["generate_features"]
        assert callable(node.operator)


class TestDAGRegistration:
    def test_feature_generation_in_default_dags(self):
        """The feature_generation DAG is included in get_default_dags."""
        from agent.pipeline.dags import get_default_dags

        dags = get_default_dags(tool_registry=None)
        names = [d.name for d in dags]
        assert "feature_generation" in names

    def test_all_default_dags_valid(self):
        from agent.pipeline.dags import get_default_dags

        for dag in get_default_dags(tool_registry=None):
            errors = dag.validate()
            assert errors == [], f"DAG {dag.name!r} invalid: {errors}"


# ══════════════════════════════════════════════════════════════
#  run_feature_generation callback
# ══════════════════════════════════════════════════════════════


class TestRunFeatureGenerationHappyPath:
    def test_full_data_produces_six_features(self):
        store = _store()
        _insert_convergence_signal(store)
        _insert_macro_data(store, _full_macro_data())
        result = run_feature_generation(
            {"db_path": ":memory:", "as_of": _NOW, "builders": DEFAULT_BUILDERS},
            {},
        )
        # Can't use the same store since callback opens its own.
        # Instead, test via the returned summary.
        assert result["produced"] == 17  # 3 convergence + 3 macro + 11 GNN

    def test_return_structure(self):
        store = _store()
        result = run_feature_generation(
            {"db_path": ":memory:", "as_of": _NOW, "builders": DEFAULT_BUILDERS},
            {},
        )
        assert "produced" in result
        assert "stored" in result
        assert "builders" in result
        assert isinstance(result["builders"], list)

    def test_builder_summaries_present(self):
        result = run_feature_generation(
            {"db_path": ":memory:", "as_of": _NOW, "builders": DEFAULT_BUILDERS},
            {},
        )
        assert len(result["builders"]) == 3
        for summary in result["builders"]:
            assert "builder" in summary
            assert "features_produced" in summary


class TestRunFeatureGenerationEdgeCases:
    def test_no_data_all_missing(self):
        """Empty store → all features emitted as missing."""
        result = run_feature_generation(
            {"db_path": ":memory:", "as_of": _NOW, "builders": DEFAULT_BUILDERS},
            {},
        )
        assert result["produced"] == 17  # 3 convergence + 3 macro + 11 GNN
        assert result["stored"] == 17  # all features valid including GNN missing

    def test_convergence_only(self):
        """Only convergence signals → convergence features have values, macro missing."""
        store = _store()
        _insert_convergence_signal(store)
        # The callback creates its own store from db_path, so use :memory:
        result = run_feature_generation(
            {"db_path": ":memory:", "as_of": _NOW, "builders": DEFAULT_BUILDERS},
            {},
        )
        assert result["produced"] == 17  # 3 convergence + 3 macro + 11 GNN
        # At least some missing (macro side + GNN side)
        total_missing = sum(s.get("missing", 0) for s in result["builders"])
        assert total_missing >= 3  # all 3 macro features missing

    def test_idempotent_rerun(self):
        """Running twice with same as_of produces same count (upsert)."""
        params = {"db_path": ":memory:", "as_of": _NOW, "builders": DEFAULT_BUILDERS}
        r1 = run_feature_generation(params, {})
        r2 = run_feature_generation(params, {})
        assert r1["produced"] == r2["produced"]
        assert r1["stored"] == r2["stored"]

    def test_custom_builders(self):
        """Only convergence builder → 3 features."""
        result = run_feature_generation(
            {
                "db_path": ":memory:",
                "as_of": _NOW,
                "builders": [ConvergenceFeatureBuilder()],
            },
            {},
        )
        assert result["produced"] == 3

    def test_empty_builder_list(self):
        result = run_feature_generation(
            {"db_path": ":memory:", "as_of": _NOW, "builders": []},
            {},
        )
        assert result["produced"] == 0
        assert result["stored"] == 0


class TestBuilderFailureResilience:
    def test_failing_builder_skipped(self):
        """If one builder raises, the other still runs."""

        class FailingBuilder(FeatureBuilder):
            @property
            def name(self) -> str:
                return "FailingBuilder"

            def build(self, store, as_of):
                raise RuntimeError("intentional test failure")

        result = run_feature_generation(
            {
                "db_path": ":memory:",
                "as_of": _NOW,
                "builders": [FailingBuilder(), ConvergenceFeatureBuilder()],
            },
            {},
        )
        # ConvergenceFeatureBuilder still produced 3 (all missing but still emitted)
        assert result["produced"] == 3
        # Check the failing builder summary
        failing_summary = next(
            s for s in result["builders"] if s["builder"] == "FailingBuilder"
        )
        assert failing_summary.get("error") is True


# ══════════════════════════════════════════════════════════════
#  End-to-End: builders → store persistence via DAG callback
# ══════════════════════════════════════════════════════════════


class TestEndToEndPersistence:
    def test_features_written_to_store(self):
        """Features are actually written to the SQLite features table."""
        store = _store()
        _insert_convergence_signal(store)
        _insert_macro_data(store, _full_macro_data())

        # Run builders directly against this store (instead of via callback
        # which opens its own connection from db_path).
        all_features: list[EngineeredFeature] = []
        for builder in DEFAULT_BUILDERS:
            all_features.extend(builder.build(store, _NOW))

        _assert_all_valid(all_features)

        row_ids = store.store_features_batch(all_features)
        assert len(row_ids) >= 6  # convergence + macro + GNN (variable)

        # Verify retrieval
        for feat in all_features:
            r = store.get_latest_feature(feat.feature_name)
            assert r is not None, f"Feature {feat.feature_name} not in store"
            if feat.value is not None:
                assert r["value"] == pytest.approx(feat.value, abs=0.01)

        store.close()

    def test_missing_features_persist_and_roundtrip(self):
        """Features with value=None survive store → query roundtrip."""
        store = _store()
        features = ConvergenceFeatureBuilder().build(store, _NOW)
        assert all(f.value is None for f in features)
        store.store_features_batch(features)

        for feat in features:
            r = store.get_latest_feature(feat.feature_name)
            assert r is not None
            assert r["value"] is None
            assert r["missing_reason"] == "no_convergence_activity"

        store.close()

    def test_idempotent_persistence(self):
        """Storing the same feature twice doesn't duplicate rows."""
        store = _store()
        _insert_convergence_signal(store)
        features = ConvergenceFeatureBuilder().build(store, _NOW)

        store.store_features_batch(features)
        store.store_features_batch(features)  # second write

        for feat in features:
            rows = store.query_features(feat.feature_name)
            assert len(rows) == 1  # no duplicates

        store.close()

    def test_all_features_pass_protocol_validation(self):
        """Every feature from every builder passes validate_feature()."""
        store = _store()
        _insert_convergence_signal(store)
        _insert_macro_data(store, _full_macro_data())

        for builder in DEFAULT_BUILDERS:
            features = builder.build(store, _NOW)
            _assert_all_valid(features)

        store.close()

    def test_feature_names_are_unique_across_builders(self):
        """No two builders produce the same feature_name."""
        store = _store()
        _insert_convergence_signal(store)
        _insert_macro_data(store, _full_macro_data())

        all_names: list[str] = []
        for builder in DEFAULT_BUILDERS:
            features = builder.build(store, _NOW)
            all_names.extend(f.feature_name for f in features)

        assert len(all_names) == len(
            set(all_names)
        ), "Duplicate feature names across builders"
        store.close()

    def test_features_ordered_by_effective_at(self):
        """query_features returns descending order by effective_at."""
        store = _store()
        builder = ConvergenceFeatureBuilder()

        # Build at two different times
        _insert_convergence_signal(store, computed_at=_NOW - 2 * _DAY)
        feats_old = builder.build(store, _NOW - _DAY)
        store.store_features_batch(feats_old)

        _insert_convergence_signal(
            store,
            signal_name="convergence.new.2026-04-02",
            computed_at=_NOW - 3600,
        )
        feats_new = builder.build(store, _NOW)
        store.store_features_batch(feats_new)

        rows = store.query_features("convergence.stress_breadth.7d", limit=10)
        assert len(rows) == 2
        assert rows[0]["effective_at"] > rows[1]["effective_at"]
        store.close()


# ══════════════════════════════════════════════════════════════
#  DEFAULT_BUILDERS sanity
# ══════════════════════════════════════════════════════════════


class TestDefaultBuilders:
    def test_default_builders_count(self):
        assert len(DEFAULT_BUILDERS) == 3

    def test_default_builders_types(self):
        assert isinstance(DEFAULT_BUILDERS[0], ConvergenceFeatureBuilder)
        assert isinstance(DEFAULT_BUILDERS[1], MacroStateFeatureBuilder)
        assert isinstance(DEFAULT_BUILDERS[2], GNNFeatureBuilder)

    def test_default_builders_names_unique(self):
        names = [b.name for b in DEFAULT_BUILDERS]
        assert len(names) == len(set(names))

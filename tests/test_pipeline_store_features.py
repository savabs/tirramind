"""Tests for PipelineStore engineered-feature persistence (step 8.2).

Covers: store_feature, store_features_batch, query_features, get_latest_feature,
schema creation, idempotent upsert, validation enforcement, edge cases.
"""

from __future__ import annotations

import json
import math
import time

import pytest

from agent.features.protocol import EngineeredFeature, VALID_HORIZONS, VALID_UNITS
from agent.pipeline.store import PipelineStore


# ── helpers ────────────────────────────────────────────────────

_NOW = time.time()


def _make_feature(**overrides) -> EngineeredFeature:
    """Build a valid EngineeredFeature with sensible defaults."""
    defaults = {
        "feature_name": "convergence.stress_breadth.7d",
        "version": 1,
        "effective_at": _NOW - 3600,
        "computed_at": _NOW,
        "horizon": "7d",
        "value": 1.23,
        "quality": 0.9,
        "source_signals": ("convergence.macro_stress",),
        "builder": "StressBreadthBuilder",
        "unit": "z_score",
    }
    defaults.update(overrides)
    return EngineeredFeature(**defaults)


# ── fixtures ───────────────────────────────────────────────────


@pytest.fixture
def store():
    s = PipelineStore(db_path=":memory:")
    yield s
    s.close()


# ── schema ─────────────────────────────────────────────────────


class TestFeaturesSchema:
    def test_features_table_exists(self, store: PipelineStore):
        conn = store._get_conn()
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "features" in tables

    def test_features_indexes_exist(self, store: PipelineStore):
        conn = store._get_conn()
        indexes = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        ]
        assert "idx_features_unique" in indexes
        assert "idx_features_lookup" in indexes

    def test_features_columns(self, store: PipelineStore):
        conn = store._get_conn()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(features)").fetchall()]
        expected = {
            "id",
            "feature_name",
            "version",
            "effective_at",
            "computed_at",
            "horizon",
            "value",
            "quality",
            "missing_reason",
            "source_signals_json",
            "builder",
            "unit",
            "metadata_json",
        }
        assert expected == set(cols)


# ── store_feature ──────────────────────────────────────────────


class TestStoreFeature:
    def test_basic_store_and_roundtrip(self, store: PipelineStore):
        feat = _make_feature()
        row_id = store.store_feature(feat)
        assert isinstance(row_id, int)

        rows = store.query_features("convergence.stress_breadth.7d")
        assert len(rows) == 1
        r = rows[0]
        assert r["feature_name"] == "convergence.stress_breadth.7d"
        assert r["version"] == 1
        assert r["value"] == pytest.approx(1.23)
        assert r["quality"] == pytest.approx(0.9)
        assert r["horizon"] == "7d"
        assert r["unit"] == "z_score"
        assert r["builder"] == "StressBreadthBuilder"
        assert r["source_signals"] == ("convergence.macro_stress",)
        assert r["missing_reason"] is None
        assert r["metadata"] is None

    def test_store_with_metadata(self, store: PipelineStore):
        feat = _make_feature(metadata={"debug": True, "n_inputs": 5})
        store.store_feature(feat)
        r = store.query_features("convergence.stress_breadth.7d")[0]
        assert r["metadata"] == {"debug": True, "n_inputs": 5}

    def test_store_missing_value(self, store: PipelineStore):
        feat = _make_feature(
            value=None,
            missing_reason="upstream_stale",
            quality=0.0,
        )
        store.store_feature(feat)
        r = store.query_features("convergence.stress_breadth.7d")[0]
        assert r["value"] is None
        assert r["missing_reason"] == "upstream_stale"

    def test_idempotent_upsert(self, store: PipelineStore):
        """Same (feature_name, version, effective_at) replaces, not duplicates."""
        feat = _make_feature(value=1.0)
        store.store_feature(feat)
        # Store again with updated value at same key
        feat2 = _make_feature(value=2.0)
        store.store_feature(feat2)

        rows = store.query_features("convergence.stress_breadth.7d")
        assert len(rows) == 1
        assert rows[0]["value"] == pytest.approx(2.0)

    def test_different_versions_coexist(self, store: PipelineStore):
        store.store_feature(_make_feature(version=1, value=1.0))
        store.store_feature(_make_feature(version=2, value=2.0))
        rows = store.query_features("convergence.stress_breadth.7d")
        assert len(rows) == 2

    def test_different_effective_at_coexist(self, store: PipelineStore):
        store.store_feature(_make_feature(effective_at=_NOW - 7200))
        store.store_feature(_make_feature(effective_at=_NOW - 3600))
        rows = store.query_features("convergence.stress_breadth.7d")
        assert len(rows) == 2

    def test_validation_rejects_invalid(self, store: PipelineStore):
        """Invalid features raise ValueError, nothing written."""
        bad = _make_feature(feature_name="BAD NAME")
        with pytest.raises(ValueError, match="failed validation"):
            store.store_feature(bad)
        rows = store.query_features("BAD NAME")
        assert rows == []

    def test_validation_rejects_nan_value(self, store: PipelineStore):
        bad = _make_feature(value=float("nan"))
        with pytest.raises(ValueError, match="failed validation"):
            store.store_feature(bad)

    def test_validation_rejects_inf_value(self, store: PipelineStore):
        bad = _make_feature(value=float("inf"))
        with pytest.raises(ValueError, match="failed validation"):
            store.store_feature(bad)

    def test_validation_rejects_missing_without_reason(self, store: PipelineStore):
        bad = _make_feature(value=None)
        with pytest.raises(ValueError, match="failed validation"):
            store.store_feature(bad)

    def test_validation_rejects_future_leakage(self, store: PipelineStore):
        bad = _make_feature(
            effective_at=_NOW + 10,
            computed_at=_NOW - 10,
        )
        with pytest.raises(ValueError, match="failed validation"):
            store.store_feature(bad)

    def test_store_with_multiple_source_signals(self, store: PipelineStore):
        feat = _make_feature(
            source_signals=("sig_a", "sig_b", "sig_c"),
        )
        store.store_feature(feat)
        r = store.query_features("convergence.stress_breadth.7d")[0]
        assert r["source_signals"] == ("sig_a", "sig_b", "sig_c")

    def test_store_zero_value(self, store: PipelineStore):
        feat = _make_feature(value=0.0)
        store.store_feature(feat)
        r = store.query_features("convergence.stress_breadth.7d")[0]
        assert r["value"] == 0.0

    def test_store_negative_value(self, store: PipelineStore):
        feat = _make_feature(value=-3.14)
        store.store_feature(feat)
        r = store.query_features("convergence.stress_breadth.7d")[0]
        assert r["value"] == pytest.approx(-3.14)

    def test_store_quality_boundary_zero(self, store: PipelineStore):
        feat = _make_feature(quality=0.0, value=None, missing_reason="low_quality")
        store.store_feature(feat)
        r = store.query_features("convergence.stress_breadth.7d")[0]
        assert r["quality"] == 0.0

    def test_store_quality_boundary_one(self, store: PipelineStore):
        feat = _make_feature(quality=1.0)
        store.store_feature(feat)
        r = store.query_features("convergence.stress_breadth.7d")[0]
        assert r["quality"] == 1.0


# ── store_features_batch ───────────────────────────────────────


class TestStoreFeaturesBatch:
    def test_batch_store(self, store: PipelineStore):
        feats = [
            _make_feature(
                feature_name="alpha.metric.7d",
                effective_at=_NOW - 3600 * i,
            )
            for i in range(5)
        ]
        ids = store.store_features_batch(feats)
        assert len(ids) == 5
        rows = store.query_features("alpha.metric.7d", limit=10)
        assert len(rows) == 5

    def test_batch_empty(self, store: PipelineStore):
        ids = store.store_features_batch([])
        assert ids == []

    def test_batch_validation_fails_atomically(self, store: PipelineStore):
        """One bad feature in a batch should reject the whole batch."""
        feats = [
            _make_feature(effective_at=_NOW - 3600),
            _make_feature(feature_name="BAD NAME", effective_at=_NOW - 1800),
            _make_feature(effective_at=_NOW - 900),
        ]
        with pytest.raises(ValueError, match="Batch validation failed"):
            store.store_features_batch(feats)
        # Nothing should be written
        rows = store.query_features("convergence.stress_breadth.7d")
        assert rows == []

    def test_batch_idempotent_upsert(self, store: PipelineStore):
        feat = _make_feature(value=1.0)
        store.store_feature(feat)
        # Batch with same key but different value
        feats = [_make_feature(value=99.0)]
        store.store_features_batch(feats)
        rows = store.query_features("convergence.stress_breadth.7d")
        assert len(rows) == 1
        assert rows[0]["value"] == pytest.approx(99.0)

    def test_batch_multiple_names(self, store: PipelineStore):
        feats = [
            _make_feature(feature_name="alpha.metric.7d"),
            _make_feature(feature_name="beta.metric.30d", horizon="30d"),
        ]
        store.store_features_batch(feats)
        assert len(store.query_features("alpha.metric.7d")) == 1
        assert len(store.query_features("beta.metric.30d")) == 1


# ── query_features ─────────────────────────────────────────────


class TestQueryFeatures:
    def test_query_since(self, store: PipelineStore):
        store.store_feature(_make_feature(effective_at=_NOW - 7200, value=1.0))
        store.store_feature(_make_feature(effective_at=_NOW - 3600, value=2.0))
        rows = store.query_features("convergence.stress_breadth.7d", since=_NOW - 5000)
        assert len(rows) == 1
        assert rows[0]["value"] == pytest.approx(2.0)

    def test_query_until(self, store: PipelineStore):
        store.store_feature(_make_feature(effective_at=_NOW - 7200, value=1.0))
        store.store_feature(_make_feature(effective_at=_NOW - 3600, value=2.0))
        rows = store.query_features("convergence.stress_breadth.7d", until=_NOW - 5000)
        assert len(rows) == 1
        assert rows[0]["value"] == pytest.approx(1.0)

    def test_query_since_and_until(self, store: PipelineStore):
        for i in range(5):
            store.store_feature(
                _make_feature(effective_at=_NOW - 1000 * i, value=float(i))
            )
        rows = store.query_features(
            "convergence.stress_breadth.7d",
            since=_NOW - 3500,
            until=_NOW - 500,
        )
        assert len(rows) == 3  # i=1,2,3

    def test_query_version_filter(self, store: PipelineStore):
        store.store_feature(_make_feature(version=1, value=1.0))
        store.store_feature(_make_feature(version=2, value=2.0))
        rows = store.query_features("convergence.stress_breadth.7d", version=2)
        assert len(rows) == 1
        assert rows[0]["version"] == 2

    def test_query_limit(self, store: PipelineStore):
        for i in range(10):
            store.store_feature(
                _make_feature(effective_at=_NOW - 100 * i, value=float(i))
            )
        rows = store.query_features("convergence.stress_breadth.7d", limit=3)
        assert len(rows) == 3

    def test_query_order_descending(self, store: PipelineStore):
        store.store_feature(_make_feature(effective_at=_NOW - 7200, value=1.0))
        store.store_feature(_make_feature(effective_at=_NOW - 3600, value=2.0))
        rows = store.query_features("convergence.stress_breadth.7d")
        assert rows[0]["effective_at"] > rows[1]["effective_at"]

    def test_query_nonexistent_name(self, store: PipelineStore):
        rows = store.query_features("does_not.exist.spot")
        assert rows == []


# ── get_latest_feature ─────────────────────────────────────────


class TestGetLatestFeature:
    def test_returns_latest(self, store: PipelineStore):
        store.store_feature(_make_feature(effective_at=_NOW - 7200, value=1.0))
        store.store_feature(_make_feature(effective_at=_NOW - 3600, value=2.0))
        r = store.get_latest_feature("convergence.stress_breadth.7d")
        assert r is not None
        assert r["value"] == pytest.approx(2.0)

    def test_returns_none_empty(self, store: PipelineStore):
        r = store.get_latest_feature("nope.nothing.spot")
        assert r is None

    def test_latest_with_version(self, store: PipelineStore):
        store.store_feature(_make_feature(version=1, value=1.0))
        store.store_feature(_make_feature(version=2, value=2.0))
        r = store.get_latest_feature("convergence.stress_breadth.7d", version=1)
        assert r is not None
        assert r["version"] == 1


# ── _feature_row_to_dict ──────────────────────────────────────


class TestFeatureRowToDict:
    def test_source_signals_roundtrip(self, store: PipelineStore):
        feat = _make_feature(
            source_signals=("a.b", "c.d"),
            metadata={"x": 1},
        )
        store.store_feature(feat)
        r = store.query_features("convergence.stress_breadth.7d")[0]
        assert isinstance(r["source_signals"], tuple)
        assert r["source_signals"] == ("a.b", "c.d")
        assert r["metadata"] == {"x": 1}
        # original json columns should be removed
        assert "source_signals_json" not in r
        assert "metadata_json" not in r

    def test_corrupt_json_graceful(self, store: PipelineStore):
        """Manually corrupt JSON and ensure dict helper doesn't crash."""
        feat = _make_feature()
        store.store_feature(feat)
        # Manually corrupt the stored JSON
        conn = store._get_conn()
        conn.execute(
            "UPDATE features SET source_signals_json='NOT JSON', "
            "metadata_json='NOT JSON' WHERE feature_name=?",
            (feat.feature_name,),
        )
        conn.commit()
        r = store.query_features("convergence.stress_breadth.7d")[0]
        assert r["source_signals"] == ()
        assert r["metadata"] is None


# ── Coexistence with existing tables ──────────────────────────


class TestCoexistence:
    def test_existing_tables_still_work(self, store: PipelineStore):
        """Ensure signals / pipeline_data / dag_runs still function."""
        sid = store.store_signal("test_sig", 3.14)
        assert isinstance(sid, int)
        did = store.store_data("test_src", {}, {"ok": True})
        assert isinstance(did, int)
        rid = store.record_run_start("test_dag")
        assert isinstance(rid, str)

    def test_features_table_in_schema_check(self, store: PipelineStore):
        conn = store._get_conn()
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"dag_runs", "pipeline_data", "signals", "features"}.issubset(tables)


# ── All valid horizons and units persist correctly ─────────────


class TestAllEnumValues:
    @pytest.mark.parametrize("horizon", sorted(VALID_HORIZONS))
    def test_all_valid_horizons(self, store: PipelineStore, horizon: str):
        feat = _make_feature(
            feature_name=f"test.h.{horizon.replace('d', 'dd') if horizon != 'spot' else 'spot'}",
            horizon=horizon,
        )
        store.store_feature(feat)
        r = store.query_features(feat.feature_name)[0]
        assert r["horizon"] == horizon

    @pytest.mark.parametrize("unit", sorted(VALID_UNITS))
    def test_all_valid_units(self, store: PipelineStore, unit: str):
        # Build a name that passes the pattern validator
        safe = unit.replace("_", "")
        feat = _make_feature(
            feature_name=f"test.{safe}.spot",
            horizon="spot",
            unit=unit,
        )
        store.store_feature(feat)
        r = store.query_features(feat.feature_name)[0]
        assert r["unit"] == unit

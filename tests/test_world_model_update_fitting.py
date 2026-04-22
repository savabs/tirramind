"""
Integration tests for world_model_update DAG parameter fitting wiring.

Tests the periodic CPD and Kalman EM fitting integration (steps 2a.2, 2b.2),
which wires WorldModel.fit_cpds() and ContinuousStateFilter.fit_filter_params()
into the world_model_update DAG via _maybe_fit_params().

Coverage:
    _should_fit          — periodicity control via stored markers
    _load_feature_history — daily snapshot grouping from flat feature rows
    _load_regime_labels  — MAP regime extraction from stored beliefs
    _build_observation_sequence — feature→observation vector mapping
    _maybe_fit_params    — full orchestration: skip, fit, fallback, marker storage
    run_world_model_update — DAG function with fitting wired in
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.features.protocol import EngineeredFeature
from agent.models.belief import BeliefState
from agent.pipeline.dags.world_model_update import (
    _build_observation_sequence,
    _DAY_SECONDS,
    _FEATURE_TO_OBS_INDEX,
    _FIT_SOURCE,
    _load_feature_history,
    _load_regime_labels,
    _maybe_fit_params,
    _OBS_DIM,
    _should_fit,
)
from agent.pipeline.store import PipelineStore

# ── Test fixtures ──────────────────────────────────────────────

_BASE_TS = 1_700_000_000.0  # Nov 2023, well after BeliefState's 2020 floor


def _make_store(tmp_path: Path) -> PipelineStore:
    db = tmp_path / "test.db"
    return PipelineStore(str(db))


def _make_feature(
    name: str,
    value: float | None,
    effective_at: float,
    *,
    horizon: str = "spot",
    quality: float = 1.0,
) -> EngineeredFeature:
    return EngineeredFeature(
        feature_name=name,
        version=1,
        effective_at=effective_at,
        computed_at=effective_at + 1.0,
        horizon=horizon,
        value=value,
        quality=quality,
        source_signals=("test_signal",),
        builder="test",
        unit="raw",
    )


def _make_belief(
    variable_name: str,
    effective_at: float,
    probabilities: dict | None = None,
    mean: float = 0.0,
) -> BeliefState:
    return BeliefState(
        variable_name=variable_name,
        version=1,
        effective_at=effective_at,
        computed_at=effective_at + 1.0,
        dist_type="categorical" if probabilities else "gaussian",
        mean=mean,
        variance=0.1,
        probabilities=probabilities,
        evidence_count=5,
        model_graph_hash="a" * 64,
        confidence=0.9,
        stale=False,
    )


def _seed_features(
    store: PipelineStore,
    n_days: int,
    base_ts: float,
) -> list[float]:
    """Seed store with n_days of features. Returns effective_at timestamps."""
    from agent.pipeline.dags.world_model_update import _FEATURE_NAMES

    timestamps: list[float] = []
    for day in range(n_days):
        ts = base_ts + day * _DAY_SECONDS + 3600  # offset 1h into each day
        timestamps.append(ts)
        features = []
        for i, feat_name in enumerate(_FEATURE_NAMES):
            features.append(
                _make_feature(feat_name, 0.5 + 0.01 * i + 0.001 * day, ts),
            )
        store.store_features_batch(features)
    return timestamps


def _seed_regime_beliefs(
    store: PipelineStore,
    timestamps: list[float],
    regime: str = "expansion",
) -> None:
    """Store regime.macro beliefs with given MAP regime for each timestamp."""
    probs: dict[str, float]
    if regime == "contraction":
        probs = {"expansion": 0.2, "contraction": 0.7, "crisis": 0.1}
    elif regime == "crisis":
        probs = {"expansion": 0.1, "contraction": 0.2, "crisis": 0.7}
    else:
        probs = {"expansion": 0.7, "contraction": 0.2, "crisis": 0.1}

    for ts in timestamps:
        store.store_belief(
            _make_belief("regime.macro", ts, probabilities=probs),
        )


# ═══════════════════════════════════════════════════════════════
# _should_fit
# ═══════════════════════════════════════════════════════════════


class TestShouldFit:
    def test_no_previous_marker_triggers(self, tmp_path):
        store = _make_store(tmp_path)
        should, reason = _should_fit(store, _BASE_TS, fit_interval_days=7)
        assert should is True
        assert "no previous fit" in reason

    def test_recent_marker_skips(self, tmp_path):
        store = _make_store(tmp_path)
        store.store_data(_FIT_SOURCE, {"as_of": _BASE_TS}, {"result": "ok"})
        should, reason = _should_fit(
            store,
            _BASE_TS + 3 * _DAY_SECONDS,
            fit_interval_days=7,
        )
        assert should is False
        assert "3.0d" in reason

    def test_old_marker_triggers(self, tmp_path):
        store = _make_store(tmp_path)
        store.store_data(_FIT_SOURCE, {"as_of": _BASE_TS}, {"result": "ok"})
        should, reason = _should_fit(
            store,
            _BASE_TS + 10 * _DAY_SECONDS,
            fit_interval_days=7,
        )
        assert should is True
        assert "10.0d" in reason

    def test_exact_boundary(self, tmp_path):
        """Exactly fit_interval_days elapsed → should fit."""
        store = _make_store(tmp_path)
        store.store_data(_FIT_SOURCE, {"as_of": _BASE_TS}, {})
        should, _ = _should_fit(
            store,
            _BASE_TS + 7 * _DAY_SECONDS,
            fit_interval_days=7,
        )
        assert should is True

    def test_uses_as_of_not_fetched_at(self, tmp_path):
        """Marker's as_of param is the reference, not wall-clock fetched_at."""
        store = _make_store(tmp_path)
        # Marker stored now but logically as_of is from 20 days ago
        store.store_data(
            _FIT_SOURCE,
            {"as_of": _BASE_TS - 20 * _DAY_SECONDS},
            {},
        )
        should, _ = _should_fit(store, _BASE_TS, fit_interval_days=7)
        assert should is True

    def test_interval_1_day(self, tmp_path):
        store = _make_store(tmp_path)
        store.store_data(_FIT_SOURCE, {"as_of": _BASE_TS}, {})
        # 23 hours later → still < 1 day
        should, _ = _should_fit(
            store,
            _BASE_TS + 23 * 3600,
            fit_interval_days=1,
        )
        assert should is False


# ═══════════════════════════════════════════════════════════════
# _load_feature_history
# ═══════════════════════════════════════════════════════════════


class TestLoadFeatureHistory:
    def test_empty_store(self, tmp_path):
        store = _make_store(tmp_path)
        snapshots = _load_feature_history(
            store,
            _BASE_TS - 90 * _DAY_SECONDS,
            _BASE_TS,
        )
        assert snapshots == []

    def test_groups_by_day(self, tmp_path):
        store = _make_store(tmp_path)
        _seed_features(store, n_days=5, base_ts=_BASE_TS)
        snapshots = _load_feature_history(
            store,
            _BASE_TS - _DAY_SECONDS,
            _BASE_TS + 6 * _DAY_SECONDS,
        )
        assert len(snapshots) == 5
        from agent.pipeline.dags.world_model_update import _FEATURE_NAMES

        for snap in snapshots:
            names = {f.feature_name for f in snap}
            assert names == set(_FEATURE_NAMES)

    def test_oldest_first(self, tmp_path):
        store = _make_store(tmp_path)
        _seed_features(store, n_days=3, base_ts=_BASE_TS)
        snapshots = _load_feature_history(
            store,
            _BASE_TS - _DAY_SECONDS,
            _BASE_TS + 4 * _DAY_SECONDS,
        )
        assert snapshots[0][0].effective_at < snapshots[-1][0].effective_at

    def test_respects_time_window(self, tmp_path):
        store = _make_store(tmp_path)
        _seed_features(store, n_days=10, base_ts=_BASE_TS)
        # Query only days 2-4 (3 days)
        snapshots = _load_feature_history(
            store,
            _BASE_TS + 2 * _DAY_SECONDS,
            _BASE_TS + 5 * _DAY_SECONDS,
        )
        assert len(snapshots) == 3

    def test_handles_malformed_rows_gracefully(self, tmp_path):
        """If a feature row can't be deserialized, it's skipped."""
        store = _make_store(tmp_path)
        # Store a valid feature
        ts = _BASE_TS + 3600
        store.store_feature(
            _make_feature("macro.rate_momentum.30d", 1.0, ts),
        )
        # Corrupt a row directly in the DB
        conn = store._get_conn()
        conn.execute(
            "INSERT INTO features "
            "(feature_name, version, effective_at, computed_at, horizon, "
            "value, quality, source_signals_json, builder, unit) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "macro.yield_curve_slope.spot",
                1,
                ts,
                ts,
                "spot",
                2.0,
                1.0,
                "[]",
                "test",
                "raw",
            ),
        )
        conn.commit()

        snapshots = _load_feature_history(
            store,
            _BASE_TS,
            _BASE_TS + 2 * _DAY_SECONDS,
        )
        # Should still get at least the valid feature
        assert len(snapshots) >= 1


# ═══════════════════════════════════════════════════════════════
# _load_regime_labels
# ═══════════════════════════════════════════════════════════════


class TestLoadRegimeLabels:
    def test_no_beliefs_gives_default(self, tmp_path):
        store = _make_store(tmp_path)
        day_keys = [int(_BASE_TS // _DAY_SECONDS) + d for d in range(5)]
        labels = _load_regime_labels(store, day_keys, default_regime="expansion")
        assert labels == ["expansion"] * 5

    def test_maps_beliefs_to_days(self, tmp_path):
        store = _make_store(tmp_path)
        timestamps = [_BASE_TS + d * _DAY_SECONDS + 3600 for d in range(3)]
        _seed_regime_beliefs(store, timestamps, regime="contraction")
        day_keys = [int(ts // _DAY_SECONDS) for ts in timestamps]
        labels = _load_regime_labels(store, day_keys)
        assert all(label == "contraction" for label in labels)

    def test_mixed_regimes_and_missing(self, tmp_path):
        store = _make_store(tmp_path)
        ts0 = _BASE_TS + 3600
        ts2 = _BASE_TS + 2 * _DAY_SECONDS + 3600
        store.store_belief(
            _make_belief(
                "regime.macro",
                ts0,
                probabilities={"expansion": 0.8, "contraction": 0.1, "crisis": 0.1},
            )
        )
        store.store_belief(
            _make_belief(
                "regime.macro",
                ts2,
                probabilities={"expansion": 0.1, "contraction": 0.2, "crisis": 0.7},
            )
        )
        day_keys = [
            int(_BASE_TS // _DAY_SECONDS),
            int((_BASE_TS + _DAY_SECONDS) // _DAY_SECONDS),
            int((_BASE_TS + 2 * _DAY_SECONDS) // _DAY_SECONDS),
        ]
        labels = _load_regime_labels(store, day_keys, default_regime="expansion")
        assert labels[0] == "expansion"  # MAP from belief
        assert labels[1] == "expansion"  # default (no belief for day 1)
        assert labels[2] == "crisis"  # MAP from belief

    def test_empty_day_keys(self, tmp_path):
        store = _make_store(tmp_path)
        assert _load_regime_labels(store, []) == []

    def test_belief_without_probabilities_ignored(self, tmp_path):
        """Gaussian beliefs for regime.macro are skipped (no probabilities)."""
        store = _make_store(tmp_path)
        ts0 = _BASE_TS + 3600
        store.store_belief(_make_belief("regime.macro", ts0, mean=0.5))
        day_keys = [int(_BASE_TS // _DAY_SECONDS)]
        labels = _load_regime_labels(store, day_keys, default_regime="expansion")
        assert labels == ["expansion"]  # fallback to default


# ═══════════════════════════════════════════════════════════════
# _build_observation_sequence
# ═══════════════════════════════════════════════════════════════


class TestBuildObservationSequence:
    def test_known_feature_mapped(self):
        feat = _make_feature("macro.rate_momentum.30d", 1.5, _BASE_TS)
        obs_seq = _build_observation_sequence([[feat]])
        assert len(obs_seq) == 1
        assert obs_seq[0][0] == pytest.approx(1.5)
        # All other positions should be NaN
        assert np.isnan(obs_seq[0][1:]).all()

    def test_unknown_feature_ignored(self):
        feat = _make_feature("unknown.feature.spot", 42.0, _BASE_TS)
        obs_seq = _build_observation_sequence([[feat]])
        assert np.all(np.isnan(obs_seq[0]))

    def test_none_value_stays_nan(self):
        feat = _make_feature("macro.rate_momentum.30d", None, _BASE_TS)
        obs_seq = _build_observation_sequence([[feat]])
        assert np.isnan(obs_seq[0][0])

    def test_full_snapshot_all_positions(self):
        feats = []
        for name, idx in _FEATURE_TO_OBS_INDEX.items():
            feats.append(_make_feature(name, float(idx), _BASE_TS))
        obs_seq = _build_observation_sequence([feats])
        for name, idx in _FEATURE_TO_OBS_INDEX.items():
            assert obs_seq[0][idx] == pytest.approx(float(idx))
        assert not np.any(np.isnan(obs_seq[0]))

    def test_multiple_snapshots(self):
        snap1 = [_make_feature("macro.rate_momentum.30d", 1.0, _BASE_TS)]
        snap2 = [_make_feature("macro.rate_momentum.30d", 2.0, _BASE_TS + _DAY_SECONDS)]
        obs_seq = _build_observation_sequence([snap1, snap2])
        assert len(obs_seq) == 2
        assert obs_seq[0][0] == pytest.approx(1.0)
        assert obs_seq[1][0] == pytest.approx(2.0)

    def test_empty_snapshot(self):
        obs_seq = _build_observation_sequence([[]])
        assert len(obs_seq) == 1
        assert np.all(np.isnan(obs_seq[0]))

    def test_output_shape(self):
        feat = _make_feature("macro.rate_momentum.30d", 1.0, _BASE_TS)
        obs_seq = _build_observation_sequence([[feat]])
        assert obs_seq[0].shape == (_OBS_DIM,)


# ═══════════════════════════════════════════════════════════════
# _maybe_fit_params — orchestration logic
# ═══════════════════════════════════════════════════════════════


class TestMaybeFitParams:
    def test_disabled(self, tmp_path):
        store = _make_store(tmp_path)
        wm = MagicMock()
        result = _maybe_fit_params(store, wm, _BASE_TS, fit_enabled=False)
        assert result["skipped"] is True
        assert "fit_enabled" in result["reason"]

    def test_skips_when_recent(self, tmp_path):
        store = _make_store(tmp_path)
        store.store_data(_FIT_SOURCE, {"as_of": _BASE_TS}, {})
        wm = MagicMock()
        result = _maybe_fit_params(
            store,
            wm,
            _BASE_TS + 2 * _DAY_SECONDS,
            fit_interval_days=7,
        )
        assert result["skipped"] is True

    def test_skips_insufficient_snapshots(self, tmp_path):
        store = _make_store(tmp_path)
        _seed_features(store, n_days=5, base_ts=_BASE_TS)
        wm = MagicMock()
        result = _maybe_fit_params(
            store,
            wm,
            _BASE_TS + 6 * _DAY_SECONDS,
            fit_interval_days=1,
            history_window_days=90,
        )
        assert result["skipped"] is True
        assert "snapshots" in result["reason"]

    def test_full_flow_cpd_and_kalman(self, tmp_path):
        store = _make_store(tmp_path)
        ts = _seed_features(store, n_days=60, base_ts=_BASE_TS)
        _seed_regime_beliefs(store, ts)

        wm = MagicMock()
        wm.fit_cpds.return_value = {
            "fitted": True,
            "n_samples": 60,
            "nodes_fitted": ["obs.a"],
        }
        wm._filter._regime_configs = {
            "expansion": MagicMock(),
            "contraction": MagicMock(),
        }
        wm._filter.fit_filter_params.return_value = {
            "fitted": True,
            "n_samples": 60,
            "iterations": 5,
            "log_likelihoods": [-100, -90],
        }

        result = _maybe_fit_params(
            store,
            wm,
            _BASE_TS + 61 * _DAY_SECONDS,
            fit_interval_days=1,
            history_window_days=90,
        )

        assert result["skipped"] is False
        assert result["cpd_result"]["fitted"] is True
        assert result["kalman_result"]["fitted"] is True
        assert result["n_snapshots"] == 60

        wm.fit_cpds.assert_called_once()
        wm._filter.fit_filter_params.assert_called_once()

        # Verify fit marker stored
        markers = store.query_data(_FIT_SOURCE, limit=1)
        assert len(markers) == 1
        assert markers[0]["params"]["cpd_fitted"] is True
        assert markers[0]["params"]["kalman_fitted"] is True

    def test_cpd_failure_doesnt_block_kalman(self, tmp_path):
        store = _make_store(tmp_path)
        ts = _seed_features(store, n_days=40, base_ts=_BASE_TS)
        _seed_regime_beliefs(store, ts)

        wm = MagicMock()
        wm.fit_cpds.side_effect = RuntimeError("pgmpy exploded")
        wm._filter._regime_configs = {"expansion": MagicMock()}
        wm._filter.fit_filter_params.return_value = {
            "fitted": True,
            "n_samples": 40,
            "iterations": 3,
            "log_likelihoods": [-50],
        }

        result = _maybe_fit_params(
            store,
            wm,
            _BASE_TS + 41 * _DAY_SECONDS,
            fit_interval_days=1,
            history_window_days=90,
        )

        assert result["cpd_result"]["fitted"] is False
        assert "error" in result["cpd_result"]
        assert result["kalman_result"]["fitted"] is True

    def test_kalman_failure_still_stores_marker(self, tmp_path):
        store = _make_store(tmp_path)
        ts = _seed_features(store, n_days=40, base_ts=_BASE_TS)
        _seed_regime_beliefs(store, ts)

        wm = MagicMock()
        wm.fit_cpds.return_value = {
            "fitted": True,
            "n_samples": 40,
            "nodes_fitted": [],
        }
        wm._filter._regime_configs = {"expansion": MagicMock()}
        wm._filter.fit_filter_params.side_effect = ValueError("EM diverged")

        result = _maybe_fit_params(
            store,
            wm,
            _BASE_TS + 41 * _DAY_SECONDS,
            fit_interval_days=1,
            history_window_days=90,
        )

        assert result["cpd_result"]["fitted"] is True
        assert result["kalman_result"]["fitted"] is False
        assert "error" in result["kalman_result"]
        # Marker should still be stored
        markers = store.query_data(_FIT_SOURCE, limit=1)
        assert len(markers) == 1

    def test_kalman_skipped_when_fewer_than_30_obs(self, tmp_path):
        store = _make_store(tmp_path)
        ts = _seed_features(store, n_days=20, base_ts=_BASE_TS)
        _seed_regime_beliefs(store, ts)

        wm = MagicMock()
        wm.fit_cpds.return_value = {
            "fitted": True,
            "n_samples": 20,
            "nodes_fitted": ["obs.a"],
        }
        wm._filter._regime_configs = {"expansion": MagicMock()}

        result = _maybe_fit_params(
            store,
            wm,
            _BASE_TS + 21 * _DAY_SECONDS,
            fit_interval_days=1,
            history_window_days=90,
        )

        assert result["cpd_result"]["fitted"] is True
        assert result["kalman_result"]["fitted"] is False
        assert "insufficient" in result["kalman_result"].get("reason", "")
        wm._filter.fit_filter_params.assert_not_called()

    def test_observation_sequence_passed_to_kalman(self, tmp_path):
        """Verify obs_seq passed to fit_filter_params is numpy arrays of correct shape."""
        store = _make_store(tmp_path)
        ts = _seed_features(store, n_days=40, base_ts=_BASE_TS)
        _seed_regime_beliefs(store, ts)

        wm = MagicMock()
        wm.fit_cpds.return_value = {"fitted": True, "n_samples": 40, "nodes_fitted": []}
        wm._filter._regime_configs = {"expansion": MagicMock()}
        wm._filter.fit_filter_params.return_value = {
            "fitted": True,
            "n_samples": 40,
            "iterations": 2,
            "log_likelihoods": [-10],
        }

        _maybe_fit_params(
            store,
            wm,
            _BASE_TS + 41 * _DAY_SECONDS,
            fit_interval_days=1,
            history_window_days=90,
        )

        call_args = wm._filter.fit_filter_params.call_args
        obs_seq = call_args[0][0]
        regime_labels = call_args[0][1]

        assert len(obs_seq) == 40
        assert all(isinstance(o, np.ndarray) for o in obs_seq)
        assert all(o.shape == (_OBS_DIM,) for o in obs_seq)
        assert len(regime_labels) == 40
        assert all(isinstance(r, str) for r in regime_labels)

    def test_marker_prevents_double_fit(self, tmp_path):
        """After a successful fit, a second call within the interval should skip."""
        store = _make_store(tmp_path)
        ts = _seed_features(store, n_days=40, base_ts=_BASE_TS)
        _seed_regime_beliefs(store, ts)

        wm = MagicMock()
        wm.fit_cpds.return_value = {"fitted": True, "n_samples": 40, "nodes_fitted": []}
        wm._filter._regime_configs = {"expansion": MagicMock()}
        wm._filter.fit_filter_params.return_value = {
            "fitted": True,
            "n_samples": 40,
            "iterations": 2,
            "log_likelihoods": [-10],
        }

        as_of = _BASE_TS + 41 * _DAY_SECONDS
        r1 = _maybe_fit_params(
            store,
            wm,
            as_of,
            fit_interval_days=7,
            history_window_days=90,
        )
        assert r1["skipped"] is False

        # Second call — same day, should skip
        r2 = _maybe_fit_params(
            store,
            wm,
            as_of + 100,
            fit_interval_days=7,
            history_window_days=90,
        )
        assert r2["skipped"] is True


# ═══════════════════════════════════════════════════════════════
# run_world_model_update — full DAG function
# ═══════════════════════════════════════════════════════════════


class TestRunWorldModelUpdateWithFitting:
    """Test that run_world_model_update integrates fitting correctly."""

    @patch("agent.pipeline.dags.world_model_update._maybe_fit_params")
    @patch("agent.pipeline.dags.world_model_update._build_world_model")
    def test_fit_result_in_output(self, mock_build, mock_fit, tmp_path):
        from agent.pipeline.dags.world_model_update import run_world_model_update

        mock_wm = MagicMock()
        mock_wm.update.return_value = []
        mock_wm.get_graph_hash.return_value = "h" * 64
        mock_build.return_value = mock_wm
        mock_fit.return_value = {"skipped": True, "reason": "test"}

        result = run_world_model_update(
            {"db_path": str(tmp_path / "t.db"), "as_of": _BASE_TS},
            {},
        )
        assert "fit_result" in result
        assert result["fit_result"]["skipped"] is True

    @patch("agent.pipeline.dags.world_model_update._maybe_fit_params")
    @patch("agent.pipeline.dags.world_model_update._build_world_model")
    def test_fit_params_forwarded(self, mock_build, mock_fit, tmp_path):
        from agent.pipeline.dags.world_model_update import run_world_model_update

        mock_wm = MagicMock()
        mock_wm.update.return_value = []
        mock_wm.get_graph_hash.return_value = "h" * 64
        mock_build.return_value = mock_wm
        mock_fit.return_value = {"skipped": False}

        run_world_model_update(
            {
                "db_path": str(tmp_path / "t.db"),
                "as_of": _BASE_TS,
                "fit_enabled": False,
                "fit_interval_days": 14,
                "history_window_days": 60,
            },
            {},
        )

        call_kwargs = mock_fit.call_args.kwargs
        assert call_kwargs["fit_enabled"] is False
        assert call_kwargs["fit_interval_days"] == 14
        assert call_kwargs["history_window_days"] == 60

    @patch("agent.pipeline.dags.world_model_update._maybe_fit_params")
    @patch("agent.pipeline.dags.world_model_update._build_world_model")
    def test_fitting_runs_before_update(self, mock_build, mock_fit, tmp_path):
        """Fitting must happen before wm.update() so fitted params are used."""
        from agent.pipeline.dags.world_model_update import run_world_model_update

        call_order: list[str] = []

        def fake_fit(*a, **kw):
            call_order.append("fit")
            return {"skipped": False}

        mock_fit.side_effect = fake_fit

        mock_wm = MagicMock()
        mock_wm.update.side_effect = lambda *a, **kw: (
            call_order.append("update") or []
        )
        mock_wm.get_graph_hash.return_value = "h" * 64
        mock_build.return_value = mock_wm

        run_world_model_update(
            {"db_path": str(tmp_path / "t.db"), "as_of": _BASE_TS},
            {},
        )
        assert call_order == ["fit", "update"]

    @patch("agent.pipeline.dags.world_model_update._maybe_fit_params")
    @patch("agent.pipeline.dags.world_model_update._build_world_model")
    def test_default_params(self, mock_build, mock_fit, tmp_path):
        """Default fit params should be True/7/90."""
        from agent.pipeline.dags.world_model_update import run_world_model_update

        mock_wm = MagicMock()
        mock_wm.update.return_value = []
        mock_wm.get_graph_hash.return_value = "h" * 64
        mock_build.return_value = mock_wm
        mock_fit.return_value = {"skipped": True, "reason": "test"}

        run_world_model_update(
            {
                "db_path": str(tmp_path / "t.db"),
                "as_of": _BASE_TS,
                "use_scheduler": False,
            },
            {},
        )

        call_kwargs = mock_fit.call_args.kwargs
        assert call_kwargs["fit_enabled"] is True
        assert call_kwargs["fit_interval_days"] == 7
        assert call_kwargs["history_window_days"] == 90

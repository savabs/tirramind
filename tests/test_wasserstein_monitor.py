"""Tests for Idea 8 — Wasserstein Distribution Shift Detector.

Covers:
    1.  WassersteinMonitor instantiates with default params
    2.  _daily_features: empty obs → zero arrays
    3.  _daily_features: single obs fills correct bin
    4.  _daily_features: count channel sums correctly over bins
    5.  _daily_features: value channel is tanh-normalised
    6.  _w1_1d_normalised: identical distributions → score ≈ 0
    7.  _w1_1d_normalised: very different distributions → score > 0
    8.  _w1_1d_normalised: normalised by baseline std (dimensionless)
    9.  _w1_1d_normalised: handles empty arrays gracefully
    10. _w1_1d_normalised: baseline constant (std≈0) → large shift if different
    11. run(): returns empty dict when store has no observations
    12. run(): skips tools with fewer than min_baseline_obs
    13. run(): returns WassersteinResult for a tool with sufficient history
    14. run(): is_alarm=False when distribution is stable
    15. run(): is_alarm=True when short window is very different from baseline
    16. run(): drift_score is non-negative
    17. run(): result fields are correctly populated
    18. store_results(): persists signals with correct names in the store
    19. store_results(): each stored signal value matches the drift_score
    20. TrainerConfig.use_wasserstein defaults False
    21. TrainerConfig.wasserstein_threshold defaults 1.0
    22. TrainerConfig.wasserstein_short_days defaults 30
    23. build_model() with use_wasserstein=True runs without error
    24. build_model() with use_wasserstein=True stores drift signals
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from agent.convergence.wasserstein_monitor import (
    WassersteinMonitor,
    WassersteinResult,
    _w1_1d_normalised,
)
from agent.models.gnn.trainer import Trainer, TrainerConfig, SyntheticGraphGenerator
from agent.pipeline.store import PipelineStore

# ── Helpers ──────────────────────────────────────────────────────────────────

_DAY = 86_400.0


def _make_store(tmp_path: Path, name: str = "wass.db") -> PipelineStore:
    return PipelineStore(str(tmp_path / name))


def _add_obs(
    store: PipelineStore,
    tool: str,
    count: int,
    t_start: float,
    t_end: float,
    value_mean: float = 1.0,
    seed: int = 0,
) -> None:
    """Add `count` observations uniformly spread over [t_start, t_end]."""
    rng = np.random.default_rng(seed)
    times = rng.uniform(t_start, t_end, count)
    entity_id = f"e_{tool}"
    store.register_entity(
        entity_type="company", canonical_name=entity_id, entity_id=entity_id
    )
    for t in times:
        store.store_entity_observation(
            entity_id=entity_id,
            source_tool=tool,
            observation_type="price",
            observed_at=float(t),
            value={"value": float(rng.normal(value_mean, 0.1))},
        )


# ═══════════════════════════════════════════════════════════════
# 1. Construction
# ═══════════════════════════════════════════════════════════════


class TestConstruction:

    def test_instantiates_defaults(self):
        m = WassersteinMonitor()
        assert m.short_days == 30
        assert m.long_days == 365
        assert m.alarm_threshold == 1.0

    def test_instantiates_custom(self):
        m = WassersteinMonitor(short_days=7, long_days=90, alarm_threshold=2.0)
        assert m.short_days == 7
        assert m.long_days == 90
        assert m.alarm_threshold == 2.0


# ═══════════════════════════════════════════════════════════════
# 2–5. _daily_features
# ═══════════════════════════════════════════════════════════════


class TestDailyFeatures:

    def test_empty_obs_returns_zeros(self):
        c, v = WassersteinMonitor._daily_features([], 0.0, _DAY * 30, n_bins=30)
        assert c.shape == (30,)
        assert v.shape == (30,)
        assert (c == 0).all()
        assert (v == 0).all()

    def test_single_obs_fills_correct_bin(self):
        t_start = 0.0
        t_end = _DAY * 10
        # Obs at 50% of span → bin 5 of 10
        obs = [{"observed_at": _DAY * 5, "value": {"value": 1.0}}]
        c, _ = WassersteinMonitor._daily_features(obs, t_start, t_end, n_bins=10)
        assert c.sum() == pytest.approx(1.0)
        assert c[5] == 1.0

    def test_count_channel_sums_correctly(self):
        t_start = 0.0
        t_end = _DAY * 4
        obs = [
            {"observed_at": _DAY * 0.5, "value": {"value": 1.0}},
            {"observed_at": _DAY * 1.5, "value": {"value": 1.0}},
            {"observed_at": _DAY * 1.6, "value": {"value": 1.0}},
        ]
        c, _ = WassersteinMonitor._daily_features(obs, t_start, t_end, n_bins=4)
        assert c.sum() == pytest.approx(3.0)

    def test_value_channel_tanh_bounded(self):
        obs = [{"observed_at": 0.0, "value": {"value": 1e6}}]
        _, v = WassersteinMonitor._daily_features(obs, 0.0, _DAY, n_bins=1)
        assert -1.0 <= float(v[0]) <= 1.0


# ═══════════════════════════════════════════════════════════════
# 6–10. _w1_1d_normalised
# ═══════════════════════════════════════════════════════════════


class TestW1Normalised:

    def test_identical_distributions_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _w1_1d_normalised(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_very_different_distributions_positive(self):
        short = np.zeros(30)  # completely silent window
        baseline = np.ones(365)  # constant active baseline
        score = _w1_1d_normalised(short, baseline)
        assert score > 0.5

    def test_normalised_by_baseline_std(self):
        rng = np.random.default_rng(1)
        baseline = rng.normal(loc=5.0, scale=2.0, size=365)
        short_same = rng.normal(loc=5.0, scale=2.0, size=30)
        short_shifted = rng.normal(loc=9.0, scale=2.0, size=30)  # 2-sigma shift
        score_same = _w1_1d_normalised(short_same, baseline)
        score_shifted = _w1_1d_normalised(short_shifted, baseline)
        # Shifted should produce a clearly larger score
        assert score_shifted > score_same

    def test_empty_arrays_return_zero(self):
        assert _w1_1d_normalised(np.array([]), np.array([1.0, 2.0])) == 0.0
        assert _w1_1d_normalised(np.array([1.0, 2.0]), np.array([])) == 0.0

    def test_constant_baseline_large_shift(self):
        short = np.array([10.0, 10.0, 10.0])
        baseline = np.zeros(30)  # std ≈ 0
        score = _w1_1d_normalised(short, baseline)
        assert score > 1.0  # large shift when baseline is constant


# ═══════════════════════════════════════════════════════════════
# 11–17. run()
# ═══════════════════════════════════════════════════════════════


class TestRun:

    def test_empty_store_returns_empty_dict(self, tmp_path):
        store = _make_store(tmp_path)
        m = WassersteinMonitor()
        result = m.run(store)
        assert result == {}

    def test_skips_tools_below_min_obs(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        # Add only 3 observations — below default min_baseline_obs=10
        _add_obs(store, "sparse_tool", 3, as_of - 30 * _DAY, as_of)
        m = WassersteinMonitor(min_baseline_obs=10)
        result = m.run(store, as_of=as_of)
        assert "sparse_tool" not in result

    def test_returns_result_for_sufficient_history(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _add_obs(store, "rich_tool", 100, as_of - 365 * _DAY, as_of)
        m = WassersteinMonitor(min_baseline_obs=10)
        result = m.run(store, as_of=as_of)
        assert "rich_tool" in result
        assert isinstance(result["rich_tool"], WassersteinResult)

    def test_is_alarm_false_stable_distribution(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        # Uniform observations throughout 365 days — no regime change
        _add_obs(store, "stable_tool", 365, as_of - 365 * _DAY, as_of, seed=1)
        m = WassersteinMonitor(alarm_threshold=5.0, min_baseline_obs=5)
        result = m.run(store, as_of=as_of)
        assert "stable_tool" in result
        assert result["stable_tool"].is_alarm is False

    def test_is_alarm_true_silent_recent_window(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        # Active in past (days 365→30), silent in last 30 days → big drift
        _add_obs(store, "dead_tool", 200, as_of - 365 * _DAY, as_of - 31 * _DAY, seed=2)
        m = WassersteinMonitor(alarm_threshold=0.5, min_baseline_obs=5)
        result = m.run(store, as_of=as_of)
        assert "dead_tool" in result
        assert result["dead_tool"].is_alarm is True

    def test_drift_score_non_negative(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _add_obs(store, "tool_a", 100, as_of - 365 * _DAY, as_of)
        m = WassersteinMonitor(min_baseline_obs=5)
        result = m.run(store, as_of=as_of)
        for r in result.values():
            assert r.drift_score >= 0.0

    def test_result_fields_populated(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _add_obs(store, "check_tool", 80, as_of - 365 * _DAY, as_of, seed=5)
        m = WassersteinMonitor(min_baseline_obs=5)
        results = m.run(store, as_of=as_of)
        assert "check_tool" in results
        r = results["check_tool"]
        assert r.tool_name == "check_tool"
        assert r.computed_at == pytest.approx(as_of, abs=5.0)
        assert r.baseline_count > 0
        assert r.threshold == m.alarm_threshold
        assert isinstance(r.w1_count, float)
        assert isinstance(r.w1_value, float)


# ═══════════════════════════════════════════════════════════════
# 18–19. store_results()
# ═══════════════════════════════════════════════════════════════


class TestStoreResults:

    def test_persists_signals_with_correct_names(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _add_obs(store, "persist_tool", 60, as_of - 365 * _DAY, as_of)
        m = WassersteinMonitor(min_baseline_obs=5)
        results = m.run(store, as_of=as_of)
        n = m.store_results(results, store)
        assert n == len(results)

        if "persist_tool" in results:
            sigs = store.query_signals("wasserstein.persist_tool.drift")
            assert len(sigs) >= 1

    def test_stored_signal_value_matches_drift_score(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _add_obs(store, "val_tool", 60, as_of - 365 * _DAY, as_of)
        m = WassersteinMonitor(min_baseline_obs=5)
        results = m.run(store, as_of=as_of)
        m.store_results(results, store)

        for tool_name, res in results.items():
            sigs = store.query_signals(f"wasserstein.{tool_name}.drift")
            assert len(sigs) >= 1
            stored_val = sigs[-1]["value"]
            assert stored_val == pytest.approx(res.drift_score, rel=1e-5)


# ═══════════════════════════════════════════════════════════════
# 20–22. TrainerConfig
# ═══════════════════════════════════════════════════════════════


class TestTrainerConfig:

    def test_use_wasserstein_defaults_false(self):
        from agent.models.gnn.trainer import TrainerConfig

        assert TrainerConfig().use_wasserstein is False

    def test_wasserstein_threshold_defaults_one(self):
        from agent.models.gnn.trainer import TrainerConfig

        assert TrainerConfig().wasserstein_threshold == pytest.approx(1.0)

    def test_wasserstein_short_days_defaults_30(self):
        from agent.models.gnn.trainer import TrainerConfig

        assert TrainerConfig().wasserstein_short_days == 30


# ═══════════════════════════════════════════════════════════════
# 23–24. build_model() integration
# ═══════════════════════════════════════════════════════════════


class TestBuildModelIntegration:

    def _make_trainer(self, tmp_path: Path, use_wasserstein: bool, tag: str) -> Trainer:
        store = _make_store(tmp_path, f"{tag}.db")
        gen = SyntheticGraphGenerator(
            num_companies=4,
            num_countries=2,
            num_vessels=2,
            time_span=3600.0 * 4,
            base_event_rate=0.001,
            seed=99,
        )
        gen.generate(store)
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            use_wasserstein=use_wasserstein,
            wasserstein_threshold=0.5,
            wasserstein_short_days=7,
        )
        return Trainer(store, cfg)

    def test_build_model_with_wasserstein_no_error(self, tmp_path):
        t = self._make_trainer(tmp_path, use_wasserstein=True, tag="wbm")
        model = t.build_model()
        assert model is not None

    def test_build_model_stores_drift_signals(self, tmp_path):
        t = self._make_trainer(tmp_path, use_wasserstein=True, tag="wsig")
        # SyntheticGraphGenerator uses epoch-relative timestamps (t_start=0).
        # Add current-time observations so they fall inside the monitor's
        # 365-day baseline window (computed from time.time()).
        as_of = time.time()
        _add_obs(t.store, "live_feed", 60, as_of - 365 * _DAY, as_of)
        t.build_model()
        sigs = t.store.query_signals("wasserstein.live_feed.drift")
        assert len(sigs) >= 1, "Expected wasserstein.live_feed.drift signal stored"

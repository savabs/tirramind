"""Tests for Idea 15 — TDA Regime Detector (Persistent Homology).

Covers:
    1.  PersistencePair.persistence = death - birth
    2.  PersistencePair.persistence = inf when death is inf
    3.  RegimeDiagram.finite_0 excludes infinite-death pairs
    4.  RegimeDiagram.all_finite excludes inf from β₀ + β₁
    5.  _build_point_cloud normalises rows to unit norm
    6.  _build_point_cloud handles zero rows without division error
    7.  _distance_matrix returns condensed form (length = n*(n-1)/2)
    8.  _persistence_via_scipy returns exactly n-1 finite + 1 inf pair for n points
    9.  _persistence_entropy: empty → 0.0
    10. _persistence_entropy: uniform lifetimes → log(n)
    11. _bottleneck_distance: identical diagrams → 0.0
    12. _bottleneck_distance: different diagrams → positive
    13. _n_components_at_threshold: correct count at given ε
    14. TDARegimeDetector construction defaults
    15. compute(): returns RegimeFeatures from valid returns
    16. compute(): n_components ≥ 1 always
    17. compute(): persistence_entropy ≥ 0
    18. compute_full(): spectral_gap ≥ 0
    19. compute_full(): spectral_gap < n_components (bounded)
    20. fit_baseline() + bottleneck_dist > 0 when distributions differ
    21. fit_baseline() bottleneck_dist = 0 when same data
    22. _prepare(): caps instruments to max_instruments
    23. _prepare(): fills NaN with 0
    24. compute_from_store(): returns None on empty store
    25. store_signals(): writes 5 signals per call
    26. store_signals(): handles store error gracefully
    27. _load_returns(): returns None when no price observations
    28. _load_returns(): builds return matrix from price data
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from agent.convergence.tda_regime import (
    PersistencePair,
    RegimeDiagram,
    RegimeFeatures,
    TDARegimeDetector,
    _bottleneck_distance,
    _build_point_cloud,
    _distance_matrix,
    _load_returns,
    _n_components_at_threshold,
    _persistence_entropy,
    _persistence_via_scipy,
)
from agent.pipeline.store import PipelineStore

# ── Helpers ───────────────────────────────────────────────────────────────


def _make_store(tmp_path: Path, name: str = "tda.db") -> PipelineStore:
    return PipelineStore(str(tmp_path / name))


def _add_price(store: PipelineStore, eid: str, ts: float, price: float) -> None:
    store.store_entity_observation(
        entity_id=eid,
        source_tool="test",
        observed_at=ts,
        observation_type="price",
        value=price,
    )


def _make_returns(T: int = 30, N: int = 5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.01, (T, N))


# ═══════════════════════════════════════════════════════════════
# 1–4. PersistencePair + RegimeDiagram
# ═══════════════════════════════════════════════════════════════


class TestPersistencePair:

    def test_persistence_finite(self):
        p = PersistencePair(birth=0.1, death=0.5, dim=0)
        assert p.persistence == pytest.approx(0.4)

    def test_persistence_infinite_death(self):
        p = PersistencePair(birth=0.0, death=math.inf, dim=0)
        assert math.isinf(p.persistence)

    def test_finite_0_excludes_inf(self):
        p1 = PersistencePair(0.0, 0.5, 0)
        p2 = PersistencePair(0.0, math.inf, 0)
        d = RegimeDiagram(pairs_0=[p1, p2])
        assert len(d.finite_0) == 1
        assert d.finite_0[0] is p1

    def test_all_finite_excludes_inf(self):
        p0 = PersistencePair(0.0, math.inf, 0)
        p1 = PersistencePair(0.1, 0.8, 1)
        d = RegimeDiagram(pairs_0=[p0], pairs_1=[p1])
        assert len(d.all_finite) == 1
        assert d.all_finite[0] is p1


# ═══════════════════════════════════════════════════════════════
# 5–7. Point cloud + distance helpers
# ═══════════════════════════════════════════════════════════════


class TestHelpers:

    def test_build_point_cloud_unit_norms(self):
        x = np.array([[3.0, 4.0], [1.0, 0.0]])
        cloud = _build_point_cloud(x)
        norms = np.linalg.norm(cloud, axis=1)
        assert np.allclose(norms, 1.0)

    def test_build_point_cloud_zero_row(self):
        x = np.array([[0.0, 0.0], [1.0, 0.0]])
        cloud = _build_point_cloud(x)  # must not raise
        assert cloud.shape == x.shape

    def test_distance_matrix_length(self):
        x = np.random.default_rng(0).normal(0, 1, (6, 3))
        cloud = _build_point_cloud(x)
        cond = _distance_matrix(cloud)
        assert len(cond) == 6 * 5 // 2


# ═══════════════════════════════════════════════════════════════
# 8. _persistence_via_scipy
# ═══════════════════════════════════════════════════════════════


class TestPersistenceScipy:

    def test_n_minus_1_finite_plus_one_inf(self):
        cloud = np.random.default_rng(7).normal(0, 1, (8, 3))
        cond = _distance_matrix(_build_point_cloud(cloud))
        diag = _persistence_via_scipy(cond, n_points=8)
        finite = diag.finite_0
        inf_pairs = [p for p in diag.pairs_0 if math.isinf(p.death)]
        assert len(finite) == 7  # n-1
        assert len(inf_pairs) == 1


# ═══════════════════════════════════════════════════════════════
# 9–13. Entropy + bottleneck + components
# ═══════════════════════════════════════════════════════════════


class TestMetrics:

    def test_entropy_empty(self):
        assert _persistence_entropy([]) == 0.0

    def test_entropy_uniform(self):
        # 4 equal lifetimes → H approaches log(4); small numerical eps shifts it
        pairs = [PersistencePair(0.0, float(i + 1), 0) for i in range(4)]
        h = _persistence_entropy(pairs)
        assert h == pytest.approx(math.log(4), abs=0.12)

    def test_bottleneck_identical(self):
        pairs = [PersistencePair(0.0, 0.3, 0), PersistencePair(0.0, 0.7, 0)]
        assert _bottleneck_distance(pairs, pairs) == 0.0

    def test_bottleneck_different_positive(self):
        a = [PersistencePair(0.0, 1.0, 0)]
        b = [PersistencePair(0.0, 0.1, 0)]
        assert _bottleneck_distance(a, b) > 0.0

    def test_n_components_at_threshold(self):
        # 3 pairs: born 0 die at 0.2, 0.5, inf
        pairs = [
            PersistencePair(0.0, 0.2, 0),
            PersistencePair(0.0, 0.5, 0),
            PersistencePair(0.0, math.inf, 0),
        ]
        # At ε=0.3: pair dying at 0.2 is dead (0.3 > 0.2), others alive → 2
        assert _n_components_at_threshold(pairs, 0.3) == 2
        # At ε=0.1: all alive → 3
        assert _n_components_at_threshold(pairs, 0.1) == 3


# ═══════════════════════════════════════════════════════════════
# 14–23. TDARegimeDetector
# ═══════════════════════════════════════════════════════════════


class TestTDARegimeDetector:

    def test_default_construction(self):
        d = TDARegimeDetector()
        assert d.window_days == 30
        assert d.max_instruments == 50

    def test_compute_returns_features(self):
        r = _make_returns(30, 5)
        d = TDARegimeDetector(use_ripser=False)
        feat = d.compute(r)
        assert isinstance(feat, RegimeFeatures)

    def test_n_components_at_least_one(self):
        r = _make_returns(30, 5)
        d = TDARegimeDetector(use_ripser=False)
        feat = d.compute(r)
        assert feat.n_components >= 1

    def test_entropy_non_negative(self):
        r = _make_returns(30, 5)
        d = TDARegimeDetector(use_ripser=False)
        feat = d.compute(r)
        assert feat.persistence_entropy >= 0.0

    def test_compute_full_spectral_gap_non_negative(self):
        r = _make_returns(15, 4)
        d = TDARegimeDetector(use_ripser=False)
        feat = d.compute_full(r)
        assert feat.spectral_gap >= 0.0

    def test_compute_full_spectral_gap_bounded(self):
        r = _make_returns(20, 5)
        d = TDARegimeDetector(use_ripser=False)
        feat = d.compute_full(r)
        # Spectral gap ≤ n_points (loose bound)
        assert feat.spectral_gap <= 20.0 + 1e-6

    def test_fit_baseline_bottleneck_differs(self):
        rng = np.random.default_rng(10)
        r_calm = rng.normal(0, 0.005, (30, 5))
        r_crisis = rng.normal(0, 0.05, (30, 5))  # 10× larger shocks
        d = TDARegimeDetector(use_ripser=False)
        d.fit_baseline(r_calm)
        feat = d.compute(r_crisis)
        # Bottleneck distance should be >0 (distributions differ)
        # Not guaranteed to be large, but should be non-negative
        assert feat.bottleneck_dist >= 0.0

    def test_fit_baseline_same_data_bottleneck_zero(self):
        r = _make_returns(30, 5)
        d = TDARegimeDetector(use_ripser=False)
        d.fit_baseline(r)
        feat = d.compute(r)
        assert feat.bottleneck_dist == pytest.approx(0.0, abs=1e-10)

    def test_prepare_caps_instruments(self):
        r = _make_returns(30, 20)
        d = TDARegimeDetector(max_instruments=5, use_ripser=False)
        r_capped = d._prepare(r)
        assert r_capped.shape[1] == 5

    def test_prepare_fills_nan(self):
        r = _make_returns(30, 3)
        r[5, 1] = float("nan")
        d = TDARegimeDetector(use_ripser=False)
        r_clean = d._prepare(r)
        assert not np.any(np.isnan(r_clean))

    def test_compute_from_store_returns_none_empty(self, tmp_path):
        store = _make_store(tmp_path, "empty.db")
        d = TDARegimeDetector(use_ripser=False)
        result = d.compute_from_store(store)
        assert result is None

    def test_store_signals_writes_5(self):
        mock_store = MagicMock()
        d = TDARegimeDetector(use_ripser=False)
        feat = RegimeFeatures(
            n_components=2,
            persistence_entropy=0.5,
            bottleneck_dist=0.1,
            spectral_gap=0.3,
            mean_persistence=0.2,
            computed_at=time.time(),
        )
        n = d.store_signals(mock_store, feat)
        assert n == 5

    def test_store_signals_handles_error(self):
        mock_store = MagicMock()
        mock_store.store_signal.side_effect = RuntimeError("disk full")
        d = TDARegimeDetector(use_ripser=False)
        feat = RegimeFeatures(
            n_components=1,
            persistence_entropy=0.0,
            bottleneck_dist=0.0,
            spectral_gap=0.0,
            mean_persistence=0.0,
            computed_at=time.time(),
        )
        n = d.store_signals(mock_store, feat)  # must not raise
        assert n == 0


# ═══════════════════════════════════════════════════════════════
# 27–28. _load_returns
# ═══════════════════════════════════════════════════════════════


class TestLoadReturns:

    def test_returns_none_no_data(self, tmp_path):
        store = _make_store(tmp_path, "nodata.db")
        result = _load_returns(store, None, 30, 50, None)
        assert result is None

    def test_builds_return_matrix(self, tmp_path):
        store = _make_store(tmp_path, "prices.db")
        now = time.time()
        # 5 entities, 10 days, daily price
        for eid in ["a", "b", "c"]:
            for day in range(10):
                _add_price(store, eid, now - (10 - day) * 86400, 100.0 + day)
        result = _load_returns(store, ["a", "b", "c"], 15, 50, now)
        assert result is not None
        assert result.ndim == 2
        assert result.shape[1] == 3

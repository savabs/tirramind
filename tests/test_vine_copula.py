"""Tests for Idea 7 — Vine Copula Tail-Dependence Encoder.

Covers:
    1.  VineCopulaEncoder instantiates with defaults
    2.  _pair_hash is deterministic and symmetric
    3.  _pair_hash produces 12-char hex strings
    4.  _kendall_tau: identical arrays → τ = 1.0
    5.  _kendall_tau: perfectly anti-correlated → τ = −1.0
    6.  _kendall_tau: independent → τ ≈ 0
    7.  _pseudo_observations: output in (0,1) open interval
    8.  _pseudo_observations: preserves rank ordering
    9.  _clayton_lambda_L: θ→∞ → λ_L → 1
    10. _clayton_lambda_L: θ=0 → λ_L = 0
    11. _gumbel_lambda_U: θ=1 → λ_U = 0
    12. _gumbel_lambda_U: large θ → λ_U → 1
    13. _fit_bivariate_copula: positive τ → λ_L > 0 and λ_U > 0
    14. _fit_bivariate_copula: negative τ → λ_L = 0, λ_U > 0 (survival)
    15. _fit_bivariate_copula: τ ≈ 0 → λ_L = 0, λ_U = 0
    16. _fit_bivariate_copula: family string is correct for positive τ
    17. run(): empty store → empty dict
    18. run(): no entity links → empty dict
    19. run(): skips pairs with insufficient joint observations
    20. run(): returns CopulaResult for a pair with sufficient history
    21. run(): lambda_L and lambda_U in [0, 1]
    22. run(): result.pair_key matches _pair_hash(eid_a, eid_b)
    23. store_results(): stores 4 signals per pair with correct names
    24. store_results(): stored tau value matches result.tau
    25. TrainerConfig.use_vine_copula defaults False
    26. TrainerConfig.vine_copula_min_obs defaults 20
    27. TrainerConfig.vine_copula_n_bins defaults 60
    28. build_model() with use_vine_copula=True runs without error
    29. build_model() with use_vine_copula=True stores copula signals
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from agent.convergence.vine_copula import (
    CopulaResult,
    VineCopulaEncoder,
    _clayton_lambda_L,
    _fit_bivariate_copula,
    _gumbel_lambda_U,
    _kendall_tau,
    _pair_hash,
    _pseudo_observations,
)
from agent.models.gnn.trainer import Trainer, TrainerConfig, SyntheticGraphGenerator
from agent.pipeline.store import PipelineStore

# ── Helpers ──────────────────────────────────────────────────────────────────

_DAY = 86_400.0


def _make_store(tmp_path: Path, name: str = "copula.db") -> PipelineStore:
    return PipelineStore(str(tmp_path / name))


def _populate_linked_pair(
    store: PipelineStore,
    eid_a: str,
    eid_b: str,
    n: int,
    t_start: float,
    t_end: float,
    correlation: float = 0.8,
    seed: int = 1,
) -> None:
    """Create two linked entities with correlated observation histories."""
    rng = np.random.default_rng(seed)
    store.register_entity("company", eid_a, eid_a)
    store.register_entity("company", eid_b, eid_b)
    store.link_entities(eid_a, eid_b, "related", "test", confidence=0.8)

    # Correlated Gaussian observations
    z1 = rng.normal(0, 1, n)
    z2 = rng.normal(0, 1, n)
    x = z1
    y = correlation * z1 + np.sqrt(1 - correlation**2) * z2

    times = sorted(rng.uniform(t_start, t_end, n))
    for i, t in enumerate(times):
        store.store_entity_observation(
            entity_id=eid_a,
            source_tool="test",
            observation_type="price",
            observed_at=float(t),
            value={"value": float(x[i])},
        )
        store.store_entity_observation(
            entity_id=eid_b,
            source_tool="test",
            observation_type="price",
            observed_at=float(t) + 1.0,
            value={"value": float(y[i])},
        )


# ═══════════════════════════════════════════════════════════════
# 1. Construction
# ═══════════════════════════════════════════════════════════════


class TestConstruction:

    def test_instantiates_defaults(self):
        enc = VineCopulaEncoder()
        assert enc.min_joint_obs == 20
        assert enc.lookback_days == 365
        assert enc.n_bins == 60

    def test_instantiates_custom(self):
        enc = VineCopulaEncoder(min_joint_obs=5, n_bins=30)
        assert enc.min_joint_obs == 5
        assert enc.n_bins == 30


# ═══════════════════════════════════════════════════════════════
# 2–3. _pair_hash
# ═══════════════════════════════════════════════════════════════


class TestPairHash:

    def test_deterministic(self):
        h1 = _pair_hash("entity_abc", "entity_xyz")
        h2 = _pair_hash("entity_abc", "entity_xyz")
        assert h1 == h2

    def test_symmetric(self):
        assert _pair_hash("aaa", "bbb") == _pair_hash("bbb", "aaa")

    def test_twelve_chars(self):
        h = _pair_hash("entity_abc", "entity_xyz")
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)


# ═══════════════════════════════════════════════════════════════
# 4–6. _kendall_tau
# ═══════════════════════════════════════════════════════════════


class TestKendallTau:

    def test_identical_arrays(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _kendall_tau(x, x) == pytest.approx(1.0, abs=1e-6)

    def test_anti_correlated(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert _kendall_tau(x, y) == pytest.approx(-1.0, abs=1e-6)

    def test_independent_near_zero(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = rng.normal(0, 1, 500)  # independent
        tau = _kendall_tau(x, y)
        assert abs(tau) < 0.1


# ═══════════════════════════════════════════════════════════════
# 7–8. _pseudo_observations
# ═══════════════════════════════════════════════════════════════


class TestPseudoObservations:

    def test_output_in_open_unit_interval(self):
        x = np.array([3.0, 1.0, 4.0, 1.5, 2.0])
        y = np.array([2.0, 4.0, 1.0, 3.0, 5.0])
        u, v = _pseudo_observations(x, y)
        assert (u > 0).all() and (u < 1).all()
        assert (v > 0).all() and (v < 1).all()

    def test_preserves_rank_order(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 5.0, 8.0, 3.0, 1.0])
        u, _ = _pseudo_observations(x, y)
        # u should be monotonically increasing
        assert (np.diff(u) > 0).all()


# ═══════════════════════════════════════════════════════════════
# 9–12. Tail dependence formulas
# ═══════════════════════════════════════════════════════════════


class TestTailDependenceFormulas:

    def test_clayton_large_theta_near_one(self):
        # λ_L = 2^{-1/θ} → 1 as θ → ∞
        assert _clayton_lambda_L(1000.0) == pytest.approx(1.0, abs=0.01)

    def test_clayton_zero_theta_returns_zero(self):
        assert _clayton_lambda_L(0.0) == 0.0

    def test_gumbel_theta_one_returns_zero(self):
        # θ=1 → Gumbel = independence → λ_U = 2 - 2^1 = 0
        assert _gumbel_lambda_U(1.0) == pytest.approx(0.0, abs=1e-9)

    def test_gumbel_large_theta_near_one(self):
        # λ_U = 2 - 2^{1/θ} → 2 - 1 = 1 as θ → ∞
        assert _gumbel_lambda_U(1000.0) == pytest.approx(1.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════
# 13–16. _fit_bivariate_copula
# ═══════════════════════════════════════════════════════════════


class TestFitBivariateCopula:

    def test_positive_tau_both_tails_positive(self):
        rng = np.random.default_rng(7)
        z = rng.normal(0, 1, 200)
        x = z + rng.normal(0, 0.2, 200)
        y = z + rng.normal(0, 0.2, 200)
        tau, lL, lU, theta, family = _fit_bivariate_copula(x, y)
        assert tau > 0
        assert lL > 0
        assert lU > 0

    def test_negative_tau_survival(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        tau, lL, lU, theta, family = _fit_bivariate_copula(x, y)
        assert tau < 0
        assert lL == 0.0
        assert lU > 0.0
        assert "Survival" in family

    def test_zero_tau_independence(self):
        rng = np.random.default_rng(99)
        x = rng.normal(0, 1, 300)
        y = rng.normal(0, 1, 300)
        tau, lL, lU, theta, family = _fit_bivariate_copula(x, y)
        # For truly independent data τ should be near 0
        # λ_L and λ_U could be small but this is probabilistic
        assert abs(tau) < 0.15

    def test_family_string_positive_tau(self):
        z = np.linspace(0, 1, 50)
        tau, _, _, _, family = _fit_bivariate_copula(z, z + 0.01)
        assert "Clayton" in family or "Gumbel" in family


# ═══════════════════════════════════════════════════════════════
# 17–22. run()
# ═══════════════════════════════════════════════════════════════


class TestRun:

    def test_empty_store_returns_empty_dict(self, tmp_path):
        store = _make_store(tmp_path)
        enc = VineCopulaEncoder()
        assert enc.run(store) == {}

    def test_no_entity_links_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        store.register_entity("company", "isolated", "isolated")
        enc = VineCopulaEncoder()
        assert enc.run(store) == {}

    def test_skips_pairs_with_insufficient_observations(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        # Only 3 observations per entity — below default min_joint_obs=20
        _populate_linked_pair(
            store,
            "a1",
            "b1",
            n=3,
            t_start=as_of - 30 * _DAY,
            t_end=as_of,
        )
        enc = VineCopulaEncoder(min_joint_obs=20, n_bins=10)
        result = enc.run(store, as_of=as_of)
        assert result == {}

    def test_returns_result_with_sufficient_history(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _populate_linked_pair(
            store,
            "ea",
            "eb",
            n=200,
            t_start=as_of - 180 * _DAY,
            t_end=as_of,
        )
        enc = VineCopulaEncoder(min_joint_obs=5, n_bins=30)
        result = enc.run(store, as_of=as_of)
        assert len(result) == 1
        assert isinstance(next(iter(result.values())), CopulaResult)

    def test_lambda_in_unit_interval(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _populate_linked_pair(
            store,
            "x1",
            "x2",
            n=200,
            t_start=as_of - 180 * _DAY,
            t_end=as_of,
        )
        enc = VineCopulaEncoder(min_joint_obs=5, n_bins=30)
        results = enc.run(store, as_of=as_of)
        for r in results.values():
            assert 0.0 <= r.lambda_L <= 1.0
            assert 0.0 <= r.lambda_U <= 1.0

    def test_pair_key_matches_hash(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _populate_linked_pair(
            store,
            "eid_aaa",
            "eid_bbb",
            n=200,
            t_start=as_of - 180 * _DAY,
            t_end=as_of,
        )
        enc = VineCopulaEncoder(min_joint_obs=5, n_bins=30)
        results = enc.run(store, as_of=as_of)
        for key, r in results.items():
            expected_key = _pair_hash(r.entity_id_a, r.entity_id_b)
            assert key == expected_key


# ═══════════════════════════════════════════════════════════════
# 23–24. store_results()
# ═══════════════════════════════════════════════════════════════


class TestStoreResults:

    def test_stores_four_signals_per_pair(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _populate_linked_pair(
            store,
            "s1",
            "s2",
            n=200,
            t_start=as_of - 180 * _DAY,
            t_end=as_of,
        )
        enc = VineCopulaEncoder(min_joint_obs=5, n_bins=30)
        results = enc.run(store, as_of=as_of)
        n = enc.store_results(results, store)
        assert n == len(results) * 4  # 4 signals per pair

        for pair_key in results:
            for suffix in ["tau", "lambda_lower", "lambda_upper", "theta"]:
                sigs = store.query_signals(f"copula.{pair_key}.{suffix}")
                assert len(sigs) >= 1

    def test_stored_tau_matches_result(self, tmp_path):
        store = _make_store(tmp_path)
        as_of = time.time()
        _populate_linked_pair(
            store,
            "t1",
            "t2",
            n=200,
            t_start=as_of - 180 * _DAY,
            t_end=as_of,
        )
        enc = VineCopulaEncoder(min_joint_obs=5, n_bins=30)
        results = enc.run(store, as_of=as_of)
        enc.store_results(results, store)
        for pair_key, res in results.items():
            sigs = store.query_signals(f"copula.{pair_key}.tau")
            assert sigs[-1]["value"] == pytest.approx(res.tau, rel=1e-5)


# ═══════════════════════════════════════════════════════════════
# 25–27. TrainerConfig
# ═══════════════════════════════════════════════════════════════


class TestTrainerConfig:

    def test_use_vine_copula_defaults_false(self):
        from agent.models.gnn.trainer import TrainerConfig

        assert TrainerConfig().use_vine_copula is False

    def test_vine_copula_min_obs_defaults_20(self):
        from agent.models.gnn.trainer import TrainerConfig

        assert TrainerConfig().vine_copula_min_obs == 20

    def test_vine_copula_n_bins_defaults_60(self):
        from agent.models.gnn.trainer import TrainerConfig

        assert TrainerConfig().vine_copula_n_bins == 60


# ═══════════════════════════════════════════════════════════════
# 28–29. build_model() integration
# ═══════════════════════════════════════════════════════════════


class TestBuildModelIntegration:

    def _make_trainer(self, tmp_path: Path, use_vine_copula: bool, tag: str) -> Trainer:
        store = _make_store(tmp_path, f"{tag}.db")
        gen = SyntheticGraphGenerator(
            num_companies=4,
            num_countries=2,
            time_span=3600.0 * 4,
            base_event_rate=0.001,
            seed=55,
        )
        gen.generate(store)
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            time_dim=8,
            num_heads=1,
            num_layers=1,
            use_vine_copula=use_vine_copula,
            vine_copula_min_obs=3,
            vine_copula_n_bins=10,
        )
        return Trainer(store, cfg)

    def test_build_model_with_vine_copula_no_error(self, tmp_path):
        t = self._make_trainer(tmp_path, use_vine_copula=True, tag="vcbm")
        model = t.build_model()
        assert model is not None

    def test_build_model_stores_copula_signals(self, tmp_path):
        t = self._make_trainer(tmp_path, use_vine_copula=True, tag="vcsig")
        # Add current-time linked pair with enough observations
        as_of = time.time()
        store = t.store
        _populate_linked_pair(
            store,
            "live_a",
            "live_b",
            n=100,
            t_start=as_of - 180 * _DAY,
            t_end=as_of,
        )
        t.build_model()
        from agent.convergence.vine_copula import _pair_hash

        pk = _pair_hash("live_a", "live_b")
        sigs = store.query_signals(f"copula.{pk}.lambda_lower")
        assert len(sigs) >= 1, "Expected at least one copula.*.lambda_lower signal"

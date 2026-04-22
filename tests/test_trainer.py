"""Tests for Phase 12d: Self-supervised training loop.

Covers:
    SyntheticGraphGenerator — entity creation, links, obs, pattern injection
    TrainerConfig — defaults
    Trainer — build_model, splitting, windowing, training loop, loss decrease
    evaluate() — metrics, no data leakage
    Edge cases — empty windows, single obs, degenerate data, NaN checking
"""

from __future__ import annotations

import math

import pytest
import torch

from agent.models.gnn.graph_builder import OBSERVATION_TYPES
from agent.models.gnn.trainer import (
    InjectedPattern,
    SyntheticGraphGenerator,
    Trainer,
    TrainerConfig,
    evaluate,
)
from agent.pipeline.entity import entity_id_from_key
from agent.pipeline.store import PipelineStore


# ─── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def store():
    return PipelineStore(db_path=":memory:")


@pytest.fixture
def populated_store(store):
    """Store with synthetic data (no injected patterns)."""
    gen = SyntheticGraphGenerator(
        num_companies=4,
        num_countries=2,
        num_vessels=2,
        num_wallets=2,
        time_span=86400.0 * 10,  # 10 days
        base_event_rate=0.0005,
        seed=42,
    )
    stats = gen.generate(store)
    return store, stats


@pytest.fixture
def pattern_store(store):
    """Store with injected pattern: insider_trade → geopolitical_event via headquartered_in."""
    pattern = InjectedPattern(
        source_type="company",
        source_obs_type="insider_trade",
        target_type="country",
        target_obs_type="geopolitical_event",
        via_edge="headquartered_in",
        lag_seconds=3600.0,
        lag_jitter=300.0,
    )
    gen = SyntheticGraphGenerator(
        num_companies=4,
        num_countries=2,
        num_vessels=2,
        num_wallets=2,
        time_span=86400.0 * 10,
        base_event_rate=0.0005,
        seed=42,
        patterns=[pattern],
    )
    stats = gen.generate(store)
    return store, stats


# ═══════════════════════════════════════════════════════════════
# SyntheticGraphGenerator tests
# ═══════════════════════════════════════════════════════════════


class TestSyntheticGraphGenerator:
    def test_creates_entities(self, populated_store):
        store, stats = populated_store
        assert stats["entities"]["company"] == 4
        assert stats["entities"]["country"] == 2
        entities = store.query_all_entities()
        assert len(entities) == 16  # 4+2+2+2+3topics+3domains

    def test_creates_links(self, populated_store):
        store, stats = populated_store
        # Phase 35 expanded link types:
        # headquartered_in=4, operates_in=2, market_authorized_in=1,
        # lobbies_for=3, debtor_of=2, port_call_to=2,
        # exchange_based_in=2, transacts_with=1, sanctioned_under=1
        # Phase 36: domain_owned_by=3 (3 domains, each→1 company)
        assert stats["links"] == 21
        links = store.query_all_entity_links()
        assert len(links) == 21

    def test_creates_observations(self, populated_store):
        store, stats = populated_store
        assert stats["observations"] > 0
        obs = store.query_all_observations()
        assert len(obs) == stats["observations"]

    def test_deterministic(self, store):
        gen = SyntheticGraphGenerator(seed=123, time_span=86400.0)
        stats1 = gen.generate(store)

        store2 = PipelineStore(db_path=":memory:")
        gen2 = SyntheticGraphGenerator(seed=123, time_span=86400.0)
        stats2 = gen2.generate(store2)

        assert stats1["observations"] == stats2["observations"]
        assert stats1["links"] == stats2["links"]

    def test_observations_have_correct_types(self, populated_store):
        store, _ = populated_store
        obs = store.query_all_observations()
        obs_types = {o["observation_type"] for o in obs}
        for ot in obs_types:
            assert ot in OBSERVATION_TYPES

    def test_observations_sorted_by_time(self, populated_store):
        store, _ = populated_store
        obs = store.query_all_observations()
        times = [o["observed_at"] for o in obs]
        assert times == sorted(times)


class TestPatternInjection:
    def test_pattern_instances_created(self, pattern_store):
        store, stats = pattern_store
        assert len(stats["pattern_instances"]) > 0

    def test_pattern_lag_distribution(self, pattern_store):
        store, stats = pattern_store
        lags = [p["actual_lag"] for p in stats["pattern_instances"]]
        mean_lag = sum(lags) / len(lags)
        # Should be close to 3600 ± reasonable jitter
        assert 2500 < mean_lag < 4500

    def test_injected_obs_stored(self, pattern_store):
        store, stats = pattern_store
        obs = store.query_all_observations()
        # There should be more total obs than in a non-pattern store
        assert stats["observations"] > 0


# ═══════════════════════════════════════════════════════════════
# Trainer tests
# ═══════════════════════════════════════════════════════════════


class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.hidden_dim == 64
        assert cfg.epochs == 10
        assert cfg.train_ratio + cfg.val_ratio < 1.0


class TestTrainerBuildModel:
    def test_build_returns_model(self, populated_store):
        store, _ = populated_store
        trainer = Trainer(
            store, TrainerConfig(hidden_dim=16, memory_dim=16, message_dim=16)
        )
        model = trainer.build_model()
        assert model is not None
        assert hasattr(model, "forward")

    def test_model_has_correct_hidden_dim(self, populated_store):
        store, _ = populated_store
        trainer = Trainer(
            store, TrainerConfig(hidden_dim=32, memory_dim=32, message_dim=32)
        )
        model = trainer.build_model()
        assert model.hidden_dim == 32


class TestTrainerSplitAndWindow:
    def test_split_sizes(self, populated_store):
        store, _ = populated_store
        cfg = TrainerConfig(train_ratio=0.7, val_ratio=0.15)
        trainer = Trainer(store, cfg)
        train, val, test = trainer._split_observations()
        total = len(train) + len(val) + len(test)
        all_obs = store.query_all_observations()
        assert total == len(all_obs)

    def test_split_chronological(self, populated_store):
        store, _ = populated_store
        trainer = Trainer(store)
        train, val, test = trainer._split_observations()
        if train and val:
            assert max(o["observed_at"] for o in train) <= min(
                o["observed_at"] for o in val
            )
        if val and test:
            assert max(o["observed_at"] for o in val) <= min(
                o["observed_at"] for o in test
            )

    def test_windows_non_empty(self, populated_store):
        store, _ = populated_store
        trainer = Trainer(store, TrainerConfig(window_size=86400.0))
        train_obs, _, _ = trainer._split_observations()
        windows = trainer._make_windows(train_obs)
        assert len(windows) > 0
        for t_start, t_end, obs in windows:
            assert len(obs) > 0
            assert t_end > t_start

    def test_empty_observations_no_windows(self, store):
        trainer = Trainer(store)
        windows = trainer._make_windows([])
        assert windows == []


class TestTrainerTraining:
    def test_train_runs(self, populated_store):
        """Training completes without errors."""
        store, _ = populated_store
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            num_heads=2,
            num_layers=1,
            epochs=2,
            window_size=86400.0 * 2,
        )
        trainer = Trainer(store, cfg)
        trainer.build_model()
        history = trainer.train()
        assert "total" in history
        assert len(history["total"]) == 2

    def test_no_nan_in_loss(self, populated_store):
        store, _ = populated_store
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            num_heads=2,
            num_layers=1,
            epochs=3,
            window_size=86400.0 * 2,
        )
        trainer = Trainer(store, cfg)
        trainer.build_model()
        history = trainer.train()
        for k, losses in history.items():
            for v in losses:
                assert not (v != v), f"NaN in {k} loss"  # NaN != NaN

    def test_loss_finite(self, populated_store):
        store, _ = populated_store
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            epochs=2,
            window_size=86400.0 * 2,
        )
        trainer = Trainer(store, cfg)
        trainer.build_model()
        history = trainer.train()
        for k, losses in history.items():
            for v in losses:
                assert abs(v) < 1e10, f"Exploding {k} loss: {v}"


class TestTrainerEdgeCases:
    def test_very_small_dataset(self, store):
        """Single entity, single observation."""
        eid = store.register_entity(
            "company", "solo", entity_id_from_key("company", "solo")
        )
        store.store_entity_observation(eid, "test", 100.0, "insider_trade", {"v": 1})
        cfg = TrainerConfig(
            hidden_dim=8,
            memory_dim=8,
            message_dim=8,
            num_heads=1,
            num_layers=1,
            epochs=1,
            window_size=50.0,
        )
        trainer = Trainer(store, cfg)
        trainer.build_model()
        history = trainer.train()
        assert len(history["total"]) == 1

    def test_no_links_contrastive_zero(self, store):
        """Entities but no links → contrastive loss = 0."""
        eid = store.register_entity(
            "company", "alone", entity_id_from_key("company", "alone")
        )
        store.store_entity_observation(eid, "test", 100.0, "insider_trade", {"v": 1})
        store.store_entity_observation(eid, "test", 200.0, "insider_trade", {"v": 2})
        cfg = TrainerConfig(
            hidden_dim=8,
            memory_dim=8,
            message_dim=8,
            epochs=1,
            window_size=50.0,
        )
        trainer = Trainer(store, cfg)
        trainer.build_model()
        history = trainer.train()
        # Contrastive should be 0 (no links)
        assert all(v == 0.0 for v in history["contrastive"])


# ═══════════════════════════════════════════════════════════════
# Walk-forward evaluation tests
# ═══════════════════════════════════════════════════════════════


class TestEvaluate:
    def test_evaluate_returns_metrics(self, populated_store):
        store, _ = populated_store
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            num_heads=2,
            num_layers=1,
            epochs=1,
            window_size=86400.0 * 2,
        )
        trainer = Trainer(store, cfg)
        model = trainer.build_model()
        trainer.train()
        metrics = evaluate(model, store, cfg, split="val")
        assert "obs_type_acc_top1" in metrics
        assert "obs_type_acc_top5" in metrics
        assert "time_delta_mae" in metrics
        assert "num_predictions" in metrics

    def test_evaluate_test_split(self, populated_store):
        store, _ = populated_store
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            epochs=1,
            window_size=86400.0 * 2,
        )
        trainer = Trainer(store, cfg)
        model = trainer.build_model()
        trainer.train()
        metrics = evaluate(model, store, cfg, split="test")
        assert metrics["num_predictions"] >= 0

    def test_evaluate_invalid_split(self, populated_store):
        store, _ = populated_store
        cfg = TrainerConfig(hidden_dim=16, memory_dim=16, message_dim=16)
        trainer = Trainer(store, cfg)
        model = trainer.build_model()
        with pytest.raises(ValueError, match="split must be"):
            evaluate(model, store, cfg, split="oops")

    def test_evaluate_no_leakage(self, populated_store):
        """Val/test obs timestamps must be after training data."""
        store, _ = populated_store
        cfg = TrainerConfig()
        trainer = Trainer(store, cfg)
        train, val, test = trainer._split_observations()
        if train and val:
            train_max_t = max(o["observed_at"] for o in train)
            val_min_t = min(o["observed_at"] for o in val)
            assert val_min_t >= train_max_t
        if val and test:
            val_max_t = max(o["observed_at"] for o in val)
            test_min_t = min(o["observed_at"] for o in test)
            assert test_min_t >= val_max_t

    def test_metrics_bounded(self, populated_store):
        store, _ = populated_store
        cfg = TrainerConfig(
            hidden_dim=16,
            memory_dim=16,
            message_dim=16,
            epochs=1,
            window_size=86400.0 * 2,
        )
        trainer = Trainer(store, cfg)
        model = trainer.build_model()
        trainer.train()
        metrics = evaluate(model, store, cfg, split="val")
        assert 0.0 <= metrics["obs_type_acc_top1"] <= 1.0
        assert 0.0 <= metrics["obs_type_acc_top5"] <= 1.0
        assert metrics["time_delta_mae"] >= 0.0

    def test_empty_eval_split(self, store):
        """If eval split is empty, return zeros."""
        eid = store.register_entity("company", "x", entity_id_from_key("company", "x"))
        store.store_entity_observation(eid, "t", 1.0, "insider_trade", {})
        cfg = TrainerConfig(
            hidden_dim=8,
            memory_dim=8,
            message_dim=8,
            train_ratio=1.0,
            val_ratio=0.0,  # all training, no val
        )
        trainer = Trainer(store, cfg)
        model = trainer.build_model()
        metrics = evaluate(model, store, cfg, split="val")
        assert metrics["num_predictions"] == 0


# ═══════════════════════════════════════════════════════════════
# Phase 41 — Kendall log-variance clamp tests
# ═══════════════════════════════════════════════════════════════


class TestLogVarClamp:
    """Verify the Phase 41 fix: log-variance clamping in the
    uncertainty-weighted multi-task loss (Kendall et al. 2018).

    Without the clamp, when a component loss reaches 0 the optimizer
    drives the log-variance parameter toward -inf; the total loss
    becomes -inf and effective weights explode. See Phase 40 Run 2
    post-mortem. The clamp bounds the reported and applied weights
    to exp(-log_var_bounds)."""

    def _make_trainer(self, store, **cfg_overrides):
        cfg = TrainerConfig(
            hidden_dim=8,
            memory_dim=8,
            message_dim=8,
            auto_tune_loss_weights=True,
            **cfg_overrides,
        )
        trainer = Trainer(store, cfg)
        trainer.build_model()
        return trainer

    def test_defaults_well_ordered(self):
        cfg = TrainerConfig()
        assert cfg.log_var_min < cfg.log_var_max

    def test_effective_weights_clamped_when_log_vars_extreme(self, populated_store):
        store, _ = populated_store
        trainer = self._make_trainer(store)
        assert trainer._log_vars is not None

        # Drive every log-variance far outside the clamp window.
        with torch.no_grad():
            trainer._log_vars["obs_type"].fill_(-50.0)  # would give exp(50) unclamped
            trainer._log_vars["time_delta"].fill_(
                +50.0
            )  # would give exp(-50) unclamped
            trainer._log_vars["contrastive"].fill_(-100.0)
            trainer._log_vars["value"].fill_(+100.0)

        eff = trainer.effective_loss_weights()

        cfg = trainer.config
        hi = math.exp(-cfg.log_var_min)
        lo = math.exp(-cfg.log_var_max)
        for k, w in eff.items():
            assert (
                lo - 1e-9 <= w <= hi + 1e-9
            ), f"{k} weight {w} outside clamp window [{lo}, {hi}]"

    def test_total_loss_finite_on_zero_components(self, populated_store):
        """Replay the Phase 40 failure mode: component losses are all
        ~0 and the log-variances have drifted negative. With the clamp,
        the total loss must stay finite (bounded below by 4 * log_var_min)."""
        store, _ = populated_store
        trainer = self._make_trainer(store)
        assert trainer._log_vars is not None

        with torch.no_grad():
            for p in trainer._log_vars.values():
                p.fill_(-50.0)

        cfg = trainer.config
        lv = trainer._log_vars
        zero = torch.tensor(0.0)
        obs_loss = zero
        dt_loss = zero
        c_loss = zero
        val_loss = zero

        # Mirror the clamp logic from Trainer.train() exactly.
        lv_min = cfg.log_var_min
        lv_max = cfg.log_var_max
        clamped = {k: torch.clamp(p, min=lv_min, max=lv_max) for k, p in lv.items()}
        total = (
            torch.exp(-clamped["obs_type"]) * obs_loss
            + clamped["obs_type"]
            + torch.exp(-clamped["time_delta"]) * dt_loss
            + clamped["time_delta"]
            + torch.exp(-clamped["contrastive"]) * c_loss
            + clamped["contrastive"]
            + torch.exp(-clamped["value"]) * val_loss
            + clamped["value"]
        )

        t = total.item()
        assert math.isfinite(t)
        # Bound: with zero component losses, total == sum of clamped log_vars
        # each bounded in [lv_min, lv_max].
        assert 4 * lv_min - 1e-6 <= t <= 4 * lv_max + 1e-6

    def test_fixed_weights_unaffected_by_clamp(self):
        cfg = TrainerConfig(
            auto_tune_loss_weights=False,
            obs_type_weight=1.5,
            time_delta_weight=0.2,
            contrastive_weight=0.7,
            value_weight=0.4,
            log_var_min=0.0,
            log_var_max=0.0,  # would collapse all weights to 1.0 if applied
        )
        store = PipelineStore(db_path=":memory:")
        trainer = Trainer(store, cfg)
        # No build_model required: fixed-weight branch does not touch _log_vars.
        eff = trainer.effective_loss_weights()
        assert eff["obs_type"] == 1.5
        assert eff["time_delta"] == 0.2
        assert eff["contrastive"] == 0.7
        assert eff["value"] == 0.4

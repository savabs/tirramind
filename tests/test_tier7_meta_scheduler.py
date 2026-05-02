"""
Tests for Meta-Learned Scheduling (Change 14, Tier 7).

Covers: Thompson Sampling convergence, reward computation, persistence,
cold start defaults, explicit overrides, diagnostics, component performance
storage, edge cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.learning.meta_scheduler import (
    DEFAULT_COMPONENTS,
    ComponentConfig,
    MetaScheduler,
    _sigmoid,
    compute_refit_reward,
)

# ── Helpers ───────────────────────────────────────────────────


def _make_scheduler(**kwargs) -> MetaScheduler:
    defaults = dict(seed=42)
    defaults.update(kwargs)
    return MetaScheduler(**defaults)


def _simple_components() -> dict[str, ComponentConfig]:
    return {
        "fast": ComponentConfig(arms=(1, 5, 10), default=5),
        "slow": ComponentConfig(arms=(30, 90), default=90),
    }


# ═══════════════════════════════════════════════════════════════
# §1 — Thompson Sampling Basics
# ═══════════════════════════════════════════════════════════════


class TestThompsonSampling:
    """Verify core Thompson Sampling mechanics."""

    def test_suggest_returns_valid_arm(self):
        s = _make_scheduler()
        for comp in s.component_names:
            arm = s.suggest(comp)
            assert arm in DEFAULT_COMPONENTS[comp].arms

    def test_suggest_unknown_component_raises(self):
        s = _make_scheduler()
        with pytest.raises(ValueError, match="Unknown component"):
            s.suggest("nonexistent")

    def test_record_unknown_component_raises(self):
        s = _make_scheduler()
        with pytest.raises(ValueError, match="Unknown component"):
            s.record_outcome("nonexistent", 7, 0.5)

    def test_record_unknown_arm_raises(self):
        s = _make_scheduler()
        with pytest.raises(ValueError, match="Unknown arm"):
            s.record_outcome("cpd_fit", 999, 0.5)

    def test_cold_start_uniform_prior(self):
        """With uniform priors, all arms should be roughly equally likely."""
        s = _make_scheduler(components=_simple_components())
        counts = {1: 0, 5: 0, 10: 0}
        for _ in range(300):
            arm = s.suggest("fast")
            counts[arm] += 1
        # Each arm should get at least 50 pulls out of 300 (uniform prior)
        for arm, c in counts.items():
            assert c > 30, f"Arm {arm} pulled only {c} times (expected ≈100)"

    def test_convergence_to_best_arm(self):
        """With consistent high reward for one arm, bandit should converge."""
        s = _make_scheduler(components=_simple_components(), seed=123)
        # Arm 5 always gives high reward, others low
        for _ in range(50):
            arm = s.suggest("fast")
            if arm == 5:
                s.record_outcome("fast", arm, 0.9)
            else:
                s.record_outcome("fast", arm, 0.1)

        # After training, arm 5 should be selected most often
        selections = [s.suggest("fast") for _ in range(100)]
        arm5_count = selections.count(5)
        assert arm5_count > 60, f"Best arm selected {arm5_count}/100 times"

    def test_reward_clamped_to_01(self):
        """Rewards outside [0,1] should be clamped."""
        s = _make_scheduler(components=_simple_components())
        s.record_outcome("fast", 5, 2.0)  # Should clamp to 1.0
        s.record_outcome("fast", 5, -1.0)  # Should clamp to 0.0
        diag = s.diagnostics()
        # alpha should be 1 + 1.0 + 0.0 = 2.0
        assert abs(diag["fast"]["arms"][5]["alpha"] - 2.0) < 0.01
        # beta should be 1 + 0.0 + 1.0 = 2.0
        assert abs(diag["fast"]["arms"][5]["beta"] - 2.0) < 0.01

    def test_beta_posterior_updates(self):
        s = _make_scheduler(components=_simple_components())
        s.record_outcome("fast", 5, 0.7)
        diag = s.diagnostics()
        assert abs(diag["fast"]["arms"][5]["alpha"] - 1.7) < 0.01
        assert abs(diag["fast"]["arms"][5]["beta"] - 1.3) < 0.01
        assert diag["fast"]["arms"][5]["pulls"] == 1


# ═══════════════════════════════════════════════════════════════
# §2 — Reward Computation
# ═══════════════════════════════════════════════════════════════


class TestRewardComputation:
    """Verify reward functions for each component."""

    def test_cpd_fit_improvement_high_reward(self):
        r = compute_refit_reward(
            "cpd_fit",
            {"total_bic": -100.0},
            {"total_bic": -80.0},  # BIC improved by 20
        )
        assert r > 0.5, f"Expected reward > 0.5 for BIC improvement, got {r}"

    def test_cpd_fit_degradation_low_reward(self):
        r = compute_refit_reward(
            "cpd_fit",
            {"total_bic": -80.0},
            {"total_bic": -100.0},  # BIC degraded
        )
        assert r < 0.5

    def test_cpd_fit_no_change_half_reward(self):
        r = compute_refit_reward(
            "cpd_fit",
            {"total_bic": -80.0},
            {"total_bic": -80.0},
        )
        assert abs(r - 0.5) < 0.01

    def test_structure_refine_with_changes(self):
        r = compute_refit_reward(
            "structure_refine",
            {},
            {"n_confident_changes": 3.0},
        )
        assert r > 0.5

    def test_structure_refine_no_changes(self):
        r = compute_refit_reward(
            "structure_refine",
            {},
            {"n_confident_changes": 0.0},
        )
        assert abs(r - 0.5) < 0.01

    def test_gnn_loss_improved(self):
        r = compute_refit_reward(
            "gnn_epochs",
            {"val_loss": 1.5},
            {"val_loss": 0.8},  # loss decreased
        )
        assert r > 0.5

    def test_gnn_loss_worsened(self):
        r = compute_refit_reward(
            "gnn_epochs",
            {"val_loss": 0.8},
            {"val_loss": 1.5},
        )
        assert r < 0.5

    def test_history_window_improvement(self):
        r = compute_refit_reward(
            "history_window",
            {"held_out_bic": -50.0},
            {"held_out_bic": -30.0},
        )
        assert r > 0.5

    def test_unknown_component_returns_half(self):
        r = compute_refit_reward("unknown_thing", {}, {})
        assert abs(r - 0.5) < 0.01

    def test_sigmoid_properties(self):
        assert abs(_sigmoid(0) - 0.5) < 1e-10
        assert _sigmoid(50) > 0.99
        assert _sigmoid(-50) < 0.01


# ═══════════════════════════════════════════════════════════════
# §3 — Persistence
# ═══════════════════════════════════════════════════════════════


class TestPersistence:
    """Test save/load round-trip."""

    def test_round_trip_json(self, tmp_path: Path):
        path = tmp_path / "scheduler.json"
        s = _make_scheduler(components=_simple_components(), persist_path=path)
        s.record_outcome("fast", 5, 0.8)
        s.record_outcome("slow", 90, 0.3)
        s.save()

        s2 = MetaScheduler(components=_simple_components(), persist_path=path)
        diag1 = s.diagnostics()
        diag2 = s2.diagnostics()
        assert diag1["fast"]["arms"][5]["alpha"] == diag2["fast"]["arms"][5]["alpha"]
        assert diag1["slow"]["arms"][90]["beta"] == diag2["slow"]["arms"][90]["beta"]

    def test_corrupted_json_uses_fresh(self, tmp_path: Path):
        path = tmp_path / "scheduler.json"
        path.write_text("not valid json {{{")
        s = MetaScheduler(components=_simple_components(), persist_path=path)
        # Should load fresh state without crashing
        assert s.diagnostics()["fast"]["total_pulls"] == 0

    def test_missing_file_uses_fresh(self, tmp_path: Path):
        path = tmp_path / "nonexistent.json"
        s = MetaScheduler(components=_simple_components(), persist_path=path)
        assert s.diagnostics()["fast"]["total_pulls"] == 0

    def test_to_dict_from_dict_round_trip(self):
        s = _make_scheduler(components=_simple_components())
        s.record_outcome("fast", 5, 0.7)
        data = s.to_dict()
        s2 = MetaScheduler.from_dict(data, seed=42)
        assert s.diagnostics() == s2.diagnostics()

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "nested" / "deep" / "scheduler.json"
        s = _make_scheduler(persist_path=path)
        s.save()
        assert path.exists()


# ═══════════════════════════════════════════════════════════════
# §4 — Diagnostics
# ═══════════════════════════════════════════════════════════════


class TestDiagnostics:
    """Verify diagnostics output."""

    def test_diagnostics_structure(self):
        s = _make_scheduler()
        diag = s.diagnostics()
        assert "cpd_fit" in diag
        assert "structure_refine" in diag
        assert "gnn_epochs" in diag
        assert "history_window" in diag
        for comp in diag.values():
            assert "default" in comp
            assert "arms" in comp
            assert "total_pulls" in comp

    def test_diagnostics_after_pulls(self):
        s = _make_scheduler(components=_simple_components())
        s.record_outcome("fast", 5, 0.6)
        s.record_outcome("fast", 5, 0.8)
        diag = s.diagnostics()
        assert diag["fast"]["arms"][5]["pulls"] == 2
        assert abs(diag["fast"]["arms"][5]["mean_reward"] - 0.7) < 0.01
        assert diag["fast"]["total_pulls"] == 2

    def test_posterior_mean(self):
        s = _make_scheduler(components=_simple_components())
        # Uniform prior: α=1, β=1 → posterior_mean = 0.5
        diag = s.diagnostics()
        assert abs(diag["fast"]["arms"][5]["posterior_mean"] - 0.5) < 0.01

    def test_component_names(self):
        s = _make_scheduler(components=_simple_components())
        assert set(s.component_names) == {"fast", "slow"}


# ═══════════════════════════════════════════════════════════════
# §5 — Default Components
# ═══════════════════════════════════════════════════════════════


class TestDefaultComponents:
    """Verify default component configuration."""

    def test_default_components_present(self):
        assert "cpd_fit" in DEFAULT_COMPONENTS
        assert "structure_refine" in DEFAULT_COMPONENTS
        assert "gnn_epochs" in DEFAULT_COMPONENTS
        assert "history_window" in DEFAULT_COMPONENTS

    def test_default_cpd_fit_arms(self):
        cfg = DEFAULT_COMPONENTS["cpd_fit"]
        assert cfg.default == 7
        assert 3 in cfg.arms
        assert 30 in cfg.arms

    def test_default_gnn_epochs_arms(self):
        cfg = DEFAULT_COMPONENTS["gnn_epochs"]
        assert cfg.default == 10
        assert 5 in cfg.arms
        assert 40 in cfg.arms

    def test_scheduler_with_defaults(self):
        s = _make_scheduler()
        for comp in DEFAULT_COMPONENTS:
            arm = s.suggest(comp)
            assert arm in DEFAULT_COMPONENTS[comp].arms


# ═══════════════════════════════════════════════════════════════
# §6 — Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestSchedulerEdgeCases:
    """Edge cases and boundary conditions."""

    def test_no_persist_path_save_is_noop(self):
        s = _make_scheduler(persist_path=None)
        s.save()  # Should not crash

    def test_single_arm_always_returns_it(self):
        single = {"only": ComponentConfig(arms=(42,), default=42)}
        s = MetaScheduler(components=single, seed=0)
        for _ in range(20):
            assert s.suggest("only") == 42

    def test_many_pulls_posterior_dominates(self):
        """After many pulls, the posterior should be heavily weighted."""
        comps = {"test": ComponentConfig(arms=(1, 2), default=1)}
        s = MetaScheduler(components=comps, seed=42)
        # Give arm 2 a lot of reward
        for _ in range(100):
            s.record_outcome("test", 2, 0.95)
            s.record_outcome("test", 1, 0.05)
        diag = s.diagnostics()
        assert diag["test"]["arms"][2]["posterior_mean"] > 0.8
        assert diag["test"]["arms"][1]["posterior_mean"] < 0.2

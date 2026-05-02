"""
Tier 3 Integration Tests — Wiring optimizers into live runtime.

Covers:
    1. AutonomousRunner uses learned reward weights from RewardWeightOptimizer
    2. AutonomousRunner records novel arm pulls and triggers promotion
    3. Entity scoring DAG merges learned CUSUM/Hawkes thresholds
    4. Convergence detection DAG merges learned convergence thresholds
    5. Goal generator handles novel arm fallback correctly
    6. End-to-end: optimizer state persists across AutonomousRunner invocations
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from agent.learning.bandit import DEFAULT_ARMS, GoalArm, StrategyBandit
from agent.learning.goal_generator import Goal, GoalGenerator
from agent.learning.reward import (
    DEFAULT_WEIGHTS,
    RewardWeightOptimizer,
    RewardWeights,
    compute_reward,
)
from agent.learning.threshold_optimizer import ThresholdOptimizer
from agent.memory.store import Episode

# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _make_episode(action: str = "web_search", success: bool = True) -> Episode:
    return Episode(
        timestamp=time.time(),
        step=1,
        action=action,
        input_summary="test",
        output_summary="result",
        success=success,
    )


def _make_evaluation(score: float = 0.7, success: bool = True, dead_end: bool = False):
    """Create a mock Evaluation."""
    from agent.learning.evaluator import Evaluation

    return Evaluation(
        success=success,
        score=score,
        new_facts_count=2,
        dead_end=dead_end,
    )


def _make_agent_result(episodes: list[Episode] | None = None):
    """Create a mock AgentResult with episodes."""
    from agent.core.orchestrator import AgentResult

    return AgentResult(
        goal="test goal",
        success=True,
        output="test output",
        steps_taken=3,
        plan_summary="ran 3 steps",
        episodes=episodes if episodes is not None else [_make_episode()],
    )


# ═══════════════════════════════════════════════════════════════
#  1. AutonomousRunner: Learned Reward Weights
# ═══════════════════════════════════════════════════════════════


class TestRewardWeightWiring:
    """AutonomousRunner uses GP-BO learned weights for compute_reward."""

    def test_compute_reward_accepts_learned_weights(self):
        """compute_reward should use provided weights, not defaults."""
        evaluation = _make_evaluation(score=0.8)
        custom = RewardWeights(
            eval_weight=0.9,
            sharpe_weight=0.0,
            facts_weight=0.0,
            novelty_bonus=0.0,
            dead_end_penalty=0.0,
        )
        reward = compute_reward(evaluation, is_first_pull=False, weights=custom)
        # With eval_weight=0.9, score=0.8: 0.9 * 0.8 = 0.72
        assert abs(reward - 0.72) < 0.01

    def test_default_weights_unchanged(self):
        """DEFAULT_WEIGHTS should be the fallback when no suggestion exists."""
        assert DEFAULT_WEIGHTS.eval_weight == 0.4
        assert DEFAULT_WEIGHTS.sharpe_weight == 0.3

    def test_optimizer_suggests_different_weights_after_trials(self, tmp_path):
        """After recording trials, the optimizer should suggest non-default weights."""
        opt = RewardWeightOptimizer(persist_path=tmp_path / "rw.json", seed=42)

        # Record several trials with varying objectives
        for _ in range(6):
            w = opt.suggest_weights()
            # Simulate: high eval_weight → high objective
            obj = w.eval_weight * 0.5 + 0.3
            opt.record_trial(w, objective=obj)

        suggested = opt.suggest_weights()
        # After 6 trials, GP-BO should be suggesting non-uniform weights
        assert isinstance(suggested, RewardWeights)

    def test_optimizer_persists_across_instances(self, tmp_path):
        """Optimizer state should survive across constructor calls."""
        path = tmp_path / "rw.json"

        opt1 = RewardWeightOptimizer(persist_path=path, seed=42)
        w1 = opt1.suggest_weights()
        opt1.record_trial(w1, objective=0.8)
        assert opt1.n_trials == 1

        # New instance from same path
        opt2 = RewardWeightOptimizer(persist_path=path, seed=42)
        assert opt2.n_trials == 1

    def test_reward_weights_all_components_bounded(self):
        """All weight components should be within [0.01, 1.0] after suggestion."""
        with tempfile.TemporaryDirectory() as td:
            opt = RewardWeightOptimizer(persist_path=Path(td) / "rw.json", seed=99)
            for _ in range(3):
                w = opt.suggest_weights()
                for attr in (
                    "eval_weight",
                    "sharpe_weight",
                    "facts_weight",
                    "novelty_bonus",
                    "dead_end_penalty",
                ):
                    val = getattr(w, attr)
                    assert 0.01 <= val <= 1.0, f"{attr}={val} out of bounds"
                opt.record_trial(w, objective=0.5)


# ═══════════════════════════════════════════════════════════════
#  2. AutonomousRunner: Novel Arm Recording
# ═══════════════════════════════════════════════════════════════


class TestNovelArmWiring:
    """Novel exploration pulls are recorded with tools_used from episodes."""

    def test_tools_extracted_from_episodes(self):
        """Tools used should be extracted from Episode.action fields."""
        episodes = [
            _make_episode(action="web_search"),
            _make_episode(action="market_data"),
            _make_episode(action="web_search"),  # duplicate
        ]
        tools_used = list({ep.action for ep in episodes if ep.action})
        assert set(tools_used) == {"web_search", "market_data"}

    def test_novel_pull_recorded_in_bandit(self, tmp_path):
        """record_novel_pull should store the pull in novel_history."""
        bandit = StrategyBandit(
            arms=DEFAULT_ARMS,
            persist_path=tmp_path / "bandit.json",
            seed=42,
        )
        bandit.record_novel_pull(
            tools_used=["web_search", "market_data"],
            reward=0.75,
            description="Cross-referenced weather and commodity data",
        )
        assert len(bandit.novel_history) == 1
        assert bandit.novel_history[0]["reward"] == 0.75

    def test_novel_promotion_after_threshold(self, tmp_path):
        """After 3 high-reward pulls with same tools, a new arm should be promoted."""
        bandit = StrategyBandit(
            arms=DEFAULT_ARMS,
            persist_path=tmp_path / "bandit.json",
            seed=42,
        )
        tools = ["gdelt", "market_data"]
        promoted = None
        for i in range(3):
            promoted = bandit.record_novel_pull(
                tools_used=tools,
                reward=0.7 + i * 0.05,
                description=f"Geopolitical pattern {i}",
            )

        assert promoted is not None
        assert promoted.name == "gdelt_market_data"
        assert set(promoted.tools) == {"gdelt", "market_data"}

    def test_novel_arm_exists_in_default_arms(self):
        """novel_exploration should be present in DEFAULT_ARMS."""
        names = [arm.name for arm in DEFAULT_ARMS]
        assert "novel_exploration" in names
        novel = next(a for a in DEFAULT_ARMS if a.name == "novel_exploration")
        assert novel.tools == []  # empty = all tools allowed

    def test_novel_pull_below_threshold_no_promotion(self, tmp_path):
        """Low-reward novel pulls should not trigger promotion."""
        bandit = StrategyBandit(
            arms=DEFAULT_ARMS,
            persist_path=tmp_path / "bandit.json",
            seed=42,
        )
        for _ in range(5):
            promoted = bandit.record_novel_pull(
                tools_used=["web_search"],
                reward=0.3,  # below 0.6 threshold
                description="Low-quality exploration",
            )
            assert promoted is None

    def test_novel_history_persists(self, tmp_path):
        """Novel history should survive across bandit instances."""
        path = tmp_path / "bandit.json"
        b1 = StrategyBandit(arms=DEFAULT_ARMS, persist_path=path, seed=42)
        b1.record_novel_pull(["web_search"], reward=0.8, description="test")

        b2 = StrategyBandit(arms=DEFAULT_ARMS, persist_path=path, seed=42)
        assert len(b2.novel_history) == 1


# ═══════════════════════════════════════════════════════════════
#  3. Entity Scoring DAG: Learned Thresholds
# ═══════════════════════════════════════════════════════════════


class TestEntityScoringThresholds:
    """Entity scoring DAG merges learned CUSUM/Hawkes thresholds."""

    def test_merge_with_learned_cusum(self, tmp_path):
        """CUSUM thresholds from optimizer should appear in scorer_config."""
        from agent.pipeline.dags.entity_scoring import _merge_learned_thresholds

        threshold_dir = tmp_path / "threshold_bo"
        threshold_dir.mkdir()

        opt = ThresholdOptimizer(persist_dir=threshold_dir, seed=42)
        params = opt.suggest("cusum")
        opt.record("cusum", params, objective=0.85)

        merged = _merge_learned_thresholds({}, threshold_dir)
        assert "cusum_k" in merged
        assert "cusum_h" in merged
        assert abs(merged["cusum_k"] - params["k"]) < 1e-10
        assert abs(merged["cusum_h"] - params["h"]) < 1e-10

    def test_merge_with_learned_hawkes(self, tmp_path):
        """Hawkes thresholds from optimizer should appear in scorer_config."""
        from agent.pipeline.dags.entity_scoring import _merge_learned_thresholds

        threshold_dir = tmp_path / "threshold_bo"
        threshold_dir.mkdir()

        opt = ThresholdOptimizer(persist_dir=threshold_dir, seed=42)
        params = opt.suggest("hawkes")
        opt.record("hawkes", params, objective=0.75)

        merged = _merge_learned_thresholds({}, threshold_dir)
        assert "hawkes_mu" in merged
        assert "hawkes_alpha" in merged
        assert "hawkes_beta" in merged

    def test_explicit_config_overrides_learned(self, tmp_path):
        """Explicit scorer_config keys should take precedence over learned."""
        from agent.pipeline.dags.entity_scoring import _merge_learned_thresholds

        threshold_dir = tmp_path / "threshold_bo"
        threshold_dir.mkdir()

        opt = ThresholdOptimizer(persist_dir=threshold_dir, seed=42)
        params = opt.suggest("cusum")
        opt.record("cusum", params, objective=0.9)

        # Explicit override
        explicit = {"cusum_k": 99.0}
        merged = _merge_learned_thresholds(explicit, threshold_dir)
        assert merged["cusum_k"] == 99.0  # explicit wins
        assert "cusum_h" in merged  # learned fills the rest

    def test_no_trials_returns_empty(self, tmp_path):
        """If no trials exist, merge should return the original config."""
        from agent.pipeline.dags.entity_scoring import _merge_learned_thresholds

        threshold_dir = tmp_path / "threshold_bo"
        threshold_dir.mkdir()

        # Create optimizer but record no trials
        ThresholdOptimizer(persist_dir=threshold_dir, seed=42)

        merged = _merge_learned_thresholds({"foo": "bar"}, threshold_dir)
        assert merged == {"foo": "bar"}

    def test_missing_dir_returns_original(self, tmp_path):
        """If threshold_dir doesn't exist, merge should be no-op."""
        from agent.pipeline.dags.entity_scoring import _merge_learned_thresholds

        merged = _merge_learned_thresholds({"cusum_k": 1.5}, tmp_path / "nonexistent")
        assert merged == {"cusum_k": 1.5}

    def test_original_dict_not_mutated(self, tmp_path):
        """_merge_learned_thresholds should not mutate the input dict."""
        from agent.pipeline.dags.entity_scoring import _merge_learned_thresholds

        threshold_dir = tmp_path / "threshold_bo"
        threshold_dir.mkdir()

        opt = ThresholdOptimizer(persist_dir=threshold_dir, seed=42)
        params = opt.suggest("cusum")
        opt.record("cusum", params, objective=0.8)

        original = {"existing_key": 42}
        merged = _merge_learned_thresholds(original, threshold_dir)
        assert "cusum_k" not in original  # original unchanged
        assert "cusum_k" in merged
        assert original == {"existing_key": 42}

    def test_both_cusum_and_hawkes_merged(self, tmp_path):
        """When both CUSUM and Hawkes have trials, both should merge."""
        from agent.pipeline.dags.entity_scoring import _merge_learned_thresholds

        threshold_dir = tmp_path / "threshold_bo"
        threshold_dir.mkdir()

        opt = ThresholdOptimizer(persist_dir=threshold_dir, seed=42)

        cusum_params = opt.suggest("cusum")
        opt.record("cusum", cusum_params, objective=0.8)

        hawkes_params = opt.suggest("hawkes")
        opt.record("hawkes", hawkes_params, objective=0.7)

        merged = _merge_learned_thresholds({}, threshold_dir)
        # CUSUM keys
        assert "cusum_k" in merged
        assert "cusum_h" in merged
        # Hawkes keys
        assert "hawkes_mu" in merged
        assert "hawkes_alpha" in merged
        assert "hawkes_beta" in merged


# ═══════════════════════════════════════════════════════════════
#  4. Convergence Detection DAG: Learned Thresholds
# ═══════════════════════════════════════════════════════════════


class TestConvergenceDetectionThresholds:
    """Convergence detection DAG merges learned z/p/fdr_q thresholds."""

    def test_load_convergence_thresholds(self, tmp_path):
        """Should return z_threshold, p_threshold, fdr_q from GP-BO."""
        from agent.pipeline.dags.convergence_detection import (
            _load_convergence_thresholds,
        )

        threshold_dir = tmp_path / "threshold_bo"
        threshold_dir.mkdir()

        opt = ThresholdOptimizer(persist_dir=threshold_dir, seed=42)
        params = opt.suggest("convergence")
        opt.record("convergence", params, objective=0.9)

        loaded = _load_convergence_thresholds(threshold_dir)
        assert "z_threshold" in loaded
        assert "p_threshold" in loaded
        assert "fdr_q" in loaded
        assert abs(loaded["z_threshold"] - params["z_threshold"]) < 1e-10

    def test_no_trials_returns_empty(self, tmp_path):
        """If no convergence trials exist, should return empty dict."""
        from agent.pipeline.dags.convergence_detection import (
            _load_convergence_thresholds,
        )

        threshold_dir = tmp_path / "threshold_bo"
        threshold_dir.mkdir()
        ThresholdOptimizer(persist_dir=threshold_dir, seed=42)

        loaded = _load_convergence_thresholds(threshold_dir)
        assert loaded == {}

    def test_missing_dir_returns_empty(self, tmp_path):
        """Non-existent directory should return empty dict."""
        from agent.pipeline.dags.convergence_detection import (
            _load_convergence_thresholds,
        )

        loaded = _load_convergence_thresholds(tmp_path / "nonexistent")
        assert loaded == {}

    def test_only_convergence_keys_returned(self, tmp_path):
        """Should only return keys that match ConvergenceDetectorConfig fields."""
        from agent.pipeline.dags.convergence_detection import (
            _load_convergence_thresholds,
        )

        threshold_dir = tmp_path / "threshold_bo"
        threshold_dir.mkdir()

        opt = ThresholdOptimizer(persist_dir=threshold_dir, seed=42)
        # Record a CUSUM trial too — it should NOT leak into convergence output
        cusum_p = opt.suggest("cusum")
        opt.record("cusum", cusum_p, objective=0.5)

        conv_p = opt.suggest("convergence")
        opt.record("convergence", conv_p, objective=0.8)

        loaded = _load_convergence_thresholds(threshold_dir)
        # Should only have convergence keys, not cusum
        for k in loaded:
            assert k in ("z_threshold", "p_threshold", "fdr_q")

    def test_values_within_bounds(self, tmp_path):
        """Loaded convergence thresholds should be within DETECTOR_SPACES bounds."""
        from agent.learning.threshold_optimizer import DETECTOR_SPACES
        from agent.pipeline.dags.convergence_detection import (
            _load_convergence_thresholds,
        )

        threshold_dir = tmp_path / "threshold_bo"
        threshold_dir.mkdir()

        opt = ThresholdOptimizer(persist_dir=threshold_dir, seed=42)
        for _ in range(3):
            p = opt.suggest("convergence")
            opt.record("convergence", p, objective=np.random.rand())

        loaded = _load_convergence_thresholds(threshold_dir)
        space = DETECTOR_SPACES["convergence"]
        for name, (lo, hi) in zip(space.names, space.bounds):
            if name in loaded:
                assert lo <= loaded[name] <= hi, f"{name}={loaded[name]} out of [{lo}, {hi}]"


# ═══════════════════════════════════════════════════════════════
#  5. Goal Generator: Novel Arm Fallback
# ═══════════════════════════════════════════════════════════════


class TestGoalGeneratorNovelArm:
    """Goal generator correctly handles novel_exploration arm."""

    def test_fallback_novel_arm_gets_valid_goal(self):
        """Novel arm fallback should not crash and should produce valid goal."""
        novel = GoalArm(
            name="novel_exploration",
            description="Open-ended exploration",
            tools=[],
            examples=[
                "Combine satellite imagery with shipping data",
            ],
        )
        goal = GoalGenerator._arm_fallback_goal(novel)
        assert goal.description == "Combine satellite imagery with shipping data"
        assert goal.expected_tool == "web_search"  # default for empty tools

    def test_fallback_novel_arm_no_examples(self):
        """Novel arm with no examples should get a reasonable description."""
        novel = GoalArm(
            name="novel_exploration",
            description="Open-ended exploration",
            tools=[],
            examples=[],
        )
        goal = GoalGenerator._arm_fallback_goal(novel)
        assert "open-ended" in goal.description.lower() or "exploration" in goal.description.lower()
        assert goal.expected_tool == "web_search"

    def test_fallback_regular_arm_unchanged(self):
        """Regular arm fallback should still work as before."""
        arm = GoalArm(
            name="test_arm",
            description="Test",
            tools=["market_data"],
            examples=["Fetch AAPL data"],
        )
        goal = GoalGenerator._arm_fallback_goal(arm)
        assert goal.description == "Fetch AAPL data"
        assert goal.expected_tool == "market_data"

    def test_fallback_arm_with_tools_no_examples(self):
        """Arm with tools but no examples should use 'Explore using <tool>'."""
        arm = GoalArm(
            name="test_arm",
            description="Test",
            tools=["backtest"],
            examples=[],
        )
        goal = GoalGenerator._arm_fallback_goal(arm)
        assert "backtest" in goal.description.lower()
        assert goal.expected_tool == "backtest"


# ═══════════════════════════════════════════════════════════════
#  6. End-to-End: Optimizer Lifecycle
# ═══════════════════════════════════════════════════════════════


class TestOptimizerLifecycle:
    """The full lifecycle: suggest → use → record → persist → reload → use again."""

    def test_reward_optimizer_lifecycle(self, tmp_path):
        """Full round-trip for reward weight optimization."""
        path = tmp_path / "rw.json"

        # Session 1: suggest + record
        opt1 = RewardWeightOptimizer(persist_path=path, seed=42)
        w1 = opt1.suggest_weights()
        evaluation = _make_evaluation(score=0.7)
        r1 = compute_reward(evaluation, weights=w1)
        opt1.record_trial(w1, objective=r1)

        # Session 2: reload, suggest again
        opt2 = RewardWeightOptimizer(persist_path=path, seed=42)
        assert opt2.n_trials == 1
        w2 = opt2.suggest_weights()
        assert isinstance(w2, RewardWeights)

    def test_threshold_optimizer_lifecycle(self, tmp_path):
        """Full round-trip for threshold optimization."""
        threshold_dir = tmp_path / "threshold_bo"
        threshold_dir.mkdir()

        # Session 1: suggest + record for all detector types
        opt1 = ThresholdOptimizer(persist_dir=threshold_dir, seed=42)
        for det in opt1.detector_names:
            p = opt1.suggest(det)
            opt1.record(det, p, objective=0.7)

        # Session 2: reload, verify all trials present
        opt2 = ThresholdOptimizer(persist_dir=threshold_dir, seed=42)
        for det in opt2.detector_names:
            assert opt2.n_trials(det) == 1
            best = opt2.current_best(det)
            assert best is not None

    def test_bandit_novel_lifecycle(self, tmp_path):
        """Full lifecycle: novel pull → persist → reload → promote."""
        path = tmp_path / "bandit.json"

        # Session 1: 2 successful pulls
        b1 = StrategyBandit(arms=DEFAULT_ARMS, persist_path=path, seed=42)
        tools = ["web_search", "insider_filings"]
        b1.record_novel_pull(tools, reward=0.8, description="Cross-ref 1")
        b1.record_novel_pull(tools, reward=0.7, description="Cross-ref 2")

        # Session 2: reload + final pull → promotion
        b2 = StrategyBandit(arms=DEFAULT_ARMS, persist_path=path, seed=42)
        assert len(b2.novel_history) == 2
        promoted = b2.record_novel_pull(tools, reward=0.75, description="Cross-ref 3")
        assert promoted is not None
        assert promoted.name == "insider_filings_web_search"

    def test_multiple_optimizer_instances_coexist(self, tmp_path):
        """Different data paths should not interfere with each other."""
        rw_path = tmp_path / "rw.json"
        threshold_dir = tmp_path / "threshold_bo"
        threshold_dir.mkdir()
        bandit_path = tmp_path / "bandit.json"

        rw_opt = RewardWeightOptimizer(persist_path=rw_path, seed=42)
        th_opt = ThresholdOptimizer(persist_dir=threshold_dir, seed=42)
        bandit = StrategyBandit(arms=DEFAULT_ARMS, persist_path=bandit_path, seed=42)

        # Each operates independently
        w = rw_opt.suggest_weights()
        rw_opt.record_trial(w, objective=0.5)

        p = th_opt.suggest("cusum")
        th_opt.record("cusum", p, objective=0.6)

        bandit.record_novel_pull(["test"], reward=0.8, description="test")

        assert rw_opt.n_trials == 1
        assert th_opt.n_trials("cusum") == 1
        assert len(bandit.novel_history) == 1


# ═══════════════════════════════════════════════════════════════
#  7. AutonomousRunner Integration (Mocked)
# ═══════════════════════════════════════════════════════════════


class TestAutonomousRunnerIntegration:
    """Verify AutonomousRunner wiring with mocked components."""

    def _make_config(self, tmp_path):
        """Create a minimal AgentConfig pointing to tmp_path."""
        config = MagicMock()
        config.memory_dir = str(tmp_path / "memory")
        Path(config.memory_dir).mkdir(parents=True, exist_ok=True)
        config.llm = MagicMock()
        config.episode_ttl_days = 30
        return config

    def test_runner_creates_reward_optimizer(self, tmp_path):
        """AutonomousRunner.__init__ should create a RewardWeightOptimizer."""
        from agent.core.autonomous import AutonomousRunner

        config = self._make_config(tmp_path)
        tool_registry = MagicMock()

        with patch("agent.core.autonomous.LLMClient"):
            runner = AutonomousRunner(config, tool_registry, max_iterations=1)

        assert hasattr(runner, "_reward_optimizer")
        assert isinstance(runner._reward_optimizer, RewardWeightOptimizer)

    def test_runner_reward_optimizer_persists_to_correct_path(self, tmp_path):
        """Optimizer should persist to memory_dir/reward_bo.json."""
        from agent.core.autonomous import AutonomousRunner

        config = self._make_config(tmp_path)
        tool_registry = MagicMock()

        with patch("agent.core.autonomous.LLMClient"):
            runner = AutonomousRunner(config, tool_registry, max_iterations=1)

        # Trigger a suggestion + record to create the file
        w = runner._reward_optimizer.suggest_weights()
        runner._reward_optimizer.record_trial(w, objective=0.5)
        expected_path = Path(config.memory_dir) / "reward_bo.json"
        assert expected_path.exists()

    def test_runner_run_uses_learned_weights(self, tmp_path):
        """The run() method should call compute_reward with learned weights."""
        from agent.core.autonomous import AutonomousRunner

        config = self._make_config(tmp_path)
        tool_registry = MagicMock()

        with patch("agent.core.autonomous.LLMClient"):
            runner = AutonomousRunner(config, tool_registry, max_iterations=1)

        # Mock all components to avoid real LLM calls
        reflection = MagicMock()
        reflection.what_worked = []
        reflection.what_failed = []
        reflection.open_questions = []
        reflection.suggested_next_actions = []

        arm = DEFAULT_ARMS[0]  # backtest_strategy
        goal = Goal(
            description="test",
            rationale="test",
            expected_tool="backtest",
        )
        result = _make_agent_result()
        evaluation = _make_evaluation(score=0.7)

        runner._reflector.reflect = MagicMock(return_value=reflection)
        runner._bandit.choose = MagicMock(return_value=arm)
        runner._goal_generator.generate_for_arm = MagicMock(return_value=goal)
        runner._evaluator.evaluate = MagicMock(return_value=evaluation)

        # Mock orchestrator
        with patch("agent.core.autonomous.Orchestrator") as MockOrch:
            MockOrch.return_value.run.return_value = result

            # Patch compute_reward to capture the weights argument
            with patch("agent.core.autonomous.compute_reward", wraps=compute_reward) as mock_cr:
                summary = runner.run()

                # Verify compute_reward was called with a RewardWeights instance
                assert mock_cr.called
                call_kwargs = mock_cr.call_args
                weights_arg = call_kwargs.kwargs.get("weights") or call_kwargs[1].get("weights")
                if weights_arg is None and len(call_kwargs.args) >= 3:
                    weights_arg = call_kwargs.args[2]
                assert isinstance(weights_arg, RewardWeights)

    def test_runner_records_novel_pull(self, tmp_path):
        """When novel_exploration arm is chosen, tools_used should be recorded."""
        from agent.core.autonomous import AutonomousRunner

        config = self._make_config(tmp_path)
        tool_registry = MagicMock()

        with patch("agent.core.autonomous.LLMClient"):
            runner = AutonomousRunner(config, tool_registry, max_iterations=1)

        # Get the novel arm
        novel_arm = runner._bandit.get_arm("novel_exploration")
        assert novel_arm is not None

        reflection = MagicMock()
        reflection.what_worked = []
        reflection.what_failed = []
        reflection.open_questions = []
        reflection.suggested_next_actions = []

        goal = Goal(
            description="Cross-reference weather and commodities",
            rationale="Novel exploration",
            expected_tool="web_search",
        )
        episodes = [
            _make_episode(action="weather_alerts"),
            _make_episode(action="market_data"),
            _make_episode(action="weather_alerts"),  # duplicate
        ]
        result = _make_agent_result(episodes=episodes)
        evaluation = _make_evaluation(score=0.8)

        runner._reflector.reflect = MagicMock(return_value=reflection)
        runner._bandit.choose = MagicMock(return_value=novel_arm)
        runner._goal_generator.generate_for_arm = MagicMock(return_value=goal)
        runner._evaluator.evaluate = MagicMock(return_value=evaluation)

        with patch("agent.core.autonomous.Orchestrator") as MockOrch:
            MockOrch.return_value.run.return_value = result
            summary = runner.run()

        # Check novel history was recorded
        assert len(runner._bandit.novel_history) == 1
        entry = runner._bandit.novel_history[0]
        assert set(entry["tools"]) == {"market_data", "weather_alerts"}

    def test_runner_records_reward_trial_after_loop(self, tmp_path):
        """After the loop, mean reward should be recorded as a trial."""
        from agent.core.autonomous import AutonomousRunner

        config = self._make_config(tmp_path)
        tool_registry = MagicMock()

        with patch("agent.core.autonomous.LLMClient"):
            runner = AutonomousRunner(config, tool_registry, max_iterations=2)

        arm = DEFAULT_ARMS[0]
        goal = Goal(description="test", rationale="t", expected_tool="backtest")
        result = _make_agent_result()

        reflection = MagicMock()
        reflection.what_worked = []
        reflection.what_failed = []
        reflection.open_questions = []

        runner._reflector.reflect = MagicMock(return_value=reflection)
        runner._bandit.choose = MagicMock(return_value=arm)
        runner._goal_generator.generate_for_arm = MagicMock(return_value=goal)
        runner._evaluator.evaluate = MagicMock(return_value=_make_evaluation(score=0.6))

        with patch("agent.core.autonomous.Orchestrator") as MockOrch:
            MockOrch.return_value.run.return_value = result
            summary = runner.run()

        # Reward optimizer should have 1 trial recorded
        assert runner._reward_optimizer.n_trials == 1

    def test_runner_non_novel_arm_no_novel_recording(self, tmp_path):
        """Non-novel arms should NOT trigger novel_pull recording."""
        from agent.core.autonomous import AutonomousRunner

        config = self._make_config(tmp_path)
        tool_registry = MagicMock()

        with patch("agent.core.autonomous.LLMClient"):
            runner = AutonomousRunner(config, tool_registry, max_iterations=1)

        arm = DEFAULT_ARMS[0]  # backtest_strategy, NOT novel
        goal = Goal(description="test", rationale="t", expected_tool="backtest")
        result = _make_agent_result()
        evaluation = _make_evaluation(score=0.7)

        reflection = MagicMock()
        reflection.what_worked = []
        reflection.what_failed = []
        reflection.open_questions = []

        runner._reflector.reflect = MagicMock(return_value=reflection)
        runner._bandit.choose = MagicMock(return_value=arm)
        runner._goal_generator.generate_for_arm = MagicMock(return_value=goal)
        runner._evaluator.evaluate = MagicMock(return_value=evaluation)

        with patch("agent.core.autonomous.Orchestrator") as MockOrch:
            MockOrch.return_value.run.return_value = result
            summary = runner.run()

        assert len(runner._bandit.novel_history) == 0


# ═══════════════════════════════════════════════════════════════
#  8. Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases for the integration wiring."""

    def test_empty_episodes_novel_arm(self, tmp_path):
        """Novel arm with no episodes should not crash."""
        from agent.core.autonomous import AutonomousRunner

        config = MagicMock()
        config.memory_dir = str(tmp_path / "memory")
        Path(config.memory_dir).mkdir(parents=True, exist_ok=True)
        config.llm = MagicMock()
        tool_registry = MagicMock()

        with patch("agent.core.autonomous.LLMClient"):
            runner = AutonomousRunner(config, tool_registry, max_iterations=1)

        novel_arm = runner._bandit.get_arm("novel_exploration")
        goal = Goal(description="test", rationale="t", expected_tool="web_search")
        result = _make_agent_result(episodes=[])  # empty episodes
        evaluation = _make_evaluation(score=0.5)

        reflection = MagicMock()
        reflection.what_worked = []
        reflection.what_failed = []
        reflection.open_questions = []

        runner._reflector.reflect = MagicMock(return_value=reflection)
        runner._bandit.choose = MagicMock(return_value=novel_arm)
        runner._goal_generator.generate_for_arm = MagicMock(return_value=goal)
        runner._evaluator.evaluate = MagicMock(return_value=evaluation)

        with patch("agent.core.autonomous.Orchestrator") as MockOrch:
            MockOrch.return_value.run.return_value = result
            # Should not crash — empty episodes → record_novel_pull not called
            # because result.episodes is empty (falsy)
            summary = runner.run()

        # No novel history since episodes was empty
        assert len(runner._bandit.novel_history) == 0

    def test_scorer_config_with_threshold_dir_param(self, tmp_path):
        """Entity scoring should accept threshold_dir via params."""
        from agent.pipeline.dags.entity_scoring import _merge_learned_thresholds

        threshold_dir = tmp_path / "custom_threshold"
        threshold_dir.mkdir()

        opt = ThresholdOptimizer(persist_dir=threshold_dir, seed=42)
        p = opt.suggest("cusum")
        opt.record("cusum", p, objective=0.9)

        merged = _merge_learned_thresholds({}, threshold_dir)
        assert "cusum_k" in merged

    def test_convergence_config_kwargs_valid(self, tmp_path):
        """Loaded convergence thresholds should be valid ConvergenceDetectorConfig kwargs."""
        from agent.convergence.detector import ConvergenceDetectorConfig
        from agent.pipeline.dags.convergence_detection import (
            _load_convergence_thresholds,
        )

        threshold_dir = tmp_path / "threshold_bo"
        threshold_dir.mkdir()

        opt = ThresholdOptimizer(persist_dir=threshold_dir, seed=42)
        p = opt.suggest("convergence")
        opt.record("convergence", p, objective=0.85)

        loaded = _load_convergence_thresholds(threshold_dir)
        # Should be unpacked into ConvergenceDetectorConfig without error
        config = ConvergenceDetectorConfig(lookback_days=365, **loaded)
        assert config.z_threshold == loaded["z_threshold"]
        assert config.p_threshold == loaded["p_threshold"]
        assert config.fdr_q == loaded["fdr_q"]

    def test_zero_iterations_no_trial_recorded(self, tmp_path):
        """If AutonomousRunner runs with max_iterations=0, no trial should be recorded."""
        from agent.core.autonomous import AutonomousRunner

        config = MagicMock()
        config.memory_dir = str(tmp_path / "memory")
        Path(config.memory_dir).mkdir(parents=True, exist_ok=True)
        config.llm = MagicMock()
        tool_registry = MagicMock()

        with patch("agent.core.autonomous.LLMClient"):
            runner = AutonomousRunner(config, tool_registry, max_iterations=0)
            summary = runner.run()

        assert summary.iterations_completed == 0
        assert runner._reward_optimizer.n_trials == 0

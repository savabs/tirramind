"""TirraMind — Edge Case Tests for ToolRoutingBandit (Change 12)

Covers: cold start, convergence, always-on enforcement, persistence,
minimum exploration, context features, DAG integration, error handling.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agent.learning.tool_router import (
    ALWAYS_ON_TOOLS,
    DEFAULT_OPTIONAL_TOOLS,
    ToolContext,
    ToolRoutingBandit,
)
from agent.pipeline.dag import DAG, Node

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def bandit() -> ToolRoutingBandit:
    return ToolRoutingBandit(seed=42)


@pytest.fixture
def tmp_path_factory() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ── Cold Start Tests ──────────────────────────────────────────


class TestColdStart:
    """Fresh bandit with uniform priors."""

    def test_all_tools_default(self, bandit: ToolRoutingBandit) -> None:
        """With uniform Beta(1,1) and exploration, most tools should run."""
        decisions = bandit.decide(ToolContext())
        # always-on tools are always True
        for tool in ALWAYS_ON_TOOLS:
            assert decisions[tool] is True

    def test_optional_tools_present(self, bandit: ToolRoutingBandit) -> None:
        decisions = bandit.decide(ToolContext())
        for tool in DEFAULT_OPTIONAL_TOOLS:
            assert tool in decisions

    def test_uniform_prior_stats(self, bandit: ToolRoutingBandit) -> None:
        stats = bandit.stats()
        for tool in DEFAULT_OPTIONAL_TOOLS:
            assert stats[tool]["alpha"] == 1.0
            assert stats[tool]["beta"] == 1.0
            assert stats[tool]["mean"] == 0.5
            assert stats[tool]["pulls"] == 0.0


# ── Convergence Tests ─────────────────────────────────────────


class TestConvergence:
    """Bandit converges to select high-reward tools."""

    def test_high_reward_tool_selected(self) -> None:
        """Tool with consistently high reward should be selected."""
        router = ToolRoutingBandit(
            tool_names=("good_tool", "bad_tool"),
            threshold=0.5,
            min_exploration_rate=0.0,  # no forced exploration
            seed=42,
        )
        # Train: good_tool always r=1, bad_tool always r=0
        for _ in range(50):
            router.record_outcome("good_tool", 1.0)
            router.record_outcome("bad_tool", 0.0)

        # Check many decisions: good_tool should be on most of the time
        good_on = sum(1 for _ in range(100) if router.decide(ToolContext()).get("good_tool", False))
        bad_on = sum(1 for _ in range(100) if router.decide(ToolContext()).get("bad_tool", False))
        assert good_on > 90, f"good_tool only selected {good_on}/100 times"
        assert bad_on < 20, f"bad_tool selected {bad_on}/100 times"

    def test_bad_tool_suppressed(self) -> None:
        """Tool with zero reward should be mostly skipped."""
        router = ToolRoutingBandit(
            tool_names=("tool_a",),
            threshold=0.5,
            min_exploration_rate=0.0,
            seed=123,
        )
        for _ in range(100):
            router.record_outcome("tool_a", 0.0)

        stats = router.stats()
        assert stats["tool_a"]["mean"] < 0.02  # Beta mean ≈ 1/102

    def test_reward_clamped_to_0_1(self) -> None:
        """Reward outside [0,1] is clamped."""
        router = ToolRoutingBandit(tool_names=("t1",), seed=1)
        router.record_outcome("t1", 2.5)  # should clamp to 1.0
        assert router.stats()["t1"]["alpha"] == 2.0  # 1.0 + 1.0
        router.record_outcome("t1", -0.5)  # should clamp to 0.0
        assert router.stats()["t1"]["beta"] == 2.0  # 1.0 + 1.0


# ── Always-On Enforcement ─────────────────────────────────────


class TestAlwaysOn:
    """Always-on tools bypass the bandit."""

    def test_fetch_instruments_always_on(self) -> None:
        router = ToolRoutingBandit(seed=0, min_exploration_rate=0.0, threshold=1.0)
        # With threshold=1.0 and no exploration, optional tools mostly OFF
        decisions = router.decide(ToolContext())
        assert decisions["fetch_instruments"] is True

    def test_always_on_regardless_of_reward(self) -> None:
        router = ToolRoutingBandit(seed=0)
        # Even if we could theoretically suppress instruments, it won't happen
        for _ in range(100):
            decisions = router.decide(ToolContext())
            assert decisions.get("fetch_instruments", True) is True


# ── Minimum Exploration ───────────────────────────────────────


class TestMinExploration:
    """Minimum exploration rate prevents total tool starvation."""

    def test_tools_occasionally_forced_on(self) -> None:
        """With min_exploration=1.0, all tools always on."""
        router = ToolRoutingBandit(
            tool_names=("t1",),
            threshold=1.0,  # high threshold → would skip
            min_exploration_rate=1.0,  # but always force on
            seed=42,
        )
        for _ in range(50):
            router.record_outcome("t1", 0.0)  # bad tool

        # Should still be on due to exploration=1.0
        for _ in range(20):
            decisions = router.decide(ToolContext())
            assert decisions["t1"] is True

    def test_zero_exploration_allows_suppression(self) -> None:
        """With exploration=0, bad tools can be fully suppressed."""
        router = ToolRoutingBandit(
            tool_names=("t1",),
            threshold=0.9,
            min_exploration_rate=0.0,
            seed=42,
        )
        for _ in range(200):
            router.record_outcome("t1", 0.0)

        # With Beta heavily weighted toward 0, most decisions should be OFF
        off_count = sum(1 for _ in range(100) if not router.decide(ToolContext()).get("t1", False))
        assert off_count > 80


# ── Persistence ───────────────────────────────────────────────


class TestPersistence:
    """Save/load round-trip."""

    def test_save_load_preserves_state(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "router.json"

            router = ToolRoutingBandit(tool_names=("t1", "t2"), persist_path=path, seed=42)
            router.record_outcome("t1", 0.8)
            router.record_outcome("t2", 0.2)

            # Load into fresh bandit
            router2 = ToolRoutingBandit(tool_names=("t1", "t2"), persist_path=path, seed=42)
            assert router2.stats()["t1"]["alpha"] == router.stats()["t1"]["alpha"]
            assert router2.stats()["t2"]["beta"] == router.stats()["t2"]["beta"]

    def test_explicit_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "router.json"

            router = ToolRoutingBandit(tool_names=("t1",), seed=42)
            router.record_outcome("t1", 0.5)
            router.save(path)

            router2 = ToolRoutingBandit(tool_names=("t1",), seed=42)
            router2.load(path)
            assert router2.stats()["t1"]["pulls"] == 1.0

    def test_corrupted_file_handled(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "router.json"
            path.write_text("not valid json {{{")
            # Should not crash — warns and uses defaults
            router = ToolRoutingBandit(tool_names=("t1",), persist_path=path, seed=42)
            assert router.stats()["t1"]["alpha"] == 1.0

    def test_missing_tool_in_state_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "router.json"
            # Save state for t1 only
            state = {
                "tool_names": ["t1"],
                "alpha": {"t1": 5.0},
                "beta": {"t1": 3.0},
                "pulls": {"t1": 7},
                "total_reward": {"t1": 4.0},
            }
            path.write_text(json.dumps(state))

            # Load but with t1 + t2
            router = ToolRoutingBandit(tool_names=("t1", "t2"), persist_path=path, seed=42)
            assert router.stats()["t1"]["alpha"] == 5.0
            assert router.stats()["t2"]["alpha"] == 1.0  # default


# ── Error Handling ────────────────────────────────────────────


class TestErrorHandling:
    """Invalid inputs and edge cases."""

    def test_unknown_tool_record(self, bandit: ToolRoutingBandit) -> None:
        with pytest.raises(ValueError, match="Unknown tool"):
            bandit.record_outcome("nonexistent_tool", 0.5)

    def test_empty_context(self, bandit: ToolRoutingBandit) -> None:
        """None context → still makes decisions."""
        decisions = bandit.decide(None)
        assert isinstance(decisions, dict)
        assert len(decisions) > 0


# ── Add Tool ──────────────────────────────────────────────────


class TestAddTool:
    """Dynamic tool addition."""

    def test_add_new_tool(self) -> None:
        router = ToolRoutingBandit(tool_names=("t1",), seed=42)
        router.add_tool("t2")
        assert "t2" in router.tool_names
        assert router.stats()["t2"]["alpha"] == 1.0

    def test_add_existing_tool_noop(self) -> None:
        router = ToolRoutingBandit(tool_names=("t1",), seed=42)
        router.record_outcome("t1", 0.8)
        router.add_tool("t1")  # should not reset
        assert router.stats()["t1"]["alpha"] > 1.0


# ── DAG Integration ───────────────────────────────────────────


class TestDAGIntegration:
    """Integration with DAG node enabling."""

    def test_build_dag_with_router(self) -> None:
        """daily_collection DAG respects router decisions."""
        from agent.pipeline.dags.daily_collection import build_daily_collection_dag

        router = ToolRoutingBandit(
            threshold=1.0,  # high threshold → skip most
            min_exploration_rate=0.0,
            seed=999,
        )
        # Suppress all optional tools
        for tool in DEFAULT_OPTIONAL_TOOLS:
            for _ in range(50):
                router.record_outcome(tool, 0.0)

        ctx = ToolContext(regime_id=0, day_of_week=2)
        dag = build_daily_collection_dag(tool_router=router, tool_context=ctx)

        # fetch_instruments must still be enabled
        assert dag.nodes["fetch_instruments"].enabled is True

        # At least some optional tools should be disabled
        disabled = [nid for nid in dag.nodes if not dag.nodes[nid].enabled and nid != "fetch_instruments"]
        assert len(disabled) > 0

    def test_build_dag_without_router(self) -> None:
        """Without router, all nodes are enabled (backward compat)."""
        from agent.pipeline.dags.daily_collection import build_daily_collection_dag

        dag = build_daily_collection_dag()
        for nid, node in dag.nodes.items():
            assert node.enabled is True

    def test_node_enabled_flag(self) -> None:
        """Node.enabled defaults to True."""
        node = Node(id="test", operator="test_op")
        assert node.enabled is True
        node.enabled = False
        assert node.enabled is False

    def test_executor_skips_disabled_node(self) -> None:
        """DAGExecutor skips disabled nodes."""
        from agent.pipeline.executor import DAGExecutor

        dag = DAG(name="test_dag")
        dag.add("enabled_node", operator=lambda params, upstream: {"ok": True})
        dag.add("disabled_node", operator=lambda params, upstream: {"ok": True})
        dag.nodes["disabled_node"].enabled = False

        executor = DAGExecutor()
        run = executor.execute(dag, trigger="test")

        assert run.node_results["enabled_node"].status == "completed"
        assert run.node_results["disabled_node"].status == "skipped"
        assert "tool router" in run.node_results["disabled_node"].error


# ── Stats & Diagnostics ──────────────────────────────────────


class TestStats:
    """Diagnostics and stats output."""

    def test_stats_format(self, bandit: ToolRoutingBandit) -> None:
        stats = bandit.stats()
        assert isinstance(stats, dict)
        for tool in DEFAULT_OPTIONAL_TOOLS:
            assert "alpha" in stats[tool]
            assert "beta" in stats[tool]
            assert "mean" in stats[tool]
            assert "pulls" in stats[tool]
            assert "total_reward" in stats[tool]

    def test_mean_updates_correctly(self) -> None:
        router = ToolRoutingBandit(tool_names=("t1",), seed=42)
        router.record_outcome("t1", 1.0)
        stats = router.stats()
        # After 1 success: α=2, β=1, mean = 2/3
        assert abs(stats["t1"]["mean"] - 2 / 3) < 1e-6

    def test_tool_names_property(self, bandit: ToolRoutingBandit) -> None:
        assert bandit.tool_names == DEFAULT_OPTIONAL_TOOLS

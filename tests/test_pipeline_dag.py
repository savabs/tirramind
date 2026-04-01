"""Tests for DAG and Node data model."""

from __future__ import annotations

import pytest

from agent.pipeline.dag import DAG, Node


# ── Node construction ──────────────────────────────────────────


class TestNode:
    def test_basic_node(self):
        node = Node(id="fetch", operator="cftc", params={"mode": "latest"})
        assert node.id == "fetch"
        assert node.operator == "cftc"
        assert node.params == {"mode": "latest"}
        assert node.depends_on == []
        assert node.retries == 1
        assert node.timeout == 60
        assert node.store_result is True
        assert node.table_name is None

    def test_node_with_deps(self):
        node = Node(id="compute", operator="fn", depends_on=["a", "b"])
        assert node.depends_on == ["a", "b"]

    def test_node_with_callable(self):
        fn = lambda x: x  # noqa: E731
        node = Node(id="proc", operator=fn)
        assert node.operator is fn

    def test_node_custom_retries_timeout(self):
        node = Node(id="slow", operator="x", retries=3, timeout=120)
        assert node.retries == 3
        assert node.timeout == 120


# ── DAG construction ───────────────────────────────────────────


class TestDAGConstruction:
    def test_empty_dag(self):
        dag = DAG(name="empty")
        assert dag.name == "empty"
        assert dag.nodes == {}

    def test_add_node(self):
        dag = DAG(name="test")
        nid = dag.add("fetch", operator="cftc", params={"mode": "latest"})
        assert nid == "fetch"
        assert "fetch" in dag.nodes
        assert dag.nodes["fetch"].operator == "cftc"

    def test_add_multiple_nodes(self):
        dag = DAG(name="test")
        dag.add("a", operator="tool_a")
        dag.add("b", operator="tool_b")
        dag.add("c", operator="tool_c", depends_on=["a", "b"])
        assert len(dag.nodes) == 3

    def test_add_duplicate_node_raises(self):
        dag = DAG(name="test")
        dag.add("a", operator="x")
        with pytest.raises(ValueError, match="Duplicate node ID"):
            dag.add("a", operator="y")

    def test_add_with_kwargs(self):
        dag = DAG(name="test")
        dag.add("n", operator="x", retries=5, timeout=300, store_result=False)
        assert dag.nodes["n"].retries == 5
        assert dag.nodes["n"].timeout == 300
        assert dag.nodes["n"].store_result is False

    def test_dag_with_schedule(self):
        dag = DAG(name="daily", schedule="0 18 * * 1-5")
        assert dag.schedule == "0 18 * * 1-5"

    def test_dag_with_description(self):
        dag = DAG(name="d", description="Daily fetch")
        assert dag.description == "Daily fetch"


# ── Roots ──────────────────────────────────────────────────────


class TestRoots:
    def test_single_node_is_root(self):
        dag = DAG(name="t")
        dag.add("a", operator="x")
        assert dag.roots() == ["a"]

    def test_all_independent_are_roots(self):
        dag = DAG(name="t")
        dag.add("a", operator="x")
        dag.add("b", operator="y")
        dag.add("c", operator="z")
        assert sorted(dag.roots()) == ["a", "b", "c"]

    def test_dependent_not_root(self):
        dag = DAG(name="t")
        dag.add("a", operator="x")
        dag.add("b", operator="y", depends_on=["a"])
        assert dag.roots() == ["a"]

    def test_no_roots_in_cycle(self):
        """If all nodes have deps, there are no roots (broken but roots() returns [])."""
        dag = DAG(name="t")
        dag.add("a", operator="x", depends_on=["b"])
        dag.add("b", operator="y", depends_on=["a"])
        assert dag.roots() == []


# ── Validation ─────────────────────────────────────────────────


class TestValidation:
    def test_valid_dag(self):
        dag = DAG(name="v")
        dag.add("a", operator="x")
        dag.add("b", operator="y", depends_on=["a"])
        assert dag.validate() == []

    def test_empty_dag_error(self):
        dag = DAG(name="empty")
        errors = dag.validate()
        assert any("no nodes" in e for e in errors)

    def test_missing_dep_error(self):
        dag = DAG(name="bad")
        dag.add("a", operator="x", depends_on=["nonexistent"])
        errors = dag.validate()
        assert any("nonexistent" in e for e in errors)

    def test_self_dep_error(self):
        dag = DAG(name="bad")
        dag.add("a", operator="x", depends_on=["a"])
        errors = dag.validate()
        assert any("self-dependency" in e for e in errors)

    def test_simple_cycle_error(self):
        dag = DAG(name="cycle")
        dag.add("a", operator="x", depends_on=["b"])
        dag.add("b", operator="y", depends_on=["a"])
        errors = dag.validate()
        assert any("cycle" in e for e in errors)

    def test_three_node_cycle(self):
        dag = DAG(name="cycle3")
        dag.add("a", operator="x", depends_on=["c"])
        dag.add("b", operator="y", depends_on=["a"])
        dag.add("c", operator="z", depends_on=["b"])
        errors = dag.validate()
        assert any("cycle" in e for e in errors)

    def test_cycle_with_valid_prefix(self):
        """DAG has valid root + cycle downstream."""
        dag = DAG(name="mixed")
        dag.add("root", operator="x")
        dag.add("a", operator="y", depends_on=["root", "b"])
        dag.add("b", operator="z", depends_on=["a"])
        errors = dag.validate()
        assert any("cycle" in e for e in errors)

    def test_diamond_dag_valid(self):
        """A → B, A → C, B → D, C → D (diamond, no cycle)."""
        dag = DAG(name="diamond")
        dag.add("a", operator="x")
        dag.add("b", operator="y", depends_on=["a"])
        dag.add("c", operator="z", depends_on=["a"])
        dag.add("d", operator="w", depends_on=["b", "c"])
        assert dag.validate() == []

    def test_wide_dag_valid(self):
        """100 independent nodes, all valid."""
        dag = DAG(name="wide")
        for i in range(100):
            dag.add(f"n{i}", operator="x")
        assert dag.validate() == []

    def test_deep_chain_valid(self):
        """50-node chain, each depending on previous."""
        dag = DAG(name="deep")
        dag.add("n0", operator="x")
        for i in range(1, 50):
            dag.add(f"n{i}", operator="x", depends_on=[f"n{i-1}"])
        assert dag.validate() == []

    def test_no_name_error(self):
        dag = DAG(name="")
        dag.add("a", operator="x")
        errors = dag.validate()
        assert any("no name" in e for e in errors)

    def test_multiple_errors(self):
        """Multiple problems detected at once."""
        dag = DAG(name="bad")
        dag.add("a", operator="x", depends_on=["missing1"])
        dag.add("b", operator="y", depends_on=["missing2", "b"])
        errors = dag.validate()
        # Should have errors for missing1, missing2, and self-dep
        assert len(errors) >= 3


# ── Topological sort ───────────────────────────────────────────


class TestTopoSort:
    def test_single_node(self):
        dag = DAG(name="single")
        dag.add("a", operator="x")
        layers = dag.topo_sort()
        assert layers == [["a"]]

    def test_two_independent(self):
        dag = DAG(name="par")
        dag.add("a", operator="x")
        dag.add("b", operator="y")
        layers = dag.topo_sort()
        assert layers == [["a", "b"]]

    def test_linear_chain(self):
        dag = DAG(name="chain")
        dag.add("a", operator="x")
        dag.add("b", operator="y", depends_on=["a"])
        dag.add("c", operator="z", depends_on=["b"])
        layers = dag.topo_sort()
        assert layers == [["a"], ["b"], ["c"]]

    def test_diamond(self):
        dag = DAG(name="diamond")
        dag.add("a", operator="x")
        dag.add("b", operator="y", depends_on=["a"])
        dag.add("c", operator="z", depends_on=["a"])
        dag.add("d", operator="w", depends_on=["b", "c"])
        layers = dag.topo_sort()
        assert layers == [["a"], ["b", "c"], ["d"]]

    def test_wide_parallel(self):
        dag = DAG(name="wide")
        for i in range(10):
            dag.add(f"n{i}", operator="x")
        layers = dag.topo_sort()
        assert len(layers) == 1
        assert len(layers[0]) == 10

    def test_complex_dag(self):
        """
        a ─→ c ─→ e
        b ─→ d ─→ e
        b ─→ c
        """
        dag = DAG(name="complex")
        dag.add("a", operator="x")
        dag.add("b", operator="y")
        dag.add("c", operator="z", depends_on=["a", "b"])
        dag.add("d", operator="w", depends_on=["b"])
        dag.add("e", operator="v", depends_on=["c", "d"])
        layers = dag.topo_sort()
        assert layers[0] == ["a", "b"]  # Both roots
        assert set(layers[1]) == {"c", "d"}  # Both depend only on roots
        assert layers[2] == ["e"]

    def test_topo_sort_deterministic(self):
        """Same DAG should always produce same layer ordering."""
        for _ in range(10):
            dag = DAG(name="det")
            dag.add("z", operator="x")
            dag.add("a", operator="x")
            dag.add("m", operator="x")
            layers = dag.topo_sort()
            assert layers == [["a", "m", "z"]]  # Sorted

    def test_topo_sort_raises_on_cycle(self):
        dag = DAG(name="cycle")
        dag.add("a", operator="x", depends_on=["b"])
        dag.add("b", operator="y", depends_on=["a"])
        with pytest.raises(ValueError, match="Invalid DAG"):
            dag.topo_sort()

    def test_topo_sort_raises_on_empty(self):
        dag = DAG(name="empty")
        with pytest.raises(ValueError, match="Invalid DAG"):
            dag.topo_sort()

    def test_deep_chain_layers(self):
        dag = DAG(name="deep")
        dag.add("n0", operator="x")
        for i in range(1, 20):
            dag.add(f"n{i}", operator="x", depends_on=[f"n{i-1}"])
        layers = dag.topo_sort()
        assert len(layers) == 20
        for i, layer in enumerate(layers):
            assert layer == [f"n{i}"]

    def test_multi_root_merge(self):
        """Multiple roots merge into one sink."""
        dag = DAG(name="fan")
        for i in range(5):
            dag.add(f"r{i}", operator="x")
        dag.add("sink", operator="y", depends_on=[f"r{i}" for i in range(5)])
        layers = dag.topo_sort()
        assert len(layers) == 2
        assert len(layers[0]) == 5
        assert layers[1] == ["sink"]

    def test_disconnected_subgraphs(self):
        """Two independent sub-DAGs in one DAG."""
        dag = DAG(name="disc")
        dag.add("a1", operator="x")
        dag.add("a2", operator="y", depends_on=["a1"])
        dag.add("b1", operator="z")
        dag.add("b2", operator="w", depends_on=["b1"])
        layers = dag.topo_sort()
        assert layers[0] == ["a1", "b1"]  # Both roots
        assert layers[1] == ["a2", "b2"]  # Both layer 1

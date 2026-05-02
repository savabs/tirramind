"""
Tests for agent/models/discovery.py — causal structure discovery.

Since tigramite is not installed (GPL-3.0, optional dependency),
these tests verify:
    - ImportError with helpful message when tigramite missing
    - Input validation (min_samples, dimensions, all-NaN columns)
    - ComparisonReport logic with mocked DiscoveryResult
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.models.discovery import (
    CausalStructureDiscovery,
    ComparisonReport,
    DiscoveredEdge,
    DiscoveryResult,
)
from agent.models.initial_graph import build_initial_graph


class TestImportError:
    def test_discover_raises_importerror(self) -> None:
        """tigramite not installed → clear ImportError."""
        disc = CausalStructureDiscovery(min_samples=10)
        data = np.random.default_rng(42).normal(size=(50, 3))
        with pytest.raises(ImportError, match="tigramite"):
            disc.discover(data, ["a", "b", "c"])


class TestValidation:
    def test_insufficient_samples(self) -> None:
        disc = CausalStructureDiscovery(min_samples=200)
        data = np.random.default_rng(42).normal(size=(50, 3))
        with pytest.raises(ValueError, match="Insufficient samples"):
            disc.discover(data, ["a", "b", "c"])

    def test_single_variable(self) -> None:
        disc = CausalStructureDiscovery(min_samples=10)
        data = np.random.default_rng(42).normal(size=(50, 1))
        with pytest.raises(ValueError, match="at least 2 variables"):
            disc.discover(data, ["a"])

    def test_name_length_mismatch(self) -> None:
        disc = CausalStructureDiscovery(min_samples=10)
        data = np.random.default_rng(42).normal(size=(50, 3))
        with pytest.raises(ValueError, match="variable_names length"):
            disc.discover(data, ["a", "b"])

    def test_1d_data(self) -> None:
        disc = CausalStructureDiscovery(min_samples=10)
        data = np.random.default_rng(42).normal(size=(50,))
        with pytest.raises(ValueError, match="2D"):
            disc.discover(data, ["a"])

    def test_all_nan_column(self) -> None:
        disc = CausalStructureDiscovery(min_samples=10)
        data = np.random.default_rng(42).normal(size=(50, 3))
        data[:, 1] = np.nan
        with pytest.raises(ValueError, match="All-NaN"):
            disc.discover(data, ["a", "b", "c"])


class TestComparisonReport:
    """Test compare_with_expert with manually constructed DiscoveryResult."""

    def _mock_discovery(
        self,
        edges: list[DiscoveredEdge],
    ) -> DiscoveryResult:
        return DiscoveryResult(
            edges=edges,
            graph_array=np.empty((0, 0, 0)),
            summary={
                "T": 300,
                "N": 3,
                "ci_test": "ParCorr",
                "alpha": 0.05,
                "max_lag": 5,
                "n_edges": len(edges),
                "runtime_seconds": 1.0,
            },
        )

    def test_confirmed_edge(self) -> None:
        """Expert edge present in discovery → confirmed."""
        graph = build_initial_graph()
        edges = [
            DiscoveredEdge("regime.macro", "obs.rate_momentum", 0, 0.001, 0.5, "-->"),
        ]
        disc_result = self._mock_discovery(edges)
        discovery = CausalStructureDiscovery()
        report = discovery.compare_with_expert(disc_result, graph)

        assert ("regime.macro", "obs.rate_momentum") in report.confirmed_edges

    def test_missing_edge(self) -> None:
        """Expert edge NOT in discovery → missing."""
        graph = build_initial_graph()
        disc_result = self._mock_discovery([])  # no edges discovered
        discovery = CausalStructureDiscovery()
        report = discovery.compare_with_expert(disc_result, graph)

        assert len(report.missing_edges) == 19  # all expert edges missing
        assert report.summary["n_confirmed"] == 0

    def test_novel_edge(self) -> None:
        """Edge in discovery but NOT in expert → novel."""
        graph = build_initial_graph()
        edges = [
            DiscoveredEdge("obs.rate_momentum", "obs.stress_breadth", 0, 0.01, 0.3, "-->"),
        ]
        disc_result = self._mock_discovery(edges)
        discovery = CausalStructureDiscovery()
        report = discovery.compare_with_expert(disc_result, graph)

        assert len(report.novel_edges) == 1
        assert report.novel_edges[0].source == "obs.rate_momentum"

    def test_lagged_edges_excluded_from_comparison(self) -> None:
        """Only lag=0 edges are compared with expert DAG."""
        graph = build_initial_graph()
        edges = [
            # This edge matches expert but has lag=1 → should not be "confirmed"
            DiscoveredEdge("regime.macro", "obs.rate_momentum", 1, 0.001, 0.5, "-->"),
        ]
        disc_result = self._mock_discovery(edges)
        discovery = CausalStructureDiscovery()
        report = discovery.compare_with_expert(disc_result, graph)

        assert ("regime.macro", "obs.rate_momentum") not in report.confirmed_edges

    def test_summary_counts(self) -> None:
        graph = build_initial_graph()
        edges = [
            DiscoveredEdge("regime.macro", "obs.rate_momentum", 0, 0.001, 0.5, "-->"),
            DiscoveredEdge("regime.macro", "obs.yield_curve_slope", 0, 0.01, 0.4, "-->"),
            DiscoveredEdge("obs.stress_breadth", "obs.stress_intensity", 0, 0.02, 0.3, "-->"),
        ]
        disc_result = self._mock_discovery(edges)
        discovery = CausalStructureDiscovery()
        report = discovery.compare_with_expert(disc_result, graph)

        assert report.summary["n_confirmed"] == 2
        assert report.summary["n_novel"] == 1
        assert report.summary["total_expert_edges"] == 19


class TestDataclasses:
    def test_discovered_edge_frozen(self) -> None:
        e = DiscoveredEdge("a", "b", 0, 0.01, 0.5, "-->")
        with pytest.raises(AttributeError):
            e.source = "c"  # type: ignore

    def test_comparison_report_frozen(self) -> None:
        r = ComparisonReport([], [], [], {})
        with pytest.raises(AttributeError):
            r.confirmed_edges = []  # type: ignore

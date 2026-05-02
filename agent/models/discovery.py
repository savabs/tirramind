"""
TirraMind — Causal Structure Discovery

Optional offline wrapper around tigramite PCMCI for data-driven causal
structure learning from historical feature data.

Tigramite (GPL-3.0) is imported lazily — all other world model functionality
works without it.  This module calls tigramite as an external tool;
no tigramite code is copied.

Design principles:
    1. Gated by minimum sample size (default 200).
    2. Produces suggestions for graph updates, not automatic changes.
    3. compare_with_expert() highlights confirmed / missing / novel edges.
    4. Lazy import with clear error message.

References:
    - Runge et al., "Detecting and quantifying causal associations in
      large nonlinear time series datasets" (2019), Sci. Adv.
    - tigramite docs: https://jakobrunge.github.io/tigramite/
    - Spec: docs/specs/world_model_spec.md (sub-phase 9.7)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from agent.models.graph import WorldModelGraph


@dataclass(frozen=True)
class DiscoveredEdge:
    """A single edge discovered by PCMCI."""

    source: str
    target: str
    lag: int
    pvalue: float
    strength: float
    link_type: str  # "-->", "o-o", "x->", etc.


@dataclass(frozen=True)
class DiscoveryResult:
    """Output of a causal discovery run."""

    edges: list[DiscoveredEdge]
    graph_array: np.ndarray  # (N, N, tau_max+1)
    summary: dict[str, Any]


@dataclass(frozen=True)
class ComparisonReport:
    """Comparison between discovered and expert graph."""

    confirmed_edges: list[tuple[str, str]]
    """Expert edges confirmed by discovery."""

    missing_edges: list[tuple[str, str, float]]
    """Expert edges NOT confirmed. (source, target, best_pvalue)."""

    novel_edges: list[DiscoveredEdge]
    """Edges found by discovery but not in expert graph."""

    summary: dict[str, Any]


class CausalStructureDiscovery:
    """Offline causal structure learning via tigramite PCMCI.

    Args:
        significance_level: Alpha for conditional independence tests.
        max_lag: Maximum time lag (tau_max) for PCMCI.
        min_samples: Minimum number of time steps required.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        max_lag: int = 5,
        min_samples: int = 200,
    ) -> None:
        self._alpha = significance_level
        self._max_lag = max_lag
        self._min_samples = min_samples

    def discover(
        self,
        data: np.ndarray,
        variable_names: list[str],
    ) -> DiscoveryResult:
        """Run PCMCI+ causal discovery.

        Args:
            data: Shape (T, N) — T timesteps, N variables.
            variable_names: Length N, names for each column.

        Returns:
            DiscoveryResult with discovered edges.

        Raises:
            ImportError: If tigramite is not installed.
            ValueError: If data fails validation.
        """
        # Validate
        if data.ndim != 2:
            raise ValueError(f"data must be 2D, got {data.ndim}D")
        T, N = data.shape
        if self._min_samples > T:
            raise ValueError(f"Insufficient samples: {T} < min_samples={self._min_samples}")
        if N < 2:
            raise ValueError(f"Need at least 2 variables, got {N}")
        if len(variable_names) != N:
            raise ValueError(f"variable_names length ({len(variable_names)}) != number of columns ({N})")
        # Check for all-NaN columns
        nan_cols = np.all(np.isnan(data), axis=0)
        if np.any(nan_cols):
            bad = [variable_names[i] for i in np.where(nan_cols)[0]]
            raise ValueError(f"All-NaN columns: {bad}")

        # Lazy import
        try:
            from tigramite import data_processing as pp
            from tigramite.independence_tests.parcorr import ParCorr
            from tigramite.pcmci import PCMCI
        except ImportError as e:
            raise ImportError(
                "tigramite is required for causal discovery but is not "
                "installed. Install it with: pip install tigramite\n"
                "Note: tigramite is GPL-3.0 licensed."
            ) from e

        start = time.time()

        # Run PCMCI+
        dataframe = pp.DataFrame(data, var_names=variable_names)
        ci_test = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ci_test)
        results = pcmci.run_pcmciplus(
            tau_min=0,
            tau_max=self._max_lag,
            pc_alpha=self._alpha,
        )

        runtime = time.time() - start

        # Extract edges
        graph_array = results["graph"]
        val_matrix = results["val_matrix"]
        p_matrix = results["p_matrix"]

        edges: list[DiscoveredEdge] = []
        for i in range(N):
            for j in range(N):
                for tau in range(self._max_lag + 1):
                    link = graph_array[i, j, tau]
                    if link != "" and link != "":
                        # Check if this is a non-empty link string
                        link_str = str(link).strip()
                        if link_str and link_str != "":
                            edges.append(
                                DiscoveredEdge(
                                    source=variable_names[i],
                                    target=variable_names[j],
                                    lag=tau,
                                    pvalue=float(p_matrix[i, j, tau]),
                                    strength=float(abs(val_matrix[i, j, tau])),
                                    link_type=link_str,
                                )
                            )

        summary = {
            "T": T,
            "N": N,
            "ci_test": "ParCorr",
            "alpha": self._alpha,
            "max_lag": self._max_lag,
            "n_edges": len(edges),
            "runtime_seconds": round(runtime, 2),
        }

        return DiscoveryResult(
            edges=edges,
            graph_array=graph_array,
            summary=summary,
        )

    def compare_with_expert(
        self,
        discovered: DiscoveryResult,
        expert: WorldModelGraph,
    ) -> ComparisonReport:
        """Compare discovered edges with expert graph.

        Only considers contemporaneous (lag=0) edges with "-->" direction
        for comparison with the expert DAG.

        Args:
            discovered: Result from discover().
            expert: The expert-specified WorldModelGraph.

        Returns:
            ComparisonReport.
        """
        expert_edges = set(expert.edges)

        # Contemporaneous directed edges from discovery
        disc_directed = {(e.source, e.target) for e in discovered.edges if e.lag == 0 and "-->" in e.link_type}

        confirmed = [e for e in expert_edges if e in disc_directed]

        # Missing: in expert but not in discovery
        missing = []
        disc_pvalues: dict[tuple[str, str], float] = {}
        for e in discovered.edges:
            key = (e.source, e.target)
            if key not in disc_pvalues or e.pvalue < disc_pvalues[key]:
                disc_pvalues[key] = e.pvalue

        for e in expert_edges:
            if e not in disc_directed:
                best_p = disc_pvalues.get(e, 1.0)
                missing.append((e[0], e[1], best_p))

        # Novel: in discovery but not in expert
        novel = [
            e
            for e in discovered.edges
            if e.lag == 0 and "-->" in e.link_type and (e.source, e.target) not in expert_edges
        ]

        summary = {
            "n_confirmed": len(confirmed),
            "n_missing": len(missing),
            "n_novel": len(novel),
            "total_expert_edges": len(expert_edges),
            "total_discovered_edges": len(disc_directed),
        }

        return ComparisonReport(
            confirmed_edges=confirmed,
            missing_edges=missing,
            novel_edges=novel,
            summary=summary,
        )

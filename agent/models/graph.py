"""
TirraMind — World Model Graph

DAG representation wrapping pgmpy BayesianNetwork.
Manages nodes, edges, CPDs, validation, and deterministic hashing.

Design principles:
    1. Thin wrapper — delegates to pgmpy for graph ops, keeps our own metadata.
    2. NodeSpec carries domain metadata (type, cardinality, feature mapping).
    3. Deterministic hashing — same structure always produces same hash.
    4. Serializable — to_dict / from_dict for persistence.
    5. Validation at boundaries — validate() returns error list.

References:
    - pgmpy.models.BayesianNetwork
    - Spec: docs/specs/world_model_spec.md (sub-phase 9.2)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork


# ── Node specification ─────────────────────────────────────────

VALID_NODE_TYPES: frozenset[str] = frozenset({"observed", "latent", "regime"})


@dataclass(frozen=True)
class NodeSpec:
    """Metadata for a single node in the world model graph."""

    name: str
    """Unique node identifier, dotted: 'regime.macro', 'obs.rate_momentum'."""

    node_type: str
    """One of 'observed', 'latent', 'regime'."""

    domain: str
    """Feature domain: 'convergence', 'macro', 'market', 'regime', 'latent'."""

    cardinality: int | None = None
    """For discrete nodes: number of states.  None for continuous proxy."""

    states: tuple[str, ...] | None = None
    """For discrete nodes: state labels matching cardinality."""

    feature_name: str | None = None
    """For observed nodes: maps to EngineeredFeature.feature_name."""

    bin_edges: tuple[float, ...] | None = None
    """For observed nodes: bin boundaries to discretize continuous values.
    Length = cardinality + 1.  E.g. (-inf, -0.5, 0.5, inf) for 3 bins."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "node_type": self.node_type,
            "domain": self.domain,
            "cardinality": self.cardinality,
            "states": list(self.states) if self.states else None,
            "feature_name": self.feature_name,
            "bin_edges": list(self.bin_edges) if self.bin_edges else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeSpec:
        return cls(
            name=d["name"],
            node_type=d["node_type"],
            domain=d["domain"],
            cardinality=d.get("cardinality"),
            states=tuple(d["states"]) if d.get("states") else None,
            feature_name=d.get("feature_name"),
            bin_edges=tuple(d["bin_edges"]) if d.get("bin_edges") else None,
        )


# ── World model graph ──────────────────────────────────────────


class WorldModelGraph:
    """DAG wrapper around pgmpy BayesianNetwork with domain metadata."""

    def __init__(
        self,
        nodes: list[NodeSpec] | None = None,
        edges: list[tuple[str, str]] | None = None,
    ) -> None:
        self._nodes: dict[str, NodeSpec] = {}
        self._bn = BayesianNetwork()

        if nodes:
            for spec in nodes:
                self.add_node(spec)
        if edges:
            for parent, child in edges:
                self.add_edge(parent, child)

    # ── Node operations ────────────────────────────────────────

    def add_node(self, spec: NodeSpec) -> None:
        """Add a node to the graph.  Idempotent if same spec."""
        if spec.name in self._nodes:
            if self._nodes[spec.name] == spec:
                return  # idempotent
            raise ValueError(f"Node '{spec.name}' already exists with different spec")
        self._nodes[spec.name] = spec
        self._bn.add_node(spec.name)

    def get_node(self, name: str) -> NodeSpec:
        """Get node spec by name.  Raises KeyError if not found."""
        return self._nodes[name]

    @property
    def node_names(self) -> list[str]:
        return list(self._nodes.keys())

    @property
    def node_specs(self) -> dict[str, NodeSpec]:
        return dict(self._nodes)

    def get_observed_nodes(self) -> list[NodeSpec]:
        return [s for s in self._nodes.values() if s.node_type == "observed"]

    def get_latent_nodes(self) -> list[NodeSpec]:
        return [s for s in self._nodes.values() if s.node_type == "latent"]

    def get_regime_nodes(self) -> list[NodeSpec]:
        return [s for s in self._nodes.values() if s.node_type == "regime"]

    # ── Edge operations ────────────────────────────────────────

    def add_edge(self, parent: str, child: str) -> None:
        """Add a directed edge.  Raises ValueError on cycle."""
        if parent not in self._nodes:
            raise ValueError(f"Parent node '{parent}' not in graph")
        if child not in self._nodes:
            raise ValueError(f"Child node '{child}' not in graph")

        # Check for cycle: adding parent→child would create a cycle if
        # child can already reach parent via existing edges.
        if parent == child:
            raise ValueError(f"Self-loop not allowed: '{parent}'")

        # Check for cycle: if child can already reach parent, adding
        # parent→child would create a cycle.
        import networkx as nx

        self._bn.add_edge(parent, child)
        if not nx.is_directed_acyclic_graph(self._bn):
            self._bn.remove_edge(parent, child)
            raise ValueError(f"Adding edge {parent} → {child} would create a cycle")

    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(self._bn.edges())

    def get_parents(self, node: str) -> list[str]:
        return list(self._bn.predecessors(node))

    def get_children(self, node: str) -> list[str]:
        return list(self._bn.successors(node))

    # ── CPD operations ─────────────────────────────────────────

    def set_cpd(self, node_name: str, cpd: TabularCPD) -> None:
        """Set or replace the CPD for a node."""
        if node_name not in self._nodes:
            raise ValueError(f"Node '{node_name}' not in graph")
        # Remove existing CPD for this node if any
        existing = [c for c in self._bn.get_cpds() or [] if c.variable == node_name]
        for c in existing:
            self._bn.remove_cpds(c)
        self._bn.add_cpds(cpd)

    def get_cpd(self, node_name: str) -> TabularCPD | None:
        """Get CPD for a node, or None."""
        for cpd in self._bn.get_cpds() or []:
            if cpd.variable == node_name:
                return cpd
        return None

    def get_all_cpds(self) -> list[TabularCPD]:
        return list(self._bn.get_cpds() or [])

    # ── Access to underlying pgmpy model ───────────────────────

    @property
    def bn(self) -> BayesianNetwork:
        """Direct access to the pgmpy BayesianNetwork (for inference)."""
        return self._bn

    # ── Validation ─────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Validate graph structure, nodes, and CPDs.

        Returns list of error strings.  Empty = valid.
        """
        errors: list[str] = []

        # Node-level checks
        for spec in self._nodes.values():
            if spec.node_type not in VALID_NODE_TYPES:
                errors.append(
                    f"Node '{spec.name}': invalid node_type '{spec.node_type}'"
                )
            if spec.node_type == "observed" and not spec.feature_name:
                errors.append(
                    f"Node '{spec.name}': observed node must have feature_name"
                )
            if spec.cardinality is not None and spec.cardinality < 1:
                errors.append(f"Node '{spec.name}': cardinality must be >= 1")
            if (
                spec.states
                and spec.cardinality
                and len(spec.states) != spec.cardinality
            ):
                errors.append(
                    f"Node '{spec.name}': states length ({len(spec.states)}) "
                    f"!= cardinality ({spec.cardinality})"
                )
            if spec.bin_edges and spec.cardinality:
                if len(spec.bin_edges) != spec.cardinality + 1:
                    errors.append(
                        f"Node '{spec.name}': bin_edges length "
                        f"({len(spec.bin_edges)}) must be cardinality + 1 "
                        f"({spec.cardinality + 1})"
                    )

        # Check CPDs exist for all nodes
        nodes_with_cpds = {c.variable for c in self._bn.get_cpds() or []}
        for name in self._nodes:
            if name not in nodes_with_cpds:
                errors.append(f"Node '{name}': missing CPD")

        # Check pgmpy model consistency (CPD dims match parents)
        if nodes_with_cpds == set(self._nodes.keys()):
            try:
                self._bn.check_model()
            except ValueError as e:
                errors.append(f"pgmpy model check failed: {e}")

        return errors

    # ── Hashing ────────────────────────────────────────────────

    def graph_hash(self) -> str:
        """Deterministic SHA-256 hash of (sorted nodes + sorted edges).

        Same structure always produces the same hash, regardless of
        insertion order.
        """
        nodes_sorted = sorted(self._nodes.keys())
        edges_sorted = sorted((p, c) for p, c in self._bn.edges())
        content = json.dumps(
            {"nodes": nodes_sorted, "edges": edges_sorted},
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    # ── Serialization ──────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph structure (nodes + edges) to dict.

        CPDs are NOT serialized here — they should be saved/loaded
        separately via pgmpy's own mechanisms.
        """
        return {
            "nodes": [s.to_dict() for s in self._nodes.values()],
            "edges": [(p, c) for p, c in self._bn.edges()],
            "graph_hash": self.graph_hash(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorldModelGraph:
        """Reconstruct from dict (inverse of to_dict).  CPDs not included."""
        nodes = [NodeSpec.from_dict(n) for n in d["nodes"]]
        edges = [(p, c) for p, c in d["edges"]]
        return cls(nodes=nodes, edges=edges)

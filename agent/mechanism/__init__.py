"""TirraMind — mechanism layer (structural half of a verification explanation).

The product returns a computed verdict *and* a computed explanation. This
package supplies the structural half of the explanation: which routes through
the entity graph connect a hypothesis's cause to its effect, how much of the
graph's mass actually flows along them, and how many genuinely independent
observations support them.

The mechanism layer is linear algebra, not deep learning. The HetTGN in
`agent/models/gnn/` collapsed twice (effective rank 2.6 of 64, 811k parameters
against 110k supervised cells); a rank-64 sparse SVD of the same graph reaches
effective rank 50.4 in under a second with zero parameters and cannot collapse,
because eigenvectors are orthogonal by construction. Do not add a neural net
here.

`agent.mechanism.graph` is the foundation every other module in this package
imports. Importing it pulls in `agent.models.gnn.graph_builder` (for the
EVENT_RELATIONS / STRUCTURAL_RELATIONS sets, which are deliberately NOT
redefined here), and that transitively imports torch and torch_geometric —
roughly 5 seconds of import cost. That is the price of having one definition of
"is this edge evidence or scaffold" rather than two that can drift apart.
"""

from __future__ import annotations

from agent.mechanism.graph import (
    DisconnectedEntityError,
    EdgeClass,
    EmptyGraphError,
    GraphLoadReport,
    HubEntry,
    MechanismEdge,
    MechanismGraph,
    MechanismGraphError,
    ScaffoldTimePolicy,
    UnknownEntityError,
    as_of_from_iso,
    load_graph,
)

__all__ = [
    "DisconnectedEntityError",
    "EdgeClass",
    "EmptyGraphError",
    "GraphLoadReport",
    "HubEntry",
    "MechanismEdge",
    "MechanismGraph",
    "MechanismGraphError",
    "ScaffoldTimePolicy",
    "UnknownEntityError",
    "as_of_from_iso",
    "load_graph",
]

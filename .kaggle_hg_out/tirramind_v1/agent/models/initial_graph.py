"""
TirraMind — Initial Expert DAG

Expert-specified causal graph with 20 nodes (3 latent/regime + 17 observed)
and 19 directed edges.  Prior CPDs are weakly informative — roughly uniform
with slight center bias.  These priors will be updated by MLE fitting once
enough historical feature data accumulates.

Causal semantics:
    - regime.macro governs macro features (rate_momentum, yield_curve_slope,
      liquidity_pressure) and influences regime.stress
    - regime.stress governs convergence features (stress_breadth,
      stress_intensity, regime_persistence) and GNN anomaly features
      (person/company/wallet/country/vessel anomaly)
    - latent.risk_appetite is driven by both regime.macro and regime.stress
      and modulates obs.liquidity_pressure, obs.stress_intensity,
      and GNN activity features (person/company/wallet activity)
    - GNN cross_entity is an unparented observed node (not caused by a single
      regime variable; provides evidence about inter-entity correlation)

Spec: docs/specs/world_model_bridge_spec.md (step 19c.1)
"""

from __future__ import annotations

import math

import numpy as np
from pgmpy.factors.discrete import TabularCPD

from agent.models.graph import NodeSpec, WorldModelGraph

# ── Node definitions ───────────────────────────────────────────

_REGIME_MACRO = NodeSpec(
    name="regime.macro",
    node_type="regime",
    domain="regime",
    cardinality=3,
    states=("expansion", "contraction", "crisis"),
)

_REGIME_STRESS = NodeSpec(
    name="regime.stress",
    node_type="regime",
    domain="regime",
    cardinality=3,
    states=("calm", "elevated", "extreme"),
)

_LATENT_RISK_APPETITE = NodeSpec(
    name="latent.risk_appetite",
    node_type="latent",
    domain="latent",
    cardinality=3,
    states=("risk_on", "neutral", "risk_off"),
)

_OBS_RATE_MOMENTUM = NodeSpec(
    name="obs.rate_momentum",
    node_type="observed",
    domain="macro",
    cardinality=3,
    states=("falling", "neutral", "rising"),
    feature_name="macro.rate_momentum.30d",
    bin_edges=(-math.inf, -0.5, 0.5, math.inf),
)

_OBS_YIELD_CURVE_SLOPE = NodeSpec(
    name="obs.yield_curve_slope",
    node_type="observed",
    domain="macro",
    cardinality=3,
    states=("inverted", "flat", "steep"),
    feature_name="macro.yield_curve_slope.spot",
    bin_edges=(-math.inf, -0.2, 0.5, math.inf),
)

_OBS_LIQUIDITY_PRESSURE = NodeSpec(
    name="obs.liquidity_pressure",
    node_type="observed",
    domain="macro",
    cardinality=3,
    states=("tight", "normal", "loose"),
    feature_name="macro.liquidity_pressure.30d",
    bin_edges=(-math.inf, -0.3, 0.3, math.inf),
)

_OBS_STRESS_BREADTH = NodeSpec(
    name="obs.stress_breadth",
    node_type="observed",
    domain="convergence",
    cardinality=3,
    states=("narrow", "moderate", "broad"),
    feature_name="convergence.stress_breadth.7d",
    bin_edges=(-math.inf, 0.3, 0.7, math.inf),
)

_OBS_STRESS_INTENSITY = NodeSpec(
    name="obs.stress_intensity",
    node_type="observed",
    domain="convergence",
    cardinality=3,
    states=("low", "medium", "high"),
    feature_name="convergence.stress_intensity.7d",
    bin_edges=(-math.inf, 0.3, 0.7, math.inf),
)

_OBS_REGIME_PERSISTENCE = NodeSpec(
    name="obs.regime_persistence",
    node_type="observed",
    domain="convergence",
    cardinality=3,
    states=("unstable", "moderate", "persistent"),
    feature_name="convergence.regime_persistence.7d",
    bin_edges=(-math.inf, 0.3, 0.7, math.inf),
)

# ── GNN entity-derived observed nodes (Phase 19c) ─────────────

_GNN_BIN_EDGES: tuple[float, ...] = (-math.inf, -1.0, 1.0, math.inf)

_OBS_PERSON_ANOMALY = NodeSpec(
    name="obs.person_anomaly",
    node_type="observed",
    domain="entity",
    cardinality=3,
    states=("low", "normal", "high"),
    feature_name="gnn.person_anomaly.spot",
    bin_edges=_GNN_BIN_EDGES,
)

_OBS_COMPANY_ANOMALY = NodeSpec(
    name="obs.company_anomaly",
    node_type="observed",
    domain="entity",
    cardinality=3,
    states=("low", "normal", "high"),
    feature_name="gnn.company_anomaly.spot",
    bin_edges=_GNN_BIN_EDGES,
)

_OBS_WALLET_ANOMALY = NodeSpec(
    name="obs.wallet_anomaly",
    node_type="observed",
    domain="entity",
    cardinality=3,
    states=("low", "normal", "high"),
    feature_name="gnn.wallet_anomaly.spot",
    bin_edges=_GNN_BIN_EDGES,
)

_OBS_COUNTRY_ANOMALY = NodeSpec(
    name="obs.country_anomaly",
    node_type="observed",
    domain="entity",
    cardinality=3,
    states=("low", "normal", "high"),
    feature_name="gnn.country_anomaly.spot",
    bin_edges=_GNN_BIN_EDGES,
)

_OBS_VESSEL_ANOMALY = NodeSpec(
    name="obs.vessel_anomaly",
    node_type="observed",
    domain="entity",
    cardinality=3,
    states=("low", "normal", "high"),
    feature_name="gnn.vessel_anomaly.spot",
    bin_edges=_GNN_BIN_EDGES,
)

_OBS_PERSON_ACTIVITY = NodeSpec(
    name="obs.person_activity",
    node_type="observed",
    domain="entity",
    cardinality=3,
    states=("low", "normal", "high"),
    feature_name="gnn.person_activity.spot",
    bin_edges=_GNN_BIN_EDGES,
)

_OBS_COMPANY_ACTIVITY = NodeSpec(
    name="obs.company_activity",
    node_type="observed",
    domain="entity",
    cardinality=3,
    states=("low", "normal", "high"),
    feature_name="gnn.company_activity.spot",
    bin_edges=_GNN_BIN_EDGES,
)

_OBS_WALLET_ACTIVITY = NodeSpec(
    name="obs.wallet_activity",
    node_type="observed",
    domain="entity",
    cardinality=3,
    states=("low", "normal", "high"),
    feature_name="gnn.wallet_activity.spot",
    bin_edges=_GNN_BIN_EDGES,
)

_OBS_COUNTRY_ACTIVITY = NodeSpec(
    name="obs.country_activity",
    node_type="observed",
    domain="entity",
    cardinality=3,
    states=("low", "normal", "high"),
    feature_name="gnn.country_activity.spot",
    bin_edges=_GNN_BIN_EDGES,
)

_OBS_VESSEL_ACTIVITY = NodeSpec(
    name="obs.vessel_activity",
    node_type="observed",
    domain="entity",
    cardinality=3,
    states=("low", "normal", "high"),
    feature_name="gnn.vessel_activity.spot",
    bin_edges=_GNN_BIN_EDGES,
)

_OBS_CROSS_ENTITY = NodeSpec(
    name="obs.cross_entity",
    node_type="observed",
    domain="entity",
    cardinality=3,
    states=("low", "normal", "high"),
    feature_name="gnn.cross_entity.spot",
    bin_edges=_GNN_BIN_EDGES,
)

ALL_NODES: list[NodeSpec] = [
    _REGIME_MACRO,
    _REGIME_STRESS,
    _LATENT_RISK_APPETITE,
    _OBS_RATE_MOMENTUM,
    _OBS_YIELD_CURVE_SLOPE,
    _OBS_LIQUIDITY_PRESSURE,
    _OBS_STRESS_BREADTH,
    _OBS_STRESS_INTENSITY,
    _OBS_REGIME_PERSISTENCE,
    # GNN entity-derived nodes (Phase 19c)
    _OBS_PERSON_ANOMALY,
    _OBS_COMPANY_ANOMALY,
    _OBS_WALLET_ANOMALY,
    _OBS_COUNTRY_ANOMALY,
    _OBS_VESSEL_ANOMALY,
    _OBS_PERSON_ACTIVITY,
    _OBS_COMPANY_ACTIVITY,
    _OBS_WALLET_ACTIVITY,
    _OBS_COUNTRY_ACTIVITY,
    _OBS_VESSEL_ACTIVITY,
    _OBS_CROSS_ENTITY,
]

ALL_EDGES: list[tuple[str, str]] = [
    ("regime.macro", "obs.rate_momentum"),
    ("regime.macro", "obs.yield_curve_slope"),
    ("regime.macro", "obs.liquidity_pressure"),
    ("regime.macro", "latent.risk_appetite"),
    ("regime.stress", "obs.stress_breadth"),
    ("regime.stress", "obs.stress_intensity"),
    ("regime.stress", "obs.regime_persistence"),
    ("regime.stress", "latent.risk_appetite"),
    ("latent.risk_appetite", "obs.liquidity_pressure"),
    ("latent.risk_appetite", "obs.stress_intensity"),
    ("regime.macro", "regime.stress"),
    # GNN anomaly edges: regime.stress → anomaly (5 edges)
    ("regime.stress", "obs.person_anomaly"),
    ("regime.stress", "obs.company_anomaly"),
    ("regime.stress", "obs.wallet_anomaly"),
    ("regime.stress", "obs.country_anomaly"),
    ("regime.stress", "obs.vessel_anomaly"),
    # GNN activity edges: latent.risk_appetite → activity (3 edges)
    ("latent.risk_appetite", "obs.person_activity"),
    ("latent.risk_appetite", "obs.company_activity"),
    ("latent.risk_appetite", "obs.wallet_activity"),
    # obs.cross_entity is unparented (root observed node)
]

# ── CPD builders ───────────────────────────────────────────────


def _weakly_informative_prior(cardinality: int) -> np.ndarray:
    """Slightly center-biased prior for root nodes.

    For card=3: [0.25, 0.50, 0.25] — favours the 'middle' state
    (expansion > contraction > crisis, calm > elevated > extreme, etc.)
    """
    if cardinality == 1:
        return np.array([1.0])
    weights = np.ones(cardinality)
    mid = cardinality // 2
    weights[mid] = 2.0
    return weights / weights.sum()


def _build_root_cpd(node: NodeSpec) -> TabularCPD:
    """CPD for a root node (no parents): weakly informative prior."""
    values = _weakly_informative_prior(node.cardinality).reshape(-1, 1)
    return TabularCPD(
        variable=node.name,
        variable_card=node.cardinality,
        values=values,
        state_names={node.name: list(node.states)},
    )


def _build_regime_stress_cpd() -> TabularCPD:
    """P(regime.stress | regime.macro)

    Domain logic:
        - expansion → mostly calm
        - contraction → elevated more likely
        - crisis → extreme more likely
    """
    # Columns: expansion, contraction, crisis  (parent states)
    # Rows: calm, elevated, extreme  (child states)
    values = np.array(
        [
            [0.70, 0.20, 0.05],  # calm
            [0.25, 0.50, 0.25],  # elevated
            [0.05, 0.30, 0.70],  # extreme
        ]
    )
    return TabularCPD(
        variable="regime.stress",
        variable_card=3,
        values=values,
        evidence=["regime.macro"],
        evidence_card=[3],
        state_names={
            "regime.stress": ["calm", "elevated", "extreme"],
            "regime.macro": ["expansion", "contraction", "crisis"],
        },
    )


def _build_risk_appetite_cpd() -> TabularCPD:
    """P(latent.risk_appetite | regime.macro, regime.stress)

    Domain logic:
        - expansion + calm → strong risk_on
        - crisis + extreme → strong risk_off
        - everything else → blended
    """
    # Parents order: regime.macro (card=3), regime.stress (card=3)
    # Total parent configs: 3 × 3 = 9 columns
    # Each column sums to 1.0
    #
    # Rows: risk_on, neutral, risk_off
    # Columns ordered by pgmpy convention: regime.macro varies slowest
    #   (expansion,calm), (expansion,elevated), (expansion,extreme),
    #   (contraction,calm), (contraction,elevated), (contraction,extreme),
    #   (crisis,calm), (crisis,elevated), (crisis,extreme)
    values = np.array(
        [
            # exp+calm  exp+elev  exp+ext   con+calm  con+elev  con+ext   cri+calm  cri+elev  cri+ext
            [0.70, 0.45, 0.20, 0.35, 0.20, 0.10, 0.15, 0.08, 0.05],  # risk_on
            [0.25, 0.40, 0.40, 0.40, 0.45, 0.30, 0.35, 0.27, 0.15],  # neutral
            [0.05, 0.15, 0.40, 0.25, 0.35, 0.60, 0.50, 0.65, 0.80],  # risk_off
        ]
    )
    return TabularCPD(
        variable="latent.risk_appetite",
        variable_card=3,
        values=values,
        evidence=["regime.macro", "regime.stress"],
        evidence_card=[3, 3],
        state_names={
            "latent.risk_appetite": ["risk_on", "neutral", "risk_off"],
            "regime.macro": ["expansion", "contraction", "crisis"],
            "regime.stress": ["calm", "elevated", "extreme"],
        },
    )


def _build_single_parent_obs_cpd(
    node: NodeSpec,
    parent: NodeSpec,
    values: np.ndarray,
) -> TabularCPD:
    """CPD for an observed node with exactly one parent."""
    return TabularCPD(
        variable=node.name,
        variable_card=node.cardinality,
        values=values,
        evidence=[parent.name],
        evidence_card=[parent.cardinality],
        state_names={
            node.name: list(node.states),
            parent.name: list(parent.states),
        },
    )


def _build_two_parent_obs_cpd(
    node: NodeSpec,
    parent1: NodeSpec,
    parent2: NodeSpec,
    values: np.ndarray,
) -> TabularCPD:
    """CPD for an observed node with exactly two parents."""
    return TabularCPD(
        variable=node.name,
        variable_card=node.cardinality,
        values=values,
        evidence=[parent1.name, parent2.name],
        evidence_card=[parent1.cardinality, parent2.cardinality],
        state_names={
            node.name: list(node.states),
            parent1.name: list(parent1.states),
            parent2.name: list(parent2.states),
        },
    )


def _build_rate_momentum_cpd() -> TabularCPD:
    """P(obs.rate_momentum | regime.macro)

    expansion → rising rates likely
    contraction → falling rates likely
    crisis → strongly falling
    """
    values = np.array(
        [
            [0.10, 0.50, 0.70],  # falling
            [0.30, 0.35, 0.20],  # neutral
            [0.60, 0.15, 0.10],  # rising
        ]
    )
    return _build_single_parent_obs_cpd(
        _OBS_RATE_MOMENTUM,
        _REGIME_MACRO,
        values,
    )


def _build_yield_curve_slope_cpd() -> TabularCPD:
    """P(obs.yield_curve_slope | regime.macro)

    expansion → steep yield curve
    contraction → flat
    crisis → inverted
    """
    values = np.array(
        [
            [0.05, 0.30, 0.65],  # inverted
            [0.25, 0.50, 0.25],  # flat
            [0.70, 0.20, 0.10],  # steep
        ]
    )
    return _build_single_parent_obs_cpd(
        _OBS_YIELD_CURVE_SLOPE,
        _REGIME_MACRO,
        values,
    )


def _build_liquidity_pressure_cpd() -> TabularCPD:
    """P(obs.liquidity_pressure | regime.macro, latent.risk_appetite)

    Two parents: regime.macro (3) × risk_appetite (3) = 9 columns.
    expansion + risk_on → loose liquidity
    crisis + risk_off → tight liquidity
    """
    # Parent order: regime.macro, latent.risk_appetite
    # pgmpy column order: regime.macro varies slowest
    values = np.array(
        [
            # exp+ro  exp+neu  exp+roff  con+ro  con+neu  con+roff  cri+ro  cri+neu  cri+roff
            [0.05, 0.10, 0.25, 0.20, 0.35, 0.55, 0.40, 0.55, 0.75],  # tight
            [0.25, 0.35, 0.45, 0.40, 0.40, 0.35, 0.40, 0.35, 0.20],  # normal
            [0.70, 0.55, 0.30, 0.40, 0.25, 0.10, 0.20, 0.10, 0.05],  # loose
        ]
    )
    return _build_two_parent_obs_cpd(
        _OBS_LIQUIDITY_PRESSURE,
        _REGIME_MACRO,
        _LATENT_RISK_APPETITE,
        values,
    )


def _build_stress_breadth_cpd() -> TabularCPD:
    """P(obs.stress_breadth | regime.stress)

    calm → narrow stress
    elevated → moderate
    extreme → broad
    """
    values = np.array(
        [
            [0.65, 0.25, 0.10],  # narrow
            [0.25, 0.50, 0.25],  # moderate
            [0.10, 0.25, 0.65],  # broad
        ]
    )
    return _build_single_parent_obs_cpd(
        _OBS_STRESS_BREADTH,
        _REGIME_STRESS,
        values,
    )


def _build_stress_intensity_cpd() -> TabularCPD:
    """P(obs.stress_intensity | regime.stress, latent.risk_appetite)

    Two parents: regime.stress (3) × risk_appetite (3) = 9 columns.
    extreme + risk_off → high intensity
    calm + risk_on → low intensity
    """
    # Parent order: regime.stress, latent.risk_appetite
    values = np.array(
        [
            # calm+ro calm+neu calm+roff elev+ro elev+neu elev+roff ext+ro  ext+neu  ext+roff
            [0.70, 0.55, 0.35, 0.30, 0.20, 0.10, 0.10, 0.08, 0.05],  # low
            [0.25, 0.35, 0.40, 0.45, 0.45, 0.35, 0.30, 0.27, 0.15],  # medium
            [0.05, 0.10, 0.25, 0.25, 0.35, 0.55, 0.60, 0.65, 0.80],  # high
        ]
    )
    return _build_two_parent_obs_cpd(
        _OBS_STRESS_INTENSITY,
        _REGIME_STRESS,
        _LATENT_RISK_APPETITE,
        values,
    )


def _build_regime_persistence_cpd() -> TabularCPD:
    """P(obs.regime_persistence | regime.stress)

    calm → persistent (stable regime)
    extreme → unstable (regime likely to change)
    """
    values = np.array(
        [
            [0.10, 0.30, 0.60],  # unstable
            [0.25, 0.45, 0.25],  # moderate
            [0.65, 0.25, 0.15],  # persistent
        ]
    )
    return _build_single_parent_obs_cpd(
        _OBS_REGIME_PERSISTENCE,
        _REGIME_STRESS,
        values,
    )


# ── GNN-derived CPDs (Phase 19c) ──────────────────────────────


def _weakly_informative_3x3(parent_name: str, parent_states: list[str]) -> np.ndarray:
    """Near-uniform CPD for a 3-state child with a 3-state parent.

    Each column sums to 1.0. Centre state slightly favoured in all
    parent configurations, with a mild alignment bias (high→high, low→low).
    """
    return np.array(
        [
            [0.40, 0.30, 0.20],  # low
            [0.35, 0.40, 0.35],  # normal
            [0.25, 0.30, 0.45],  # high
        ]
    )


def _build_anomaly_cpd(node: NodeSpec) -> TabularCPD:
    """P(obs.{type}_anomaly | regime.stress)

    Weakly informative: extreme stress slightly raises anomaly probability.
    CPD columns: calm, elevated, extreme.
    CPD rows: low, normal, high.
    """
    values = np.array(
        [
            [0.45, 0.30, 0.15],  # low
            [0.35, 0.40, 0.35],  # normal
            [0.20, 0.30, 0.50],  # high
        ]
    )
    return _build_single_parent_obs_cpd(node, _REGIME_STRESS, values)


def _build_activity_cpd(node: NodeSpec) -> TabularCPD:
    """P(obs.{type}_activity | latent.risk_appetite)

    Weakly informative: risk_on slightly raises activity.
    CPD columns: risk_on, neutral, risk_off.
    CPD rows: low, normal, high.
    """
    values = np.array(
        [
            [0.15, 0.30, 0.45],  # low
            [0.35, 0.40, 0.35],  # normal
            [0.50, 0.30, 0.20],  # high
        ]
    )
    return _build_single_parent_obs_cpd(node, _LATENT_RISK_APPETITE, values)


def _build_cross_entity_cpd() -> TabularCPD:
    """P(obs.cross_entity) — root observed node (no parents).

    Weakly informative prior: slightly favours "normal" correlation.
    """
    values = np.array([[0.25], [0.50], [0.25]])
    return TabularCPD(
        variable="obs.cross_entity",
        variable_card=3,
        values=values,
        state_names={"obs.cross_entity": ["low", "normal", "high"]},
    )


# ── Public builder ─────────────────────────────────────────────


def build_initial_graph() -> WorldModelGraph:
    """Construct the expert-specified initial world model graph.

    Returns a fully wired WorldModelGraph with 20 nodes, 19 edges,
    and weakly informative prior CPDs.  The graph passes validate()
    with zero errors.
    """
    graph = WorldModelGraph(nodes=ALL_NODES, edges=ALL_EDGES)

    # Root node priors
    graph.set_cpd("regime.macro", _build_root_cpd(_REGIME_MACRO))

    # Conditional CPDs — original 6 observed nodes
    graph.set_cpd("regime.stress", _build_regime_stress_cpd())
    graph.set_cpd("latent.risk_appetite", _build_risk_appetite_cpd())
    graph.set_cpd("obs.rate_momentum", _build_rate_momentum_cpd())
    graph.set_cpd("obs.yield_curve_slope", _build_yield_curve_slope_cpd())
    graph.set_cpd("obs.liquidity_pressure", _build_liquidity_pressure_cpd())
    graph.set_cpd("obs.stress_breadth", _build_stress_breadth_cpd())
    graph.set_cpd("obs.stress_intensity", _build_stress_intensity_cpd())
    graph.set_cpd("obs.regime_persistence", _build_regime_persistence_cpd())

    # GNN anomaly nodes — P(anomaly | regime.stress) (5 nodes)
    graph.set_cpd("obs.person_anomaly", _build_anomaly_cpd(_OBS_PERSON_ANOMALY))
    graph.set_cpd("obs.company_anomaly", _build_anomaly_cpd(_OBS_COMPANY_ANOMALY))
    graph.set_cpd("obs.wallet_anomaly", _build_anomaly_cpd(_OBS_WALLET_ANOMALY))
    graph.set_cpd("obs.country_anomaly", _build_anomaly_cpd(_OBS_COUNTRY_ANOMALY))
    graph.set_cpd("obs.vessel_anomaly", _build_anomaly_cpd(_OBS_VESSEL_ANOMALY))

    # GNN activity nodes — P(activity | latent.risk_appetite) (3 nodes)
    graph.set_cpd("obs.person_activity", _build_activity_cpd(_OBS_PERSON_ACTIVITY))
    graph.set_cpd("obs.company_activity", _build_activity_cpd(_OBS_COMPANY_ACTIVITY))
    graph.set_cpd("obs.wallet_activity", _build_activity_cpd(_OBS_WALLET_ACTIVITY))

    # GNN activity nodes — unparented (2 remaining activity + cross_entity)
    # country_activity and vessel_activity have no parent edges (not enough
    # causal justification for risk_appetite → vessel/country activity).
    # They get marginal priors like cross_entity.
    graph.set_cpd("obs.country_activity", _build_root_cpd(_OBS_COUNTRY_ACTIVITY))
    graph.set_cpd("obs.vessel_activity", _build_root_cpd(_OBS_VESSEL_ACTIVITY))

    # Cross-entity: root observed node (no parents)
    graph.set_cpd("obs.cross_entity", _build_cross_entity_cpd())

    # Validate — fail fast if priors are broken
    errors = graph.validate()
    if errors:
        raise RuntimeError("Initial graph validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return graph

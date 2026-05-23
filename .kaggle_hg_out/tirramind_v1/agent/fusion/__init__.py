"""
TirraMind — Signal Fusion Module

Self-supervised entity micro-alpha pipeline. The GNN's prediction surprise
is the primary anomaly signal. Statistical monitors (CUSUM, Hawkes, Event Study)
serve as node feature enrichment inputs to the GNN, not as anomaly outputs.

Paradigm: unique observation × advanced math = edge.

Layer: Fusion (Layer 4 in the 7-layer computation stack).
"""

from agent.fusion.alert import EntityAlert
from agent.fusion.convergence import ConvergenceCluster, ConvergenceDetector
from agent.fusion.cusum import CUSUMMonitor
from agent.fusion.entity_baseline import EntityBaseline
from agent.fusion.entity_scorer import EntityAnomalyScorer, ScorerConfig
from agent.fusion.hawkes import HawkesIntensity
from agent.fusion.surprise import EntitySurprise, SurpriseExtractor

__all__ = [
    "CUSUMMonitor",
    "ConvergenceCluster",
    "ConvergenceDetector",
    "EntityAlert",
    "EntityAnomalyScorer",
    "EntityBaseline",
    "EntitySurprise",
    "HawkesIntensity",
    "ScorerConfig",
    "SurpriseExtractor",
]

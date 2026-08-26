"""Evidence Graph — document → entities → relationships → searchable graph.

The infrastructure layer for the deeper-intelligence direction. Ingests
unstructured documents (PDF / news / CSV / text), extracts entities and
relationships with explicit confidence, and exposes a searchable evidence
graph over HTTP.

Deterministic and auditable: no LLM, every extraction records *why* (the
source sentence). Reuses the existing entity-graph normalization.
"""

from agent.evidence.graph import build_graph, degree_centrality, neighbors
from agent.evidence.ingest import EvidenceIngestor, Extraction, ingest_to_store
from agent.evidence.store import EvidenceGraphStore

__all__ = [
    "EvidenceGraphStore",
    "EvidenceIngestor",
    "Extraction",
    "ingest_to_store",
    "build_graph",
    "degree_centrality",
    "neighbors",
]

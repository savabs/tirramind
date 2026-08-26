---
title: "Checkpoint 2026-08-25 — Evidence Graph API (Option A foundation)"
tags:
  - doc/checkpoint
  - phase/1
  - topic/evidence-graph
  - topic/product
  - status/active
---

# Checkpoint: Evidence Graph API — Option A foundation built

**Date:** 2026-08-25

## Direction

Per the user's direction ("Evidence Graph API — my favorite"), oriented the product
toward **infrastructure**: ingest unstructured documents → structure into
entities + relationships + confidence → expose a searchable evidence graph. This
moves up the value stack (deeper intelligence, not just a brief), targeting
hedge funds / analysts / AI startups.

## What was built — `agent/evidence/`

| Module | Purpose |
|---|---|
| `store.py` | `EvidenceGraphStore` — SQLite (`.tirra_pipeline/evidence.db`): `evidence_documents`, `evidence_mentions`, `evidence_links`, each with confidence. Append-only, dedup by text hash. |
| `ingest.py` | `EvidenceIngestor` — deterministic extraction (no LLM, auditable): split sentences → match seed entities (high conf 0.95) / capitalized-org fallback (0.6) → build relations (`same_sentence` 0.9, `nearby` 0.4). Reuses `normalize_company_name` from the existing entity graph. `ingest_pdf` (pypdf) / `ingest_csv` / `ingest_text`. |
| `__init__.py` | Public surface. |

## API (wired into `brief_server.py`)

- `POST /evidence/ingest` — body `{doc_id, text|path, doc_type, source, title}` → returns stats.
- `GET /evidence/graph?q=<entity>` → mentions + links (with confidence + evidence).
- `GET /evidence/stats` → document/mention/link counts.

## Verification

- **72 tests passed** (5 new evidence tests) — store dedup, search, same-sentence high-confidence link, stats.
- **ruff clean**.
- **Live HTTP smoke**: POST'd a news doc → mentions=4, links=6; `GET /evidence/graph?q=nvidia` returned mention + same_sentence link to microsoft (conf 0.9) + nearby links (0.4); `/evidence/stats` correct.

## Honest status

- **Deterministic extraction** is a solid, honest foundation (auditable, zero-cost, no dependency).
- **Not yet**: LLM-backed extraction for harder entity types, real networkx graph export, PDF page-level evidence, and the analytics (which entities co-occur across many docs → strong signal). Those are next layers.
- The existing entity graph (15K links) can feed `SEED_ENTITIES` to make extraction far richer without new parsing.

## Next (natural)
1. ~Seed `SEED_ENTITIES` from the existing entity registry (leverage the 4.8K-entity graph).~ ✅ **DONE (this session)**
2. ~Add cross-document co-occurrence analytics (a signal: same pair in many docs = stronger).~ ✅ **DONE (this session)**

## Rich-evidence update (same session)

Made the evidence graph rich:
- `seed_entities_from_registry()` — builds a seed dict from the real entity registry (764 companies, 840 persons, 502 vessels, etc.). `EvidenceIngestor.from_registry()` seeds from it (test confirmed 497 seeded entities).
- **Cross-document analytics** on `EvidenceGraphStore`:
  - `co_occurrences(entity)` — entities that co-occur across DISTINCT documents (n_docs = strength). A pair in many docs = genuine recurring signal.
  - `cross_doc_pairs(min_docs)` — strongest recurring entity pairs.
- New HTTP endpoint: `GET /evidence/analytics?q=<entity>` (co_occurrences + cross_doc_pairs).

### Live demo (HTTP)
Ingested 3 docs of "NVIDIA and Microsoft..." news →
```
co_occurrences for nvidia: microsoft n_docs=4 (strongest), analysts n_docs=3
cross_doc_pairs: microsoft <-> nvidia: 4 docs (the recurring relationship)
```
This is the hedge-fund-worthy signal: **the same pair recurring across documents**, not just one mention.

### Tests
- `tests/test_evidence.py` — 8 passed (incl. registry seed richness, cross-doc co-occurrence, recurring pairs)
- Full regression — **75 passed**; ruff clean.

## Next after rich
- ~~`/evidence/graph/export` → networkx adjacency list for real graph analytics.~~ ✅ **DONE (same session)**

## Graph analytics update (same session)

Added real graph layer on top of the evidence graph:
- **`agent/evidence/graph.py`** (networkx):
  - `build_graph(store)` — weighted undirected graph from evidence links (weight = n_docs, recurring pairs dominate).
  - `degree_centrality(store)` — most-connected entities (hubs).
  - `neighbors(store, entity)` — an entity's immediate weighted neighborhood.
- **Store export**: `graph_export()` → adjacency list `{nodes, edges[{source,target,n_docs,confidence,evidence}]}`.
- New HTTP endpoints:
  - `GET /evidence/graph/export` → full adjacency graph.
  - `GET /evidence/graph/centrality?top=N` → top hubs; `?q=<entity>` → its neighbors.

### Live demo (HTTP)
```
/evidence/graph/export   → nodes=5 edges=9; top edge nvidia<->microsoft n_docs=4
/evidence/graph/centrality → hubs: nvidia(4), microsoft(4), qualcomm(3), tesla(3)
```
This is genuine graph intelligence: which entities are central hubs in the evidence
network, and which relationships recur across documents.

### Tests
- `tests/test_evidence.py` — **10 passed** (incl. graph export n_docs weight, degree centrality, neighbors).
- Full regression — **77 passed**; ruff clean.

## Next after graph analytics
- LLM-backed extraction for harder entity types (optional, later — determinism is a feature).
- Connect the evidence graph to the existing market signal tools (cross-domain: evidence + CFTC positioning).


## Related
- [[deep_intelligence_roadmap]]
- [[checkpoint_2026-08-24_front_door_deploy]]
- [[revenue_plan_2026-05-08]]

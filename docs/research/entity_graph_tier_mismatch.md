---
title: "Research: Entity Graph tier sells the wrong dataset"
tags:
  - doc/research
  - topic/product
  - topic/api
  - status/active
date: 2026-08-26
---

# Research: Entity Graph tier sells the wrong dataset

> Filed by api-backend-engineer while reviewing `agent/brief_server.py`'s
> `/evidence/*` routes and their gating. No implementation code beyond the
> stopgap disclosure fix (see bottom) — this is facts + design space, per
> `RulesForAI.md` §3.

## 1. The mismatch, confirmed

`products/brief_subscription/pricing.html` markets the **Entity Graph** tier as:

> "A learned, queryable graph of **4,800+ entities** — companies, people,
> vessels, countries — with confidence-scored relationships and cross-document
> co-occurrence analytics. Every link traces back to source evidence."
> with example calls `GET /evidence/graph?q=nvidia`, `/evidence/analytics`,
> `/evidence/graph/centrality`.

`brief_server.py` gates exactly those routes (`/evidence/graph`,
`/evidence/stats`, `/evidence/analytics`, `/evidence/graph/export`,
`/evidence/graph/centrality`) behind `_ENTITY_GRAPH_TIERS`. So the tier
customers pay for is defined, in code, as "whatever `/evidence/*` returns."

What `/evidence/*` actually returns comes from `agent/evidence/` — a document
store (`EvidenceGraphStore`, `.tirra_pipeline/evidence.db`) built in a
different session (see `docs/memory/checkpoint_2026-08-25_evidence_graph.md`)
for a different purpose: a deterministic, regex-based extraction demo over
manually-POSTed documents (`POST /evidence/ingest`).

Live counts, right now, from the two databases:

| | `agent/evidence/` (what's served) | `agent/pipeline/store.py` (what's marketed) |
|---|---|---|
| Source | `.tirra_pipeline/evidence.db` | `.tirra_pipeline/pipeline.db` |
| Entities | **5 documents ingested, 155 distinct entity strings matched** | **5,628** rows in `entities` |
| Relationships | 31,706 rows in `evidence_links` (mostly low-value same-doc/nearby co-occurrence pairs generated from those 5 docs) | 16,870 rows in `entity_links` (the real relationship graph `agent/models/gnn/graph_builder.py` trains on) |
| Backing evidence | 283 `evidence_mentions` rows | 365,739 rows in `entity_observations` |
| Extraction method | two regex patterns (`_SENT_SPLIT`, `_UPPER_WORDS` in `agent/evidence/ingest.py`) plus a seed vocabulary of ~497 names pulled from the real entity registry — used only as a name list to recognize, not as linked data | GNN training pipeline, built from live tool ingestion over time |
| Growth | Append-only, but only grows when someone manually calls `POST /evidence/ingest` with the admin `X-Ingest-Token` — nothing auto-feeds it | Grows continuously from the DAG-scheduled pipeline |

The "4,800+" in the marketing copy traces to `seed_entities_from_registry()`
(see the 2026-08-25 checkpoint: "764 companies, 840 persons, 502 vessels...").
That function seeds the regex matcher's *vocabulary* — i.e., which entity
names it's capable of recognizing if they show up in an ingested document — it
does not mean 4,800 entities are queryable today. Only entities present in the
5 ingested sample documents (155 of them) return anything.

**Net: a customer paying for "Entity Graph" gets a 5-document demo dataset
with noisy regex co-occurrence "relationships," not the 5,628-entity /
16,870-link production graph the copy describes, and nothing auto-refreshes
it.**

## 2. Why this wasn't wired correctly in the first place

Two features were built independently and never reconciled:

- `agent/pipeline/store.py` + `agent/models/gnn/graph_builder.py` — the real
  entity graph, built for GNN training (Layer 3, world model).
- `agent/evidence/` — a standalone "evidence graph" product feature, explicitly
  framed at the time as "Option A foundation," "not yet" the real graph
  (checkpoint's own "Honest status" section says as much), pointed at pricing
  before the "not yet" items were closed.

The pricing copy appears to have been written to describe the aspirational
end-state (the real graph, sized off the real entity registry) rather than
what `/evidence/*` actually serves.

## 3. Why "just wire the real graph in" is not a same-file fix

`PipelineStore` already exposes read methods that make routing straightforward
in principle: `query_all_entities()`, `query_all_entity_links()`,
`get_entity()`, `query_entity_observations()` — the same pattern
`/api/v1/data` already uses for the Data Platform tier.

But `entities` / `entity_links` (canonical name, type, confidence, link type)
look reasonably safe to expose as "the entity graph." `entity_observations`
(365K rows, `value_json`) is a different matter — that's raw pipeline
signal/feature data, plausibly the actual alpha the rest of the system is
built on. Whether *any* of that belongs behind a $19–99/mo tier, and if so
which fields, is a product/security decision (what's the real product-market
fit of "Entity Graph," does exposing real relationship data leak the
proprietary signal, does this need field-level redaction) — not something to
decide unilaterally inside `brief_server.py`. That's out of api-backend-
engineer's remit (route/gating contract, not data-exposure policy) and belongs
with whoever owns the tier's product definition, informed by a security review
of what `entity_observations` actually contains.

## 4. What was done now (stopgap, in scope)

Per api-backend-engineer's actual remit — the route/response contract — and
without inventing new paid infrastructure (`CLAUDE.md` §7):

- Added a `dataset_scope` block to every `/evidence/*` JSON response
  (`agent/brief_server.py`), computed fresh per request from the real store
  stats, explicitly stating this is a small document-evidence sample, not the
  production entity graph, and naming this doc. This stops the mismatch from
  being *silent* — a customer or auditor calling the API now sees the truth
  regardless of what the marketing copy says — without touching pricing copy,
  tier definitions, or payments logic (none of which are this file's remit).
- Left a prominent code comment above `_evidence_store` in `brief_server.py`
  pointing back here, so the gap can't be "fixed" away by deleting the
  disclosure without someone reading why it's there.
- Verified live: gating still denies/allows correctly (403 no key, 200 valid
  key, unchanged), and the 10 existing `tests/test_evidence.py` tests still
  pass — this change only adds a field to the HTTP response layer.

## 5. What's still needed (not done here — needs an owner + a spec)

1. **Immediate, urgent, not mine to fix**: `products/brief_subscription/
   pricing.html`'s "4,800+ entities" claim is inaccurate for what
   `/evidence/*` serves today. This is a customer-facing misrepresentation
   risk (refund/chargeback/reputational exposure on an active paid tier) —
   flagging for whoever owns pricing copy / tier definitions
   (`customer-lifecycle`, `payments-auditor`, or the product owner) to correct
   or pull immediately, independent of any engineering fix timeline.
2. **Real fix, needs a spec first** (`RulesForAI.md` §3 — this touches
   multiple files, exposes previously-internal data, is not trivial):
   - Decide what subset of `entities` / `entity_links` is safe to expose
     (redact `entity_observations`'s `value_json` unless deliberately decided
     otherwise).
   - Add read routes to `brief_server.py` following the existing
     `/api/v1/sources` + `/api/v1/data` pattern, reusing `PipelineStore`'s
     existing query methods, gated by `_ENTITY_GRAPH_TIERS`.
   - Either retire `agent/evidence/`'s HTTP surface or keep it as a distinct,
     honestly-named, separately-priced (or free) feature — it's a legitimate
     deterministic-extraction tool on its own, it's just not what's being sold
     as "Entity Graph."
   - Re-price/re-describe the tier once its real dataset is known.

## Related
- [[deep_intelligence_roadmap]]
- [[checkpoint_2026-08-25_evidence_graph]]

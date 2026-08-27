---
title: "Research: The tier ladder is inverted — $50 buys more than $500"
tags:
  - doc/research
  - topic/product
  - topic/api
  - status/active
date: 2026-08-27
---

# Research: The tier ladder is inverted

Found 2026-08-27 while planning the $19 tier. The owner's positioning is
explicit: *"we sell the integration intelligence — the maths, finance, systems,
architecture is our infra."* When the product is the integration, the tier
ladder **is** the product definition. It is currently backwards.

## The defect

`agent/brief_server.py:52-54`:

```python
_ENTITY_GRAPH_TIERS  = {"entity", "data", "scheduler"}
_DATA_PLATFORM_TIERS = {"data", "scheduler"}
_SCHEDULER_TIERS     = {"scheduler"}
```

Resolving that against the price list gives:

| Tier | Price | Surfaces reachable |
|---|---|---|
| brief | $19 | `/brief.*` only |
| **scheduler** | **$50** | **entity-graph + data-platform + dag-runs** |
| entity | $300 | entity-graph |
| data | $500 | entity-graph + data-platform |

**A $50 subscriber reaches strictly more than a $500 subscriber.** The comment
above the constants says scheduler and data "are treated as a superset (they
paid for more surface)" — true of `data`, false of `scheduler`, which is the
cheapest infrastructure tier on the page.

`data` also cannot reach `/api/v1/dag/runs`, so the most expensive tier is
missing a surface the cheapest one has.

## The second defect: the $19 product is given away

`agent/brief_server.py:572-575` gates `/brief.*` on `_valid_key`, and
`_valid_key` (`:263-265`) is `_authorized_for(key, allowed_tiers=None)`.
`_authorized_for` (`:317-319`) returns `True` unconditionally when
`allowed_tiers is None`:

```python
if allowed_tiers is None:
    return True
return store.tier_of_key(key) in allowed_tiers
```

So **every active subscriber of every tier receives the brief**. The string
`"brief"` appears in no tier set at all — it exists only as `_DEFAULT_TIER` in
`agent/payments/handler.py:42`. The $19 tier has no gate of its own; it is a
free add-on to the other three.

## Why this was not caught

The three infrastructure tiers were added incrementally, each adding its own
constant, and no test asserts a relationship *between* tiers. Every existing
test checks one tier against one route ("an entity key reaches
`/evidence/stats`", "a brief key gets 403"), which all pass individually while
the ladder as a whole is incoherent.

## Blast radius

Currently zero: production `subscribers.json` does not exist, so there are no
subscribers of any tier. That is the only reason this is a design defect rather
than a revenue incident. It must be fixed before the first sale, not after.

## Design decision

Access should be monotonic in price — a higher tier reaches a superset of every
lower tier. That is the only arrangement that makes an upgrade path meaningful
when the product being sold is the integrated surface itself.

The alternative reading — that `scheduler` was *intended* as a top tier and is
merely mispriced at $50 — is possible but contradicted by `pricing.html`, which
presents Data Platform at $500 as the flagship and Scheduler at $50 as
"read-only visibility into the DAG execution engine". The copy and the code
disagree; the copy is the customer-facing promise, so the code is what is wrong.

## Related

- [[evidence_ingest_path_traversal]] — the other case this session where two
  individually-correct components composed into a defect nobody owned.

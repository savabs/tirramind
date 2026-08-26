---
title: "Spec: Product value reconciliation (2026-08-27)"
tags:
  - doc/spec
  - topic/product
  - topic/api
  - topic/security
  - status/active
date: 2026-08-27
---

# Spec: Product value reconciliation

Ordered, atomic steps closing gaps found between what each pricing tier
promises (`products/brief_subscription/pricing.html`) and what the product
verifiably delivers. Research: [[data_platform_telemetry_leak]],
[[entity_graph_tier_mismatch]].

1. **Security fix** — deny-list internal DAG telemetry sources out of
   `/api/v1/sources` and `/api/v1/data` in `agent/brief_server.py`. Add a
   regression test reproducing the exact live leak. — DONE, this pass.
2. **Entity Graph tier** — wire `/api/v1/entity-graph/entities`, `/entity`,
   `/links` to the real production graph (`agent/pipeline/store.py`
   `entities`/`entity_links` tables), field-whitelisted to exclude
   `entity_observations` and `metadata_json`, paginated. — DONE, this pass.
3. **Pricing copy accuracy** — correct `pricing.html` claims that outrun the
   code: Data Platform tagline honesty, Entity Graph tier copy reflects the
   new real endpoints, Scheduler tier's false "closed beta: submit custom
   DAGs" claim removed, Opportunity Brief's "learned win-probability" reworded
   to describe the actual prior-based mechanism, FAQ's "trained model with
   auditable weights" claim corrected to state no billed tier currently
   depends on the graph model's output. — DONE, this pass.
4. **Retrain status** — document that GNN evaluation shows no edge over a
   momentum baseline and that a real Phase D retrain requires the owner to
   run it on Kaggle (local sandbox has no GPU). — DONE, this pass (task file
   only; the retrain itself is owner-executed, out of scope for this spec).

## Verification

- Full relevant test suite green: `test_brief_server.py`, `test_evidence.py`,
  `test_graph_builder.py`, `test_graph_builder_expanded.py`.
- New entity-graph routes independently re-verified against
  `agent/pipeline/store.py` to confirm the claimed field exclusions are real,
  not just documented.

## Related

- [[data_platform_telemetry_leak]]
- [[entity_graph_tier_mismatch]]

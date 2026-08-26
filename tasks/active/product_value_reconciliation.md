---
title: Task — Product Value Reconciliation
tags:
  - doc/task
  - topic/product
  - topic/security
  - status/active
---

# Task — Product Value Reconciliation

Research: `docs/research/data_platform_telemetry_leak.md`
Spec: `docs/specs/product_value_reconciliation_spec.md`

Started 2026-08-27.

**Context in one line:** owner asked the team to close every gap between what
each pricing tier promises and what the product actually, verifiably
delivers. Surfaced a live security leak (internal DAG telemetry readable
through the paid Data Platform API) alongside the marketing/reality gaps.

---

## Checklist

- [x] Fix internal-telemetry leak in `agent/brief_server.py` + regression test
- [x] Wire real Entity Graph routes (`/api/v1/entity-graph/*`) + independently
      verify the claimed `entity_observations`/`metadata_json` exclusions
- [x] Correct `pricing.html` copy to match verified reality
- [x] Document GNN retrain status (needs owner to run on Kaggle — no local GPU)
- [ ] Owner sign-off on whether Data Platform checkout should pause pending
      the leak fix reaching production (API backend isn't deployed yet, so
      the live static site was never actually exposed — but flag for
      awareness before the backend goes live)

✓ DONE — leak fixed and tested, entity-graph routes verified, pricing copy
corrected, retrain status documented for owner action. Remaining item is an
owner decision, not implementation work.

---
title: "Task: Deep Surveillance Phase 10b.2 — form144 L2"
tags:
  - doc/task
  - status/done
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Task: Deep Surveillance Phase 10b.2 — form144 L2 Upgrade

Status: completed
Research: [[form144_l2]]
Spec: [[form144_l2_spec]]

## Steps

- [x] 10b.2.1: Extract reporter_cik in `_parse_filings()` — add both CIKs to all filing dicts
- [x] 10b.2.2: Accept optional PipelineStore in constructor
- [x] 10b.2.3: Implement `_persist_entities()` — entity registration + observation storage
- [x] 10b.2.4: CIK-based dedup in `_find_best_sell_cluster()`
- [x] 10b.2.5: Add `entity_ids` mapping to cluster data
- [x] 10b.2.6: Edge case test suite
- [x] 10b.2.7: MI measurement integration test

## Notes

- Pattern established by [[deep_surveillance_10b|insider_filings L2 (10b.1)]]
- All infrastructure ready (PipelineStore entity tables, entity_id_from_key, normalize_company_name, depth_eval)
- CIK swap logic in `_parse_filings()` means reporter_cik must be derived AFTER the swap resolves issuer_cik
- Metadata-only records still get CIKs from EFTS
- observation_type = "sell_intent" (vs "purchase" for insider_filings)

## Related

- [[form144_l2]]
- [[form144_l2_spec]]
- [[deep_surveillance_tools]]
- [[deep_surveillance_10b]]

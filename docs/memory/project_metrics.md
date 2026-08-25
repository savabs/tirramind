---
title: "Project Metrics — Canonical Owner"
tags:
  - doc/memory
  - status/active
  - topic/pipeline
canonical_facts:
  - fact_key: "test_pass_count"
    pattern: "\\b(9[,\\s]?\\d{3})\\s*pass(?:ing|ed)"
  - fact_key: "dag_node_count"
    pattern: "\\b(\\d{1,3})[- ]node\\s+DAG\\b"
  - fact_key: "enrichment_dim"
    pattern: "ENRICHMENT_DIM\\s*[=:]\\s*(\\d+)"
---

# Project Metrics — Canonical Owner

This is the **single source of truth** for all numeric project metrics.
All other active files must reference this file via `[[project_metrics]]` rather than copying raw values.

See `[[AGENTS]]` and `[[copilot-instructions]]` for the Single-Owner Rule.

## Current Metrics (updated 2026-08-23)

| Metric | Value | Notes |
|--------|-------|-------|
| Test pass count | 10461 passing | Fast suite (excludes slow/live/integration), verified 2026-08-23 after fresh 3.12 venv |
| Test fail count | 59 failing | Env/network-key + stale structural-count tests; not signal-blocking |
| DAG node count | 52-node DAG | Per pipeline_registry TestPhase453Nodes expectations |
| ENRICHMENT_DIM | 55 | After Phase 34+ observation type additions |
| Data tool count | 63 | Free no-auth public-API data tools in `agent/tools/` |
| Live-fired signals | 3 families verified | `gov_contracts` (US awards), `cftc` (commodity positioning), `ais_vessel` (shipping) — all real data 2026-08-23 |

## Related

- [[quant_training_ground]] — roadmap and phase ordering
- [[tirramind_structure]] — repo memory (phase history, build sequence)
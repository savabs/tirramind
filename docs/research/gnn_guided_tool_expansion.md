---
title: "Research: GNN-Guided Tool Expansion"
tags:
  - doc/research
  - phase/16
  - topic/surveillance
  - topic/world-model
  - layer/surveillance
  - layer/world-model
---

# Research: GNN-Guided Tool Expansion

## Goal

Use the GNN diagnostics from Phase 15d (`compute_diagnostics()`) to make data-driven decisions about which tools to upgrade to L2 or add next, instead of expanding blindly. The diagnostic pipeline produces five streams of evidence; this phase turns them into a ranked priority list.

Business problem: expanding the surveillance surface costs engineering time, and tools that add a fourth observation channel to an already-dense entity type are lower priority than tools that create the first link between disconnected entity clusters or provide the first observations to a starved entity type.

## Search Log

No external search required. This phase consumes outputs already built in Phases 12–15 and applies them to the tool catalog in [[l2_tool_expansion]].

## Current Architecture

### Diagnostic API (Phase 15d)

`compute_diagnostics(model, store, crystallized?)` in `agent/models/gnn/integration.py` returns:

| Stream | Key | Meaning |
|--------|-----|---------|
| Entity-type density | `entity_type_density` | `{type: count}` — how many entities of each type exist |
| Observation density | `observation_density` | `{obs_type: count}` — how many observations per type |
| Edge-type attention | `edge_type_attention` | `{edge_type: mean_attention}` — HGT attention per relation |
| Neighborhood sparsity | `neighborhood_sparsity` | `{entity_type: mean_degree}` — average degree per type |
| Supervised confidence | `supervised_confidence` | `{entity_type: mean_prob}` — supervised head confidence |

### Current Entity Graph (Phase 13)

9 entity types: company, country, domain, organization, person, protocol, topic, vessel, wallet.

15 observation types: btc_transfer, cert_issued, cross_entity_pattern, dns_change, form144_filing, geopolitical_event, insider_trade, lobbying_spend, pageview_spike, patent_filing, port_call, project_status, sell_intent, tvl_change, vessel_position.

12 L2 tools feed the graph: insider_filings, form144, whale_alert, ais_vessel, gdelt, cert_transparency, dns_monitor, wikipedia_pageviews, lobbying, patent_filings, defi_flows, interconnection_queue.

### Candidate Tools (from [[l2_tool_expansion]])

**Company-entity tools** (high potential): gov_contracts, creditor_filings, bankruptcy_court, sanctions_monitor, drug_regulatory, job_postings, finra_short_volume, supply_chain_monitor.

**Aggregate tools** (likely stay L1 conditioning): treasury_receipts, consumer_sentiment, central_bank_balance, global_pmi, energy_supply, food_security, macro_data, capital_flows.

**Physical/geo tools** (potential new entity links): weather_alerts, disease_surveillance, earthquake_proximity, satellite_activity, transport_throughput.

## Observations

### What each diagnostic stream tells us operationally

1. **entity_type_density** — Types with very few entities lack training signal. If `domain` has 2 entities while `company` has 50, domain neighborhoods are statistically unreliable. Fix: add more tools that produce domain entities, or lower the priority of learning from domain nodes until density improves.

2. **observation_density** — Obs types with low counts starve the self-supervised next-event predictor. If `patent_filing` has 5 observations while `insider_trade` has 200, the model barely learns to predict patent filings. Fix: either add tools that produce that obs type more frequently, or reclassify rare obs types as features of a more common type.

3. **edge_type_attention** — Low mean attention on a relation type means the HGT is ignoring it. Possible causes: (a) the linked observations on both sides don't co-occur temporally — the link exists but carries no temporal predictive signal; (b) the link exists but the entity type on one side has too few observations for the model to learn anything. Fix depends on cause.

4. **neighborhood_sparsity** — Mean degree < 1.0 for a type means most entities of that type have zero or one link. The GNN cannot propagate useful information through isolated nodes. Fix: add tools that create links to/from that type, or add linking heuristics (e.g., sector-based company–company links).

5. **supervised_confidence** — Low mean probability for a type means the supervised head is uncertain about co-occurrence predictions involving that entity type. Combined with sparsity, this flags where adding data would most improve predictive quality.

### Operational interpretation thresholds (provisional)

These are starting points. They should be revised after the first synthetic + real diagnostic runs.

| Stream | Threshold for "needs attention" | Rationale |
|--------|--------------------------------|-----------|
| entity_type_density | < 5 entities | Too few for reliable gradient signal |
| observation_density | < 10 observations per type | Self-supervised loss barely trains on it |
| edge_type_attention | < 0.05 mean attention | Model is ignoring this relation |
| neighborhood_sparsity | mean degree < 1.0 | Most entities of this type are isolated |
| supervised_confidence | mean prob clustered near 0.5 | Model has no discriminative signal |

### Decision rules

1. **Sparse + low-confidence entity type** → highest priority: find tools that add both entities and observations of that type, and links connecting it to denser neighborhoods.
2. **Low-attention edge type** → investigate: if the endpoint types are well-populated, the edge type may just be uninformative. If one endpoint is sparse, fixing sparsity may fix attention.
3. **Dense entity type + dense obs + high attention** → no action needed; this part of the graph is healthy.
4. **Aggregate tools** → do not force into entity graph. Use as global conditioning features (feed into model as a global context vector or time-varying bias), not as entity nodes.
5. **New tool creation** → only if diagnostics reveal a gap that no existing tool can fill. Prefer upgrading existing L1 tools to L2 before building new tools.

## Ranking Rubric

For each candidate tool, score on five dimensions:

| Dimension | Weight | Metric |
|-----------|--------|--------|
| Graph connectivity gain | 0.30 | Δ mean_degree for the target entity type |
| Signal uniqueness | 0.25 | Does it add an obs type not yet in the graph, or increase density of a starved one? |
| Implementation effort | 0.20 | Estimated hours; L2 upgrade of existing tool ≈ 2–4h; new tool ≈ 8–16h |
| Data quality risk | 0.15 | Free API stability, update frequency, historical depth |
| Overlap with dense neighborhoods | 0.10 | Penalty if target entity type already has ≥ 3 obs types |

Weighted score → Tier 1 (top 3), Tier 2 (next 5), Tier 3 (rest).

## Risks

- **Synthetic data may not reflect real sparsity.** SyntheticGraphGenerator creates uniform entity distributions. Real data will be skewed (many companies, few protocols). Synthetic runs validate the diagnostic workflow; real data drives final ranking.
- **Threshold sensitivity.** Provisional thresholds may over- or under-flag entity types. Plan to revise after the first diagnostic run.
- **Circular dependency.** Diagnostics depend on a trained model; model quality depends on data density. First iteration bootstraps from self-supervised pre-training only; supervised confidence becomes useful only after fine-tuning.

## Data Requirements

- Trained HetTGN model (from `Trainer` or `retrain_and_discover`)
- PipelineStore with L2 entity data (12 tools already feeding)
- Crystallized patterns (from Phase 14/15 pipeline) for supervised confidence stream
- No new external data sources needed for the diagnostic pass itself

## Math/Algorithm Survey

No new math. Phase 16 is an application of the diagnostics already implemented. The ranking rubric uses a simple weighted linear score, not a learned model. If the number of candidate tools grows beyond ~20, consider a multi-criteria decision framework (AHP or TOPSIS), but for now the weighted score is sufficient and more interpretable.

## Related

- [[gnn_guided_tool_expansion_spec]] — Phase 16 spec
- [[gnn_guided_tool_expansion]] — Phase 16 task
- [[gnn_pattern_and_finetuning]] — Phase 14/15 research (diagnostic outputs defined here)
- [[gnn_pattern_and_finetuning_spec]] — Phase 14/15 spec
- [[l2_tool_expansion]] — Phase 13 research (candidate tool catalog)
- [[l2_tool_expansion_spec]] — Phase 13 spec

---
title: "GNN Diagnostics: Synthetic Baseline"
tags:
  - doc/research
  - phase/16
  - topic/surveillance
  - topic/world-model
  - layer/world-model
---

# GNN Diagnostics: Synthetic Baseline

**Generated**: Phase 16a.3
**Config**: `SyntheticGraphGenerator(num_companies=6, num_countries=3, num_vessels=3, num_wallets=3, seed=42)`
**Model**: `TrainerConfig(hidden_dim=16, memory_dim=16, message_dim=16, num_heads=2, num_layers=1, epochs=5, window_size=172800.0)`

---

## Entity Density (threshold: < 5 flagged)

| Entity Type | Count | Flagged? |
|-------------|-------|----------|
| company     | 6     | —        |
| country     | 3     | **YES**  |
| vessel      | 3     | **YES**  |
| wallet      | 3     | **YES**  |

**Interpretation**: 3 of 4 entity types below density threshold. Expected on synthetic data with small entity counts. Real data should have more entities per type from L2 tools.

## Observation Density (threshold: < 10 flagged)

| Observation Type      | Count | Flagged? |
|-----------------------|-------|----------|
| port_call             | 3,914 | —        |
| btc_transfer          | 7,860 | —        |
| vessel_position       | 3,880 | —        |
| insider_trade         | 5,064 | —        |
| sell_intent           | 5,144 | —        |
| geopolitical_event    | 7,597 | —        |
| form144_filing        | 5,237 | —        |

**Interpretation**: All observation types well above threshold. Synthetic generator produces uniform event rates across types — real data will vary based on actual tool activity and API volume.

## Edge Type Attention (threshold: < 0.05 flagged)

| Edge Type                          | Mean Attention | Flagged? |
|------------------------------------|----------------|----------|
| company→headquartered_in→country   | 0.500          | —        |
| vessel→port_call_to→country        | 0.000          | **YES**  |
| wallet→exchange_based_in→country   | 0.000          | **YES**  |

**Interpretation**: Model concentrates all attention on `headquartered_in` edges. `port_call_to` and `exchange_based_in` edges receive zero attention — model finds no useful signal in those relationships. This is consistent with synthetic data where all observations are generated uniformly (no structural difference between edge types). With real data, tools that produce richer vessel/wallet→country interactions may shift these weights.

## Neighborhood Sparsity (threshold: mean degree < 1.0 flagged)

| Entity Type | Mean Degree | Flagged? |
|-------------|-------------|----------|
| company     | 1.0         | —        |
| country     | 4.0         | —        |
| vessel      | 1.0         | —        |
| wallet      | 1.0         | —        |

**Interpretation**: No entity types flagged for sparsity. Country nodes have degree 4.0 (hub role connecting via `headquartered_in`, `port_call_to`, `exchange_based_in`). Other types have degree 1.0 — each connected to exactly one country.

## Supervised Confidence

| Pattern Type | Mean Confidence | Flagged? |
|--------------|-----------------|----------|
| *(empty)*    | —               | —        |

**Interpretation**: No outcome labels were generated in this run. `generate_outcome_labels()` requires crystallized patterns with matching entity pairs in the graph — the synthetic data with uniform input distributions doesn't reliably produce labelable patterns. This stream will be meaningful only on real data with actual market outcomes.

## Summary

| Metric                 | Count |
|------------------------|-------|
| Flagged entity types   | 3     |
| Flagged obs types      | 0     |
| Flagged edge types     | 2     |
| Flagged sparse types   | 0     |
| Flagged uncertain types| 0     |

## Caveats

1. **Synthetic data has uniform distributions.** Entity counts, event rates, and observation signals are symmetric across types. Real data will be asymmetric — some entity types will have far more data than others.
2. **Edge type attention is heavily model-dependent.** With only 5 epochs on synthetic data, attention weights are not stable estimates. They establish baseline behavior, not converged attention.
3. **Entity density thresholds are provisional.** The threshold of 5 was chosen to flag obviously sparse types. After real-data diagnostic runs, thresholds may need adjustment.
4. **The supervised confidence stream requires market outcome data** (resolved Polymarket events, realized insider trade outcomes). Synthetic data cannot produce this signal.

## Related

- [[gnn_guided_tool_expansion]] — Phase 16 research
- [[gnn_guided_tool_expansion_spec]] — Phase 16 spec
- [[gnn_guided_tool_expansion]] — Phase 16 task
- [[gnn_pattern_and_finetuning]] — Phase 14/15 (diagnostic API source)

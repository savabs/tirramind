---
title: "Phase 42 — Entity Diversity Expansion: From Stock-Indexing Model to Multi-Domain Economic Intelligence"
tags:
  - doc/research
  - phase/42
  - topic/gnn
  - topic/pipeline
  - topic/surveillance
  - topic/diversity
  - layer/surveillance
  - layer/world-model
---

# Phase 42 — Entity Diversity Expansion

## Problem Statement

Phase 41 delivered a hardened, statistically clean HetTGN checkpoint: 87.0% test top-1 accuracy, 1.9s `time_delta` MAE. That headline number hides a structural issue in what the model is actually learning.

**The observation distribution is pathologically imbalanced.**

```
┌─────────────────┬────────┬────────────────┬───────────────────────────────┐
│ source_tool     │  count │ share of total │ entity type it writes         │
├─────────────────┼────────┼────────────────┼───────────────────────────────┤
│ instrument_universe  │ 68,890 │     97.0%   │ instrument (89 nodes)         │
│ polymarket           │  1,458 │      2.0%   │ topic (729), country (82)     │
│ gdelt                │    735 │      1.0%   │ country, topic                │
│ cftc                 │     40 │      0.06%  │ cftc_contract (20)            │
└─────────────────┴────────┴────────────────┴───────────────────────────────┘

Entity coverage gaps:
  company  — 7 nodes, ~0 observations      (DEAD)
  protocol — 2 nodes, ~0 observations      (DEAD)
  country  — 82 nodes, only GDELT events   (SPARSE)
```

TirraMind is designed as a **general economic and financial predictor** operating across Layers 0-3 (physical reality → human decisions → information flows → markets). Today the GNN is 97% indexing asset prices. That is a stock-market calendar model wearing a HetTGN hat — exactly the kind of Layer-3-only system the project doctrine rejects.

**We have the raw capability to fix this.** 40+ data tools exist and are registered with `pipeline_store` wiring. Only 9 are scheduled in `daily_collection`. The gap between "tool exists" and "data flows into the graph" is the entire source of the imbalance.

## Why Entity Diversity Matters for a GNN

The heterogeneous GNN (HetTGN) uses typed message passing: each edge type `(src_type, rel, dst_type)` has its own learned transformation. When an entity type has zero observations, every attention head that routes to that type sees only a zero-initialized memory vector. The model learns:

1. **Nothing** about the type's intrinsic dynamics (no temporal signal to shape its memory).
2. **Nothing** about how it relates to other types (no gradient flows from dead nodes during the self-supervised next-observation task).

The degradation is not linear. A type with zero observations is effectively **deleted from the model's world view**. The 7 company nodes and 2 protocol nodes in the current graph contribute no information to prediction. The 82 country nodes contribute only news-event counts from GDELT — no macro, no yields, no central bank positioning, no capital flows.

### The Math: Effective Participation

Let $n_k$ be the number of observations for entity type $k$, $N = \sum_k n_k$ the total. The share $p_k = n_k/N$ is the fraction of training-signal mass that updates type $k$'s embeddings.

Shannon entropy of the observation distribution:
$$H(\mathbf{p}) = -\sum_k p_k \log p_k$$

With 6 types, $H_{\max} = \log 6 \approx 1.79$ nats (perfectly balanced).

Current state (Phase 41):
$$H \approx -(0.97\ln 0.97 + 0.02\ln 0.02 + 0.01\ln 0.01 + \ldots) \approx 0.17 \text{ nats}$$

We are at **~9.5% of maximum entropy**. The model is essentially univariate at the type level. Any claim of "multi-domain reasoning" is marketing until that entropy rises meaningfully — target ≥ 1.0 nats (56% of max) as a concrete phase goal.

### Information-Theoretic Framing

This is a classic class-imbalance problem dressed in graph clothing. The literature (Johnson & Khoshgoftaar 2019 — "Survey on deep learning with class imbalance") documents that extreme imbalance causes:

- **Representation collapse** on minority classes (the learned embedding for rare types degenerates toward a type-prior centroid).
- **Gradient starvation** (minority-type parameters receive $O(p_k)$ of the aggregate gradient per epoch).

For heterogeneous GNNs specifically, Lv et al. 2021 ("Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks") show that minor tweaks to message-passing on balanced graphs beat complex architectures on imbalanced ones. **Fixing the data distribution beats tuning the model.** This is the Phase 42 thesis.

## Current Tool Inventory vs. Scheduled Tools

Audit of `agent/tools/` vs `agent/pipeline/dags/daily_collection.py`:

| Category | Built (has `pipeline_store`) | Currently Scheduled |
|---|---|---|
| Macro / Sovereign | central_bank_balance, sovereign_debt, global_pmi, capital_flows, macro_data, consumer_sentiment, treasury_receipts | macro_data |
| Corporate / Filings | insider_filings, form144, patent_filings, lobbying, gov_contracts | — |
| Crypto / On-chain | whale_alert, defi_flows | whale_alert |
| Attention / Social | wikipedia_pageviews, polymarket, gdelt | polymarket, gdelt |
| Trade / Logistics | comtrade, transport_throughput | — |
| Regulatory / Legal | bankruptcy_court, creditor_filings, sanctions_monitor, regulatory_gazette, drug_regulatory, foia_requests | — |
| Market structure | cftc, finra_short_volume, instrument_universe | all three |
| Energy / Grid | power_grid, electricity_monitor, interconnection_queue, energy_supply | power_grid ×2 |
| Other physical | ais_vessel, weather_alerts, earthquake_proximity, disease_surveillance, food_security, labor_disruptions, internet_outages, political_risk, migration_flows | — |

**Nine scheduled tools. ~25 L2-capable tools dormant.**

## Phase 42 Scope: 8-Tool Activation

We will wire 8 tools into `daily_collection` in one coherent batch. Selection criteria:

1. **Activate every dormant entity type** — `company`, `protocol`, `person` must all gain observations.
2. **Densify country nodes** — move from GDELT events only to macro + monetary + fiscal coverage.
3. **Cross-domain link creation** — tools that produce edges bridging disconnected clusters get priority.
4. **Free, no-auth-failure-risk** — minimise operational fragility in one batch.
5. **Daily cadence makes sense** — tools whose underlying data refreshes monthly/quarterly still get wired (observation arrives on a daily DAG; underlying source emits less frequently, that's fine — the next-observation prediction head learns the inter-arrival time).

### Selected Tools and Rationale

| # | Tool | Entity types fed | Signal type | Trusted source |
|---|---|---|---|---|
| 1 | `insider_filings` | company, person | SEC Form 4 open-market purchases. Project-memory flagship: *"executives reveal private info through their trades."* | [SEC EDGAR submissions API](https://www.sec.gov/edgar/sec-api-documentation) |
| 2 | `central_bank_balance` | country | Fed/ECB/BOJ/BOE/PBOC balance sheets, FX-normalised. Global liquidity = single biggest driver of all risk assets. | [FRED API](https://fred.stlouisfed.org/docs/api/fred/) |
| 3 | `sovereign_debt` | country | US/EU/JP/UK yield curves — growth + inflation expectations encoded per country. | FRED constant-maturity series |
| 4 | `global_pmi` | country | OECD Composite Leading Indicators + Business/Consumer Confidence for G7 + major EM. | [OECD SDMX API](https://data-explorer.oecd.org/) |
| 5 | `capital_flows` | country | US TIC data: foreign holdings of Treasuries, net purchases, reserve flows. | FRED TIC series |
| 6 | `defi_flows` | protocol | DeFi TVL, stablecoin supply, DEX volumes. Activates the two dead `protocol` nodes (Uniswap, Aave). | [DefiLlama public API](https://defillama.com/docs/api) |
| 7 | `wikipedia_pageviews` | topic | Article-level pageview spikes as attention leading indicator. Densifies the 729 sparse topic nodes beyond GDELT event counts. | [Wikimedia Pageviews API](https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews) |
| 8 | `lobbying` | company | US Senate LDA filings — who pays whom to change rules. Adds strategic-intent observations to company nodes. | [US Senate LDA public disclosures](https://lda.senate.gov/system/public/) |

### What's Intentionally Excluded from Phase 42

- `form144` (planned sales filings) — redundant with `insider_filings` for the same entity type; wait for Phase 43 GNN audit to confirm marginal signal.
- `comtrade` / `transport_throughput` — these produce flow data without a clear entity_type writer today; need a small extractor audit before scheduling.
- `bankruptcy_court` / `creditor_filings` / `sanctions_monitor` / `patent_filings` / `gov_contracts` — all useful but the Phase 42 batch is already large; keep Phase 43 for a second pass.
- `ais_vessel`, `satellite_activity`, `weather_alerts`, `earthquake_proximity`, `disease_surveillance` — physical-world tools requiring more careful entity design (vessel entity type not yet in registry; need a dedicated phase).
- `job_postings`, `building_permits`, `treasury_receipts`, `energy_supply` — no `pipeline_store` writes verified or FRED-only without entity linkage; keep as L1 support tools for now.

### Tool Parameter Decisions (Verified Against Source Code)

From `grep -oE 'kwargs\.get\("[^"]+"[^)]*\)'` audits:

| Tool | Mode we'll schedule | Other params | Justification |
|---|---|---|---|
| insider_filings | *(kwargless, days_back defaults to 30)* | `days_back=30, min_cluster_size=3` | Default window captures weekly to monthly clustering. |
| central_bank_balance | `mode="balance_sheets"` | `period="1y"` | Snapshot all CBs. `liquidity_index` is a derived mode we can schedule in Phase 43 once raw data is flowing. |
| sovereign_debt | `mode="us_yields"` + another node `mode="eu_yields"` | — | Only yields modes hit FRED/Ecb directly; `spreads` is derived. Two nodes keep fan-out simple. |
| global_pmi | `mode="cli"` | `countries=<default G7+>` | Composite Leading Indicator is the broadest single OECD series. |
| capital_flows | `mode="holdings"` | `country="all"` or default | TIC holdings is the densest of the three modes. |
| defi_flows | `mode="tvl"` | `limit=20` | TVL across top 20 protocols is the standard default. |
| wikipedia_pageviews | `mode="spike"` | `days_back=30, z_threshold=2.0, limit=50` | Spike mode gives anomaly observations; denser than flat series. |
| lobbying | `mode="spending"` | `year=<current>` | Spending mode gives numeric amounts per company; richer than `issues` or `search`. |

### Secondary Concern: Operator Resolution

The DAG executor resolves `operator="<string>"` by looking up `<string>` in the `ToolRegistry` (see `agent/pipeline/operators.py:ToolOperator`). All 8 tools are already registered in `agent/cli.py`:

```
InsiderFilingsTool       → name="insider_filings"       ✓
CentralBankBalanceTool   → name="central_bank_balance"  ✓
SovereignDebtTool        → name="sovereign_debt"        ✓
GlobalPmiTool            → name="global_pmi"            ✓
CapitalFlowsTool         → name="capital_flows"         ✓
DefiFlowsTool            → name="defi_flows"            ✓
WikipediaPageviewsTool   → name="wikipedia_pageviews"   ✓
LobbyingTool             → name="lobbying"              ✓
```

No extra registration work needed.

### Risks & Mitigations

1. **API rate limits / failures** — several tools hit third-party APIs (OECD SDMX, DefiLlama, Wikimedia). Mitigation: all nodes use `retries=2` and `timeout=120s`. DAG executor already handles node-level failures without poisoning the run.
2. **Data volume surprise** — `wikipedia_pageviews` in spike mode with `limit=50` could produce hundreds of observations per run. Acceptable — the graph is observation-starved.
3. **FRED API key** — `central_bank_balance`, `sovereign_debt`, `capital_flows` all require `TIRRA_FRED_API_KEY`. If unset, they return a structured error rather than crashing. The DAG continues.
4. **Single-batch risk** — wiring 8 tools at once is riskier than one at a time. Mitigation: DAG tests + a dry-run assertion that every new node has a registered operator before we commit to a live collection.
5. **Training time** — more observations → larger graph → longer retrain. Current 5-epoch retrain is 10 min. Expect 15-20 min post-expansion. Budget for it.

## Acceptance Criteria

After Phase 42 completes:

- [ ] `daily_collection` has **17 nodes** (current 9 + 8 new).
- [ ] All DAG structure tests pass (`tests/test_pipeline_registry.py`).
- [ ] Executing `daily_collection` end-to-end populates data for every new tool (non-zero row counts in their respective `entity_observations` rows — verify in the audit step).
- [ ] Observation entropy $H(\mathbf{p}) \geq 1.0$ nats (vs 0.17 today) — target 56% of max 6-type entropy.
- [ ] Every entity type has ≥ 100 observations (no dead types).
- [ ] GNN retrain: monotonic loss, weights in [0.05, 20], test top-1 holds ≥ 70% (lower bar than Phase 41's 87% because we're predicting over a much more diverse observation space now; 70% still hugely beats the ~random 2.17% baseline).
- [ ] Walk-forward backtest still runs cleanly; no regression in Sharpe baselines.

## Depth Roadmap (Per Doctrine)

Per the Signal Depth Doctrine in copilot-instructions.md, each tool should have an L1/L2/L3 roadmap.

| Tool | L1 (aggregate) | L2 (entity-level — today) | L3 (cross-entity) |
|---|---|---|---|
| insider_filings | # of buys, total $ | Per-executive trading history; per-company filer density | Cluster buys across companies in same sector; cross-exec cluster within one company (the "conviction" signal) |
| central_bank_balance | Fed total assets (FRED) | Per-country CB balance & rates | Policy-divergence edges: country↔country relative stance |
| sovereign_debt | 10Y yield chart | Per-country curve points (2Y/5Y/10Y/30Y) | Cross-country spreads; curve-inversion detection as country-level edge |
| global_pmi | OECD world CLI | Per-country CLI level & change | Lead-lag relationships country→country |
| capital_flows | Total TIC holdings | Per-country holdings/flows | Bilateral concentration edges (who holds whose debt) |
| defi_flows | Total DeFi TVL | Per-protocol TVL/users/volume | Cross-protocol flow edges (Uniswap→Aave liquidity migration) |
| wikipedia_pageviews | Daily total pageviews | Per-article pageview time-series + z-score spike detection | Article↔instrument attention transfer, article↔country crisis signals |
| lobbying | Total $ spent / quarter | Per-company lobbying time-series | Company→issue_area edges; lobbying-intent→policy-outcome tracking |

Phase 42 lands us at L2 for all 8 tools. L3 is Phase 43+ work driven by the GNN-guided expansion audit (per AGENTS.md §"Signal Depth Doctrine rule 6").

## Math-Heavy Note: Why We're Not Weighting the Loss Instead

A reasonable alternative to fixing the data distribution is re-weighting the training loss to upweight rare-type observations:

$$\mathcal{L}' = \sum_i w(k_i) \cdot \mathcal{L}_i,\quad w(k) = 1 / \sqrt{n_k}$$

This is the standard imbalanced-softmax fix. We are **not** doing this in Phase 42 because:

1. Re-weighting can't create information that doesn't exist. If `company` has 0 observations, no weight scheme helps. Data first, then calibration.
2. The Kendall auto-tune loss (Phase 41) already implements per-head variance-weighted loss across the 3 heads (`obs_type`, `time_delta`, `contrastive`). Adding entity-type reweighting on top stacks two orthogonal calibration schemes and complicates debugging.
3. The project doctrine is explicit: "Maximize learnable structure; minimize hand-coded intelligence." Rebalancing the observation stream is data-engineering; loss-reweighting is a hand-tuned hack.

If, after Phase 42 completes, the retrained model shows persistent per-type accuracy gaps, **that** is when loss-reweighting or focal loss becomes justified — and will be the Phase 43 discussion.

## Related

- [[phase41_model_refresh_hardening]] — immediate predecessor, cleared the clamp runaway
- [[phase41_model_refresh_hardening_spec]]
- [[chat_checkpoint_2026-04-21_phase41_complete]]
- [[project_memory]]
- [[SCHEMA]]

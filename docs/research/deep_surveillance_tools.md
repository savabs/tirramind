---
title: "Research: Deep Surveillance Tool Upgrades"
tags:
  - doc/research
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Research: Deep Surveillance Tool Upgrades

## Goal

Upgrade TirraMind's 60+ data tools from surface-level aggregates (L1) to entity-level resolution (L2) and cross-domain entity linking (L3), then prepare a future L4 inference layer that turns linked entity patterns into latent-state signals, per the [[project_memory|Deep Surveillance Doctrine]].

---

## Current Architecture

### Pipeline Store Schema (SQLite)

| Table | Purpose | Entity-Aware? |
|-------|---------|---------------|
| `dag_runs` | Pipeline execution tracking | No |
| `pipeline_data` | Raw fetched data (JSON blobs) | No — data_json is opaque |
| `signals` | Named scalar time-series | No — no entity dimension |
| `features` | Engineered features (EngineeredFeature protocol) | No — keyed by feature_name only |
| `beliefs` | World model posteriors | No — keyed by variable_name |

**Key gap:** No table has an `entity_id` or `entity_type` column. Every row represents a market-wide or tool-wide aggregate. There is no way to ask "what do we know about entity X across all data sources?"

### Tool Inventory (60 files in `agent/tools/`)

**11 Deep (L2-capable already):**
insider_filings, finra_short_volume, cftc, whale_alert, gdelt, bankruptcy_court, form144, ais_vessel, disease_surveillance, patent_filings, central_bank_balance

**13 with entity-tracking potential (from moderate tools):**
polymarket, polymarket_whales, lobbying, sanctions_monitor, political_risk, drug_regulatory, interconnection_queue, dns_monitor, cert_transparency, gov_contracts, foia_requests, creditor_filings, earthquake_proximity

**Remaining ~36:** Aggregate/macro tools (market_data, macro_data, power_grid, etc.) — L1 by nature, entity dimension less applicable.

---

## Observations

### What "L2" Actually Means Per Tool

Each tool has a natural entity type. L2 means resolving to that entity and building time-series per entity.

| Tool | Natural Entity | Current State | L2 Upgrade |
|------|---------------|---------------|------------|
| `insider_filings` | Person (insider) + Company (CIK) | ✅ Clusters insiders, names + roles | Track per-insider purchase history, build conviction scores over time |
| `form144` | Person (insider) + Company (CIK) | ✅ Clusters sell-intent by insider | Link Form 144 → Form 4 execution, track pre/post portfolio ratio |
| `whale_alert` | Wallet address (BTC) | ✅ Resolves individual addresses | Build wallet profiles: age, tx frequency, exchange deposit patterns, clustering |
| `gdelt` | Actor (country/org/person) dyads | ✅ Resolves actor pairs | Build per-actor escalation curves, tone trajectories, event frequency baselines |
| `ais_vessel` | Vessel (MMSI/IMO) | ✅ Individual vessel tracking | Build route fingerprints, dark-period detection, port-call anomaly scores |
| `bankruptcy_court` | Company (debtor) + Court | ✅ Debtor names, case numbers | Track creditor claim lists, DIP lender identity, trustee patterns |
| `patent_filings` | Company (assignee) + Inventor | ✅ Per-assignee filing velocity | Add citation networks, inventor mobility tracking, CPC pivot detection |
| `disease_surveillance` | Geography (state/country) + Pathogen strain | ✅ Per-state/pathogen data | Add variant-level metadata, cross-source temporal correlation |
| `finra_short_volume` | Ticker (aggregate across facilities) | ⚠️ Per-ticker only, no facility breakdown | Add facility-level granularity, borrow fee integration |
| `cftc` | Category (managed money/producer/swap) | ❌ Category-level aggregates | CFTC COT is structurally L1 — no individual trader data in public reports |
| `central_bank_balance` | Institution (7 CBs) | ❌ Institution-level aggregates | Add FOMC dealer data, repo haircuts, swap line usage, ELA |
| `polymarket_whales` | Wallet address (Polygon) | ✅ Per-whale scoring | Already L2 — extend with cross-market whale linking |
| `lobbying` | Registrant + Client + Issue | ✅ Entity-level spend | Track spend trajectories, detect sudden sector-specific surges |
| `sanctions_monitor` | Sanctioned entity (person/org/vessel) | ✅ Individual entities | Cross-link to AIS vessels, EDGAR filings, wallet addresses |
| `gov_contracts` | Contractor + Agency | ✅ Per-contractor awards | Track award velocity, sole-source concentration, sector shifts |

### What "L3" Actually Means — Cross-Domain Patterns

L3 is combining entities across tools to surface patterns invisible within any single source. The entity resolution layer is the prerequisite.

**Pattern 1: Insider + Vessel + Positioning (Sanctions Evasion Detection)**
- `insider_filings`: Defense/energy company insiders selling before sanctions announcement
- `ais_vessel`: Tankers turning off AIS transponders near sanctioned ports
- `cftc`: Managed money pivoting crude oil positioning
- `sanctions_monitor`: New entity additions to OFAC SDN
- **Combined signal:** Convergence of these four on the same 2-week window predicts sanctions regime change

**Pattern 2: Drug Approval Lifecycle**
- `disease_surveillance`: ClinicalTrials.gov phase transition (Phase 2 → Phase 3)
- `patent_filings`: Clustering of therapeutic-area patents by the pharma company
- `insider_filings` + `form144`: Insider buying/selling patterns at the company
- `drug_regulatory`: FDA filing language specificity
- **Combined signal:** Phase transition + patent acceleration + insider buying = high approval probability

**Pattern 3: Industrial Production Proxy**
- `electricity_monitor`: Facility-level power consumption anomalies
- `ais_vessel`: Port-call frequency changes for cargo/bulk carriers
- `transport_throughput`: Trucking/rail volumes near specific facilities
- `job_postings`: Hiring surges at specific companies/regions
- **Combined signal:** Energy + logistics + hiring convergence reveals production ramp before official data

**Pattern 4: Geopolitical Escalation Timing**
- `gdelt`: Actor-pair conflict event intensity curves
- `ais_vessel`: Military/government vessel route changes
- `lobbying`: Defense contractor lobbying spend surges
- `sanctions_monitor`: Draft designation lists
- **Combined signal:** Multi-domain escalation convergence predicts kinetic events

**Pattern 5: Bankruptcy Contagion**
- `bankruptcy_court`: Chapter 11 filing with creditor list
- `creditor_filings`: Same creditors appearing across multiple bankruptcies
- `insider_filings`: Insiders at creditor companies selling
- `finra_short_volume`: Short volume spike on creditor-company tickers
- **Combined signal:** Shared-creditor clustering predicts next-to-fail company

### What "L4" Could Mean — Inferred Latent Structure

L4 should not be "more complicated L3". It should be a distinct layer that consumes L3 linked-entity patterns and produces explicit inferred state variables, coordination motifs, or regime-transition precursors.

The clean distinction is:
- L1: aggregate observations
- L2: entity-resolved observations
- L3: linked entity patterns across tools
- L4: inferred hidden structure built from those linked patterns

Examples of valid L4 outputs:

**Pattern 6: Hidden Coordination State**
- Inputs: recurring insider, creditor, and vessel co-movements over time
- L3 object: repeated cross-tool motifs involving the same entities
- L4 output: posterior probability of a hidden coordination state such as `distress_preannouncement`, `sanctions_evasion`, or `inventory_ramp`

**Pattern 7: Regime Transition Precursors**
- Inputs: GDELT escalation curves, sanctions additions, vessel rerouting, lobbying surges
- L3 object: aligned multi-domain escalation windows
- L4 output: hazard score or transition probability for a geopolitical regime shift

**Pattern 8: Counterparty Stress Graph**
- Inputs: bankruptcy creditor overlap, insider selling, short-volume spikes, contract loss signals
- L3 object: connected subgraph of firms and creditors under stress
- L4 output: latent contagion score for the subgraph, not just the individual entities

### L4 Design Constraints

To count as L4, a signal must satisfy all of the following:
- It is computed from linked L3 structures, not directly from raw tool payloads.
- It produces an explicit state, score, motif label, or transition probability.
- It can be backtested or otherwise evaluated against a downstream target.
- It is storable as a first-class object, not just embedded ad hoc inside a report.

Non-examples:
- Adding more fields to an entity observation
- More pairwise joins without state inference
- A larger cluster report with no new latent variable or decision statistic

### L4 Storage Direction (Future)

The current Phase 10 registry is enough for L2 and most L3 work, but a proper L4 likely needs one of these representations:
- `motif_observations`: timestamped inferred motifs over sets of entities
- `entity_graph_snapshots`: graph-derived features or communities at a point in time
- `latent_state_observations`: posterior probabilities, hazard rates, or regime labels keyed by entity, subgraph, or market

Recommendation: do not add new schema for L4 until at least two L3 workflows are operational. Premature schema design here would be guesswork.

---

## Entity Resolution Architecture — Design Options

### Option A: Lightweight ID Registry (Recommended for Phase 10)

A simple SQLite table mapping cross-source IDs to a canonical entity:

```sql
CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY,       -- UUID or deterministic hash
    entity_type TEXT NOT NULL,        -- 'company', 'person', 'vessel', 'wallet', 'country'
    canonical_name TEXT NOT NULL,     -- normalized display name
    created_at REAL NOT NULL,
    metadata_json TEXT                -- flexible attributes
);

CREATE TABLE entity_aliases (
    alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    source TEXT NOT NULL,             -- 'edgar_cik', 'mmsi', 'btc_address', 'ticker', 'fips'
    external_id TEXT NOT NULL,        -- the actual ID from that source
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    UNIQUE(source, external_id)
);

CREATE TABLE entity_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    source_tool TEXT NOT NULL,        -- 'insider_filings', 'gdelt', etc.
    observed_at REAL NOT NULL,        -- when the observation occurred
    ingested_at REAL NOT NULL,        -- when we stored it
    observation_type TEXT NOT NULL,   -- 'filing', 'transaction', 'event', 'position'
    value_json TEXT NOT NULL,         -- the actual data point
    metadata_json TEXT
);

CREATE INDEX idx_entity_obs_lookup
    ON entity_observations(entity_id, source_tool, observed_at);
```

**Pros:** Simple, no new dependencies, fits existing SQLite pipeline store, can deploy incrementally per tool.
**Cons:** Fuzzy matching is manual, no ML-powered dedup, bottleneck at million+ entities.

### Option B: Probabilistic Record Linkage (splink / recordlinkage)

Use an open-source probabilistic matching library for cross-source entity resolution:

- **splink** (UK MoJ, BSD-3, Python, DuckDB backend) — Fellegi-Sunter model, fast, scales to millions. Most actively maintained. Can run on DuckDB (we could add DuckDB later).
- **recordlinkage** (BSD-3, Python, pandas-based) — Modular toolkit, good for <100K records. Blocking + comparison + classification pipeline. Last release 2023.
- **dedupe** (MIT, Python) — Active learning loop for building training sets. Good for name matching.

**Verdict:** splink is the best fit — BSD-3 license, Python, scales, active development, supports DuckDB. But overkill for Phase 10 where entity counts are low (<10K). Start with Option A, add splink when entity volume justifies it.

### Option C: Knowledge Graph (NetworkX / neo4j)

Full graph database for entity relationships.

**Verdict:** Premature. We need entity identity and time-series first, not graph traversal. Build the registry, then consider graph representation when L3 cross-entity patterns require it.

### Recommendation

**Phase 10a:** Option A — lightweight SQLite entity registry in `PipelineStore`.
**Phase 10b:** Upgrade tools one at a time to write entity_observations.
**Phase 11 (future):** Add splink for fuzzy matching when entity volume >10K.
**Phase 12 (future):** Graph representation when L3 patterns are formalized.
**Phase 13 (future):** L4 latent-structure layer — inferred motif/state objects computed from validated L3 patterns.

---

## Risks

1. **Scope creep** — 60 tools × 3 depth layers = 180 possible upgrades. Must prioritize ruthlessly.
2. **Entity name normalization** — Company names differ across sources (e.g., "Apple Inc." vs "APPLE INC" vs CIK 0000320193). Need deterministic normalization before probabilistic matching.
3. **Storage growth** — Entity observations accumulate fast. Need TTL/pruning strategy.
4. **API rate limits** — Deeper tool queries hit more endpoints. Must respect rate limits.
5. **Schema migration** — Adding entity tables to existing PipelineStore requires migration logic.
6. **Cross-source timing** — Different tools update at different frequencies (GDELT: 15min, CFTC: weekly, SEC: filing-driven). Time alignment for L3 patterns is non-trivial.

---

## Priority Ordering — Which Tools to Upgrade First

Prioritize by: (1) uniqueness of signal, (2) entity resolution feasibility, (3) existing L2 readiness, (4) cross-domain linking potential.

### Tier 1 — Highest Priority (unique + entity-ready + high L3 potential)

| Tool | Why First | L2 Work Required |
|------|-----------|------------------|
| `insider_filings` | Already L2, strongest free equity signal, cross-links to every company domain | Add per-insider historical baseline, cross-company board interlocks |
| `form144` | Complements insider_filings (sell-intent before execution), same CIK entity space | Link Form 144 → Form 4 execution tracking, 10b5-1 plan detection |
| `whale_alert` | Already L2, unique behavioral leakage, wallet clustering is tractable | Build wallet profiles, exchange deposit patterns, BTC→ETH bridge tracking |
| `ais_vessel` | L0 physics signal, can't be faked, strong L3 links (sanctions, commodities) | Dark vessel detection (AIS gaps), route fingerprinting, IMO→owner mapping |
| `gdelt` | 15-min update cycle, actor-pair resolution, geopolitical escalation curves | GKG enrichment, actor tone trajectories, event causality chains |

### Tier 2 — Medium Priority (good entity basis, moderate L3 links)

| Tool | Why | L2 Work Required |
|------|-----|------------------|
| `patent_filings` | Innovation pipeline velocity, inventor mobility | Citation networks, inventor tracking, international filing expansion |
| `bankruptcy_court` | Contagion detection, creditor network analysis | Creditor claim extraction, DIP lender tracking, cross-debtor linking |
| `sanctions_monitor` | Direct L3 link to vessels, wallets, companies | Cross-source entity matching (OFAC SDN name → MMSI/CIK/wallet) |
| `lobbying` | Strategic intent leakage, regulatory outcome prediction | Spend trajectory per registrant/client, sector-specific surge detection |
| `polymarket_whales` | Already L2, strong behavioral signal | Cross-market whale linking, accuracy decay monitoring |

### Tier 3 — Lower Priority (aggregate or limited entity dimension)

| Tool | Why Lower | Notes |
|------|-----------|-------|
| `finra_short_volume` | No individual trader data available publicly | Facility-level breakdown is the max depth possible |
| `cftc` | Structurally L1 — CFTC only publishes category aggregates | Options COT and basis analysis are the only upgrades |
| `central_bank_balance` | 7 institutions is the entire entity space | Dealer-level repo data and swap line usage are the only upgrades |
| `disease_surveillance` | Geographic entities, lower financial signal density | Variant-level metadata and animal surveillance are the upgrades |

---

## Data Requirements

### New Storage Needed

1. **Entity registry tables** (entities, entity_aliases, entity_observations) — in PipelineStore
2. **Entity name normalization** — deterministic: lowercase, strip suffixes (Inc., Corp., Ltd.), CIK→ticker mapping via SEC company tickers JSON
3. **Per-entity time-series** — via entity_observations table with `(entity_id, source_tool, observed_at)` index

### External Data for Entity Linking

| Mapping | Source | Cost | Format |
|---------|--------|------|--------|
| CIK → Ticker → Company Name | SEC company_tickers.json | Free | JSON (13K+ entities) |
| MMSI → IMO → Vessel Name → Owner | ITU MMSI registry + Equasis | Free (basic) | Web scrape or CSV |
| BTC address → Exchange wallet tags | blockchain.info, walletexplorer.com | Free | API |
| OFAC SDN → Names, Aliases, IDs | OFAC SDN XML | Free | XML (12K+ entities) |
| FIPS → Country Name → ISO code | CIA World Factbook / GDELT codebook | Free | CSV |

---

## Depth Evaluation Framework — How Deep Is Deep Enough?

**Core problem:** L3 might not be enough, or L2 might already be enough. We don't guess — we measure. Depth is an information theory question: *does going deeper actually reduce uncertainty about what we're trying to predict?*

### Metric 1: Conditional Mutual Information Gain

For each tool at each depth level, measure how much additional information the deeper data provides about target variables (regime state, belief posteriors, asset returns):

$$I(X_{Lk}; Y \mid X_{L1}, \ldots, X_{L(k-1)})$$

where $X_{Lk}$ is the data at depth $k$ and $Y$ is the prediction target.

- $I(X_{L1}; Y)$ = how much the aggregate data tells us about the target
- $I(X_{L2}; Y \mid X_{L1})$ = how much ADDITIONAL info entity-level data gives, given we already have the aggregate
- $I(X_{L3}; Y \mid X_{L1}, X_{L2})$ = marginal info gain from cross-entity combinations

**Decision rule:** If $I(X_{Lk}; Y \mid X_{<k}) < \epsilon$, the upgrade to depth $k$ is redundant for that tool. Stop at depth $k-1$.

**Estimation method:** Discrete MI via empirical histogram binning for categorical targets (regime states). For continuous targets, use KSG estimator (Kraskov-Stögbauer-Grassberger, 2004) — already the standard for noisy, finite-sample MI estimation.

- **Trusted source:** Kraskov, Stögbauer, Grassberger (2004). "Estimating Mutual Information." Physical Review E 69(6). The KSG estimator uses k-nearest-neighbor distances to avoid binning artifacts.
- **OSS implementation:** `sklearn.feature_selection.mutual_info_classif` / `mutual_info_regression` (uses KSG internally). Also `dit` library (BSD) for multivariate info-theoretic measures.

### Metric 2: Belief Update Magnitude (KL Divergence)

We already have the world model (Phase 9). Use it as the measurement instrument:

1. Record the world model's posterior before injecting tool evidence: $P_{prior}$
2. Inject L1 aggregate evidence → record posterior: $P_{L1}$
3. Inject L2 entity-level evidence → record posterior: $P_{L2}$
4. Compute KL divergence at each step:
   - $D_{KL}(P_{L1} \| P_{prior})$ = how much L1 moves beliefs
   - $D_{KL}(P_{L2} \| P_{L1})$ = how much L2 ADDITIONALLY moves beliefs

**Decision rule:** If $D_{KL}(P_{L2} \| P_{L1}) < \delta$, then L2 barely shifts beliefs beyond what L1 already provided. Going deeper is not worth it for that tool.

**Advantage:** This metric uses the world model as a natural compression of "what matters." If the Bayesian network posteriors don't change, the deeper data isn't adding predictive signal — regardless of how interesting it looks on paper.

### Metric 3: Walk-Forward Sharpe Delta

The hardest test — does adding the deeper signal actually improve out-of-sample risk-adjusted returns?

1. Run walk-forward backtest with L1 features only → Sharpe $S_1$
2. Run same backtest adding L2 features → Sharpe $S_2$
3. $\Delta S = S_2 - S_1$

**Decision rule:** If $\Delta S$ is not statistically significant (bootstrap confidence interval includes 0), the depth upgrade has no economic value.

**When to use:** Only after the full pipeline (features → world model → fusion → policy) is operational. This is the final gate, not the first check.

### The Depth Evaluation Loop

We don't set depth targets upfront. We build a **measurement loop** into the pipeline:

```
For each tool:
  1. Measure MI of current depth against target variables
  2. Upgrade tool one depth level
  3. Measure conditional MI gain from the upgrade
  4. Measure belief update magnitude (KL divergence) from the upgrade
  5. If gain > threshold → keep the upgrade, try going deeper
  6. If gain < threshold → stop, this depth is sufficient for now
  7. Log the measurement to entity_observations for trend tracking
```

### Why This Changes the Implementation Order

The depth evaluation framework must be built BEFORE upgrading tools — otherwise we're flying blind, spending engineering effort on upgrades we can't test. The revised implementation priority:

1. **First:** Build depth measurement infrastructure (MI computation, KL divergence measurement, entity observation logging with metadata sufficient for MI estimation)
2. **Then:** Upgrade Tier 1 tools with measurement on each upgrade
3. **Continuously:** Let the measurements tell us which tools deserve further depth and which can stop

### Practical Requirements for Measurement

Each `entity_observation` must store enough metadata to compute MI:

- **Timestamp** (observed_at) — for temporal alignment with targets
- **Source tool + depth level** — which tool and at which depth this observation came from
- **Observation value** — the actual data point (quantized for MI computation)
- **Target variable snapshot** — what the world model posteriors or asset returns were at the observation time

This means the entity_observations schema needs a `depth_level` column and the pipeline needs to record target-variable snapshots alongside observations.

```sql
-- Addition to entity_observations table
ALTER TABLE entity_observations ADD COLUMN depth_level INTEGER NOT NULL DEFAULT 1;

-- New table for depth evaluation metrics
CREATE TABLE depth_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    depth_level INTEGER NOT NULL,
    evaluated_at REAL NOT NULL,
    target_variable TEXT NOT NULL,      -- which belief/return we measured against
    mi_gain REAL,                       -- conditional MI gain from this depth
    kl_divergence REAL,                 -- belief update magnitude
    sharpe_delta REAL,                  -- walk-forward improvement (null until backtest available)
    sample_size INTEGER NOT NULL,       -- how many observations the estimate is based on
    metadata_json TEXT
);
```

---

## Math/Algorithm Survey

### Entity Resolution Methods (for when fuzzy matching is needed)

| Method | Library | Complexity | When to Use |
|--------|---------|-----------|-------------|
| **Exact match** on canonical ID (CIK, MMSI, ticker) | Built-in | O(1) lookup | Default for structured ID matching |
| **Deterministic rules** (normalize + exact) | Custom | O(n) | Company name normalization |
| **Fellegi-Sunter probabilistic model** | splink (BSD-3) | O(n log n) with blocking | When names are fuzzy across sources |
| **Active learning** | dedupe (MIT) | O(n log n) | When we need human-in-loop training |
| **Embedding similarity** | entity-embed (BSD-3) + sentence-transformers | O(n) encode + O(k) ANN | When entity descriptions are rich text |

**Recommendation:** Start with exact ID matching (CIK, MMSI, ticker, wallet address) — covers >80% of cross-source links without any fuzzy logic. Add splink only when we encounter entity types that lack deterministic IDs.

### Anomaly Detection Per Entity

Once entities have time-series, we can detect behavioral regime changes. Methods:

| Method | What It Detects | Already in Repo? |
|--------|----------------|-----------------|
| **BOCPD** (Bayesian Online Changepoint Detection) | When an entity's behavior regime shifts | ✅ `agent/quant/changepoint.py` |
| **Z-score against entity baseline** | When current observation deviates from entity's own history | ❌ Need to implement |
| **Hawkes process** | Self-exciting event clustering (e.g., insider filing bursts) | ❌ Need to implement |
| **Transfer entropy** | Information flow between entities (who leads whom?) | ❌ Need to implement |

---

## OSS References

| Library | License | Notes |
|---------|---------|-------|
| [splink](https://github.com/moj-analytical-services/splink) | BSD-3 | Probabilistic record linkage, DuckDB/Spark backend. Best for scaled fuzzy matching. |
| [recordlinkage](https://github.com/J535D165/recordlinkage) | BSD-3 | Modular Python toolkit. Good for <100K records. Academic quality. |
| [dedupe](https://github.com/dedupeio/dedupe) | MIT | Active learning for entity dedup. Good for name matching. |
| [entity-embed](https://github.com/vintasoftware/entity-embed) | BSD-3 | PyTorch entity embeddings + ANN. Good for text-rich entities. |
| [linktransformer](https://github.com/dell-research-harvard/linktransformer) | BSD-3 | Transformer-based record linkage. Harvard. |

All are commercially safe (BSD-3 or MIT). Zingg (AGPL-3) is excluded — concept source only.

---

## Implementation Strategy

### Phase 10a: Depth Evaluation Framework + Entity Registry (foundation)
1. Add entity tables to PipelineStore schema (entities, entity_aliases, entity_observations with depth_level)
2. Add depth_evaluations table to PipelineStore
3. Build entity name normalization utilities
4. Load SEC company_tickers.json as seed entities
5. Implement MI computation module (KSG estimator for continuous, histogram for discrete)
6. Implement belief update magnitude measurement (KL divergence from world model posteriors)
7. Wire `insider_filings` to write entity_observations on each scan — first tool, test the full measurement loop

### Phase 10b: Tier 1 Tool Upgrades (5 tools, each measured)
For each tool: upgrade → measure conditional MI gain → keep/revert based on threshold.

8. `insider_filings` → per-insider time-series + historical baseline → measure MI gain
9. `form144` → Form 144→Form 4 execution linking → measure MI gain
10. `whale_alert` → wallet profile builder + exchange deposit detection → measure MI gain
11. `ais_vessel` → dark vessel detection + route fingerprinting → measure MI gain
12. `gdelt` → GKG enrichment + actor escalation curves → measure MI gain

### Phase 10c: Tier 2 Tool Upgrades (5 tools, each measured)
13. `patent_filings` → citation networks + inventor mobility
14. `bankruptcy_court` → creditor extraction + contagion linking
15. `sanctions_monitor` → cross-source entity matching
16. `lobbying` → spend trajectory + sector surge detection
17. `polymarket_whales` → cross-market whale linking

### Phase 10d: Cross-Domain Combinators (L3)
18. Build L3 pattern detectors using entity_observations from multiple tools
19. Measure conditional MI of L3 combinations vs L2-only signals
20. Wire validated L3 patterns into world model as new evidence nodes

---

## Phase 10b.1: insider_filings L2 Upgrade — Detailed Analysis

### Current Code Architecture

`InsiderFilingsTool(Tool)` in `agent/tools/insider_filings.py` (~550 lines):
- **Constructor:** `__init__(self, cache: DataCache | None = None)` — optional cache, no pipeline store
- **execute():** Fetches EFTS search → parses XML → detects clusters → returns ToolResult
- **_parse_filings():** Extracts CIKs/names from EFTS hits, fetches Form 4 XML, calls `_parse_form4_xml()`
- **_parse_form4_xml():** Extracts open-market purchases (`transactionCode == "P"`) with ticker, company, name, role, shares, price, date
- **_detect_clusters():** Groups transactions by ticker, calls `_find_best_cluster()`
- **_find_best_cluster():** Sliding 14-day window, 3+ distinct insiders = cluster. Dedup by `name.upper()`

### Key Data Available But Unused

EFTS response `_source` fields for each filing:
- `ciks`: `["<reporter_cik>", "<issuer_cik>"]` — **reporter CIK is extracted in `_parse_filings()` line ~237 but never stored in the transaction dict**
- `display_names`: `["<reporter_name> (CIK <reporter_cik>)", "<company_name> (<ticker>) (CIK <issuer_cik>)"]`
- `sics`: SIC codes for the company
- `biz_locations`: geographic info

The reporter CIK is the strongest entity identifier available — unique per insider, persistent across filings, directly usable as entity resolution key. Currently discarded.

### L2 Upgrade Design

**Minimal changes principle:** The tool must continue to work standalone (no PipelineStore required). Entity wiring is additive only — existing behavior preserved when `pipeline_store=None`.

**Change 1: Add reporter_cik to transaction dicts**
In `_parse_filings()`, add `reporter_cik = ciks[0]` to the transaction dict passed to `_parse_form4_xml()`. This flows through to clusters without changing the cluster detection logic.

**Change 2: Accept optional PipelineStore**
Add `pipeline_store: PipelineStore | None = None` to `__init__()`. Store as `self._store`.

**Change 3: Register entities + store observations post-parse**
After `_parse_filings()` returns transactions, add a new private method `_persist_entities(transactions)` that:
1. Deduplicates by (issuer_cik, reporter_cik) to avoid redundant registrations
2. For each unique company: `register_entity(type="company", key=issuer_cik)` + aliases (sec_cik, ticker)
3. For each unique insider: `register_entity(type="person", key=reporter_cik)` + alias (sec_cik)
4. For each transaction: `store_entity_observation(entity_id=insider_eid, source_tool="insider_filings", depth_level=2, observation_type="purchase", value={...})`
5. Wrapped in try/except so persistence failures don't break the tool's primary result

**Change 4: CIK-based dedup in clustering**
In `_find_best_cluster()`, prefer `reporter_cik` for dedup when present, fall back to `name.upper()` for backward compatibility.

**Change 5: Enrich ToolResult.data with entity context**
Add `entity_ids` mapping to cluster data: `{insider_name: entity_id}` so downstream consumers can link to the entity graph.

### What NOT to Change (Deferred)

- Historical baseline queries (needs accumulated observations first)
- Conviction score enhancement using history (same reason)
- Cross-domain linking to form144 (that's Phase 10b.2)
- MI measurement integration into the tool itself (measurement is external via depth_eval module)

### Risk Analysis

1. **Performance:** Entity registration adds SQLite writes per transaction. Cost: ~1ms per INSERT OR IGNORE. For typical scan (200-500 filings), adds <1 second. Acceptable.
2. **Error isolation:** Entity persistence failures must not prevent ToolResult from returning. Wrap in try/except with logging.
3. **Schema compatibility:** PipelineStore already has entity tables from Phase 10a. No migration needed.
4. **Testing:** Must test both paths: with PipelineStore (entities registered) and without (existing behavior preserved).
5. **reporter_cik reliability:** Some EFTS entries have malformed CIK arrays. `_parse_filings()` already skips entries with `< 2` CIKs. The reporter_cik will be None only in edge cases; fall back to name-based dedup.

### EFTS Data Confirmed

Fetched live EFTS response — confirmed fields available per hit:
- `ciks`: `["reporter_cik", "issuer_cik"]` — both always present as 10-digit zero-padded strings
- `display_names`: parallel array with names + CIKs
- `adsh`: accession number for XML fetch

---

## Related

- [[project_memory]]
- [[world_model]]
- [[world_model_spec]]
- [[signal_protocol_feature_engineering]]
- [[signal_protocol_feature_engineering_spec]]
- [[deep_surveillance_tools_10b_spec]]
- [[deep_surveillance_10b]]
- [[7b-V_ucc_creditor_filings]]

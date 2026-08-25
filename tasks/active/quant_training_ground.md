---
title: "Task: quant_training_ground"
tags:
  - doc/task
  - layer/feature-engineering
  - layer/surveillance
  - layer/world-model
  - phase/25
  - status/active
  - topic/quant
---

# Task: quant_training_ground

Status: active
Latest completed phase: Phase 46 — Living System: Online GNN + EWC continuous learning COMPLETE (2026-04-23)
Next immediate work: Phase 47 — Historical Backfill Runner (backfill ALL 51 tools for 2–5 years of history, then run Phase 40)
Next after that: Phase 47 — Historical Backfill Runner (backfill ALL 51 tools for 2–5 years of history, then train)
Next gated phase: Phase 40 — Real Data Model Refresh (UNGATED after Phase 47 completes — no longer waiting for live accumulation)

**Revised sequence (Phase 47 unlocks Phase 40 early):**
1. Phase 46 — EWC online learning layer (CPU, $0, <1 week)
2. Phase 47 — Historical backfill all 51 tools → years of observations in DB → density audit → patch sparse entities → run Phase 40 immediately after
3. Phase 40 — Real GNN retrain on years of real history + density report (which entities are sparse, which tools need extended backfill)
4. Phase 48 — Transformer World Model + Model-Based RL — **GATED: do not start until density audit passes (≥500 observations per entity type average, no entity type below 100)**

**Density mandate (standing rule before Phase 48):** Transformers are data-hungry. Sparse input = broken attention = garbage predictions. Every phase from 47 onward must track observation density per entity type. Any entity type below threshold gets extended backfill, synthetic augmentation, or additional tool wiring before Phase 48 begins. Density is a first-class exit condition, not an afterthought.
Active task: [[quant_training_ground]]
Latest checkpoint: [[chat_checkpoint_2026-04-22_phase45_3_complete]]
Prior phases: [[agent_autonomy]], [[scoring_validation]], [[liquidity_regime_detection]], [[observational_surface]], [[convergence_detection]], [[signal_protocol_feature_engineering]], [[world_model_bridge]], [[signal_fusion]], [[rl_policy]], [[adversarial]], [[gnn_guided_expansion_r2]], [[e2e_global_integration]]

## Overview

Build an **advanced self-improving agent** whose core asset is a **cross-domain entity embedding space** (ℰ). **Markets are playground #1** — the first harsh environment that scores the agent. Two tracks: **Agent Intelligence** (ℰ, memory, world model, alignment, RL) and **Playground infrastructure** (global sensors, `agent/quant/` math as constraints/readouts, walk-forward scoring). **ML-first:** discovery lives in representation learning, not hand-built factor shops. Canonical doctrine: [[agent_playground_doctrine]].

**Product direction:** TirraMind is not only a trading system. It is a predictive intelligence platform whose outputs can serve traders, quant teams, enterprises, operators, and strategic decision-makers. Trading remains an important downstream application, but the core product is the prediction engine itself: forecasts, anomaly alerts, regime changes, entity-risk shifts, cross-domain link signals, and decision-support surfaces.

**Product hierarchy:** the base asset is the intelligence engine. That comes first. On top of it we build custom intelligence layers for specific users and tasks: customer-specific APIs, alerting workflows, dashboards, scoring frameworks, and service packages. The moat is not the packaging; it is the underlying predictive engine.

**Commercial doctrine:** sell the outcome through a tool-shaped product. The preferred SKU is not model seats or generic AI access. It is recurring delivery of decision advantage: monitored signals, probability updates, anomaly detection, regime-change alerts, and customer-specific predictive workflows. The customer can experience that through software surfaces such as dashboards, APIs, workbenches, and alerts, but those surfaces should package the outcome rather than expose interchangeable model access. Models, prompts, agents, and internal tooling should be replaced whenever a better stack improves quality or lowers cost.

**Margin logic:** better models should widen margin, not threaten the business. If TirraMind is priced around outcome value and embedded in customer workflows, model improvements reduce delivery cost and improve reliability while customer willingness-to-pay remains tied to the value of being right early. The software layer matters because it embeds workflow, historical context, telemetry, and organizational habit, which makes the product harder to swap out than a raw model endpoint.

**Commercial niche:** **N1 + N4 + microstructure** (2026-06-02). Spec: [[n1_n4_playground_spec]]. Task: [[n1_n4_microstructure_playground_task]]. Quant-desk math bar — no sentiment SKU. Buyers: macro retail, indie quant per [[revenue_plan_2026-05-08]].

**Decision filter:** Does this improve ℰ / the agent loop, or only a one-off playground hack? See [[agent_playground_doctrine]].

**Worldview:** Markets are outputs. Reality is the input. TirraMind operates at Layer 0 (physical: shipping, weather, factories) and Layer 1 (behavioral: policy, trades, production) to predict Layers 2-3 (information and prices). No country excluded. No data source irrelevant until proven so.

**Firm identity:** TirraMind is an **advanced agent / predictive AI company** — not a quant fund. Mission: learn ℰ over heterogeneous reality; markets score it first. Edge: unconventional observation × ML on a graph × living system (HetTGN, belief propagation, Kalman, RL, EWC, alignment). Quant math in `agent/quant/` is **instrumentation**, not the primary discovery engine. Full doctrine: [[agent_playground_doctrine]].

**POMDP doctrine:** The global system is a Partially Observable Markov Decision Process. States are partially hidden, the environment is non-stationary, actors have latent intentions, and rewards are sparse and delayed. The full stack is designed around this: GNN perceives the partially observable state → world model represents hidden-state uncertainty → Kalman fusion integrates noisy evidence → RL policy acts under uncertainty. Use RL and world model components exactly where this sequential, uncertain, partially observable structure demands it. The current RL layer (SAC, model-free) learns from experience. The Phase 48 target (Dreamer, model-based) plans by imagining trajectories through the world model — the natural solver for a POMDP at scale.

**Calibration:** Best predictive AI company in the world. The benchmark is not a quant fund — it is: does this advance the frontier of what a machine can know about reality before humans do? That is the only bar that matters.

**Model agnosticism (standing rule):** No model is sacred. The current stack is the best-justified choice for the current data volume and scale. After Phase 40 (first real GNN retrain on live history), evaluate every layer by one metric: does it produce genuine out-of-sample predictive edge? If a component is weak, replace or upgrade it — no sentiment, no sunk cost. Upgrade triggers already defined: world model DAG → PyMC variational at >500 nodes; SAC hidden_dim 128→256 when replay saturates; GNN sparse attention at entity count >500K. Until those triggers fire, hold the current architecture and focus on data quality.

**Execution rule for speed:** data first, schemas now, abstractions after coverage. Continue expanding high-value surveillance tools first; enforce stable machine-readable outputs while doing so; defer Bloomberg-like context layers, dashboards, and other commodity abstractions until the evidence surface is broad enough to justify them.

## Phases

- [x] Phase 0: Agent end-to-end
- [x] **Phase 1: Data Foundation** ✅ COMPLETE
- [x] **Phase 2: Global Liquidity Regime Detection** ✅ COMPLETE
- [x] **Phase 3: Scoring & Validation** ✅ COMPLETE
- [x] **Phase 4: Agent Autonomy** ✅ COMPLETE
- [x] **Phase 4b: RL Layer** ✅ COMPLETE
- [x] **Phase 5: Full Observational Surface** ✅ COMPLETE
- [x] **Phase 6: Extended Observational Surface** ✅ COMPLETE (6h folded into 7b-M)
- [x] **Phase 7: Pipeline Layer** ✅ COMPLETE (356/356 tests, 9 steps, deterministic DAG scheduler — no LLM, scheduled triggers)
- [x] **Phase 7b: Global Deep Surveillance** ✅ COMPLETE
- [x] **Phase 7c: Convergence Detection Layer** ✅ COMPLETE (883 tests — BOCPD, HMM, spectral, convergence scoring)
- [x] **Phase 8: Signal Protocol + Feature Engineering** ✅ COMPLETE (294 tests — EngineeredFeature protocol, FeatureBuilder ABC, 2 builders)
- [x] **Phase 10a: Deep Surveillance Framework** ✅ COMPLETE (entity tables, MI computation, depth eval)
- [x] **Phase 10b: L2 Tool Upgrades** ✅ COMPLETE (insider_filings, form144, whale_alert L2)
- [x] **Phase 12: Temporal Heterogeneous GNN** ✅ COMPLETE (HetTGN architecture, trainer, pattern extractor — 242 tests)
- [x] **Phase 13: L2 Tool Expansion** ✅ COMPLETE (12 L2 tools, graph builder expansion — 147 tests)
- [x] **Phase 14–15: Pattern Recovery + Fine-Tuning** ✅ COMPLETE (self-supervised + supervised pipeline, diagnostics)
- [x] **Phase 16: GNN-Guided Tool Expansion** ✅ COMPLETE (run_diagnostics, Tier 1/2/3 ranking — 34 tests)
- [x] **Phase 17: Entity Linking Layer** ✅ COMPLETE (8 link types, 95 entity linking tests, GNN edges live)
- [x] **Phase 18: Tier 1 Tool Expansion** ✅ COMPLETE (sanctions_monitor L2, gov_contracts L2, supply_chain_monitor L2 — per [[tool_priority_ranking]])
- [x] **Phase 19: GNN ↔ World Model Bridge** ✅ COMPLETE (GNN features, expanded DAG, inference DAG, belief-driven backtest)
- [x] **Phase 20: Signal Fusion** ✅ COMPLETE (entity micro-alpha via GNN prediction surprise — 249 tests)
- [x] **Phase 21: RL Policy** ✅ COMPLETE (surprise-driven portfolio allocation, SAC actor-critic — 276 tests)
- [x] **Phase 22: Adversarial** ✅ COMPLETE — 148 tests (manipulation detection, edge decay)
- [x] **Phase 23: GNN-Guided Expansion R2** ✅ COMPLETE — 3 L2 upgrades (finra_short_volume, creditor_filings, drug_regulatory), 21 obs types, dim was 30 pre-Phase-28 (now 41), new link types (debtor_of, market_authorized_in) — 174 tests
- [x] **Phase 24: End-to-End Global Multi-Asset Integration** ✅ COMPLETE — instruments as GNN entities (90), multi-asset strategy (SAC + equal-weight + buy-hold), inference DAG (4 nodes), walk-forward backtest + attribution, paper trade with alerts (drawdown/concentration/Sharpe/edge-decay) — 223 tests. Task: [[e2e_global_integration]]
- [x] **Phase 25: Cross-Domain Entity Linking** ✅ COMPLETE — instrument nodes connected to company/country/CFTC/topic/wallet structure; instruments no longer graph-isolated. Task: [[phase25_cross_domain_entity_linking]]
- [x] **Phase 27: FX Country Wiring + Central Bank L2** ✅ COMPLETE — FX two-country metadata/links + `central_bank_balance` L2 (monetary_balance, policy_rate on country nodes). Task: [[phase27_fx_country_monetary_linking]]
- [x] **Phase 28: Country Node Enrichment (Macro)** ✅ COMPLETE — `sovereign_debt` L2 (sovereign_yield) + `capital_flows` L2 (capital_flow) + `global_pmi` L2 (economic_activity) on country nodes. Country goes from 3→6 obs types.
- [x] **Phase 29: Company + Investigative L2** ✅ COMPLETE — `bankruptcy_court` L2 (bankruptcy_status) + `foia_requests` L2 (investigation_signal) + `academic_preprints` L2 (research_velocity) on company/topic nodes.
- [x] **Phase 30: Crypto Islands + Cross-Domain Linking** ✅ COMPLETE — BTC/ETH → protocol links + wallet → instrument links from `whale_alert`. Crypto instruments no longer graph-isolated.
- [x] **Phase 31: Remaining Country Signals** ✅ COMPLETE — `consumer_sentiment` L2 + `food_security` L2 + `internet_outages` L2 + `migration_flows` L2 on country nodes. Country reaches 10 obs types.
- [x] **Phase 32: Trade + Disease + Political L2** ✅ COMPLETE — `comtrade` L2 (bilateral trade_flow) + `transport_throughput` L2 (border_throughput) + `disease_surveillance` L2 (pathogen_level) + `political_risk` L2 (campaign_finance).
- [x] **Phase 33: Organization + Grid Enrichment** ✅ COMPLETE — `regulatory_gazette` L2 (regulatory_velocity on organization) + `electricity_monitor` L2 (grid_demand on region). Organization entities gain first real observations.
- [x] **Phase 46: Living System — Online GNN (EWC Continuous Learning)** ✅ COMPLETE (2026-04-23) — `agent/models/gnn/ewc.py` (EWCState, compute_fisher, ewc_penalty), `TrainerConfig` extended (ewc_lambda=1000.0, online_batch_threshold=100), Fisher diagonal computed after train(), save/load backward-compatible, `_loss_from_window` + `online_update` added to Trainer, wired into `gnn_inference.py` DAG. 65 regression tests pass + 13 new EWC tests. $0 compute (CPU). Research: [[living_system_online_gnn]]. Spec: [[living_system_online_gnn_spec]].
- [x] **Phase 47: Historical Backfill Runner** ✅ COMPLETE (2026-04-22) — `scripts/backfill.py` (68-entry BACKFILL_PLAN, BackfillCheckpoint, Group A/B/C, dry-run, retry, 429 handling, DB-lock retry), `scripts/density_audit.py` (per entity_type/source_tool report, Shannon entropy, SPARSE flagging, exit gate), `_backfill=True` bypass added to 6 capped tools (earthquake_proximity, disease_surveillance, insider_filings, form144, sanctions_monitor, internet_infrastructure). 34 new tests (12 bypass + 12 runner + 10 audit) — all pass. Run `python scripts/backfill.py --dry-run` to preview plan. Run `python scripts/density_audit.py` to check Phase 40 readiness (exit 0 = ready). Research: [[historical_backfill]]. Spec: [[historical_backfill_spec]].
- [x] **Phase 49: GNN Downstream Alignment** ✅ COMPLETE (2026-04-24) — `agent/models/gnn/alignment.py` (compute_belief_log_likelihood_delta, store_entity_alignment, load_alignment_weights), 15 tests pass. Research: [[gnn_downstream_alignment]].

- [x] **Phase 49b: Convergence Detection as Control Signal** ✅ COMPLETE (2026-04-24) — `agent/pipeline/regime_gate.py` (get_current_regime, is_high_changepoint, world_model_prior_decay, feature_trust_scale, sac_entropy_scale), 12 tests pass. Wired into `gnn_inference.py` DAG (force retrain when is_high_changepoint()). Research: [[convergence_as_control]].

- [ ] **Phase 48: Transformer World Model + Model-Based RL** — replace the pgmpy Bayesian DAG world model with a transformer over entity-observation sequences (attention learns causal structure from data rather than hand-coded edges). Replace SAC MLP policy with a Dreamer-style model-based RL agent that plans over imagined rollouts of the transformer world model. This is the target production architecture. **Hard prerequisites before a single line of Phase 48 code is written: (1) density audit passes — ≥500 obs/entity type average, no type below 100; (2) Phase 40 walk-forward complete — failure modes identified per layer; (3) Phase 40 shows the current stack has hit its ceiling on at least one layer.** If density fails, extend Phase 47 backfill window (try days_back=3650 for 10 years on confirmed tools) or wire new tools before proceeding. Research: [[transformer_world_model]]. Spec: [[transformer_world_model_spec]].
- [x] **Phase 37: First Live Pipeline Run** ✅ COMPLETE — DAG scheduler runs daily_collection → convergence_detection → feature_generation. Discovered downstream integration gap (0 evidence, all-None features). Task: [[phase37_first_live_pipeline]]
- [x] **Phase 38: Downstream Pipeline Integration** ✅ COMPLETE — Fixed source name mismatch (table_name on DAG nodes), added fetch_macro node, 59 tests pass. Task: [[phase38_downstream_pipeline_integration]]
- [x] **Phase 34: Commodity Country Links + Diagnostic Sweep** ✅ COMPLETE — `primary_exchange_country` field + `exchange_country` link type + `graph_diagnostics.py` utility. No instrument class has zero entity links.
- [x] **Phase 35: GNN Retrain on Expanded Entity Graph** ✅ COMPLETE — SyntheticGraphGenerator expanded to full 11-type/45-obs/18-link schema. 6 cross-domain injected patterns. HetTGN retrained: 32.9% top-1 accuracy (15x random), 80.1% top-5. Attention analysis reveals starved edges (market_authorized_in, sanctioned_under, exchange_country, located_in, exchange_based_in) and disconnected entity types (domain, topic have 0 degree). 60 tests (27 existing + 33 Phase 35). Research: [[phase35_gnn_retrain_expanded_graph]]. Spec: [[phase35_gnn_retrain_expanded_graph_spec]].
- [x] **Phase 36: Connect Disconnected Entity Types** ✅ COMPLETE — domain→company (domain_owned_by) + topic→instrument (topic_relates_to_instrument) links. 33 Phase 36 tests. Domain and topic entities no longer graph-isolated. Research: [[phase36_connect_disconnected_entities]]. Spec: [[phase36_connect_disconnected_entities_spec]].

Full roadmap: [[l2_expansion_roadmap]]. Starved class audit: [[starved_class_audit]].

**Architecture decision (2026-03-24):** TirraMind runs two execution engines:
- **Agent Layer** (LLM-driven): orchestrator, exploration, research, hypothesis generation
- **Pipeline Layer** (deterministic): DAG scheduler, fetch→feature→model→signal, no LLM, scheduled triggers
- **Shared State**: world model (numpy arrays, not JSONL text)
Data tools (Phase 6) serve both layers. Pipeline Layer (Phase 7) after all data tools exist. Phases 8-12 run on Pipeline.

Only the current phase is decomposed. Later phases get broken down when they're next.

## Phase 6: Extended Observational Surface — Atomic Steps

**Goal:** Security hardening + schema validation, then expand surveillance with 7 more data tools. All tools dual-interface (agent tool + pipeline-ready).

### Sub-phase 6-infra: Security + Validation ✅ COMPLETE

- [x] 6-infra-A: Sandbox execution — `_safe_env()`, env filtering, timeout config
- [x] 6-infra-B: JSON schema validation on tool-call arguments
- [x] 6-infra-C: Verification — 7/7 tests passed (env isolation, timeout, cwd, schema validation, config)

### Sub-phase 6a: GDELTTool (Geopolitical Event Surveillance) ✅ COMPLETE

- [x] 6a.1: Research — `[[gdelt]]` (raw files, DOC API, schema, CAMEO codes)
- [x] 6a.2: Spec — `[[gdelt_spec]]` (8 atomic steps)
- [x] 6a.3: Implement `agent/tools/gdelt.py` — full tool: events mode + articles mode
- [x] 6a.4: Register in `agent/cli.py`, add `geopolitical_intelligence` bandit arm
- [x] 6a.5: Live test — 4 batches, 4956 events, all filters verified

### Sub-phase 6b: CFTC Commitments of Traders ✅ COMPLETE

- [x] 6b.1: Research — `[[cftc]]` (URLs, column layout, signal survey)
- [x] 6b.2: Spec — `[[cftc_spec]]` (9 atomic steps)
- [x] 6b.3: Implement `agent/tools/cftc.py` — full tool: latest + historical modes, 191-col parser, signal computation
- [x] 6b.4: Register in `agent/cli.py`, add `futures_positioning` bandit arm
- [x] 6b.5: Edge case tests — 65/65 passed (live fetch, filters, signals, malformed data, dot handling)
### Sub-phase 6c: Whale Alert (Crypto Whale Transfers) ✅ COMPLETE

- [x] 6c.1: Research — `[[whale_alert]]` (tested free sources, Whale Alert is $30+/month — rejected)
- [x] 6c.2: Spec — `[[whale_alert_spec]]`
- [x] 6c.3: Implement `agent/tools/whale_alert.py` — two free modes (mempool + confirmed block), both via blockchain.com, $0, no API key
- [x] 6c.4: Register in `agent/cli.py`, add `crypto_whale_flows` bandit arm (no config key needed)
- [x] 6c.5: Live tested both modes — mempool (291 BTC whale tx), confirmed block #942014 (1548 BTC largest, 82 whales ≥10 BTC)
- [x] 6c.6: Edge case tests — 36/36 passed (mode validation, mempool+confirmed normal/empty/malformed/HTTP error, min_btc filter, limit, caching, sorting, satoshi boundary, multi-output sum, formatting, 2 live tests, bandit/config integration)

### Sub-phase 6d: SEC Form 144 (Insider Sell Intent Detection) ✅ COMPLETE

- [x] 6d.1: Research — `[[form144]]` (EFTS API, XML schema, signal theory, acquisition taxonomy, urgency signals)
- [x] 6d.2: Spec — `[[form144_spec]]` (8 atomic steps)
- [x] 6d.3: Implement `agent/tools/form144.py` — full tool: EFTS fetch (pagination, 429 retry, 500 graceful degradation), two-phase XML parse (metadata grouping → selective fetch for cluster candidates), acquisition classifier (open_market/private_placement/vesting/gift/other), urgency classifier (immediate/near_term/planned/unknown), sell-intent cluster detection (14-day window, distinct insiders, conviction scoring)
- [x] 6d.4: Register in `agent/cli.py`, extend `insider_flow` bandit arm (added form144 tool, updated description/examples)
- [x] 6d.5: Live tested — 47 sell-intent clusters in 5 days (NVDA $315M 3 insiders, ALM $169M 3.5% of outstanding, DELL $96M 6 insiders/Silver Lake, RDW $70M 3.9% outstanding, COP $66M, ROST $16M, CRWD $27M). 325 EFTS hits, 186 tickers, 46 cluster candidates.
- [x] 6d.6: Edge case tests — 53/53 passed, 4 skipped (2 hmmlearn dep, 2 live network). 57 total tests across 11 classes: input validation, EFTS fetch, XML parser, acquisition classifier, date parsing, urgency, helpers, cluster detection, full pipeline, integration, live network.

### Sub-phase 6e: FINRA Short Volume & Short Interest ✅ COMPLETE

- [x] 6e.1: Research — `[[finra_short_data]]` (probed all FINRA endpoints: ATS weekly stale 2023 only, TRACE 401 auth-gated, Reg SHO daily + Short Interest both free/live)
- [x] 6e.2: Spec — `[[finra_short_volume_spec]]` (8 atomic steps)
- [x] 6e.3: Implement `agent/tools/finra_short_volume.py` — full tool: Reg SHO daily short volume (single ticker trend + all-ticker scan) and consolidated short interest. Signal computation: z-score anomaly detection, trend (rising/falling/flat), squeeze risk (DTC>5), building/covering flags. 3-facility aggregation, pagination (6×5000), fractional par float handling.
- [x] 6e.4: Register in `agent/cli.py` (21 tools total), add `institutional_flow` bandit arm (tools: finra_short_volume, market_data, cftc)
- [x] 6e.5: Live tested — AAPL short_volume 3 days (43.5% ratio, rising trend), scan 2026-03-24 (26,300 records, 2,289 tickers >500k vol), AAPL short_interest (SI=113.6M, DTC=2.6)
- [x] 6e.6: Edge case tests — 95/95 passed, 2 skipped (hmmlearn dep). 97 total tests across 15 classes: mode/param validation, tool metadata, ticker mode (single/multi-day/normalization), scan (basic/filter/limit/pagination), short interest (signals/squeeze/building/covering/multi-period), aggregation (multi-facility/zero-vol/fractional), signal computation (trend/zscore/anomaly/zero-stdev), date helpers, _safe_float, API errors (204/400/429/500/timeout/non-JSON/non-list), cache interaction (hit/miss/empty/API signature), output formatting, CLI registration, bandit arm.

### Sub-phase 6f: ADS-B Jet Tracking — SKIPPED

- [x] 6f.1: Research — `[[adsb_jet_tracking]]` (OpenSky Network probed: 3-5% coverage globally, NYC area 4 aircraft all commercial, historical flights 403, Teterboro 2 flights/2h)
- **Decision: SKIP** — Insufficient coverage for anomaly detection. Cannot construct baseline with <5% visibility. No historical lookback API. Documented in research doc.

### Sub-phase 6g: Power Grid Demand (NYISO) ✅ COMPLETE

- [x] 6g.1: Research — `[[power_grid]]` (probed 7 ISOs: NYISO free CSV, CAISO XML/ZIP, EIA needs key, PJM/ERCOT/MISO/ISO-NE blocked)
- [x] 6g.2: Spec — `[[power_grid_spec]]` (8 atomic steps, 4 modes)
- [x] 6g.3: Implement `agent/tools/power_grid.py` — full tool: 4 modes (demand/fuel_mix/pricing/forecast), NYISO MIS CSV fetch with monthly ZIP archive fallback, zone filtering + normalization, signal computation (peak/trough/avg demand, fuel proportions, DA-RT LBMP spread, forecast deviation %). LBMP `_zone` file suffix, columnar forecast parsing, hour truncation for forecast-actual matching.
- [x] 6g.4: Register in `agent/cli.py` (22 tools total), add `energy_demand` bandit arm (tools: power_grid, market_data, macro_data)
- [x] 6g.5: Live tested — demand (11 zones, peak 15,705 MW, NYC 4,900 MW), fuel mix (Gas 31.4%, Hydro 23.3%, Nuclear 16.0%), pricing (15 zones DA/RT with spreads, no stress), forecast (11 zones with deviation %, Millwood +10.0% flagged), archive fallback (Feb 15 2026 from ZIP), zone alias (NYC→N.Y.C.)
- [x] 6g.6: Edge case tests — 98/98 passed, 0 skipped. 17 test classes: input validation (8), tool metadata (5), zone normalization (8), CSV parsing (6), demand mode (8), fuel mix mode (7), pricing mode (7), forecast mode (6), HTTP/fetch (5), archive ZIP parsing (3), cache interaction (4), _safe_float (12), hour truncation (6), output formatting (3), NYISO zones (3), bandit integration (3), live network (4).

### Sub-phase 6h: ClinicalTrials.gov — FOLDED INTO 7b-M

---

## Phase 7: Pipeline Layer (Deterministic DAG Scheduler) ← CURRENT

**Goal:** Build a second execution engine — deterministic, scheduled, parallel, persistent. No LLM. Shares tools with Agent Layer. SQLite for structured storage. APScheduler for cron triggers.

**Research:** `[[pipeline_layer]]`
**Spec:** `[[pipeline_layer_spec]]`

### Step 7.1: PipelineStore (SQLite persistence)
- [x] 7.1: Create `agent/pipeline/__init__.py` + `agent/pipeline/store.py` — SQLite wrapper (WAL mode, dag_runs/pipeline_data/signals tables, CRUD methods, context manager)
- [x] 7.1t: Unit tests for PipelineStore (in-memory SQLite, CRUD, query, schema) — 56/56 passed

### Step 7.2: DAG + Node data model
- [x] 7.2: Create `agent/pipeline/dag.py` — Node/DAG dataclasses, validate (cycle detection via Kahn's), topo_sort (execution layers), roots
- [x] 7.2t: Unit tests for DAG (cycles, missing deps, topo sort, diamond/wide/deep graphs) — 39/39 passed

### Step 7.3: Operators (Tool + Function wrappers)
- [x] 7.3: Create `agent/pipeline/operators.py` — Operator ABC, ToolOperator, FunctionOperator
- [x] 7.3t: Unit tests for operators (mock tool delegation, function calls, error handling) — 22/22 passed

### Step 7.4: DAGExecutor (parallel execution engine)
- [x] 7.4: Create `agent/pipeline/executor.py` — topo sort → parallel layers via ThreadPoolExecutor, retry + backoff, timeout, upstream result injection, run tracking
- [x] 7.4t: Unit tests for executor (linear/diamond/wide DAGs, retry, timeout, failure cascading, concurrent writes) — 32/32 passed

### Step 7.5: PipelineConfig + CLI integration
- [x] 7.5a: Add `PipelineConfig` to `agent/config/settings.py` (frozen dataclass, TIRRA_PIPELINE_* env vars, embedded in AgentConfig)
- [x] 7.5b: Add `--pipeline` subcommand to `agent/cli.py` (run/list/status/start — lazy imports for registry/scheduler)
- [x] 7.5t: Tests for config + CLI arg parsing — 36/36 passed

### Step 7.6: PipelineScheduler (cron triggers) ✅ COMPLETE
- [x] 7.6a: Create `agent/pipeline/scheduler.py` — wraps APScheduler BackgroundScheduler, cron triggers, blocking start via threading.Event, idempotent stop, manual trigger, auto-register from registry
- [x] 7.6b: Add `apscheduler>=3.10,<4.0` to `pyproject.toml`
- [x] 7.6t: Tests — 44/44 passed (9 classes: lifecycle, registration, trigger, cron scheduling, error handling, blocking, DAGProvider protocol)

### Step 7.7: DAGRegistry + daily_collection DAG ✅ COMPLETE
- [x] 7.7a: Create `agent/pipeline/registry.py` — DAGRegistry (CRUD, load_defaults, DAGProvider protocol)
- [x] 7.7b: Create `agent/pipeline/dags/__init__.py` + `daily_collection.py` — 6-node all-parallel DAG (cftc, finra_scan, power_demand, power_fuel, gdelt, polymarket), schedule="0 18 * * 1-5"
- [x] 7.7t: Tests — 39/39 passed (7 classes: CRUD, load_defaults, DAG validation, daily_collection structure)

### Step 7.8: pipeline_query tool (Agent ↔ Pipeline bridge) ✅ COMPLETE
- [x] 7.8a: Create `agent/tools/pipeline_query.py` — 3 modes (data/signals/runs), relative time parsing (7d/24h/2w), limit clamped 1-500
- [x] 7.8b: Register in `agent/cli.py` (now 23 tools)
- [x] 7.8t: Tests — 47/47 passed (9 classes: modes, time parsing, limit, whitespace, empty results, integration)

### Step 7.9: Comprehensive edge case test suite ✅ COMPLETE
- [x] 7.9: Full edge case tests — 41/41 passed (7 classes: DAGStress 8 tests, ExecutorEdgeCases 7 tests, StoreResilience 10 tests, SchedulerEdgeCases 4 tests, OperatorEdgeCases 5 tests, RegistryEdgeCases 2 tests, FullIntegration 5 tests). Covers: 100-node flat/deep/diamond DAGs, cycle detection, concurrent writes (per-thread stores), SQL injection, large blobs, signal precision, all-fail/partial/cascade/retry/timeout execution, scheduler lifecycle, end-to-end pipeline flow.

**Phase 7 TOTAL: 356/356 tests passing across 8 test files.**

---

## Phase 7b: Global Deep Surveillance (Post-Pipeline)

**Goal:** Build the most advanced global Layer 0/Layer 1 surveillance surface in existence. Every data source must observe physical reality or committed human decisions across ALL major economies — things that can't be faked, retracted, or manipulated. US-only is never acceptable as a final state. Every tool supports multi-country queries, cross-country anomaly detection, and synchronized-event alerting across jurisdictions.

**Why global:** It's a globalized world — everything flows across borders. The edge comes from seeing the SAME hidden state emerge across independent observation channels in independent countries. A drought in Brazil, a ship diversion in Hormuz, and a hiring freeze in Germany — individually noise, together = supply chain crisis forming.

**Why after Phase 7:** Several sources (whale tracking, satellite polling, on-chain indexing) require persistent state and scheduled polling — exactly what the Pipeline Layer provides.

**CRITICAL PRINCIPLE — Cause Data Only:**
We want Layer 0 (physical reality) and Layer 1 (human decisions with legal/financial commitment) data.
We do NOT want Layer 3 (market prices, derivatives of prices) as predictive features — those are the consequence, not the cause, and are the most manipulated layer. Prices are needed only as TARGET variables (what we predict), never as input features.

Classification:
- ✅ **CAUSE (L0/L1):** Physical reality, on-chain immutable, legal filings, committed human actions
- ⚠️ **MIXED (L1-2):** Real data but interpretation needs care (spoofing, lag, methodological manipulation)
- ❌ **CONSEQUENCE (L2-3):** Derived from prices, manipulated by central banks/MMs, what everyone else already has
- 💰 **SUBSCRIPTION (L0/L1, future):** Cause data behind a paywall. Architecture the interface NOW (abstract tool class, params, parsing). When we have proven edge, plug in the API key and it works day one.

**Ordering principle:** Free cause data first → prove edge → unlock paid cause data. No sub-phase blocks another.

**GLOBAL PRINCIPLE:** TirraMind is a global system. Every source must cover all major economies where data is available. US-only is never acceptable as a final state. Every tool should support multi-country queries, cross-country anomaly detection, and synchronized-event alerting across jurisdictions. The edge comes from seeing the SAME hidden state emerge across independent observation channels in independent countries.

---

### Existing Source Globalization (parallel with new sources)

These tools already exist but are US-centric. Each needs global expansion as we build new sources:

| Existing Tool | Current Coverage | Global Expansion |
|---|---|---|
| **CFTC** | US exchanges | **ICE** (London), **Eurex** (Frankfurt), **SGX** (Singapore), **TOCOM** (Tokyo), **MCX** (India), **B3** (Brazil) |
| **FINRA** | US equities | **ESMA** (EU-wide short selling disclosures), **HKEX** (HK), **ASX** (AU), **LSE** (UK), **TSE** (Japan) |
| **Power grid** | NYISO only | **ENTSO-E** (36 European countries), **AEMO** (AU), **POSOCO** (India), **JEPX** (Japan), **ERCOT/PJM/CAISO** (rest of US) |
| **SEC EDGAR** | US issuers | **Companies House** (UK), **SEBI** (India), **EDINET** (Japan), **HKEX** (HK), **Bundesanzeiger** (DE), **ASX** (AU), **SEDAR+** (CA) |
| **FDA** | US drugs/devices | **EMA** (Europe), **PMDA** (Japan), **TGA** (Australia), **NMPA** (China), **Health Canada** |
| **GDELT** | Already global ✅ | No change |
| **Whale Alert** | Already global ✅ | Blockchain is borderless |
| **Polymarket whales** | Already global ✅ | On-chain, global wallets |

---

### Research & Globalization Audit (2026-03-28)

**Full audit doc:** `[[tool_audit_research_globalization]]`

All 20 data tools audited. Standalone research docs now exist for all tools.

| # | Tool | Research | Globalization | Research Doc |
|---|------|----------|---------------|--------------|
| 1 | academic_preprints | `[R:FULL]` | `[G:GLOBAL]` | `[[7b-M_academic_preprints]]` |
| 2 | ais_vessel | `[R:FULL]` | `[G:REGIONAL]` | `[[7b-D_ais_vessel_tracking]]` |
| 3 | cftc | `[R:FULL]` | `[G:INHERENT]` | `[[cftc]]` |
| 4 | defi_flows | `[R:FULL]` | `[G:GLOBAL]` | `[[7b-L_defi_flows]]` |
| 5 | earthquake_proximity | `[R:FULL]` | `[G:GLOBAL]` | `[[7b-U_earthquake_proximity]]` |
| 6 | finra_short_volume | `[R:FULL]` | `[G:INHERENT]` | `[[finra_short_data]]` |
| 7 | form144 | `[R:FULL]` | `[G:INHERENT]` | `[[form144]]` |
| 8 | gdelt | `[R:FULL]` | `[G:GLOBAL]` | `[[gdelt]]` |
| 9 | gov_contracts | `[R:FULL]` | `[G:EXPANDED]` ✅ US+UK | `[[7b-G_gov_contracts]]` |
| 10 | insider_filings | `[R:FULL]` | `[G:INHERENT]` | `[[insider_filings]]` |
| 11 | macro_data | `[R:FULL]` | `[G:EXPANDED]` ✅ FRED+ECB+WB | `[[macro_data]]` |
| 12 | market_data | `[R:FULL]` | `[G:GLOBAL]` | `[[market_data]]` |
| 13 | polymarket | `[R:FULL]` | `[G:GLOBAL]` | `[[polymarket]]` |
| 14 | polymarket_whales | `[R:FULL]` | `[G:GLOBAL]` | `[[polymarket_whale]]` |
| 15 | power_grid | `[R:FULL]` | `[G:NEEDS-EXPANSION]` | `[[power_grid]]` |
| 16 | regulatory_gazette | `[R:FULL]` | `[G:NEEDS-EXPANSION]` | `[[7b-Q_regulatory_gazette]]` |
| 17 | transport_throughput | `[R:FULL]` | `[G:NEEDS-EXPANSION]` | `[[7b-R_transport_throughput]]` |
| 18 | weather_alerts | `[R:FULL]` | `[G:REGIONAL]` | `[[7b-C_weather_alerts]]` |
| 19 | whale_alert | `[R:FULL]` | `[G:GLOBAL]` | `[[whale_alert]]` |
| 20 | wikipedia_pageviews | `[R:FULL]` | `[G:GLOBAL]` | `[[7b-O_wikipedia_pageviews]]` |

**Summary:** 20/20 `[R:FULL]` ✅ | 9 Global, 4 Inherent (US-locked by nature), 3 Need Expansion, 2 Regional, 2 Expanded ✅

**International API alternatives researched:** `[[international_api_alternatives]]` (60+ endpoints probed live, 2026-03-28)

**Verified working (no auth):** UK Contracts Finder (OCDS JSON), ECB Data API (SDMX JSON), Eurostat Transport (SDMX JSON), World Bank (JSON), OECD SDMX (XML), UK legislation.gov.uk (Atom XML), DWD Germany (CAP XML), JMA Japan (JSON)

**Auth-gated (free key):** ENTSO-E (36 EU countries), UK Companies House, EDINET Japan

**No working alternative found:** FINRA short selling (ESMA/FCA/ASX all dead/blocked)

---

### Tier 1 — CAUSE DATA, FREE (highest priority — build these)

#### 7b-A: Polymarket Whale Tracker ✅ COMPLETE — CAUSE (L1, on-chain immutable)

Research: `[[polymarket_whale]]`
Spec: `[[polymarket_whale_spec]]`

**API discovery (2026-03-27):** `data-api.polymarket.com` is fully public, $0, no key — returns all trades with wallet addresses, sizes, prices, tx hashes, user names. Also has `/positions` (P&L per wallet) and `/activity` (USDC values). Max 1000/request, no rate limiting.

- [x] 7b-A.1: Research — probed CLOB (401 auth-gated), data-api (public), Gamma (resolved events), Polygon RPC (unnecessary). Research doc written.
- [x] 7b-A.2: Spec — 10 atomic steps, wallet scoring schema, 4 signal types, 2 pipeline DAGs.
- [x] 7b-A.3: Implement `fetch_recent_trades()` — data-api /trades, micro-market filter
- [x] 7b-A.4: Implement `index_trades()` — PipelineStore insert, tx_hash dedup
- [x] 7b-A.5: Implement `score_wallets()` — Bayesian accuracy, composite score, top-500
- [x] 7b-A.6: Implement `track_resolutions()` — Gamma resolved events, outcome parsing
- [x] 7b-A.7: Implement `detect_signals()` — consensus/whale_alert/contrarian signals
- [x] 7b-A.8: Build whale_tracking + whale_scoring DAGs, register in dags/__init__.py
- [x] 7b-A.9: Implement PolymarketWhalesTool — 4 modes, cold-start fallback
- [x] 7b-A.10: CLI registration + bandit arm update
- [x] 7b-A.11: Live validation — full DAG run against real API. ISP DNS poisoning detected (Indian BSNL redirecting polymarket.com → 49.44.79.236). Built `agent/data/dns_bypass.py` — auto-detects poisoning via Cloudflare DoH, patches `socket.getaddrinfo`. All 4 spec validation steps passed: fetch_recent_trades (389 trades, 229 wallets), track_resolutions (96 resolved markets), full whale_tracking DAG (completed in 0.5s), PolymarketWhalesTool cold-start (top-10 wallets by volume).
- [x] 7b-A.12: Edge case tests — 100/100 passed

---

#### 7b-B: Congressional & Political Insider Trading ✅ CAUSE (L1, legal disclosure + real money) — DEPRIORITIZED

**Skipped for now.** 45-day reporting delay, range-based $ reporting, crowded signal (everyone scrapes this now), politicians have adapted. Data quality too poor for the edge we want. May revisit if unique angle found.

---

#### 7b-C: Weather / Climate / Natural Disaster — GLOBAL ✅ CAUSE (L0, physics) ✅ COMPLETE

**Layer 0 physical reality across all continents.** Drought → crop failure → commodity spike. Freeze → energy demand. Hurricane → supply chain. Wildfire → utility liability. Flooding → logistics.

- [x] 7b-C.1: Research — probed NOAA NWS (free, GeoJSON, severity filter, no `limit` param, User-Agent required), NASA FIRMS MODIS (free, CSV, 16K+ fires/day, no auth), TSA (DNS fail), Eurocontrol (404), Port of LA (SSL error). NWS + FIRMS = working combo.
- [x] 7b-C.2: Implement — `agent/tools/weather_alerts.py` (~400 lines). 3 modes: `alerts` (NOAA NWS active severe weather, 20 market-relevant event types, severity filter, state filter), `fires` (NASA FIRMS near 12 infrastructure zones — Permian Basin, Gulf Coast, ERCOT, Corn Belt, etc.), `summary` (combined NWS + FIRMS overview). Cache: NWS 600s, FIRMS 3600s.
- [x] 7b-C.3: Register + tests — tool #25, bandit arm: weather_disruption, 92 edge case tests.

#### 7b-D: AIS Vessel Tracking — GLOBAL ✅ CAUSE (L0, physical ship movement) ✅ COMPLETE

**Physical movement of goods across all oceans and straits.** Oil tankers, grain carriers, LNG carriers, container ships. Can't fake a 300m tanker's position.

- [x] 7b-D.1: Research — probed 6 AIS APIs (MarineTraffic, AISHub, BarentsWatch, NOAA MarineCadastre, Datalastic, Finland Digitraffic). **Finland Digitraffic = winner**: zero auth, 18K+ vessels, Baltic coverage, rich metadata. See `[[7b-D_ais_vessel_tracking]]`.
- [x] 7b-D.2: Implement — `agent/tools/ais_vessel.py` (~400 lines). 4 modes: `area` (vessels in named zone or custom bbox, ship type filter), `vessel` (MMSI lookup with position+metadata), `port_calls` (Finnish port activity), `destination_flow` (aggregate destination distribution — detects Suez/Russia trade shifts). 9 named areas (danish_straits, gulf_of_finland, st_petersburg, gotland, skagerrak, kiel, gulf_of_bothnia, riga_gulf, full_baltic). Nav status mapping, ship type classification (tanker/cargo/passenger/fishing/tug). Strategic destination grouping (Suez, Russia, Rotterdam, Antwerp). Caching: positions 5min, metadata 6hr, port calls 1hr. Key insight: Baltic captures destination INTENT globally — 580+ ships heading to Suez, 280+ to Russia.
- [x] 7b-D.3: Register + tests — registered in cli.py (tool #23). `tests/test_ais_vessel_edge.py`: 89 tests across 15 classes (TestInBbox, TestShipTypeLabel, TestShipTypeMatches, TestModeRouting, TestAreaMode, TestNamedAreas, TestVesselMode, TestPortCallsMode, TestDestinationFlowMode, TestCacheIntegration, TestToolSchema, TestNavStatus, TestRegistryIntegration, TestExceptionSafety, TestDataShapeEdgeCases). Full suite: 910 passed, 0 failed, 2 skipped.

#### 7b-E: Bankruptcy, Court Filings & Regulatory Actions — GLOBAL ✅ CAUSE (L1, legal reality)

**Legal filings with irreversible consequences across major jurisdictions.**

- [x] 7b-E.1: Research — probed 30+ endpoints across US/UK/EU/Asia. **Winners**: PACER RSS (6 courts, free, real-time XML), SEC Admin Proceedings RSS, SEC EFTS 8-K Item 1.03, UK Gazette Atom feed, GOV.UK SFO API. **Dead**: DOJ (404), FTC (404), FCA (403), Companies House (auth), ESMA/BaFin/AMF/CONSOB/JFSA/ASIC/SEBI (all 404), WTO (needs key), Canada OSB (auth). See `[[7b-E_bankruptcy_court]]`.
- [x] 7b-E.2: Implement — `agent/tools/bankruptcy_court.py` (~480 lines). 4 modes: `us_bankruptcy` (PACER RSS, 6 courts parallel via ThreadPoolExecutor, chapter detection regex), `sec_enforcement` (Admin + Litigation RSS merged/sorted), `sec_bankruptcy` (EFTS 8-K Item 1.03 with date range), `uk_insolvency` (Gazette Atom + GOV.UK SFO combined). Browser-like UA (SEC blocks custom UAs). Caching: PACER 10min, SEC enforcement 30min, EFTS/UK 1hr.
- [x] 7b-E.3: Register + tests — registered in cli.py (tool #33). `tests/test_bankruptcy_court_edge.py`: 106 tests across 17 classes (TestParseChapter, TestParsePubDate, TestKeywordMatch, TestTitleRegex, TestModeRouting, TestParameterValidation, TestPACERParsing, TestPACERCourtFiltering, TestPACERErrorHandling, TestSECEnforcementParsing, TestSECEnforcementErrors, TestSECBankruptcyParsing, TestSECBankruptcyErrors, TestGazetteAtomParsing, TestGovUKParsing, TestUKInsolvencyCombined, TestCacheIntegration, TestToolSchema, TestRegistryIntegration, TestResultFormat, TestPACERCourtsDict). Full suite: 1928 passed, 2 failed (pre-existing NYISO weekend), 6 skipped.

#### 7b-F: Job Postings / Hiring Intent — GLOBAL ✅ CAUSE (L1, real business decisions)

**Companies commit real money when they hire. Leading indicator across all economies.**

- [x] 7b-F.1: Research — probed BLS JOLTS via FRED (JTSJOL, JTSQUL, JTSHIL, JTSLDR) and BLS Public API (POST, no auth, 25 queries/day free). Sector-level series IDs for 11 NAICS super-sectors. Composite labor market = JOLTS + UNRATE + ICSA + PAYEMS.
  - **US:** BLS JOLTS (FRED), H-1B LCA filings (DOL iCERT, free)
  - **Global:** Indeed Hiring Lab (public data, multi-country), OECD employment indicators, LinkedIn Economic Graph (limited public data)
  - **EU:** Eurostat job vacancy statistics, national employment agencies
  - **UK:** ONS vacancy data
  - **Japan:** MHLW job-to-applicant ratio (leading indicator, published monthly)
  - **India:** EPFO payroll data (monthly, free), NCS portal
  - **China:** PMI employment sub-index (NBS monthly)
- [x] 7b-F.2: Implement — `agent/tools/job_postings.py` (~350 lines). 3 modes: `jolts` (headline JOLTS via FRED or BLS fallback), `sector` (11-sector breakdown via BLS), `labor_market` (composite: JOLTS + unemployment + claims + payrolls, requires FRED key). Quits/layoffs ratio, market tightness computation.
- [x] 7b-F.3: Register + tests — tool #40 (job_postings), bandit arm: labor_market, 60 edge case tests.

#### 7b-G: Government Contract Awards — GLOBAL ✅ CAUSE (L1, committed taxpayer money) — COMPLETE

**Government procurement across major economies. Committed revenue before Wall Street notices.**

- [x] 7b-G.1: Research — probed USASpending.gov API (POST /api/v2/search/spending_by_award/ — free, no auth, structured JSON)
- [x] 7b-G.2: Implement — `agent/tools/gov_contracts.py` (GovContractsTool, 4 modes: recent/top/agency/search)
- [x] 7b-G.3: Register + tests — tool #29, bandit arm `government_spending`, 60+ edge case tests

#### 7b-H: Patent & Trademark Filings — GLOBAL ✅ CAUSE (L1, legal commitment to innovation) — COMPLETE

**Patent filings across all major patent offices reveal global innovation pipeline.**

- [x] 7b-H.1: Research — probe:
  - **US:** USPTO Open Data Portal (free), PatentsView API (free)
  - **EU:** EPO Open Patent Services (free API), Espacenet
  - **Japan:** J-PlatPat (JPO, free)
  - **China:** CNIPA (China National IP Administration — world's largest patent office by volume)
  - **Korea:** KIPRIS (free)
  - **Global:** WIPO PATENTSCOPE (free, PCT applications), Madrid trademark system
  - **India:** IP India (CGPDTM)
- [x] 7b-H.2: Implement — `agent/tools/patent_filings.py`. Modes: `search` (keyword / assignee / CPC search), `trends` (filing volume trends by CPC class), `assignee` (portfolio + filing velocity by company). Initial implementation uses USPTO / PatentsView-style API coverage as production-ready baseline.
- [x] 7b-H.3: Register + tests — registered in `cli.py`, bandit arm: `innovation_pipeline`, comprehensive edge case suite added

#### 7b-I: Satellite-Derived Physical Activity — GLOBAL ✅ CAUSE (L0, photons hitting sensors)

**Free satellite data — inherently global.** Nighttime lights = economic output. Thermal hotspots = factory activity. Vegetation = crop health. Already global by nature.

- [x] 7b-I.1: Research — NASA FIRMS (fire/thermal hotspots), MODIS Web Service (NDVI vegetation), NASA EONET (natural events). See `[[batch7_satellite_electricity_queue]]`.
- [x] 7b-I.2: Implement — `agent/tools/satellite_activity.py`. Modes: `fire` (NASA FIRMS hotspots with FRP stats + grid clustering), `vegetation` (MODIS NDVI time series with health classification + anomaly detection), `events` (NASA EONET natural disaster events with category counts).
- [x] 7b-I.3: Register + tests — registered in `cli.py`, bandit arm: `satellite_surveillance`, comprehensive edge case suite added (83 tests)

#### 7b-J: Lobbying Expenditure — GLOBAL ✅ CAUSE (L1, committed money to change policy) — COMPLETE

**Companies spend money lobbying BEFORE policy changes. This exists in every democracy.**

- [x] 7b-J.1: Research — probe:
  - **US:** Senate Lobbying Disclosure Act database (free XML), OpenSecrets.org API (free tier)
  - **EU:** EU Transparency Register (free, all EU-level lobbying), national registers
  - **UK:** ORCR (Office of the Registrar of Consultant Lobbyists)
  - **Canada:** Office of the Commissioner of Lobbying (free, structured)
  - **Australia:** Register of Lobbyists
- [x] 7b-J.2: Implement — `agent/tools/lobbying.py`. Modes: `search` (registrant/client/year), `spending` (multi-year spend aggregation + anomaly detection), `issues` (issue-area activity and spender concentration). Initial implementation uses LDA.gov as the production baseline.
- [x] 7b-J.3: Register + tests — registered in `cli.py`, bandit arm: `lobbying_intelligence`, comprehensive edge case suite added

#### 7b-K: Utility Interconnection Queue — GLOBAL ✅ CAUSE (L1, committed capital for future generation)

**Extension of power_grid tool into the future. Shows what's being built where.**

- [x] 7b-K.1: Research — EIA electricity/operating-generator-capacity API v2 (free with key). See `[[batch7_satellite_electricity_queue]]`.
- [x] 7b-K.2: Implement — `agent/tools/interconnection_queue.py`. Modes: `queue` (search planned/construction generators by state/fuel/status/min_mw), `summary` (aggregate MW by fuel type/state/status), `datacenter` (hyperscaler power project detection via 24 regex patterns — Amazon, AWS, Microsoft, Google, Meta, Equinix, etc.).
- [x] 7b-K.3: Register + tests — registered in `cli.py`, bandit arm: `energy_infrastructure_pipeline`, comprehensive edge case suite added (67 tests)

#### 7b-L: DeFi Protocol On-Chain Flows — GLOBAL ✅ CAUSE (L1, immutable blockchain) — COMPLETE

**Full on-chain economy — inherently borderless.**

- [x] 7b-L.1: Research — probed DefiLlama API (3 endpoints: /protocols, /stablecoins, /overview/dexs — free, no auth)
- [x] 7b-L.2: Implement — `agent/tools/defi_flows.py` (DefiFlowsTool, 4 modes: tvl/stablecoins/dex_volume/chain)
- [x] 7b-L.3: Register + tests — tool #28, bandit arm `defi_liquidity`, 70+ edge case tests

#### 7b-M: Academic Preprint Signals — GLOBAL ✅ CAUSE (L1, researchers committing to public claims) — COMPLETE

**Preprints are inherently global. Add non-English sources for true global coverage.**

- [x] 7b-M.1: Research — probed arXiv API (Atom XML, https, follow_redirects) + ClinicalTrials.gov v2 (JSON, free, no auth)
- [x] 7b-M.2: Implement — `agent/tools/academic_preprints.py` (AcademicPreprintsTool, 3 modes: papers/trending/trials)
- [x] 7b-M.3: Register + tests — tool #30, bandit arm `research_pipeline`, 55+ edge case tests

#### 7b-N: FCC / Spectrum & Telecom Filings — GLOBAL ⏭️ SKIPPED (all APIs dead)

**Spectrum auctions and equipment authorization exist in every country.**

- [x] 7b-N.1: Research — SKIPPED. All FCC endpoints dead: ECFS 403, ULS timeout, Equipment Auth timeout, Auction Data DNS failure, Socrata stale broadband only. International alternatives (ETSI 404, Ofcom 404, ITU 404, MIC 404) also dead. See `[[7b-N_fcc_spectrum]]`.
- [N/A] 7b-N.2: Skipped — no viable free API found
- [N/A] 7b-N.3: Skipped — no implementation

---

### Tier 1-NEW — "BORING GOLD" SOURCES (free, global, nobody watches)

**The thread connecting all of these: bureaucratic, ugly, boring data that requires effort to parse — but represents irreversible real-world commitments.** A FOIA request is filed. A UCC lien is recorded. A certificate is issued. A railroad car is loaded. These aren't opinions — they're actions already taken.

#### 7b-O: Wikipedia / Wikimedia Page Views — GLOBAL ✅ CAUSE (L1, behavioral trace) ✅ COMPLETE

**The "somebody knows something" detector.** People research before they act. Wikimedia REST API, free, zero auth, per-article daily granularity, 300+ language editions. Japanese Wikipedia spike on a chemical compound = Japanese research interest. Arabic Wikipedia spike on a political figure = regional instability forming.

- [x] 7b-O.1: Research — Wikimedia REST API (pageviews endpoint), rate limits, language edition coverage. Confirmed: per-article daily, top-1000/day, top-by-country, all work. ~0.6s/query, no auth, User-Agent required.
- [x] 7b-O.2: Implement — `agent/tools/wikipedia_pageviews.py`. Three modes: `spike` (z-score anomaly detection across watchlist, 30d trailing baseline), `top` (trending articles with evergreen filtering), `series` (raw timeseries with inline spike flagging). Default watchlist: 37 major companies/entities. Near-zero-variance handling. Politeness delay.
- [x] 7b-O.3: Register in cli.py + 77 edge case tests (math, spike detection, HTTP mocking, cache, mode routing, parameter clamping, evergreen filtering, schema validation, registry integration). 819/819 full suite.

#### 7b-P: Certificate Transparency Logs — GLOBAL ✅ CAUSE (L1, strategic intent leaked via domain registration)

**Every SSL certificate on Earth is logged.** When Apple registers `apple-car-services.com`, that's a product reveal. When a company registers domains in 15 countries = global product rollout. Phishing cert detection = breach incoming.

- [ ] 7b-P.1: Research — Google CT, Cloudflare Nimbus, crt.sh (free Postgres interface). Volume, filtering, real-time vs batch
- [ ] 7b-P.2: Implement — new domain monitoring for tracked companies (global), brand-adjacent domain detection (product launches), phishing cert detection (impersonation = fraud/breach signal), multi-country registration clustering
- [ ] 7b-P.3: Register + tests

#### 7b-Q: Regulatory Gazette / Official Journal — GLOBAL ✅ CAUSE (L1, the future regulatory pipeline)

**The entire future of regulation is published before it happens. Almost nobody monitors it programmatically across countries.**

- [x] 7b-Q.1: Research — probe:
  - **US:** Federal Register API (free) ✅ Excellent — JSON, no auth, 470 agencies
  - **EU:** EUR-Lex / Official Journal of the EU (free SPARQL endpoint) ❌ SPARQL returns 0 results, HTML endpoints return 202 SPA
  - **UK:** legislation.gov.uk API (free) ⚠️ Atom XML feed partial — inconsistent fields
  - **Japan:** e-Gov (free)
  - **India:** Gazette of India (egazette.gov.in)
  - **Brazil:** Diário Oficial da União (free) ❌ Connection refused
  - **Australia:** Federal Register of Legislation (free API) ❌ SPA, no JSON API
  - **China:** State Council (gov.cn)
- [x] 7b-Q.2: Implement — US Federal Register API with 4 modes (recent/search/agency/upcoming), 20 curated market agencies, significant rule detection, comment period tracking. EUR-Lex deferred (broken API).
- [x] 7b-Q.3: Register + tests — 144 edge case tests, tool #24, bandit arm: regulatory_pipeline

#### 7b-R: Transportation Throughput — GLOBAL ✅ CAUSE (L0, physical movement of humans and goods) ✅ COMPLETE

**Real-time GDP components measured by physical movement across all countries.**

- [x] 7b-R.1: Research — probed TSA (DNS fail), Eurocontrol (404), Port of LA (SSL cert error), BTS Airline On-Time (404), USACE Lock (404), AAR Rail (HTML only). **BTS Border Crossing (Socrata) = winner**: free, no auth, JSON API, SoQL queries, 333K+ records since 1996. 10 measures: Trucks (37K records), Trains (30K), Rail Containers Loaded/Empty (30K each), Personal Vehicles (39K), Buses (32K), Pedestrians (33K), Passengers. 2 borders: US-Canada (254K), US-Mexico (79K). Monthly.
- [x] 7b-R.2: Implement — `agent/tools/transport_throughput.py` (~350 lines). 4 modes: `recent` (latest month aggregate by border + measure), `trend` (time series with MoM % change), `port` (port-level detail by state), `compare` (Canada vs Mexico side-by-side with ratio). 14 measure aliases, 4 border aliases. Key insight: Trucks = trade proxy, Rail Containers loaded/empty ratio = trade balance direction. Cache: 7200s.
- [x] 7b-R.3: Register + tests — tool #27, bandit arm: transport_flow, 62 edge case tests.

#### 7b-S: FOIA / FOI / RTI Request Logs — GLOBAL ✅ CAUSE (L1, the investigation detector)

**Cluster of information requests about the same entity = investigation forming. Works in any country with FOI laws.**

- [x] 7b-S.1: Research — probed MuckRock API (US, free, no auth, paginated REST), WhatDoTheyKnow (UK, Alaveteli API, free). FOIA.gov lacks request search API. See `[[7b-S_foia_logs]]`.
- [x] 7b-S.2: Implement — `agent/tools/foia_requests.py` (~450 lines). 3 modes: `search` (keyword search across FOIA/FOI requests), `agency_activity` (request volume + surge detection), `entity_cluster` (cross-agency/jurisdiction investigation convergence detection). Sources: MuckRock (US) + WhatDoTheyKnow (UK).
- [x] 7b-S.3: Register + tests — tool #37 (foia_requests), bandit arm: investigation_signals (#25), 108 edge case tests passing. Full suite: 2386 passed, 0 failed, 6 skipped.

#### 7b-T: Government Bond / Sovereign Debt Markets — GLOBAL ✅ CAUSE (L1, fiscal stress pricing)

**Bond markets are smarter than equity markets. They price fiscal stress months/years before equity investors notice. Detroit munis screamed 18 months before bankruptcy.**

- [x] 7b-T.1: Research — probed MSRB EMMA, ECB sovereign data, DMO gilt auctions, MOF JGB data, BIS debt statistics, World Bank IDS. See `[[7b-T_sovereign_debt]]`.
- [x] 7b-T.2: Implement — `agent/tools/sovereign_debt.py` (726 lines). 4 modes: `us_treasury` (yields, curve), `sovereign_spreads` (cross-country EM/DM spreads), `muni_stress` (US municipal bond stress), `auction_results` (global sovereign auctions). Fixed cache API bug (4 occurrences).
- [x] 7b-T.3: Register + tests — tool #28 (sovereign_debt), bandit arm: sovereign_stress, 94 edge case tests passing.

#### 7b-U: Earthquake + Geophysical Proximity — GLOBAL ✅ CAUSE (L0, physics) ✅ COMPLETE

**Everyone sees the earthquake. Nobody auto-cross-references it with the specific factory, pipeline, or mine in the blast radius.**

- [x] 7b-U.1: Research — probed USGS Earthquake API (free, GeoJSON, global, real-time). Params: minmagnitude, starttime, endtime, limit=500, orderby=magnitude. Properties: mag, place, time (epoch ms), alert (green/yellow/orange/red), tsunami (0/1), sig.
- [x] 7b-U.2: Implement — `agent/tools/earthquake_proximity.py` (~380 lines). 3 modes: `recent` (significant quakes worldwide with infra overlay), `monitor` (check specific zone for seismic activity), `infrastructure` (list 19 monitored zones). 19 critical infrastructure zones across 8 sectors: semiconductor (TSMC Hsinchu/Tainan, Samsung), mining (Escondida, Chuquicamata, Grasberg, Indonesia Nickel), nuclear (Fukushima, Kashiwazaki-Kariwa, Turkey Point), energy (BTC Pipeline, Permian Basin, Oklahoma Injection), logistics (Port of LA, Port of Shanghai, Strait of Hormuz), agriculture (NZ Dairy), industrial (Japan Pacific Coast), tech (Northern Virginia DCs). Cache: 1800s.
- [x] 7b-U.3: Register + tests — tool #26, bandit arm: seismic_risk, 83 edge case tests.

#### 7b-V: UCC / Secured Creditor Filings — GLOBAL ✅ CAUSE (L1, financial stress before news)

**When creditors file security claims, they're getting nervous about getting paid. This precedes bankruptcies by months.**

- [x] 7b-V.1: Research — probe:
  - **US:** State-level UCC filings (NY, DE, CA — bulk download)
  - **UK:** Companies House charges register (free API)
  - **Canada:** PPSA registries (provincial)
  - **Australia:** PPSR (Personal Property Securities Register, free search)
  - **India:** MCA charge registrations (free)
- [x] 7b-V.2: Implement — creditor filing surge detection per entity, multi-country filing spikes on same parent company = global financial distress, sector-wide stress clustering
- [x] 7b-V.3: Register + tests

#### 7b-W: Drug / Medical Regulatory — GLOBAL ✅ CAUSE (L1, health authority actions)

**Same drug getting warning letters in EU + Japan + Australia = global safety signal, not a local anomaly.**

- [ ] 7b-W.1: Research — probe: FDA warning letters/483s (US), EMA (EU), PMDA (Japan), TGA (Australia), NMPA (China), Health Canada, WHO prequalification, EU CTR, JAPIC
- [ ] 7b-W.2: Implement — warning letter / safety signal monitoring across all major health authorities, cross-jurisdiction signal convergence (same drug flagged in multiple countries), drug supply disruption detection (sole-source manufacturer warning = shortage)
- [ ] 7b-W.3: Register + tests

#### 7b-X: Academic Preprint Surge — covered by 7b-M global expansion above

---

### Tier 1-GLOBAL — NET-NEW CROSS-COUNTRY SOURCES

**These sources only exist at the cross-country level. They are the connective tissue of the global economy.**

#### 7b-Y: UN Comtrade — Global Bilateral Trade Flows ✅ CAUSE (L1, customs reality)

**Every bilateral trade flow between every country, by HS commodity code.** When China rare earth exports to Japan drop 40% → supply chain crisis. When Russian wheat exports stop → food price spike. When semiconductor imports surge → stockpiling ahead of sanctions.

- [x] 7b-Y.1: Research — probed UN Comtrade API (free preview at comtradeapi.un.org/public/v1/preview, premium with subscription key). M49 country codes, HS commodity taxonomy, 10 records/request free tier.
- [x] 7b-Y.2: Implement — `agent/tools/comtrade.py` (~370 lines). 3 modes: `flows` (bilateral trade between reporter/partner), `commodity` (search by HS code), `partners` (top trading partners). 34 M49 country codes, 18 strategic HS commodities. Premium/public API switching via TIRRA_UN_COMTRADE_KEY. Bug fix: empty string country resolution.
- [x] 7b-Y.3: Register + tests — tool #39 (comtrade), bandit arm: global_trade, 60 edge case tests.

#### 7b-Z: Global Central Bank Balance Sheets & Rate Decisions ✅ CAUSE (L1, monetary policy reality) ✅ COMPLETE

**Individual CB data is watched. Cross-CB relative positioning is not systematically computed by almost anyone outside top macro funds.**

- [x] 7b-Z.1: Research — probed FRED API (WALCL, ECBASSETSW, JPNASSETS + FX series) and ECB SDW API (free, no auth, confirmed working). See `[[7b-Z_central_bank_balance_sheets]]`.
- [x] 7b-Z.2: Implement — `agent/tools/central_bank_balance.py` (~650 lines). 4 modes: `balance_sheets` (cross-CB snapshot in USD), `liquidity_index` (net global liquidity = CB assets - RRP - TGA), `policy_divergence` (expanding vs contracting, rate differentials), `rate_monitor` (current rates + last change detection). 7 CBs: Fed, ECB, BOJ, BOE, SNB, BOC, RBA. FX normalization to USD.
- [x] 7b-Z.3: Register + tests — tool #36 (central_bank_balance), bandit arm: global_liquidity, 84 edge case tests passing.

#### 7b-AA: Global PMI / Leading Indicators ✅ CAUSE (L1, forward-looking survey + commitment)

**When PMIs fall across Asia + Europe simultaneously = synchronized global slowdown.**

- [ ] 7b-AA.1: Research — S&P Global PMI (headline free via press release), OECD CLI (free API, 38 countries), China NBS/Caixin PMI, India IHS Markit, Conference Board LEI
- [ ] 7b-AA.2: Implement — PMI by country with trend, cross-country synchronization detection (PMI falling in 10+ countries = global slowdown), leading indicator composite across countries, manufacturing vs services divergence
- [ ] 7b-AA.3: Register + tests

#### 7b-AB: Global Building Permits / Construction ✅ CAUSE (L0, physical construction)

**Construction boom across multiple countries = synchronized credit expansion. Permits crashing = recession signal 12-18 months out.**

- [x] 7b-AB.1: Research — FRED API for US permits (PERMIT, PERMITNSA, PERMIT1), regional series (4 Census regions × 2 = 8 series), housing starts (HOUST, HOUST1F).
- [x] 7b-AB.2: Implement — `agent/tools/building_permits.py` (~340 lines). 3 modes: `permits` (national trends with MoM/YoY/consecutive declines warning, SF/MF share), `regional` (8 regional series with divergence detection), `housing_starts` (starts/permits ratio = builder confidence). Requires FRED key.
- [x] 7b-AB.3: Register + tests — tool #41 (building_permits), bandit arm: construction_cycle, 59 edge case tests.

#### 7b-AC: Cross-Border Capital Flows ✅ CAUSE (L1, money moving across borders) — COMPLETE

**When Japan + China + Saudi all sell US Treasuries = coordinated de-dollarization. The direction of capital across borders IS the macro signal.**

- [x] 7b-AC.1: Research — US Treasury TIC data (monthly), Japan MOF international transactions (weekly — only major country with weekly data), BIS banking statistics, IIF capital flows tracker
- [x] 7b-AC.2: Implement — `agent/tools/capital_flows.py`. Modes: `holdings` (major foreign holders of US Treasuries), `flows` (net capital flow direction / reversals), `reserves` (reserve accumulation / drawdown stress). Includes coordinated selling/buying detection and reserve stress alerts.
- [x] 7b-AC.3: Register + tests — registered in `cli.py`, bandit arm: `capital_flow_monitor`, comprehensive edge case suite added

#### 7b-AD: Global Electricity Consumption × Nighttime Lights ✅ CAUSE (L0, physics cross-reference)

**Cross-references grid data + satellite imagery for ground truth on economic activity.**

- [x] 7b-AD.1: Research — EIA API v2 electricity/rto endpoints (demand, generation, interchange). See `[[batch7_satellite_electricity_queue]]`.
- [x] 7b-AD.2: Implement — `agent/tools/electricity_monitor.py`. Modes: `demand` (hourly load with peak/trough/avg by region), `generation` (fuel mix proportions + renewable/fossil share %), `interchange` (bidirectional power flows between BAs + net import/export).
- [x] 7b-AD.3: Register + tests — registered in `cli.py`, bandit arm: `electricity_demand`, comprehensive edge case suite added (66 tests)

---

### Tier 1-GAP — DEEP COVERAGE, GAP-CLOSING SOURCES (free, global)

**These 10 sources close the remaining blind spots. Without them, entire prediction domains (pandemics, food crises, labor shocks, political disruptions) have zero observability. Raw physics/science/biology data — never filtered, never ignored. Small signals compose into edge.**

#### 7b-AE: Disease & Pandemic Surveillance ✅ CAUSE (L0-L1, biological/epidemiological reality)

**A single wastewater sample detecting a novel pathogen 3 weeks before hospitals fill up. A 0.3°C sea temperature anomaly shifting disease vectors. This is where pandemics, pharma demand, and travel disruptions start — in biology, not in news.**

- [x] 7b-AE.1: Research — probed 16 endpoints live (2026-03-31). **Winners:** CDC NWSS wastewater (6 pathogen datasets: SARS-CoV-2/Flu-A/RSV/Mpox/Measles/Avian-H5, 2.28M records total, Socrata API, no auth, 51 US jurisdictions, plant-level), WHO DON (OData, 3175 global outbreak entries, no auth), ECDC Open Data (3 EU datasets: cases/variants/hospital, 183K records), NCBI E-utilities (genomic sequence velocity, 12K+ SARS-CoV-2 sequences 2026). **Dead:** ProMED (all 404), GISAID (auth-gated), HealthMap (shutdown), Global.health (DNS fail), India IDSP (no API), Brazil InfoGripe (timeout). See `[[7b-AE_disease_surveillance]]`.
- [x] 7b-AE.2: Implement — `agent/tools/disease_surveillance.py` (~600 lines). 4 modes: `wastewater` (CDC NWSS Socrata, 6 pathogen datasets + aggregate metrics, state filter, detection rate/concentration stats, multi-state wave alert), `outbreaks` (WHO DON OData, disease/country title parsing, frequency analysis), `eu_surveillance` (ECDC 3 datasets: cases/variants/hospital, country filter), `genomics` (NCBI E-utilities, sequence submission velocity, YoY ratio, ACCELERATING/ELEVATED/DECLINING/STABLE signals). Pathogen alias resolution (covid→sars-cov-2, flu→influenza_a, h5n1→avian_h5, etc.), hyphen/underscore equivalence. Cache TTLs: CDC 2hr, WHO 6hr, ECDC 12hr, NCBI 24hr.
- [x] 7b-AE.3: Register + tests — registered in `cli.py`, bandit arm: `pandemic_surveillance` (tools: disease_surveillance, weather_alerts, web_search). `tests/test_disease_surveillance_edge.py`: 113/113 passed across 15 classes (TestToolMetadata, TestInputValidation, TestResolvePathogen, TestParseWhoTitle, TestSafeFloat, TestSafeInt, TestWastewaterPathogen, TestWastewaterAggregate, TestCDCSocrataErrors, TestOutbreaksMode, TestWHOErrors, TestEUSurveillanceMode, TestECDCErrors, TestGenomicsMode, TestNCBIErrors, TestCacheInteraction, TestOutputFormatting, TestDataConstants, TestRegistryIntegration, TestModeRouting, TestEdgeCombinations).

#### 7b-AF: Sanctions & Export Control Lists ✅ CAUSE (L1, legal/regulatory reality)

**When OFAC adds a company to the SDN list, every bank in the world must stop transacting with it within 24 hours. When BIS adds entities to the Entity List, their supply chain breaks. These lists ARE the geopolitical weapon system.**

- [ ] 7b-AF.1: Research — probe:
  - **OFAC SDN/SSI List** (US Treasury, free): Specially Designated Nationals, blocked persons/entities, updated regularly
  - **OFAC Consolidated Sanctions** (free XML/CSV): Full US sanctions list with reasons, dates, identifiers
  - **EU Consolidated Sanctions** (free): EU Council sanctions, updated per Official Journal
  - **UN Security Council Sanctions** (free XML): Global sanctions regime, all member states must comply
  - **BIS Entity List** (US Commerce Dept, free): Export control — entities banned from receiving US technology
  - **OFSI** (UK HM Treasury, free): UK financial sanctions list
  - **Australia DFAT** (free): Australian sanctions list
  - **Japan METI/MOF** (free): Japanese export control lists
- [ ] 7b-AF.2: Implement — entity addition/removal detection (new sanctions = immediate supply chain disruption), cross-list correlation (entity appears on US + EU + UN simultaneously = coordinated action), sector targeting analysis (which industries are being sanctioned?), geographic concentration (Russia/Iran/China/DPRK breakdown), sanctions escalation velocity (additions per week trending up = geopolitical deterioration)
- [ ] 7b-AF.3: Register + tests

#### 7b-AG: Agricultural & Food Security Pipeline ✅ CAUSE (L0, physics + biology of food supply)

**Soil moisture is physics. Locust swarms are biology. Crop yield estimates are measurements. When USDA WASDE revises corn ending stocks down by 15%, that's not an opinion — it's a count of physical grain. Food crises cascade into social instability, migration, and regime change.**

- [x] 7b-AG.1: Research — probe:
  - **USDA WASDE** (free, monthly): World Agricultural Supply and Demand Estimates — crop production, ending stocks, trade, prices for every major commodity
  - **FAO GIEWS** (free): Global Information and Early Warning System — crop conditions, food price alerts, country briefs
  - **NASA SMAP** (free): Soil Moisture Active Passive satellite — global soil moisture at 9km resolution. Physical L0.
  - **FAO Desert Locust Watch** (free): Locust swarm tracking, breeding conditions, forecast
  - **USDA FAS PSD** (free): Production, Supply, Distribution — historical crop data for every country
  - **NDVI vegetation index** (MODIS/Sentinel, free): Normalized Difference Vegetation Index — satellite measurement of crop health
  - **GEOGLAM Crop Monitor** (free): G20 initiative, global crop condition assessment
  - **India PMGKAY / FCI stocks** (free): Indian food grain buffer stock levels
- [x] 7b-AG.2: Implement — crop condition by country/commodity (wheat, corn, soy, rice), soil moisture anomaly detection (drought = crop failure 2-3 months out), locust swarm proximity to agricultural zones, WASDE revision surprise detection, cross-source validation (SMAP drought + NDVI browning + WASDE revision = high confidence food shock), food price stress by country (import-dependent nations are most vulnerable)
- [x] 7b-AG.3: Register + tests

#### 7b-AH: Elections & Political Risk ✅ CAUSE (L1, committed political actions + physical conflict)

**Election filings are committed. Political ad spend is committed capital. Armed conflict events are physical reality. When ad spend surges in swing states while approval ratings diverge from prediction markets — that's edge.**

- [x] 7b-AH.1: Research — probe:
  - **FEC filings** (US, free): Campaign finance — contributions, expenditures, donor networks, all downloadable
  - **Meta Ad Library** (global, free): Political ad spending by country, advertiser, demographics, spend amount
  - **ACLED** (Armed Conflict Location & Event Data, free for researchers): Every political violence event, protest, battle globally. Geolocated, classified, daily updates
  - **IDEA International** (free): Global election calendar, election system types, voter turnout
  - **V-Dem** (Varieties of Democracy, free): Democracy quality indicators by country — autocratization trends
  - **European Parliament / national election commissions** (free): Filing data, party funding, results
  - **India ECI** (Election Commission, free): Candidate affidavits, spending, results
- [x] 7b-AH.2: Implement — election calendar with economic relevance scoring, political ad spend velocity (acceleration = uncertainty), armed conflict intensity by region (ACLED event count + fatality trends), regime stability index (V-Dem democratization/autocratization trends), campaign finance anomaly detection (sudden large donations, unusual donor patterns), protest frequency and escalation tracking
- [x] 7b-AH.3: Register + tests

#### 7b-AI: Internet Infrastructure & Digital Outages ✅ CAUSE (L0, physical network reality)

**When a country's internet goes dark, that's a coup, a natural disaster, or censorship — all market-moving. When BGP routes shift, that's physical cable cuts or state-directed rerouting. Digital infrastructure IS physical infrastructure.**

- [x] 7b-AI.1: Research — probe:
  - **IODA** (Internet Outage Detection & Analysis, Georgia Tech, free): Country-level internet connectivity, BGP, DNS, active probing
  - **CAIDA BGP RouteViews** (free): Global BGP routing tables, AS path changes, route hijacks, prefix announcements
  - **Cloudflare Radar** (free API): Global internet traffic patterns, outages, attack trends, country-level stats
  - **RIPE Atlas** (free): Distributed measurement network, ping/traceroute to any target, network health
  - **Google Transparency Report** (free): Traffic disruption data by country/product
  - **OONI** (Open Observatory of Network Interference, free): Censorship measurement, VPN blocking, website blocking by country
  - **Submarine Cable Map** (TeleGeography, free data): Cable landing points, owners, capacity — physical infrastructure
- [x] 7b-AI.2: Implement — country-level connectivity score (percent of normal traffic), outage detection with type classification (full shutdown vs partial, state-directed vs infrastructure failure), BGP anomaly detection (route hijacks, unusual AS path changes), censorship escalation tracking (OONI blocking events trending up), submarine cable cut detection (traffic rerouting patterns)
- [x] 7b-AI.3: Register + tests

#### 7b-AJ: Labor Disruptions & Strike Activity ✅ CAUSE (L1, committed collective action)

**When 10,000 workers walk off a production line, that's a physical reality that shows up in output 1-2 quarters later. Strike funds being drawn down, union authorization votes — these are committed, not speculative.**

- [x] 7b-AJ.1: Research — BLS WSU001/WSU002 confirmed (free, no auth, monthly). Dead: NLRB (timeout), Cornell ILR (404), FRED DEMO_KEY (rejected). See [[batch8_labor_migration_energy]]
- [x] 7b-AJ.2: Implement — agent/tools/labor_disruptions.py (3 modes: work_stoppages, idle_days, overview). BLS API, signal computation with alerts, trend analysis. Bug fix: NEW_ACTIVITY trend was unreachable due to 0.01 fallback.
- [x] 7b-AJ.3: Register + tests — cli.py + bandit.py updated. 77 edge-case tests pass (tests/test_labor_disruptions_edge.py)

#### 7b-AK: Migration & Refugee Flows ✅ CAUSE (L0-L1, physical human movement)

**People moving across borders is physics. When UNHCR reports 500K displaced in 3 weeks, that's a housing crisis, labor market disruption, and political destabilization cascade. Remittance flows are committed financial transfers — real money crossing borders.**

- [x] 7b-AK.1: Research — UNHCR Refugee Statistics API v1 confirmed (population, asylum-decisions, demographics — free, no auth). World Bank remittances BX.TRF.PWKR.CD.DT confirmed. Dead: UNHCR ODP (antibot 404), IDMC (404). See [[batch8_labor_migration_energy]]
- [x] 7b-AK.2: Implement — agent/tools/migration_flows.py (3 modes: displacement, asylum, remittances). UNHCR + World Bank APIs. _safe_int handles mixed UNHCR types (int, "0", "-", None). Signals: displacement thresholds, acceptance rate, remittance YoY/trend.
- [x] 7b-AK.3: Register + tests — cli.py + bandit.py updated. 87 edge-case tests pass (tests/test_migration_flows_edge.py)

#### 7b-AL: Energy Supply Side (Production, Rigs, Nuclear) ✅ CAUSE (L0-L1, physical energy production)

**Baker Hughes rig count is physical metal in the ground. Nuclear plant outages are physics — reactor trips, maintenance schedules, fuel loading. EIA weekly petroleum is inventory measurement. The supply side of energy is where price shocks originate.**

- [x] 7b-AL.1: Research — EIA API v2 confirmed (DEMO_KEY works). petroleum/stoc/wstk (weekly stocks), petroleum/sum/sndw (weekly S&D), natural-gas/enr/drill (monthly rigs). Dead: NRC (all URLs 404/timeout), EIA drilling endpoint (404). See [[batch8_labor_migration_energy]]
- [x] 7b-AL.2: Implement — agent/tools/energy_supply.py (3 modes: petroleum_stocks, petroleum_supply, rig_count). EIA API v2 sole source. Series: WCESTUS1 (crude), WGTSTUS1 (gasoline), WDISTUS1 (distillate), WPRSTUS1 (SPR). Signals: WoW change, consecutive draws/builds, surprise detection (>5M bbl), rig trend.
- [x] 7b-AL.3: Register + tests — cli.py + bandit.py updated. 84 edge-case tests pass (tests/test_energy_supply_edge.py)

#### 7b-AM: Consumer Confidence & Sentiment Surveys ✅ CAUSE (L1, forward-looking committed survey data)

**These are not derived — they're direct measurements of consumer intent. When UMichigan 1-year inflation expectations spike while EU consumer confidence collapses, that's a real divergence in how humans plan to spend. Survey data = committed expectations, not market prices.**

- [x] 7b-AM.1: Research — Eurostat ei_bsco_m confirmed working (28 EU countries, balance %, JSON-stat 2.0). FRED UMich (UMCSENT, MICH) needs TIRRA_FRED_API_KEY. BLS CPI works (free, no auth). OECD CCI already covered by global_pmi.py. See `[[7b-AM_consumer_sentiment]]`.
- [x] 7b-AM.2: Implement — `agent/tools/consumer_sentiment.py` (ConsumerSentimentTool). 3 modes: eu_confidence (Eurostat, 28 countries), us_sentiment (FRED UMich, graceful degradation without key), inflation_reality (BLS CPI + expectation gap analysis). Signals: cross-country divergence, synchronized decline, unanchored expectations, expectation gap. See `[[7b-AM_consumer_sentiment_spec]]`.
- [x] 7b-AM.3: Register + tests — cli.py + bandit.py updated (`consumer_sentiment_monitor` arm). 138 edge-case tests pass across both new tools (tests/test_consumer_sentiment_edge.py).

#### 7b-AN: Government Tax Receipts & Fiscal Data ✅ CAUSE (L1, actual money collected by governments)

**Tax receipts are the most honest economic data that exists. Governments can lie about GDP, massage employment numbers, but tax receipts are audited cash flows. When withholding tax receipts drop mid-quarter, the employment situation is worse than reported. When corporate tax receipts surge, profits are better than estimates.**

- [ ] 7b-AN.1: Research — probe:
  - **US Treasury Daily Statement** (free, daily): Daily tax receipts by type (individual withholding, corporate, excise, customs), outlays, debt
  - **US Treasury Monthly Statement** (free): Monthly receipts/outlays, budget surplus/deficit
  - **HMRC Monthly Receipts** (UK, free): Tax receipts by type — income tax, VAT, corporation tax, national insurance
  - **India GST Revenue** (free, monthly press release): Goods & Services Tax collections — direct measure of economic activity
  - **Australia ATO** (free): Tax statistics, company tax, individual tax, GST collections
  - **Brazil Receita Federal** (free): Federal tax revenue by type, monthly
  - **Japan MOF Tax Revenue** (free): Monthly tax revenue statistics
  - **Eurostat Government Finance Statistics** (free): Government revenue/expenditure by EU country
- [ ] 7b-AN.2: Implement — daily withholding tax tracker (US — best real-time employment proxy), corporate tax receipt surprise (mid-quarter deviation from seasonal = earnings surprise), cross-country tax receipt growth comparison (synchronized decline = global recession), VAT/GST receipt trend (direct consumption measurement), fiscal deficit trajectory by country, tax receipt vs official GDP growth divergence (receipts tell the truth)
- [ ] 7b-AN.3: Register + tests

#### 7b-AO: Supply Chain Price Monitor (pivoted from Commerce Inventory) ✅ CAUSE (L1, producer prices + import costs)

**Original commerce inventory vision impossible (no free APIs). Pivoted to BLS PPI supply chain price monitoring — tracks producer and import prices across semiconductors, steel, machinery, petroleum, computers, and chemicals. Price pressure signals margin squeeze, cost-push inflation, and trade flow disruption.**

- [x] 7b-AO.1: Research — PIVOTED. Original commerce inventory vision impossible (Octopart 404, DigiKey/Mouser auth-gated, Census EITS 204 empty). Pivoted to BLS PPI supply chain price monitoring. Verified PPI series: PCU334413334413 (semiconductors), PCU334111334111 (computers), PCU333120333120 (machinery), PCU331110331110 (iron/steel), PCU324110324110 (petroleum), PCU325130325130 (chemicals). Import prices via EIUIR. See `[[7b-AO_supply_chain_monitor]]`.
- [x] 7b-AO.2: Implement — `agent/tools/supply_chain_monitor.py` (SupplyChainMonitorTool). 3 modes: producer_prices (6 PPI sectors, sector filter), import_prices (3 import indices), pressure_index (composite 0-100 supply chain pressure score). Signals: sector MoM, 3-month trends, broad inflation detection, cross-sector divergence. See `[[7b-AO_supply_chain_monitor_spec]]`.
- [x] 7b-AO.3: Register + tests — cli.py + bandit.py updated (`supply_chain_pressure` arm). Edge-case tests in tests/test_supply_chain_monitor_edge.py.

---

### Tier 1-S — CAUSE DATA, SUBSCRIPTION (architect now, pay later)

**Pattern:** Each tool has `_fetch_data()` that checks for API key in config (`TIRRA_<SOURCE>_API_KEY`). If no key → returns `ToolResult(success=False, output="<source> requires API key. Set TIRRA_<SOURCE>_API_KEY. Free tier: <url>")`. If key present → full functionality.

#### 7b-S1: Satellite Analytics (Orbital Insight / Kayrros / RS Metrics / Ursa Space) 💰 L0
- [ ] 7b-S1.1: Design abstract interface — modes: `retail_traffic`, `oil_storage`, `construction`, `economic_activity`. Provider-agnostic.
- [ ] 7b-S1.2: Implement with mock data + provider adapter pattern
- [ ] 7b-S1.3: Tests with synthetic data

#### 7b-S2: Credit Card / Transaction Data (Second Measure / YipitData) 💰 L1
- [ ] 7b-S2.1: Design interface — modes: `company_revenue`, `sector_spend`, `geographic_trend`
- [ ] 7b-S2.2: Implement with mock data + adapter
- [ ] 7b-S2.3: Tests

#### 7b-S3: Geolocation / Foot Traffic (Placer.ai / SafeGraph) 💰 L0
- [ ] 7b-S3.1: Design interface — modes: `location_visits`, `competitor_comparison`, `new_location_detection`
- [ ] 7b-S3.2: Implement with mock data + adapter
- [ ] 7b-S3.3: Tests

#### 7b-S4: Supply Chain Intelligence (Panjiva / ImportGenius / Descartes) 💰 L1
- [ ] 7b-S4.1: Design interface — modes: `imports_by_company`, `exports_by_country`, `commodity_flow`, `supplier_network`
- [ ] 7b-S4.2: Implement with mock data + adapter
- [ ] 7b-S4.3: Tests

#### 7b-S5: Web Traffic & App Intelligence (SimilarWeb / Sensor Tower) 💰 L1
- [ ] 7b-S5.1: Design interface — modes: `web_traffic`, `app_downloads`, `app_revenue`, `engagement_trend`
- [ ] 7b-S5.2: Implement with mock data + adapter
- [ ] 7b-S5.3: Tests

#### 7b-S6: ESG & Emissions Data (CDP / Trucost) 💰 L0-L1
- [ ] 7b-S6.1: Design interface — modes: `carbon_emissions`, `water_usage`, `waste_generation`, `resource_intensity`
- [ ] 7b-S6.2: Implement with mock data + adapter (also investigate CDP open data, EU ETS registry which is free)
- [ ] 7b-S6.3: Tests

#### 7b-S7: Corporate Jet Tracking (FlightAware + FAA) 💰 L0
- [ ] 7b-S7.1: Build N-number → company ownership mapper using FAA registry (free)
- [ ] 7b-S7.2: Design interface — modes: `executive_flights`, `convergence_detection`, `anomalous_destinations`
- [ ] 7b-S7.3: Tests

---

### Tier 2 — MIXED DATA (useful with care)

#### 7b-MX1: Polymarket Order Book Depth ⚠️ MIXED (L1-2, real liquidity but spoofable)
- [ ] 7b-MX1.1: Add `book` mode to PolymarketTool — bid/ask depth, imbalance signals
- [ ] 7b-MX1.2: Tests

#### 7b-MX2: Social Sentiment / Dark Web Monitoring ⚠️ MIXED (L1-2, noisy + bot-polluted)
- [ ] 7b-MX2.1: Research — Reddit API, Have I Been Pwned (breach detection, free), Telegram Bot API
- [ ] 7b-MX2.2: Implement — breach/ransomware targeting detection, Reddit ticker volume surges, Telegram crypto pump groups
- [ ] 7b-MX2.3: Tests

---

### Tier 3 — CONSEQUENCE DATA (optional context only, NOT predictive features)

#### 7b-CQ1: Treasury/Sovereign Yield Curve ❌ CONSEQUENCE — optional World Model context
#### 7b-CQ2: Credit Spreads ❌ CONSEQUENCE — optional World Model context
#### ~~VIX Term Structure~~ — DROPPED (famously manipulated, pure L3 derivative)

---

### Source Count Summary

| Category | Count | Status |
|---|---|---|
| **Original Tier 1 (A-N)** — globalized | 13 active + 1 deprioritized (B) | 1 complete (A), 12 to build |
| **New "Boring Gold" (O-W)** — global | 9 | All to build |
| **Cross-Country Only (Y-AD)** — global | 6 | All to build |
| **Gap-Closing Deep Coverage (AE-AN)** — global | 10 | All to build |
| **Paid/Subscription (S1-S7)** | 7 | All to architect |
| **Mixed (MX1-MX2)** | 2 | All to build |
| **Consequence (CQ1-CQ2)** | 2 | Deferred |
| **Existing tools to globalize** | 8 | Parallel track |
| **TOTAL UNIQUE SURVEILLANCE SOURCES** | **~50 free + 7 paid** | |

---

### Build Order (8 batches, maximizing convergence detection power)

**Batch 1 — Fastest to build, highest convergence value: ✅ COMPLETE**
1. **7b-O** (Wikipedia page views) — trivial API, global, instant "someone knows" detector ✅
2. **7b-D** (AIS vessel tracking) — physical L0, global shipping ✅
3. **7b-Q** (Regulatory gazette) — Federal Register, regulatory pipeline ✅

**Batch 2 — Physical reality, cross-country:**
4. **7b-C** (Weather/climate) — NOAA + ECMWF + NASA FIRMS, global
5. **7b-U** (Earthquake proximity) — USGS, already global, infrastructure overlay
6. **7b-R** (Transportation throughput) — TSA + Eurocontrol + rail + ports

**Batch 3 — Financial stress, legal commitment:**
7. **7b-T** (Bond/sovereign markets) — EMMA + global sovereign yields
8. **7b-Z** (Central bank balance sheets) — global liquidity index
9. **7b-E** (Bankruptcy/court filings) — global legal reality

**Batch 4 — Boring bureaucratic gold:**
10. **7b-P** (Certificate transparency) — domain registrations, global
11. **7b-S** (FOIA/FOI logs) — investigation detector, multi-country
12. **7b-V** (UCC/creditor filings) — financial stress detector

**Batch 5 — Economic activity + trade:**
13. **7b-Y** (UN Comtrade) — global bilateral trade flows
14. **7b-F** (Job postings/hiring) — global hiring intent
15. **7b-G** (Government contracts) — global procurement
16. **7b-AB** (Building permits) — global construction cycle
17. **7b-AC** (Cross-border capital flows) — money movement

**Batch 6 — Forward-looking + specialized:**
18. **7b-H** (Patents/trademarks) — global innovation pipeline
19. **7b-J** (Lobbying) — global regulatory anticipation
20. **7b-K** (Interconnection queues) — energy build pipeline
21. **7b-L** (DeFi on-chain) — full on-chain economy
22. **7b-M** (Academic preprints) — global research pipeline
23. **7b-N** (FCC/spectrum) — global telecom
24. **7b-W** (Drug/medical regulatory) — global pharma safety
25. **7b-AA** (Global PMI) — synchronized slowdown detection
26. **7b-AD** (Electricity × nightlights) — ground truth economic activity

**Batch 7 — Biological, geopolitical, food security (closes critical blind spots):**
27. **7b-AE** (Disease/pandemic surveillance) — ProMED, wastewater, GISAID, WHO — biology is L0
28. **7b-AF** (Sanctions & export controls) — OFAC SDN, EU, UN, BIS Entity List — geopolitical weapon system
29. **7b-AG** (Agricultural & food security) — USDA WASDE, FAO, NASA SMAP soil moisture, locust watch
30. **7b-AH** (Elections & political risk) — FEC, ACLED armed conflict, Meta Ad Library, V-Dem
31. **7b-AI** (Internet infrastructure) — IODA, BGP routing, Cloudflare Radar — digital = physical

**Batch 8 — Economic fundamentals, real-time fiscal, human movement:**
32. **7b-AJ** (Labor disruptions & strikes) — BLS, ILO, ETUI — committed collective action

### Raw-First Priority Overlay (do not remove commoditized tools)

**Rule:** Keep the commoditized / institutional tools in backlog for breadth, but prioritize raw, harder-to-arbitrage observation layers first.

**Immediate raw-first additions / priority lift:**
1. **7b-I** (Satellite-derived physical activity) — raw sensor layer, global, high moat
2. **7b-AD** (Electricity × nighttime lights) — cross-validates two physical signals
3. **7b-K** (Utility interconnection queue) — committed capex / future load and generation
4. **7b-AI** (Internet infrastructure & digital outages) — physical network state, outage and routing reality
5. **7b-AJ** (Labor disruptions & strike activity) — direct production impairment, not sentiment
6. **7b-AK** (Migration & refugee flows) — physical human movement, under-modeled macro stress signal
7. **7b-AO** (Commerce inventory & hardware marketplace signals) — supply bottlenecks via observable availability / lead times
8. **7b-AG** (Agricultural & food security) — strongest when anchored on soil moisture / NDVI / locust / crop stress, not just reports
9. **7b-AN** (Government tax receipts & fiscal data) — strongest when anchored on daily receipt flows, especially withholding tax
10. **7b-AL** (Energy supply side) — strongest when anchored on rig counts, outages, refinery utilization, field-level production

**Keep in scope, but lower priority because more commoditized / interpreted by default:**
1. **7b-AA** (Global PMI / leading indicators)
2. **7b-AM** (Consumer confidence / sentiment)
3. **7b-W** (Drug / medical regulatory)

**Implementation note:** For mixed tools, bias toward the rawest layer available. Example: prefer wastewater over official outbreak summaries, daily tax receipts over monthly fiscal summaries, and satellite / grid measurements over headline survey products.
33. **7b-AK** (Migration & refugee flows) — UNHCR, remittances, IOM, Frontex — human physics
34. **7b-AL** (Energy supply side) — Baker Hughes rigs, EIA petroleum, IAEA nuclear — supply chain origin
35. **7b-AM** (Consumer confidence surveys) — UMichigan, EU, Japan, India, China — direct measurement of intent
36. **7b-AN** (Government tax receipts) — US Treasury Daily, HMRC, India GST — most honest economic data

**Parallel track:** Globalize existing tools + architect paid-tier interfaces (S1-S7)

**After 7b:** Phase 7c — Convergence Detection Layer (signal taxonomy, evidence tagging, coincidence detector, causal chain templates). Every source above feeds evidence into the convergence engine. Must be hypersensitive to weak-signal coincidences — a single wastewater sample + a shipping route deviation + an unusual sanctions addition = early pandemic/geopolitical signal before anyone else sees it.

## Phase 5: Observational Surface — Atomic Steps

**Goal:** Two zero-cost data tools — Polymarket prediction market snapshot + SEC insider filing cluster detection. Informed money leaves traces in public data. We harvest them.

**Two sub-phases:** 5a (Polymarket) → 5c (Insider Filings) → 5.7 (Integration with bandit arms)

### Sub-phase 5a: PolymarketTool

- [x] 5a.1: Create `agent/tools/polymarket.py` skeleton — class, params schema, `to_openai_tool()` test
- [x] 5a.2: Implement `_fetch_events()` — Gamma API GET, httpx, cache, mock test
- [x] 5a.3: Implement `_parse_markets()` — extract prices/volume/category from nested JSON, fixture test
- [x] 5a.4: Implement `execute()` — cache check, fetch, parse, filter, format output, error handling test
- [x] 5a.5: Register PolymarketTool in `cli.py`, verify in registry

### Sub-phase 5c: InsiderFilingsTool

- [x] 5c.1: Create `agent/tools/insider_filings.py` skeleton — class, params schema, test
- [x] 5c.2: Implement `_fetch_recent_filings()` — EFTS API, pagination, rate limiting, User-Agent, mock test
- [x] 5c.3: Implement `_fetch_filing_detail()` — Form 4 XML parsing, transaction extraction, fixture test
- [x] 5c.4: Implement `_detect_clusters()` — group by company, sliding 14-day window, conviction scoring, synthetic test
- [x] 5c.5: Implement `execute()` — full pipeline, mock test
- [x] 5c.6: Register InsiderFilingsTool in `cli.py`

### Integration

- [x] 5.7: Add `prediction_markets` and `insider_flow` arms to bandit DEFAULT_ARMS
- [x] 5.8: Update task file, mark Phase 5 complete

## Phase 2: Global Liquidity Regime Detection — Atomic Steps

**Goal:** Build a global liquidity composite from central bank data, detect regime changes with real math (BOCPD, HMM, spectral), and validate against asset returns. RenTech-grade, not toy.

**Three layers:** Data (construct composite) → Math (detect regimes) → Validation (prove it works).

### Research & Spec (no code)

- [x] 2.1: Write `[[liquidity_regime_detection]]` — data series survey (WALCL, WTREGEN, RRPONTSYD, M2SL, ECBASSETSW, JPNASSETS, BOEBSTAUKA). Confirmed all on FRED. PBOC excluded (no reliable free source)
- [x] 2.2: Write `[[liquidity_regime_detection]]` — math survey: BOCPD (implement from scratch), HMM (use hmmlearn), spectral (use scipy). New deps: scipy>=1.11, hmmlearn>=0.3, matplotlib>=3.7
- [x] 2.3: Write `[[liquidity_regime_detection_spec]]` — 22 atomic implementation steps, files, edge cases, testing plan

### Data Layer: Build the Global Liquidity Composite

- [x] 2.4: Create `agent/quant/__init__.py` — new package for math/quant primitives (empty init)
- [x] 2.5: Create `agent/quant/liquidity.py` — skeleton class `LiquidityComposite`
- [x] 2.6: Implement `fetch_us()` — pull the core US series via MacroDataTool/FRED: WALCL, WTREGEN, RRPONTSYD, M2SL
- [x] 2.7: Test `fetch_us()` — all 4 series validated, correct date ranges and magnitudes
- [x] 2.8: Implement `compute()` — weekly alignment, net liquidity formula, first-diff → 52-week rolling z-score. 836 obs.
- [x] 2.9: Visual validation — composite overlaid with SPY price, COVID/QE/QT periods align. Plot: `docs/research/us_composite_vs_spy.png`
- [x] 2.10: Implement `fetch_global()` — ECB + BOJ (BOE excluded: BOEBSTAUKA is annual % GDP, UKASSETS discontinued 2014). FX conversion to USD.
- [x] 2.11: Global composite validated — correlation with US-only ρ=0.517 (spec: >0.5, <1.0) ✅

### Math Layer: Regime Detection Primitives

- [x] 2.12: Implement BOCPD in `agent/quant/changepoint.py` — Normal-Inverse-Gamma model, constant hazard, run-length truncation. Uses `scipy.special.gammaln` (numpy 2.x compat).
- [x] 2.13: BOCPD tested on 3 synthetic datasets — single CP ✅, two CPs ✅, zero false positives ✅. Detection via expected-run-length drops (not raw P(r=0)).
- [x] 2.14: Implement RegimeHMM in `agent/quant/regime.py` — wraps hmmlearn, relabels by ascending mean
- [x] 2.15: HMM tested on synthetic 3-state data — means within 0.06, 100% accuracy ✅
- [x] 2.16: Implement spectral analysis in `agent/quant/spectral.py` — custom CWT (scipy.signal.cwt removed in scipy 1.12+)
- [x] 2.17: Spectral tested — FFT finds exact frequencies ✅, CWT localizes in time ✅
- [x] 2.18: BOCPD on real data — λ=200: 3 structural breaks (2014-01, 2020-03, 2023-03). Stable across all lambdas. Plot: `docs/research/bocpd_liquidity.png`
- [x] 2.19: HMM on real data — K=3 wins by BIC (2323 vs 2330). States split by variance, not level (z-scoring artifact). Plot: `docs/research/hmm_regimes_liquidity.png`

### Validation Layer: Prove It Works

- [x] 2.20: Benchmark returns fetched — SPY, TLT, GLD, BTC-USD, DX-Y.NYB. 836 obs aligned to composite W-WED dates.
- [x] 2.21: Regime-conditional returns — no significant t-tests for expansion vs contraction on SPY/TLT/GLD/DX. Key finding: z-scored rate-of-change ≠ bull/bear signal.
- [x] 2.22: Walk-forward backtest — train 2009-2020, test 2021-2024. Buy&Hold SPY: Sharpe 0.78, +66.8%. Avoid-crisis: Sharpe 0.82, +71.0%. Only-neutral: Sharpe 0.38, +26.3%. Plot: `docs/research/walkforward_backtest.png`
- [x] 2.23: Implement `agent/quant/scoring.py` — sharpe_ratio, max_drawdown, information_ratio, hit_rate. All tested.
- [x] 2.24: Strategy scored — results documented in `[[liquidity_regime_results]]`
- [x] 2.25: Registered `LiquidityRegimeTool` in `agent/tools/liquidity_regime.py` and `agent/cli.py`. Tested: returns current regime, z-score, transitions, changepoints.
- [x] 2.26: Task file updated, Phase 2 marked complete.

### Phase 2 Key Findings

1. **BOCPD works well.** Finds real structural breaks at known policy shifts. λ=200 is a good default.
2. **HMM classifies variance regimes**, not directional regimes — because the composite is z-scored first-differences, not levels.
3. **Naive regime→return mapping fails.** "Contraction" state had the best SPY Sharpe (2.63 OOS). The composite measures *rate of policy change*, not market direction.
4. **Tool value is situational awareness**, not direct trading signals. Knowing you're in a liquidity tail event is valuable context for an agent, even if it doesn't translate to simple long/short.
5. **BOE data gap.** No usable FRED series for BOE balance sheet. Proceed with 3-CB (Fed+ECB+BOJ).
6. **scipy/numpy 2.x compat issues.** `np.lgamma` → `scipy.special.gammaln`. `scipy.signal.cwt` removed → custom implementation.

## Phase 3: Scoring & Validation Framework — Atomic Steps

**Goal:** Build a reusable backtesting and scoring engine. Replace Phase 2's ad-hoc validation with composable primitives. Expose to the agent as a tool.

**Three layers:** Metrics (extended scoring) → Engine (walk-forward backtest) → Integration (agent tool).

Research: `[[scoring_validation]]`
Spec: `[[scoring_validation_spec]]`

### Research & Spec (no code)
- [x] 3.1: Write research doc — survey current gaps, math/algorithm choices, scope control
- [x] 3.2: Write spec doc — 24 atomic steps, files affected, edge cases, testing plan

### Extended Scoring Metrics (`agent/quant/scoring.py`)
- [x] 3.3: Add `sortino_ratio()` — downside-deviation denominator. Test: asymmetric returns with known answer.
- [x] 3.4: Add `calmar_ratio()` — ann. return / |max DD|. Test: synthetic drawdown series.
- [x] 3.5: Add `value_at_risk()` + `cvar()` — historical percentile VaR + CVaR. Test: known distribution.
- [x] 3.6: Add `drawdown_duration()` — max consecutive periods below peak. Test: step-function series.
- [x] 3.7: Add `turnover()` — mean absolute weight change. Test: known position series.
- [x] 3.8: Add `score_returns()` — computes all metrics, returns dict. Test: verify all keys present.

### Bootstrap Confidence Intervals
- [x] 3.9: Add `block_bootstrap_ci()` — block bootstrap for any metric. Configurable block length + n_bootstrap.
- [x] 3.10: Test bootstrap CIs — 95% CI covers true Sharpe ≥90% over 100 Monte Carlo runs.

### Strategy Protocol & Walk-Forward Engine (`agent/quant/backtest.py`)
- [x] 3.11: Create `backtest.py` — define `Strategy` ABC: `generate_weights(train_data, test_dates) -> weights`
- [x] 3.12: Implement `WalkForward` class — expanding-window backtester. Input: Strategy + data → per-fold + aggregate results.
- [x] 3.13: Test WalkForward — trivial always-long strategy matches buy-and-hold returns.
- [x] 3.14: Implement `BacktestResult` dataclass — per-fold metrics, aggregate metrics, equity curve.
- [x] 3.15: Implement `RegimeConditionalAnalysis` — per-regime metrics via `score_returns()`.

### Built-in Strategies (re-implement Phase 2 as Strategy objects)
- [x] 3.16: `BuyAndHoldStrategy` — always weight=1.0, baseline.
- [x] 3.17: `RegimeAvoidStrategy` — weight=0 during specified states, weight=1 otherwise.
- [x] 3.18: `RegimeOnlyStrategy` — weight=1 during specified states, weight=0 otherwise.

### Validation: Re-run Phase 2 through the engine
- [x] 3.19: Re-run liquidity-regime backtest — results consistent with Phase 2 (BH Sharpe 0.84, AvoidCrisis 1.20, OnlyNeutral 1.41 over 6 folds, 312 test weeks).
- [x] 3.20: Bootstrap CIs — all three strategies significant at 95%: BH [0.05,1.85], Avoid [0.45,2.02], Only [0.67,2.17].

### Agent Integration
- [x] 3.21: Create `agent/tools/backtest.py` — BacktestTool the agent can invoke.
- [x] 3.22: Register BacktestTool in `agent/cli.py`.
- [x] 3.23: Test end-to-end — agent tool returns scored results.

### Wrap-up
- [x] 3.24: Update task file, mark Phase 3 complete.

### Phase 3 Key Findings
1. **All three strategies statistically significant.** Bootstrap 95% CIs on Sharpe exclude zero for all strategies.
2. **Regime-only (state 1) is strongest.** Sharpe 1.41, max DD -17.5% — the neutral/calm regime captures most of the equity premium.
3. **Regime-avoid (state 2) improves on B&H.** Sharpe 1.20 vs 0.84. Simply avoiding the extreme-vol regime (1% of time) meaningfully reduces drawdowns.
4. **HMM state allocation shifted in OOS.** State 1 (neutral) covers 93% of OOS data — the model is well-calibrated (extreme events are rare, as expected).
5. **SPY date alignment requires W-WED resampling.** yfinance weekly bars use Monday anchoring; composite uses W-WED. Must fetch daily and resample.

## Phase 4: Agent Autonomy — Atomic Steps

**Goal:** Make the agent self-directed: reflect → generate goals → execute → evaluate → learn, in a loop. Replace "human types goal" with "agent decides what to do next."

**Four layers:** Reflection (review past) → Generation (propose next) → Evaluation (score outcome) → Loop (tie it together).

Research: `[[agent_autonomy]]`
Spec: `[[agent_autonomy_spec]]`

### Research & Spec (no code)
- [x] 4.1: Write research doc — architecture gaps, autonomy loop design, guardrails
- [x] 4.2: Write spec doc — 25 atomic steps, files affected, edge cases

### Reflection Engine (`agent/learning/reflection.py`)
- [x] 4.3: Create `Reflector` class — `reflect(episodes, semantic_facts) -> ReflectionResult`
- [x] 4.4: Define `ReflectionResult` dataclass — what_worked, what_failed, open_questions, suggested_next_actions
- [x] 4.5: Test Reflector — cold_start, fallback_reflection, format_episodes all pass

### Goal Generator (`agent/learning/goal_generator.py`)
- [x] 4.6: Create `GoalGenerator` class — `generate(reflection, tools, attempted) -> Goal`
- [x] 4.7: Define `Goal` dataclass — description, rationale, expected_tool, priority, is_novel
- [x] 4.8: Implement goal deduplication — retries up to max_retries, marks is_novel=False if all dupes
- [x] 4.9: Test GoalGenerator — fallback goal, parse with invalid tool fallback, all pass

### Run Evaluator (`agent/learning/evaluator.py`)
- [x] 4.10: Create `Evaluator` class — `evaluate(agent_result, goal) -> Evaluation`
- [x] 4.11: Define `Evaluation` dataclass — success, score, new_facts_count, strategy_metrics, dead_end, lessons
- [x] 4.12: Implement quantitative evaluation — regex extraction of Sharpe/Sortino/MaxDD/Calmar/HitRate from output text
- [x] 4.13: Test Evaluator — quant extraction ✅, heuristic eval (success + dead end) ✅

### Learning Memory Extensions (`agent/memory/store.py`)
- [x] 4.14: Add `LearningEntry` dataclass — goal, score, success, dead_end, lessons, timestamp
- [x] 4.15: Add `store_learning()`, `get_attempted_goals()`, `get_dead_ends()`, `get_learnings()` to SemanticMemory
- [x] 4.16: Test learning memory — in-memory store/retrieve/dedup ✅, disk persistence roundtrip ✅

### Autonomous Loop (`agent/core/autonomous.py`)
- [x] 4.17: Create `AutonomousRunner` class
- [x] 4.18: Implement loop: reflect → generate → orchestrator.run() → evaluate → store → repeat
- [x] 4.19: Add guardrails: max_iterations, goal dedup (via attempted_goals), stuck detection (consecutive_failures >= 3 → pause)
- [x] 4.20: AutonomousRunSummary with full reporting

### CLI Integration
- [x] 4.21: Add `--autonomous` / `-a` flag to CLI
- [x] 4.22: Add `--max-goals` flag (default: 5)
- [x] 4.23: Test: `--help` shows autonomous flags ✅, import chain verified ✅

### Wrap-up
- [x] 4.24: Update `agent/learning/__init__.py` with exports (Reflector, GoalGenerator, Evaluator + dataclasses)
- [x] 4.25: Task file updated, Phase 4 complete

### Phase 4 Key Findings
1. **All 4 learning components import cleanly and pass tests.** 12 unit tests: cold start, fallback paths, metric extraction, persistence roundtrip, CLI flags.
2. **Reflector handles empty history gracefully.** Cold start returns sensible exploratory goals without LLM call.
3. **Evaluator has dual path.** LLM scoring for qualitative assessment + regex extraction for quant metrics (Sharpe, Sortino, etc.) — metrics override LLM score when found.
4. **GoalGenerator deduplicates.** Checks exact match against attempted_goals. Retries up to 3 times. Falls back with is_novel=False.
5. **Autonomous loop guardrails.** max_iterations cap, consecutive failure detection (default: 3 → pause), per-iteration callback for UI.
6. **Learning memory persists.** LearningEntries stored in separate `learnings.jsonl` alongside `semantic.jsonl`. Survives restart.

## Phase 4b: RL Layer — Atomic Steps

**Goal:** Replace Phase 4's LLM-only decision-making with actual RL. A Thompson Sampling bandit learns which goal categories produce the most reward over time. The bandit decides WHAT TYPE of work; the LLM fills in specifics. Parameters update, behavior changes — this is real learning.

**Architecture:** Bandit = brain (strategic decisions, trained by reward). LLM = tongue (language tasks, goal wording, evaluation parsing). Separation means learning survives even if the LLM is swapped.

Research: `[[rl_layer]]`
Spec: `[[rl_layer_spec]]`

### Research & Spec (no code)
- [x] 4b.1: Write research doc — role split (bandit vs LLM), why Thompson Sampling, action space (5 arms), reward design, persistence, risks
- [x] 4b.2: Write spec doc — 17 atomic steps

### Strategy Bandit (`agent/learning/bandit.py`)
- [x] 4b.3: Create `GoalArm` dataclass — name, description, tools, examples
- [x] 4b.4: Define `DEFAULT_ARMS` — 5 categories: backtest_strategy, tune_parameters, explore_asset, fetch_macro_data, research_market
- [x] 4b.5: Create `ArmStats` dataclass — alpha, beta, pulls, total_reward, mean_reward, uncertainty
- [x] 4b.6: Create `StrategyBandit` class — Thompson Sampling with Beta(α,β), choose(), update(arm, reward)
- [x] 4b.7: Implement bandit persistence — save/load α,β,pulls,total_reward to JSON
- [x] 4b.8: Test convergence — 3 arms, known reward rates, 200 pulls: best arm gets 94% of pulls ✅
- [x] 4b.9: Test persistence roundtrip — save, load into new instance, alpha/beta match ✅
- [x] 4b.10: Test edge cases — reward clamping, unknown arm, is_first_pull, get_arm ✅

### Reward Computation (`agent/learning/reward.py`)
- [x] 4b.11: Create `RewardWeights` dataclass — configurable component weights
- [x] 4b.12: Implement `compute_reward()` — 5 components: eval_score, sharpe_quality, knowledge_gain, novelty_bonus, dead_end_penalty. Clamped [0,1].
- [x] 4b.13: Test reward — perfect outcome=0.96, dead end=0.00, moderate=0.32, novelty=+0.1, neg Sharpe=0.12, custom weights ✅

### Goal Generator Extension (`agent/learning/goal_generator.py`)
- [x] 4b.14: Add `generate_for_arm()` — arm-constrained goal generation with dedup
- [x] 4b.15: Add `_ARM_GOAL_PROMPT` — constrains LLM to arm's tools and category
- [x] 4b.16: Add `_arm_fallback_goal()` — uses arm examples when LLM fails
- [x] 4b.17: Test generate_for_arm — valid response ✅, LLM failure fallback ✅, dedup ✅, backward compat ✅

### Autonomous Loop Rewrite (`agent/core/autonomous.py`)
- [x] 4b.18: Wire bandit into loop: reflect → bandit.choose() → generate_for_arm() → execute → evaluate → compute_reward() → bandit.update()
- [x] 4b.19: Update `LoopIteration` — add `arm: GoalArm` and `reward: float` fields
- [x] 4b.20: Update `AutonomousRunSummary` — add `bandit_report: str`, report includes arm names + reward per iteration
- [x] 4b.21: Extend `LearningEntry` — add `arm: str = ""` and `reward: float = 0.0` (backward compatible)

### Integration & Testing
- [x] 4b.22: Update `agent/learning/__init__.py` — export GoalArm, StrategyBandit, DEFAULT_ARMS, ArmStats, RewardWeights, compute_reward
- [x] 4b.23: Update CLI callback — show arm name and reward per iteration
- [x] 4b.24: Integration test — 10-iteration simulated loop: bandit learns correct arm ranking, persistence roundtrip works, all fields serialize ✅
- [x] 4b.25: Task file + checkpoint updated

### Phase 4b Key Findings
1. **Thompson Sampling converges fast.** After 200 pulls on 3-arm test (true rates 0.8/0.5/0.2), best arm gets 94% of pulls. After 10 iterations on real 5-arm config, correct ranking emerges.
2. **Bandit is 30 lines of core logic.** No external ML libraries needed. Beta distribution sampling is in Python's `random` stdlib. Simple = auditable = reliable.
3. **Reward function has 5 components.** eval_score (0.4), sharpe_quality (0.3), knowledge_gain (0.2), novelty_bonus (0.1), dead_end_penalty (-0.3). All configurable via `RewardWeights`. Dead ends clamp to 0.
4. **LLM still handles language tasks.** Goal wording, evaluation parsing, reflection summarization. Bandit doesn't touch text. Clean separation of concerns.
5. **Persistence is simple JSON.** Bandit state survives sessions. New arms auto-initialize with Beta(1,1) → high uncertainty → automatic exploration.
6. **Backward compatible.** `LearningEntry` defaults `arm=""` and `reward=0.0`. Old `generate()` still works. No breaking changes.

---

## THE ARCHITECTURE SHIFT: From LLM-Wrapper to Mathematical Intelligence System

**Phase 5 completed the data pipe era. Phases 6-12 build the real brain.**

The system transitions from "LLM reads data and gives recommendations" to "mathematical models produce probability distributions over future states, RL policy optimizes actions under uncertainty, and the LLM is demoted to a support role for text parsing and hypothesis generation only."

**Core principle: Physics and blockchains can't lie, and they can't delay their disclosures.** A jet either flew or it didn't. A whale wallet either moved BTC or it didn't. A factory consumed electricity or it didn't. These unforgeable real-time traces are primary signals. Human disclosures (Form 4, 13F, congressional trades) are confirmation signals.

**Output is NEVER a recommendation. Output is ALWAYS a probability distribution.**
```
P(FMBM > $35 | 30d) = 0.72, CI: [0.58, 0.83]
Conviction drivers: insider cluster (0.85), sector momentum (0.62)
Recommended position: 2.1% of capital (Kelly-optimal given portfolio correlation)
```

**The full computation stack (bottom to top):**
```
Layer 1: SURVEILLANCE SURFACE (data tools — free APIs, sensors, public traces)
Layer 2: FEATURE ENGINEERING (structured signals — OFI, VPIN, Hurst, transfer entropy)
Layer 3: WORLD MODEL (Bayesian network — causal graph, belief propagation)
Layer 4: SIGNAL FUSION (Kalman/particle filter — fuse noisy multi-source into state beliefs)
Layer 5: RL POLICY + PORTFOLIO OPTIMIZER (model-based RL, Kelly, Black-Litterman, robust opt)
Layer 6: ADVERSARIAL LAYER (manipulation detection, edge decay, game theory)
Layer 7: LLM (support only — unstructured→structured, hypothesis generation, narrative)
```

---

## Phase 6: Extended Surveillance Surface

**Goal:** Triple the data pipe count. Prioritize real-time behavioral/physical traces over stale disclosures. Every new tool must output the same `Signal` struct so the math layers can ingest uniformly.

**Principle: The more real-time the source, the higher the priority.** A corporate jet landing at an investment bank is real-time. A 13F filing is 45 days stale. Build the telescope before hiring the astronomer.

### 6a: GDELT (Global Event Database) — The Poor Man's Palantir
- [x] 6a.1: Research GDELT BigQuery/API — event taxonomy (300+ CAMEO codes), GKG, 15-min updates. Findings: raw event files (no auth, no rate limit, ~1200 events/15min, 61 columns), DOC API (article search, rate limited), BigQuery (deferred). Two-mode tool design: `events` (raw files, primary) + `articles` (DOC API, secondary). See `[[gdelt]]`.
- [ ] 6a.2: Create `agent/tools/gdelt.py` — GDELTTool fetching structured geopolitical events (conflicts, diplomacy, protests, embargoes, sanctions). Every event is geocoded and timestamped. This is what Palantir charges governments millions for — free via BigQuery.
- [ ] 6a.3: Event clustering — detect escalation patterns (Hawkes process on GDELT event streams in a region). When conflict events in a commodity-producing region accelerate, supply disruption probability rises.
- [ ] 6a.4: Register tool, bandit arm `geopolitical_intelligence`

### 6b: CFTC Commitments of Traders — Who's Actually Hedging
- [ ] 6b.1: Research CFTC disaggregated report format (CSV, column mapping, update schedule)
- [ ] 6b.2: Create `agent/tools/cftc_cot.py` — CftcCotTool fetching weekly positioning by commercials vs managed money vs leveraged funds. Commercials are the "insiders" of commodity markets — they hedge their own production and know their own supply/demand reality before anyone else.
- [ ] 6b.3: Positioning divergence detection — when commercials are net long and specs are net short, price reversals statistically follow. The cross-commodity version (crude vs natgas vs grains) is even more powerful because almost nobody computes it.
- [ ] 6b.4: Register tool, add `commodity_positioning` bandit arm

### 6c: Whale Alert — Real-Time On-Chain Money Movement
- [ ] 6c.1: Research Whale Alert API (free tier: 10 req/min, transaction types, chain coverage)
- [ ] 6c.2: Create `agent/tools/whale_alert.py` — WhaleAlertTool tracking large crypto transfers. When someone moves $50M BTC to an exchange, they're about to sell. When they move it off, they're accumulating. Cross-reference with Polymarket crypto markets for double confirmation.
- [ ] 6c.3: Wallet behavior classification — distinguish exchange deposits (likely sells) from cold storage (accumulation) from inter-exchange transfers (arbitrage/noise)
- [ ] 6c.4: Register tool, extend `prediction_markets` arm

### 6d: SEC Form 144 — Intent to Sell (Pre-Trade Signal)
- [x] 6d.1: Research Form 144 on EDGAR EFTS (same API as Form 4, different form type)
- [x] 6d.2: Create `agent/tools/form144.py` — Form144Tool. Two-phase approach: EFTS metadata grouping → selective XML fetch for cluster candidates. Acquisition/urgency/conviction classification.
- [x] 6d.3: Cross-reference with Form 4 cluster data — insider_flow bandit arm now covers both Form 4 and Form 144
- [x] 6d.4: Register tool, extend `insider_flow` arm

### 6e: FINRA Dark Pool + TRACE — Institutional Flow
- [ ] 6e.1: Research FINRA ATS weekly data format + TRACE corporate bond data
- [ ] 6e.2: Create `agent/tools/finra_data.py` — FinraDataTool. Dark pool volume anomalies reveal institutional accumulation/distribution before it shows in lit-market prices. TRACE bond spread widening for a company while equity price hasn't moved = credit traders know something equity traders don't yet. Credit markets are the canary.
- [ ] 6e.3: Anomaly detection — flag when dark pool volume for a ticker exceeds 2σ of its 30-day average, especially if concurrent with insider cluster buying from our Form 4 tool
- [ ] 6e.4: Register tool, add `institutional_flow` bandit arm

### 6f: ADS-B Jet Tracking — Physical Pre-M&A Signal
- [ ] 6f.1: Research ADS-B Exchange API (free tier), FAA aircraft registry (tail numbers → corporate owners)
- [ ] 6f.2: Create `agent/tools/flight_tracker.py` — FlightTrackerTool. Corporate jets to investment banks before acquisitions, pharma jets to FDA before approval decisions, CEO jets to competitor HQ before partnerships. Build a mapping of Fortune 500 tail numbers from FAA registry, then monitor unusual routes.
- [ ] 6f.3: Anomaly scoring — flag routes that differ from historical patterns for a given tail number. A jet that normally flies NYC→SF suddenly flying NYC→Basel (pharma HQ cluster) is a signal.
- [ ] 6f.4: Register tool, add `physical_observables` bandit arm

### 6g: Power Grid Demand — Real-Time Factory Utilization
- [ ] 6g.1: Research ISO/RTO APIs (PJM, ERCOT, CAISO — all free, 5-min interval data)
- [ ] 6g.2: Create `agent/tools/grid_demand.py` — GridDemandTool. Electricity consumption at industrial load zones reveals factory utilization in real-time. Physics can't lie — either the factory consumed megawatts or it didn't. When ERCOT industrial load drops 15%, Texas manufacturing is slowing before any PMI report says so. Semiconductor fab power draw correlates with chip production volumes.
- [ ] 6g.3: Register tool, extend `physical_observables` arm

### 6h: ClinicalTrials.gov — Biotech Catalyst Prediction
- [ ] 6h.1: Research ClinicalTrials.gov API (free, structured JSON)
- [ ] 6h.2: Create `agent/tools/clinical_trials.py` — ClinicalTrialsTool. PDUFA dates are public (everyone knows them). But micro-signals are overlooked: trial status changing from "recruiting" to "active, not recruiting" means enrollment is complete and results are coming. Mid-trial primary endpoint modifications are a bad sign. These status changes happen weeks before any analyst note.
- [ ] 6h.3: Register tool, add `biotech_catalysts` bandit arm

---

## Phase 7: Standardized Signal Protocol + Feature Engineering

**Goal:** Every tool must output the same `Signal` dataclass so all downstream math layers consume a uniform interface. Then build the feature engineering primitives that transform raw data into quantitative features.

**Key insight: Raw data is not a signal. A signal is a structured, timestamped, confidence-weighted observation that the math layers can reason over.**

### Signal Protocol
- [ ] 7.1: Define `Signal` dataclass in `agent/quant/signals.py` — `{ticker, signal_type, value, confidence, timestamp, source, metadata}`. Every tool adapter converts its output into one or more Signals.
- [ ] 7.2: Define `SignalBus` — pub/sub in-memory bus. Tools publish Signals, models subscribe. When a new GDELT event fires, it propagates to the world model immediately. Event-driven, not batch.
- [ ] 7.3: Write adapters for all existing tools (polymarket, insider_filings, market_data, macro_data, liquidity_regime) to emit Signals
- [ ] 7.4: Write adapters for all Phase 6 tools

### Feature Engineering Primitives
- [ ] 7.5: **Order Flow Imbalance (OFI)** in `agent/quant/microstructure.py` — measures aggressive buying vs selling pressure. When 70% of recent volume is buy-initiated, the ask side of the order book is being consumed and price must rise.
- [ ] 7.6: **VPIN (Volume-synchronized Probability of Informed Trading)** — decomposes trade flow into informed vs uninformed components. High VPIN = someone with private info is trading aggressively. This metric predicted the 2010 Flash Crash hours before it happened.
- [ ] 7.7: **Hurst Exponent** in `agent/quant/fractal.py` — measures persistence vs mean-reversion of a time series. H>0.5 = trending, H<0.5 = reverting, H=0.5 = random walk. Determines which model class (momentum vs mean-reversion) to apply.
- [ ] 7.8: **Transfer Entropy** in `agent/quant/information.py` — directional information flow between two time series. Does GDELT conflict intensity actually predict oil price moves, or just correlate? Transfer entropy answers the causal direction question and kills false signals mathematically.
- [ ] 7.9: **Mutual Information** — non-linear dependency measure. Two series can have zero correlation but high mutual information (e.g., U-shaped relationship). Feature selection: keep signals with high MI to target, discard those without.
- [ ] 7.10: **Hawkes Process Intensity** in `agent/quant/hawkes.py` — models self-exciting event dynamics. One insider buy makes the next more likely. One conflict event makes escalation more likely. Hawkes intensity at time t gives you the instantaneous rate of future events — critical for timing decisions.

---

## Phase 8: World Model (Bayesian Network)

**Goal:** Build a causal graph where nodes represent states of the world (geopolitical tension, sector health, liquidity regime, informed flow level, macro cycle) and edges represent causal relationships. Evidence propagates through the graph via belief propagation to produce posterior probability distributions over outcomes.

**This replaces "LLM reads data and opines" with "mathematical model ingests evidence and computes posteriors."**

- [ ] 8.1: Research pgmpy vs pymc vs numpyro for Bayesian network implementation — choose based on inference speed, dynamic graph support, and GPU acceleration
- [ ] 8.2: Define initial causal graph skeleton (~15-20 nodes) in `agent/quant/world_model.py`. Nodes include: `us_liquidity_regime`, `geopolitical_tension_level`, `commodity_supply_disruption_probability`, `sector_rotation_direction`, `informed_trading_intensity`, `credit_stress_level`, `volatility_regime`, `macro_growth_trajectory`, etc. Edges encode domain knowledge: liquidity regime → credit stress → equity vol. Geopolitical tension → commodity supply → inflation expectation.
- [ ] 8.3: Implement evidence injection — when a Signal arrives from the SignalBus, it updates the corresponding node's observed state. Belief propagation computes new posteriors across the whole graph.
- [ ] 8.4: Implement conditional probability tables (CPTs) — initially from historical data, then continuously updated with Bayesian updating as new evidence arrives. The graph structure is domain knowledge; the parameters are learned.
- [ ] 8.5: Add scenario simulation — `world_model.simulate(intervention={node: value})` uses do-calculus to answer counterfactual questions: "If Iran blocks the Strait of Hormuz, what happens to oil → inflation → equity vol?" This is the causal inference layer that separates correlation from causation.
- [ ] 8.6: Test on historical scenarios — backtest the world model against known events (COVID crash, Russia-Ukraine, SVB collapse) to validate that evidence propagation produces reasonable posteriors.

---

## Phase 9: Signal Fusion (Kalman/Particle Filter)

**Goal:** Multiple noisy signals about the same underlying state need to be fused into a single optimal estimate with quantified uncertainty. A Kalman filter (for linear/Gaussian cases) or particle filter (for non-linear/non-Gaussian) does this.

**Example: What is the "true" probability of an oil supply disruption?** GDELT says 0.35, AIS vessel data says 0.28, CFTC commercial positioning implies 0.41, satellite imagery of tanker queues suggests 0.33. The Kalman filter fuses these into a single estimate (maybe 0.34) with a confidence interval, weighted by each source's historical reliability.

- [ ] 9.1: Implement `SignalFuser` in `agent/quant/fusion.py` using filterpy (Kalman) — state vector = hidden true states, observation vector = noisy signals from tools. Process noise = how fast the true state changes. Measurement noise = how accurate each source is (learned from history).
- [ ] 9.2: Implement particle filter variant for non-Gaussian cases — when signals have fat tails or multi-modal distributions, the Kalman assumption fails. Particle filter handles arbitrary distributions via sequential Monte Carlo.
- [ ] 9.3: **Information-theoretic signal quality scoring** — compute mutual information between each signal and the target outcome. Signals with high MI get more weight in fusion. Signals with low MI get downweighted or dropped. This is automatic feature selection that runs continuously.
- [ ] 9.4: **Uncertainty quantification at every step** — every output from the fusion layer is a distribution, not a point estimate. The width of the distribution tells the RL policy HOW CONFIDENT to be, which maps directly to position sizing via Kelly criterion.
- [ ] 9.5: Test on synthetic multi-source data — known true state, 5 noisy sources with different bias/variance profiles, verify the fuser recovers the true state more accurately than any single source

---

## Phase 10: Probabilistic Output Engine

**Goal:** Every system output is a full probability distribution, never a point estimate. This layer runs Monte Carlo simulations through the world model and fusion state to produce outcome distributions, tail risk estimates, and optimal position sizes.

**The output vocabulary changes from "buy FMBM" to "P(FMBM > $35 | 30d) = 0.72 [0.58, 0.83], Kelly-optimal position: 2.1% of capital."**

- [ ] 10.1: **Monte Carlo simulation engine** in `agent/quant/monte_carlo.py` — propagate current state beliefs (from world model + fusion) through stochastic dynamics to produce outcome distributions. 10,000+ scenario paths per asset per query. Parallelized via numpy vectorization or Ray for large portfolios.
- [ ] 10.2: **Copula models** in `agent/quant/copulas.py` — model tail dependencies between assets. Two stocks may be 30% correlated normally but 90% correlated in a crash. Gaussian copula misses this (2008 proved that). Use Clayton/Gumbel copulas for asymmetric tail dependence. Critical for risk management — you need to know that your "diversified" portfolio actually concentrates in a tail event.
- [ ] 10.3: **Hawkes process integration** — feed Hawkes intensity (from Phase 7) into Monte Carlo simulations. Insider buying clusters are self-exciting — one buy makes the next more likely. Model this endogenous feedback rather than assuming independence between events.
- [ ] 10.4: **Kelly criterion position sizing** — given the posterior distribution over returns and the current portfolio correlation structure, compute the Kelly-optimal position for each signal. Fractional Kelly (0.25-0.5) for real deployment to account for model uncertainty.
- [ ] 10.5: **Risk budget allocation** — VaR/CVaR constraints, max drawdown limits, sector exposure caps, correlation-aware sizing. The portfolio optimizer must respect these constraints while maximizing expected risk-adjusted return. Use convex optimization (cvxpy) for tractable solutions.
- [ ] 10.6: Test output pipeline end-to-end — mock world model state → Monte Carlo → distribution → Kelly sizing → risk check → final output format with full uncertainty quantification

---

## Phase 11: RL Policy (Model-Based, Replaces Bandit)

**Goal:** Replace the Thompson Sampling bandit (which only picks goal categories) with a full model-based RL agent that optimizes trading actions. The RL agent uses the world model to simulate thousands of futures, plans within those simulated trajectories, and selects optimal actions.

**This is the difference between "explore which data source is useful" (current bandit) and "given my beliefs about the world, what is the optimal trade to make right now" (real RL policy).**

- [ ] 11.1: Define **state space** — current beliefs (posteriors from world model + fusion), portfolio positions & PnL, capital available, market conditions (vol regime, liquidity), time-of-day/week effects
- [ ] 11.2: Define **action space** — position sizing (continuous: -1 to +1 of max for each instrument), entry timing (now vs wait for better fill), hedging decisions (which hedges to add/remove), information gathering actions (which tools to query next — formalizes the explore/exploit tradeoff)
- [ ] 11.3: Define **reward function** — risk-adjusted returns (Sharpe/Sortino), with penalties for drawdown, excessive turnover, and concentration risk. Reward is realized PnL adjusted for risk, not raw PnL — this prevents the agent from taking reckless bets.
- [ ] 11.4: Implement **model-based planning** in `agent/learning/policy.py` — use the world model (Phase 8) as the environment simulator. For each candidate action, simulate N future trajectories through the world model, compute expected reward across trajectories, pick the action with best risk-adjusted expected outcome. This is the Dreamer/MuZero approach adapted for financial markets.
- [ ] 11.5: Implement **robust optimization** — don't optimize for the most likely future; optimize for the worst case within an uncertainty set. If the world model says P(crash)=0.05, the robust optimizer ensures the portfolio survives that 5% tail. This prevents the classic quant failure mode of "the model said it was safe."
- [ ] 11.6: **Thompson Sampling bandit preserved as exploration heuristic** — the bandit continues to choose which data-gathering goals to pursue (which tools to run, what to investigate). The RL policy decides what to DO with the information. Two levels of decision-making: strategic exploration (bandit) + tactical execution (RL policy).
- [ ] 11.7: Test on historical replay — feed historical data through the pipeline, verify the RL policy produces reasonable actions at each timestep, compare against simple baselines (buy & hold, regime-conditional)

---

## Phase 12: Adversarial Intelligence Layer

**Goal:** The system must reason about other players in the market. Detect manipulation, model counterparty behavior, estimate information asymmetry, and monitor whether our own edge is decaying as others discover the same signals.

**Mindset: Assume smart adversaries. If you can see a signal, eventually someone else will too. The question is: how long until your edge decays, and can you detect signals that others are actively trying to hide?**

- [ ] 12.1: **Spoofing detection** in `agent/quant/adversarial.py` — large orders placed and cancelled within milliseconds to manipulate the book. Detectable via order-to-trade ratio anomalies. When we detect spoofing on the bid side, the "support" is fake and price will drop once the spoof is pulled.
- [ ] 12.2: **Stop hunting detection** — price pushed through a known stop-loss cluster then immediately reversed. Detectable via the liquidity cascade model (Phase 9) + reversal timing analysis. When we detect stop hunting, the reversal is the trade.
- [ ] 12.3: **Pump and dump detection** — social media velocity (Reddit/Twitter mention spike) + micro-cap volume surge + insider selling timing. When all three align, it's a pump and the dump is coming. Stay away or short it.
- [ ] 12.4: **PIN model (Probability of Informed Trading)** in `agent/quant/microstructure.py` — decompose order flow into informed vs uninformed components using the Easley-Kiefer-O'Hara model. High PIN = someone with private info is trading. Cross-reference with our insider filing clusters — if PIN is high AND Form 4 cluster buying is happening, the informed traders are definitely insiders.
- [ ] 12.5: **Edge decay monitoring** — track the Sharpe ratio of each signal over rolling windows. When a signal's Sharpe degrades, it's being arbitraged away by other participants who discovered the same pattern. The system should reduce weight on decaying signals and increase investment in discovering new ones.
- [ ] 12.6: **Game-theoretic counterparty modeling** — for each detected pattern, estimate: how many other participants are likely trading this? What is the crowding risk? What happens if everyone exits simultaneously? Model the unwind scenario via agent-based simulation. Factor crowding risk into position sizing.
- [ ] 12.7: **Adversarial robustness testing** — can the system's signals be reverse-engineered by an adversary observing our trades? If so, the adversary can front-run us. Design signals that are difficult to reverse-engineer (e.g., multi-source fusion signals that require access to the same data combination).
- [ ] 12.8: Test on known historical manipulation events — Flash Crash 2010, GameStop 2021, LUNA/UST 2022. Verify the adversarial layer would have detected the anomaly in real-time.

---

## Revised Roadmap — Approved 2026-04-22

**Why revised:** Phase 40 (Real Data Model Refresh) is DATA-GATED, not code-gated. The 27-node DAG only
became fully populated on 2026-04-22. Running Phase 40 against 3-day-old data produces meaningless
walk-forward results and junk GNN embeddings. The 4–6 week data accumulation window must be used
productively — not idled waiting for GNN guidance that can't exist until Phase 40 is done.

### Tier 1 — Immediate (this week)

1. **Phase 45.1 — Fix 9 trivially-broken pre-existing test failures** (✓ DONE 2026-04-22)
2. **Phase 45.2 — cert_transparency + dns_monitor DAG wiring** (✓ DONE 2026-04-22)
   - `FINANCIAL_DOMAINS`: 20 domains (10 banks/funds, 3 regulators, 3 exchanges, 4 infra)
   - `fetch_cert_domains`: callable operator iterating per-domain (CT log recent mode, 30d lookback)
   - `fetch_dns_domains`: `dns_monitor` string operator, `bulk_resolve` mode, 20 domains
   - DAG: 27 → 29 nodes. 9 new tests (67/67 pass). Regression: baseline unchanged.
3. **Phase 26** (MCP agent upgrade) and **Phase 38** (DB architecture) — active task files exist, non-blocking background work

### Tier 2 — Data accumulation window (next 2–4 weeks)

3. **cert/dns domain strategy** — ✓ DONE (Phase 45.2). `FINANCIAL_DOMAINS` + 2 DAG nodes wired.
4. **Phase 46 — Living System: Online GNN with EWC** ⭐ HIGH PRIORITY — implement during accumulation window so the system is ready to learn continuously the moment Phase 40 real data arrives.
   - Add incremental gradient update step to `HeteroMemory.update_memory_from_events()` after each batch of ≥100 new observations
   - Add EWC regularisation: $L_{total} = L_{new} + \lambda \sum_i F_i(\theta_i - \theta_i^*)^2$ where $F_i$ is the Fisher diagonal
   - Fisher computed once after each full retrain; stored alongside model weights (~1.7 MB overhead)
   - Compute cost: 1 gradient step per 100 obs, <1 second on CPU, $0/month
   - Prevents catastrophic forgetting when market regime changes
   - Research: [[living_system_online_gnn]]. Spec: [[living_system_online_gnn_spec]].
5. **Audit remaining 24 unwired tools** — categorise each as:
   - (a) L1 aggregate (global-conditioning, skip entity wiring) — expected ~3
   - (b) L2-ready, wire now — expected ~15–18
   - (c) Needs research/config before wiring — expected ~3–5
   One focused session produces a prioritised list.
6. **Wire top 5–8 from audit** based on entity type coverage gaps visible by inspection — vessel tools,
   additional country tools, org tools, etc. Do NOT wait for GNN guidance; the GNN has no real data yet.

### Phase 47 Backfill Final State (2026-04-24)

**Density audit result after full Group A + Group B backfill + 10-year extended backfill:** FAIL (4 sparse types — supplementary only, override below)

DB stats: 977,863 obs | 2,450 entities  (up from 77,421 after GDELT 10-year backfill 2026-04-24)

Core GNN entity types — all OK:
- instrument: 89 entities, 69,424 obs, 780 obs/ent, 1099d span ✓
- topic: 1,235 entities, 5,394 obs, 4.4 obs/ent, 639d span ✓
- country: 233 entities, 901,766 obs, 3870 obs/ent, 38830d span ✓  ← GDELT 10yr backfill (+900K obs)
- person: 459 entities, 731 obs, 1.6 obs/ent, 1053d span ✓
- cftc_contract: 20 entities, 300 obs, 15.0 obs/ent, 4116d span ✓  ← CFTC 2011–2019 added

Supplementary entity types — sparse (density audit flags these, override documented below):
- wallet: 33 entities, 122 obs, 3.7 obs/ent, 3d span — whale_alert has no historical loop
- protocol: 21 entities, 60 obs, 2.9 obs/ent, 3d span — defi_flows snapshot-only
- company: 25 entities, 50 obs, 2.0 obs/ent, 19754d span — drug_regulatory ~25 companies, no loop
- organization: 8 entities, 16 obs, 2.0 obs/ent, 0d span — regulatory_gazette single-day

**Phase 40 override (2026-04-24):** The 4 sparse types are supplementary cross-domain entity types
not yet used as GNN training nodes (wallet, protocol, company, organization together total <250 obs
vs 77K+ total). The 5 core GNN entity types (instrument, topic, country, person, cftc_contract)
all pass density gate. Phase 40 GNN training does not depend on wallet/protocol/company/organization
density because the current GNN architecture uses instrument+country+topic as the primary entity
types for the world model. The sparse types will be addressed in a targeted post-Phase-40 backfill
pass or via GNN attention diagnostics identifying which cross-domain edges most improve signal.
Proceeding to Phase 40 with this override is valid — the missing density is supplementary, not core.

### Tier 3 — Phase 40: Real Data Model Refresh (target: mid-May 2026 at earliest)

⚠️ **DO NOT START PHASE 40 BEFORE THE DAG HAS RUN FOR 3–4 WEEKS MINIMUM.**

6. Connect real surveillance observations + real price series into GNN training pipeline
7. Run walk-forward backtest against real entity observations (not synthetic)
8. First real signal-vs-noise read — this is the first meaningful validation of the entire stack

### Tier 4 — After Phase 40

9. **GNN attention diagnostic** — now has real meaning. Use attention weights to guide remaining tool wiring.
10. **Paper trade launch** — first real capital-at-risk test.

### Post-Phase-40 Debug Protocol (if results are below bar)

**Decision (2026-04-22):** Do NOT make changes to any layer before Phase 40 runs. We have no real
signal-vs-noise read yet. Everything trains on synthetic data. We cannot tell good from bad until
Phase 40 produces a walk-forward result against real observations.

**If Phase 40 results are weak, the debugging sequence is:**

1. **Isolate which layer is the failure point.** Check each layer independently before assuming
   the problem is cross-layer:
   - L1 (Surveillance): Are observations actually accumulating? Check entity counts, obs counts,
     tool success rates in the pipeline store. If data is sparse/malformed, fix ingestion first.
   - L2 (Features): Are BOCPD/HMM/spectral features producing meaningful signal distributions?
     Inspect feature histograms — degenerate outputs (all-zero, constant, NaN) indicate a
     feature engineering failure, not a model failure.
   - L3 (GNN): Are entity embeddings differentiating between entity types and states?
     Check attention weights — if all heads are uniform, the GNN is not learning structure.
     Run the GNN attention diagnostic (Phase 24b.5 equivalent on real data) to rank starved nodes.
   - L4 (Fusion): Is the Kalman filter tracking real state or diverging? Check residual
     autocorrelation — white noise residuals = good filter; correlated residuals = model mismatch.
   - L5 (RL Policy): Is SAC receiving non-degenerate reward signal? If reward variance is near
     zero, the signal hasn't reached the policy layer — fix upstream before tuning RL.
   - L6 (Adversarial): Are edge decay rates reasonable? High decay across all entities may
     indicate data sparsity, not true signal decay.

2. **Identify failing layer combinations.** Some failures are interaction failures, not single-layer
   failures. Common patterns:
   - L1 sparse → L3 isolated nodes → L4 no signal to fuse → L5 random policy (fix: wire more tools)
   - L2 degenerate features → L3 learns nothing → L4 noise amplification (fix: debug feature builders)
   - L3 good embeddings but L4 mismatched noise model → poor fusion (fix: re-tune process/obs noise)

3. **Extend intelligence in the failing layer only.** Do not touch layers that are performing.
   The layer boundary is the blast radius limit. One layer at a time, one change at a time.

4. **Re-run Phase 40 walk-forward after each fix.** The walk-forward result is the ground truth.
   Do not judge a fix by unit tests alone — judge it by whether the backtest IC improves.

**What "below bar" means for Phase 40:**
- Information Coefficient (IC) < 0.03 across all folds: essentially no predictive signal
- IC > 0.05 on ≥3 of 5 folds: weak but real signal — investigate which entity types contribute
- IC > 0.10 on ≥3 folds: meaningful signal — proceed to paper trade with small size
- IC variance > 0.15 across folds: signal is unstable — likely L1 data sparsity or L3 overfitting

---

## Phase Architecture (all 7 layers complete as of Phase 44)

```
Phases 0–44 ── ALL COMPLETE
    │
    ├── L1 Surveillance: 51 tools built, 27 wired in DAG (26 unwired)
    ├── L2 Feature Engineering: BOCPD, HMM, FFT+CWT, scoring ── COMPLETE
    ├── L3 World Model: HetTGN GNN ── COMPLETE
    ├── L4 Signal Fusion: Kalman/particle filter ── COMPLETE
    ├── L5 RL Policy: Thompson Sampling + SAC ── COMPLETE
    ├── L6 Adversarial: edge decay, manipulation detection ── COMPLETE
    └── L7 LLM Support: narration only ── COMPLETE

⚠️  PIPELINE BREAK: Data flows into SQLite from 27 DAG nodes, but NOTHING trains on it.
    GNN, backtester, and RL policy all run on synthetic/mock data.
    Phase 40 closes this break — but needs 3–4 weeks of real data first.
```

**Key metrics (2026-04-22):**
- Data tools built: 51 | Wired in DAG: 27 nodes (25 unique tools) | Unwired: 26 tools
- Tests: 9,685 passing (see [[project_metrics]]) | Pre-existing failures: 27 (9 trivially fixable, 5 need Phase 40, 13 other)
- Full regression baseline confirmed: 27 failed, 9685 passed, 11 skipped

**GNN training status (2026-04-28):**
- Epochs 1–5 trained (1–3 local, 4–5 Kaggle T4). Checkpoint: `.tirra_pipeline/checkpoints/epoch_005.pt`. Active model: `.tirra_pipeline/gnn_model.pt`.
- ✅ `get_memory()` off-by-one FIXED (`het_tgn.py`): clamp + zero-fill for out-of-range global_ids after entity count grew from 2450→2451.
- ✅ `_contrastive_loss()` FIXED (`trainer.py`): embeddings were not L2-normalised before distance computation. Embedding magnitudes driven by MSE losses grew to ~166K scale, making margin=1.0 permanently inactive. Fix: normalise embeddings to unit sphere (`F.normalize`) before computing pairwise distances + use `random.sample` for negative indices. Verified: contrastive_loss now = 0.874 (was 0.0).
- Loss at epoch 5: total=287.48, obs_type=5.46, time_delta=126.31, contrastive=0.0 (was broken), value=918.07
- Attention: 5 STARVED edges (country↔country, topic→instrument, exchange_country, wallet→instrument, market_authorized_in). Fix via data depth, not model changes.

**GNN training status (2026-05-26) — V16 READY:**
- Kaggle v15 ran epochs 28–40 (~12hr, cancelled at limit). Best checkpoint: `epoch_040.pt` at `/tmp/hg_v15_out/tirramind_v1/.tirra_pipeline/checkpoints/h_g/epoch_040.pt`
- IC at epoch_040 = −0.033 (GNN-EmbNorm), −0.0165 (GNN-ValueHead). WEAK, NOT SIGNIFICANT.
- ✅ EWC sidecar fix committed (`5ba8b2c`): `ewc_state.pt` written after Fisher computation, loaded on resume. Eliminates 5-epoch loss spikes in multi-block training.
- ✅ All 16 EWC tests pass: `PYTHONPATH=. ~/.local/bin/pytest tests/test_ewc.py`
- ✅ `scripts/kaggle_watch.py` live terminal dashboard created.
- ✅ Deep architecture review complete: `[[architecture_review_2026]]` — 12-section analysis, 8 risks ranked, all core components validated with sources.
- ✅ Fix time_delta NaN — done 2026-05-27 (see below).
- ⬜ **NEXT: V34 Kaggle push** — `--resume 40 --epochs 50`, `use_listnet_return_loss=True`, target IC → +0.03–+0.07.
- ⬜ Complete Phase 17 entity linking before Phase 40 final evaluation run.

**GNN training status (2026-05-27) — PRE-V34 FIXES COMPLETE:**
- ✅ `time_delta NaN` fixed: `torch.isfinite()` guard added in both the main training path and EWC `_loss_from_window` in `agent/models/gnn/trainer.py`.
- ✅ `xsnorm_price_feats` added: cross-sectional z-score normalisation of price feature block [14:23] in `graph_builder.py`. Applied consistently in `trainer.py`, `ic_check.py`, `quant_benchmark.py`.
- ✅ `freeze_backbone` flag: `TrainerConfig.freeze_backbone` + separate raw-head optimizer/clip in `trainer.py`; `--freeze-backbone` CLI in `retrain_gnn.py`.
- ✅ `return_raw_head` corrected eval: ICIR = **+0.467** with 26-week history. Trained checkpoint: `epoch_021_rawhead.pt`. This is a factor-model floor, independent of GNN quality.
- ✅ Repo cleanup: 1,149 MagicMock test-DB files removed; 4 old notebooks removed; 417 MB stale nested copy removed; old upload zips removed; logs moved to `logs/`.
- ✅ `.gitignore` updated: MagicMock patterns, root `*.log`, old upload zips, nested `tirramind_v1/` all blocked.
- ⬜ **NEXT: V34 push** — stage ep40 → upload tirramind-data → re-upload tirramind-code → update kernel-metadata.json → push. See `[[kaggle_runbook]]` checklist.
- Known pytest workaround: `PYTHONPATH=. ~/.local/bin/pytest` (3.10.12); system `python3.11 -m pytest` BROKEN.

**All phases are $0.** Every library is open source. Every data source is a free public API.

## Related

- [[project_memory]]
- [[tool_priority_ranking]] — Phase 16 diagnostic-driven priority order
- [[math_stack_roadmap]] — Full applied math stack (SDE, options, Greeks, IV surface, rough vol, CVaR, RL hedging, microstructure, information theory)
- [[l2_tool_expansion]] — Phase 13 L2 expansion task
- [[gnn_guided_tool_expansion]] — Phase 16 research
- [[chat_checkpoint_2026-04-10_session3]] — latest checkpoint

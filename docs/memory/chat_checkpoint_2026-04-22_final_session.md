---
title: "Checkpoint: 2026-04-22 Final Session — Identity, Phase 46/47, Cost Analysis, Architecture Review"
tags:
  - doc/checkpoint
  - phase/46
  - phase/47
  - topic/living-system
  - topic/backfill
  - topic/cost
  - status/done
---

# Checkpoint: 2026-04-22 — Full Session Summary

## 1. Verified System State (ground truth, this session)

| Metric | Value |
|---|---|
| DB size | 24.2 MB |
| Entities | 1,087 |
| Entity observations | 74,030 |
| Entity links | 357 |
| DAG runs completed | 9 |
| Active tools | 59 (51 data + 8 utility) |
| Tests passing | 9,663 |
| GNN parameters | ~220K (HetTGN, hidden_dim=64) |
| SAC hidden_dim | 128, 2 layers |
| World model | pgmpy Bayesian DAG + Kalman filter (hybrid) |
| DAG nodes | 29 |
| Unwired tools | 24 of 51 (47.1%) |

---

## 2. Identity — reframed this session

**TirraMind is a predictive AI company. Not a quant fund. Not an information-arbitrage firm.**

The mission: build the most capable real-world prediction system ever constructed. The product is the prediction engine itself — probability distributions over what will happen next, across every domain, every country.

The edge: **unconventional observation × SOTA math × living system architecture.**

Files updated:
- `.github/copilot-instructions.md` — opening definition rewritten
- `[[quant_training_ground]]` — Firm identity + Calibration + Model agnosticism sections
- `/memories/repo/tirramind_structure.md` — Identity section rewritten

**New benchmark:** Does this advance the frontier of what a machine can know about reality before humans do?

---

## 3. Strategic decisions locked this session

### 3.1 Model agnosticism doctrine (now in canonical files)

No model is sacred. After Phase 40 backtests, every layer is evaluated on one criterion: genuine out-of-sample predictive edge. If a component doesn't produce real edge, it gets replaced — no sunk cost, no sentiment.

Pre-defined upgrade triggers (do not upgrade before these fire):
- World model DAG → PyMC variational inference: when DAG exceeds 500 nodes
- SAC hidden_dim 128 → 256: when replay buffer saturates / policy plateaus
- GNN full attention → sparse attention: when entity count exceeds 500K
- Add copula tail model: when correlated crisis risk becomes dominant

Until triggers fire: hold the current architecture. Feed it data.

### 3.2 Current architecture is correct for current scale

The system is not underbuilt. It is starved of data:
- pgmpy Bayesian DAG with Kalman hybrid = correct for sparse structured data
- SAC with Kalman-augmented actor + CVaR + entropy regularisation = sophisticated, right-sized
- HetTGN 220K params trained in 41 seconds = appropriate for current entity count

The architecture is built for expert-structured causal inference, not brute-force neural scaling. That is a deliberate advantage: the structure is known, so you don't need millions of samples to learn it.

### 3.3 Phase 47 (Historical Backfill) replaces the 6-week wait

Key insight: 27 of 51 tools support historical date-range parameters. One backfill run with `days_back=1825` (5 years) across all Group A tools transforms the DB from 3 days of observations to years of real history — including COVID shock, supply chain crisis, rate cycle, sanctions waves, geopolitical shifts. The GNN trains on real causal cycles.

**Revised phase sequence:**
```
Phase 46 → Phase 47 → Phase 40
  EWC          backfill     real GNN retrain
  online        all 51       on years of
  learning      tools        genuine history
  ($0, CPU)     (1–2 days)   (41 sec CPU)
```

Phase 40 no longer waits for mid-May. It runs immediately after Phase 47 completes.

---

## 4. Cost to live production — final realistic estimate

### Infrastructure

| Component | Dev now | Month 1 live | Month 6 scaled |
|---|---|---|---|
| Compute (VPS) | $0 (laptop) | $14/mo (Hetzner CX22, 4GB RAM) | $38/mo (Hetzner CPX41, 16GB RAM) |
| Storage | $0 | included | included |
| Domain + SSL | $0 | $15/mo | $15/mo |
| Monitoring | $0 | $0 (BetterUptime free) | $0 |
| Bandwidth | $0 | $0–$5/mo | $5–$10/mo |
| LLM (local Ollama) | $0 | $0 | $0 |
| LLM (Groq free tier) | $0 | $0 | $0 |
| GNN retrain (CPU) | $0 | $0 | $0 |
| GNN retrain (GPU, monthly) | — | $0.60/retrain | $0.60/retrain |
| EWC online updates | $0 | $0 | $0 |
| **Subtotal infrastructure** | **$0** | **$30–$35/mo** | **$59–$64/mo** |

### ML Training

| Training component | Cost | Frequency |
|---|---|---|
| GNN full retrain (CPU, 41 sec) | $0 | Monthly |
| GNN full retrain (Lambda A10 GPU, 5 min) | $0.60 | Monthly |
| EWC online update (<1 sec per batch) | $0 | Per 100 new obs |
| SAC RL update (CPU) | $0 | Per new transitions |
| Bayesian world model update | $0 | Per DAG run |
| Thompson bandit update | $0 | Per tool call |
| **Total training cost** | **<$1/month** | — |

### Optional data subscriptions (only if free data proves insufficient)

| Subscription | What it adds | Monthly |
|---|---|---|
| Polygon.io Starter | Real-time US equity + options flow | $29 |
| Glassnode | On-chain blockchain depth | $39 |
| Unusual Whales | Options flow L2 | $50 |
| MarineTraffic | AIS vessel tracking beyond free | $50 |
| Whale Alert Pro | Crypto movement depth | $149 |
| Diffbot | Corporate entity resolution | $200 |

**None of these are required to launch.** All 51 tools run on free/public APIs. Buy a subscription only after Phase 40 backtests show that specific data type would materially improve signal quality.

### Total cost from now to 1 month of live production

```
Phase 46 + 47 dev (2–3 weeks, laptop):   $0
Phase 40 GNN retrain (1 GPU run):         $0.60
Server setup + domain (one-time):         $15
Month 1 running (server + domain):        $35/mo
─────────────────────────────────────────────────
Total from now to end of month 1 live:    ~$51
```

**Under $55 total to go from current state to a live, self-updating, production predictive AI system.**

The spend does not scale meaningfully until:
- Entity count passes 100K (upgrade server to 16GB RAM: +$24/mo)
- A paid data subscription proves its edge (cheapest useful one: Polygon $29/mo)
- External API serving at volume (add $0 — FastAPI on same box)

---

## 5. What the system already does — verified capabilities

| Layer | Component | Status |
|---|---|---|
| Layer 1 — Surveillance | 51 data tools, 29-node DAG | ✅ LIVE, running daily |
| Layer 1 — Living | HeteroMemory GRU (per-entity state) | ✅ LIVE, updates every observation |
| Layer 2 — Features | 41 ENRICHMENT_DIM, 32 obs types, FeatureBuilder | ✅ LIVE |
| Layer 2 — Signal | BOCPD changepoint, HMM regime, spectral | ✅ LIVE |
| Layer 2 — Convergence | 12 causal chain templates, Fisher/BH FDR | ✅ LIVE |
| Layer 3 — World model | pgmpy Bayesian DAG + Kalman hybrid | ✅ LIVE, updates every DAG run |
| Layer 3 — GNN | HetTGN 220K params, 15 edge types | ✅ BUILT, needs Phase 40 retrain on real data |
| Layer 4 — Fusion | Kalman signal fusion | ✅ LIVE |
| Layer 5 — RL | SAC with Kalman-augmented actor | ✅ LIVE, 276 tests |
| Layer 5 — Bandit | Thompson Sampling, 48 arms | ✅ LIVE, continuous learning |
| Layer 6 — Adversarial | Manipulation detection, edge decay | ✅ LIVE, 148 tests |
| Layer 7 — LLM | Ollama/Groq support, research/synthesis only | ✅ LIVE |
| Memory | Reviewed memory, lesson promotion | ✅ LIVE |

Missing pieces (Phase 46 + 47 address both):
- GNN **weight** evolution between retrains → Phase 46 (EWC)
- GNN trained on real multi-year history → Phase 47 (backfill) → Phase 40

---

## 6. All 51 data tools

```
academic_preprints     ais_vessel             bankruptcy_court
building_permits       capital_flows          central_bank_balance
cert_transparency      cftc                   comtrade
consumer_sentiment     creditor_filings       defi_flows
disease_surveillance   dns_monitor            drug_regulatory
earthquake_proximity   electricity_monitor    energy_supply
finra_short_volume     foia_requests          food_security
form144                gdelt                  global_pmi
gov_contracts          insider_filings        instrument_universe
interconnection_queue  internet_infrastructure internet_outages
job_postings           labor_disruptions      liquidity_regime
lobbying               macro_data             market_data
migration_flows        patent_filings         political_risk
polymarket             polymarket_whales      power_grid
regulatory_gazette     sanctions_monitor      satellite_activity
sovereign_debt         supply_chain_monitor   transport_throughput
treasury_receipts      weather_alerts         whale_alert
wikipedia_pageviews
```

Backfill categories:
- **27 confirmed backfillable** (Group A — call with days_back=1825)
- **2 live-only** (cert_transparency, dns_monitor, internet_outages — skip backfill)
- **~22 partial** (Group B — verify endpoint before backfill)

---

## 7. Next session — what to build

### Step 1: Phase 46 — EWC Online Learning (do first)

**What:** Add continuous GNN weight updates between full retrains using Elastic Weight Consolidation.

**Math:**
$$L_{total} = L_{new\_data} + \lambda \sum_i F_i(\theta_i - \theta_i^*)^2$$

- $F_i$ = Fisher information diagonal (weight importance, computed after each full retrain)
- $\theta_i^*$ = weights at last full retrain
- Prevents catastrophic forgetting when making small gradient steps on new event batches
- Storage overhead: 2 × 860 KB = 1.7 MB. Compute: <1 second per batch, CPU, $0/month

**Files to create first (mandatory preflight):**
1. `[[living_system_online_gnn]]` — EWC theory, Kirkpatrick 2017 anchor
2. `[[living_system_online_gnn_spec]]` — ordered atomic implementation steps

**Files to implement:**
- `agent/models/gnn/het_tgn.py` — add `compute_fisher()` and `online_update()` methods
- `agent/models/gnn/trainer.py` — add `trigger_online_update()` when obs batch ≥ 100
- `tests/test_gnn_online_learning.py` — 1 happy path + 2 failure cases minimum

### Step 2: Phase 47 — Historical Backfill Runner (do second)

**What:** One script that calls all 51 tools with years of lookback, writes timestamped observations to DB, then triggers Phase 40 immediately.

**Files to create first (mandatory preflight):**
1. `[[historical_backfill_spec]]`

**Files to implement:**
- `scripts/backfill.py` — main runner: iterate Group A tools, days_back=1825, resume checkpoint
- Verify timestamp correctness on historical writes (no `now()` for historical records)
- Verify Group B tools one by one

**Then immediately:** Phase 40 — GNN retrain on years of real history. No waiting.

---

## 8. Key file map — everything needed to resume cold

### Canonical planning owners
| File | What it owns |
|---|---|
| `[[quant_training_ground]]` | Roadmap, phase list, firm identity, model agnosticism doctrine |
| `/memories/repo/tirramind_structure.md` | Current metrics, DB counts, phase summaries, Tier 1/2 roadmap |
| `.github/copilot-instructions.md` | Agent workflow, firm identity, architecture priority, model agnosticism |
| `AGENTS.md` | Agent definitions, tool permissions, available prompts |

### Phase 46 (Living System — EWC)
| File | Status |
|---|---|
| `[[living_system_online_gnn]]` | NOT YET CREATED — first thing next session |
| `[[living_system_online_gnn_spec]]` | NOT YET CREATED |
| `agent/models/gnn/het_tgn.py` | Exists — target: add `compute_fisher()`, `online_update()` |
| `agent/models/gnn/trainer.py` | Exists — target: add online update trigger |

### Phase 47 (Historical Backfill)
| File | Status |
|---|---|
| `[[historical_backfill]]` | ✅ CREATED this session |
| `[[historical_backfill_spec]]` | NOT YET CREATED |
| `scripts/backfill.py` | NOT YET CREATED — main implementation |

### Core architecture (do not modify unless upgrade trigger fires)
| File | Layer | Purpose |
|---|---|---|
| `agent/pipeline/dags/daily_collection.py` | L1 | 29-node production DAG, all 51 tools |
| `agent/pipeline/store.py` | L1 | DB read/write, PipelineStore |
| `agent/models/gnn/het_tgn.py` | L3 | HetTGN + HeteroMemory (GRU per entity) |
| `agent/models/world_model.py` | L3 | pgmpy Bayesian DAG + Kalman hybrid |
| `agent/learning/policy/sac.py` | L5 | SAC actor-critic, Kalman-augmented |
| `agent/learning/bandit.py` | L5 | Thompson Sampling, 48 arms |
| `agent/models/gnn/trainer.py` | L3 | Batch GNN trainer (Phase 40 target) |
| `agent/data/graph_builder.py` | L1/L3 | Builds PyG graph from DB for GNN |

---

## 9. How to cold-start next session

```
1. Read this checkpoint (you're reading it)
2. Read [[quant_training_ground]] — current sequence + phases
3. Start Phase 46 preflight:
   a. Create [[living_system_online_gnn]]
   b. Create [[living_system_online_gnn_spec]]
4. Implement Phase 46 (EWC in het_tgn.py + trainer.py)
5. Then Phase 47 preflight → implement scripts/backfill.py
6. Run backfill → run Phase 40 GNN retrain → evaluate edge
```

---

## 10. The one-sentence summary

**TirraMind is a $0/month predictive AI system watching 51 live data sources across every country and domain, with a living entity graph, Bayesian world model, RL policy, and self-updating memory — currently starved of historical data, which Phase 47 fixes in 1–2 days, after which Phase 40 produces the first real trained GNN and the system becomes genuinely production-ready for under $55 total spend.**

## Related

- [[quant_training_ground]]
- [[historical_backfill]]
- [[living_system_online_gnn]]
- [[chat_checkpoint_2026-04-22_identity_phase47]]

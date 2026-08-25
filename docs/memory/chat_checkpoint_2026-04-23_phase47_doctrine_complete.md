---
title: "Checkpoint: Phase 47 Doctrine + Data Strategy Research Complete (2026-04-23)"
tags:
  - doc/checkpoint
  - phase/47
  - topic/backfill
  - topic/data-strategy
  - topic/training-data
  - layer/surveillance
  - status/done
---

# Checkpoint: Phase 47 Doctrine + Data Strategy Research Complete

**Date:** 2026-04-23
**Session focus:** Data strategy doctrine, backfill time analysis, cloud/local deployment, full Phase 47 research synthesis
**Prior checkpoint:** [[chat_checkpoint_2026-04-22_phase45_3_complete]]
**Canonical task:** [[quant_training_ground]]

---

## 1. What was completed this session

### 1a. Phase 46 (confirmed at session start — completed prior session)
- `agent/models/gnn/ewc.py` — `EWCState` dataclass, `compute_fisher()`, `ewc_penalty()` fully implemented
- `TrainerConfig` extended: `ewc_lambda=1000.0`, `online_batch_threshold=100`
- Fisher diagonal computed once after full `train()`, stored alongside model weights
- `Trainer._loss_from_window()` and `Trainer.online_update()` wired
- `gnn_inference.py` DAG step wired to call `online_update()` on each new event batch
- Save/load backward-compatible (no Fisher = pre-EWC checkpoint, gracefully handled)
- 65 regression GNN tests pass + 13 new EWC-specific tests
- $0 compute, CPU-only, <1 second per batch update
- Phase 46 marked COMPLETE in `[[quant_training_ground]]`

### 1b. Phase 49 and 49b research stubs filed (this session, early)
Two research stubs were created so the insights from an architectural review are not lost:

**`[[gnn_downstream_alignment]]`** — Phase 49 stub
- Problem: GNN optimises "predict next observation" but the downstream objective is world-model belief quality. These are not the same objective — embeddings can be self-supervised-optimal but downstream-suboptimal.
- Three candidate approaches documented:
  1. World-model likelihood auxiliary loss (weak signal from belief log-likelihood delta per entity fed back into GNN training as `alignment_weights`)
  2. RL advantage-weighted replay (upweight GNN training examples where downstream policy made high-variance decisions)
  3. Slow-fast dual memory (fast head fine-tuned on recent regime; slow head EWC-protected; world model chooses which to query)
- **Gate: do NOT implement before Phase 40.** There is no real downstream signal until GNN is trained on real data.
- Phase 49 added to `[[quant_training_ground]]` before Phase 48.

**`[[convergence_as_control]]`** — Phase 49b stub
- Problem: convergence detection output (`regime_label`, `changepoint_posterior`) is currently a feature. It should be a control signal.
- Five wiring targets documented:
  1. SAC entropy coefficient — raise in high-changepoint regimes to increase exploration
  2. World model prior decay — soften beliefs when `regime_label` changes (distrust stale CPTs)
  3. GNN retrain trigger — force full retrain when `changepoint_posterior > 0.9`
  4. Feature trust factor — scale ENRICHMENT_DIM by regime stability duration
  5. New shared helper: `agent/pipeline/regime_gate.py` — single entry point all wiring targets call
- **Can run in parallel with Phase 47 — does NOT need Phase 40.**
- Phase 49b added to `[[quant_training_ground]]` before Phase 48 (after Phase 49).

### 1c. Phase 47 spec filed (`[[historical_backfill_spec]]`)
Full implementation spec with verified tool call signatures:
- `insider_filings`: cap=90d → sliding window needed
- `form144`: cap=60d → sliding window needed
- `finra_short_volume`: cap=20d → sliding window needed
- `sanctions_monitor`: cap=365d → sliding window needed
- `cftc`: year-based historical mode → `for year in range(2021, now.year+1): cftc.execute(mode="historical", year=year)`
- `market_data`: `period="5y"` (confirmed valid)
- `macro_data`: `start_date` / `end_date` string params
- All other Group A tools: `days_back=1825`
- Live-only skip list: `whale_alert`, `defi_flows`, `polymarket_whales`, `dns_monitor`, `power_grid`, `interconnection_queue`
- Utility skip list: `base`, `code_executor`, `file_manager`, `shell_runner`, `web_browse`, `web_search`, `backtest`, `pipeline_query`

Two scripts to be created:
- `scripts/backfill.py` — resumable orchestrator (checkpoint JSON, per-tool error isolation, `--dry-run`, `--delay`, `--db`)
- `scripts/density_audit.py` — post-backfill reporter (obs per entity type, FAIL/WARN/OK per type, exit 1 on any FAIL)

Design decision deferred into spec: **capped tools will use sliding-window loops** — requires adding `as_of_date` param to the 4 capped tools (~5 lines each). This is a code prerequisite before the backfill runner is written.

### 1d. MCP stack secured and health-checked (prior session, confirmed)
- `.vscode/mcp.json` updated: top-level `"inputs": [{"type":"promptString","id":"tavily-api-key","password":true}]`
- Tavily env now uses `"TAVILY_API_KEY": "${input:tavily-api-key}"` — key entered at runtime via VS Code prompt, never stored in files or transits the agent
- 7 MCP servers verified healthy: tavily, github, context7, sequential-thinking, git, playwright, memory

### 1e. `[[data_strategy_doctrine]]` — COMPLETE (18 sections, this session)
This is the governing research document for Phase 47. It was synthesised from 10 parallel Tavily research searches plus free `fetch_webpage` pulls. It defines what "enough data" means for every layer of the stack.

Full section index:
1. **Objective** — why data quality dominates model quality; POMDP belief-state contamination argument
2. **Prediction target** — raw observations are primary; prices are derivative; entity-resolved > aggregate
3. **Four backfill dimensions** — Depth, Density, Coverage, Diversity (each fails independently)
4. **Sample complexity per layer** — TGN (2+ business cycles), world model (≥10K obs for CPT), EWC (≤18 tasks before divergence, proto-replay solution), TV-POMDP (MPSE recency weighting, exact timestamps mandatory), HMM (≥30 obs/state, ≥10y for 2 cycles), CMI feature selection (N≥k^d)
5. **Modal diversity + information balance** — 93.8% instrument_daily is a structural crisis; target shares per category; exogenous:endogenous ≥ 2:1 rule (arXiv:2509.05779)
6. **Temporal principles** — timestamp fidelity (`observed_at` = event time not ingestion time), no look-ahead bias (`available_at` field to be added Phase 47b), resolution matching (daily price ≠ quarterly filing, different Kalman process noise)
7. **Depth targets per tool** — verified source availability, backfill target years per tool
8. **Core risks and mitigations** — 10-row table (modal collapse, regime amnesia, survivor bias, timestamp corruption, look-ahead bias, API failure, disk exhaustion, duplicate writes, quota hits, silent source drift)
9. **Invariant: preserve structure > maximize volume** — always choose 100K new-type observations over 10M more price observations
10. **GNN-guided expansion link** — backfill does not upgrade tools to L2; that is post-Phase-40 GNN-diagnostic work
11. **Exit conditions for Phase 47** — 7 criteria: Group A/B complete, density audit pass, modal balance ±5pp, regime coverage (≥2 VIX regimes), timestamp spot-check, ≥2M total observations, per-entity depth meets §7 targets
12. **CMI plateau** — within-source saturation vs cross-source orthogonality; CMI gain <5% → redirect budget; formal bound $N \geq k^d$
13. **Graph structural requirements** — 7 hard minimums: distinct entity types ≥5, node degree ≥3, cross-type edge ratio ≥50%, temporal edge density ≥1/7d, obs per entity type ≥30, entity survival ratio ≥60%, regime span ≥2 per type
14. **System boundary** — explicit out-of-scope: HFT tick data, paid/proprietary data, synthetic data, social fire-hose, sub-hourly streaming, commercial satellite
15. **Statistical integrity** — 6 rules: no imputation in raw observations, unit consistency (USD, decimal rates, UTC), FRED vintage tracking, upsert deduplication, survivorship bias disclosure, timestamp gate
16. **Final principle: the self-updating doctrine** — doctrine becomes historical record after Phase 40; the 30% cap on any single observation category is the one invariant that never relaxes; doctrine must be tested empirically not theoretically

External sources cited (15+ verified references):
- arXiv:2006.10637 (TGN, Rossi 2020)
- arXiv:2510.09416 (Hayes/Schumacher/Strohmaier 2025 — temporal GNN property learning)
- arXiv:2002.07962 (TGAT)
- Kirkpatrick et al. 2017 PNAS (EWC)
- Jones & Sprague 2018 JMU (EWC diverges at ~18 tasks)
- arXiv:2602.09720 (prototype replay for non-stationary continual learning)
- Mornik et al. 2024 PLMO24 (TV-POMDP, MPSE)
- Peng et al. 2005 IEEE PAMI (mRMR)
- Brown et al. 2012 JMLR (CMI unifying framework)
- arXiv:2207.08476 (high-order CMI maximisation)
- Song & Eraker via Wiley JAE (infinite HMM regime detection)
- Wang et al. WSDM 2021 (GraphSMOTE)
- Liu & Fang KDD 2021 (Tail-GNN)
- Qian et al. NeurIPS 2022 (CM-GCL co-modality contrastive)
- arXiv:2509.05779 (exogenous-aware spatio-temporal forecasting)
- Hu et al. WWW 2020 arXiv:2003.01332 (Heterogeneous Graph Transformer — basis for HGTConv)

Data source availability verified:
- SEC EDGAR: 1993-present (efts.sec.gov)
- GDELT v1: 1979+, v2: Feb 2015+ (gdeltproject.org)
- FRED: 840,454+ series (fred.stlouisfed.org)
- CFTC COT: futures-only 1986, F+O 1995, disaggregated 2006 (cftc.gov)
- USPTO PatentsView: 1976+ (patentsview.org)
- UN Comtrade: reliable 2011+ (comtradeplus.un.org)
- USASpending: 2001+, reliable post-FFATA 2008 (usaspending.gov)

### 1f. Backfill time analysis (this session — analytical, no code changes)

Key finding: **the constraint is Comtrade, not compute or network.**

| Phase | Estimated wall-clock |
|---|---|
| Code prerequisite: `as_of_date` patch for 4 capped tools | ~30 min |
| Phase 47a — all tools except Comtrade + GDELT | 4–6 hours |
| Comtrade — aggregate design (15y × 20 reporters = 300 calls @ 100/hr) | +3 hours |
| Comtrade — per-commodity design (15y × 20 × 15 commodities = 4,500 calls) | +45 hours |
| Phase 47b — GDELT (5,475 daily CSV files, separate runner) | 1–3 days |

**Comtrade design decision pending** — must be locked into `[[historical_backfill_spec]]` before writing `scripts/backfill.py`. Recommendation: use aggregate design (300 calls) for the initial backfill; per-commodity can be a Phase 47c extension if GNN diagnostics after Phase 40 show commodity-level structure is needed.

Other constraints:
- Disk: 2M rows × ~200 bytes = ~400 MB SQLite. Fine.
- Memory: tools run serially. Peak ~200 MB (market_data with 100 tickers). Fine.
- No GPU needed at any point through Phase 47. CPU-only throughout.
- The checkpoint/resume design in the spec is **mandatory** — a 7-hour job with no resume is a single-point-of-failure.

### 1g. Cloud vs local analysis (this session — analytical, no code changes)

- Phase 47a (all tools except GDELT) → **run locally, overnight**. SQLite is a file on disk.
- Phase 47b (GDELT, 1–3 days) → **cloud VM preferred** (Oracle Always Free, Google Cloud free tier, or ~$5 DigitalOcean for 1 day). Background job, don't babysit.
- Phase 40 training (220K params, CPU) → **local is fine**. GPU not needed yet.
- If daily DAG needs to run unattended: move to a small cloud VM or GitHub Actions cron. Not needed now.
- Critical constraint: SQLite file must be transferred back (rsync/S3) before Phase 40 GNN training if the backfill ran in the cloud.

### 1h. Architectural principle added to repo memory (prior session, confirmed)
`memories/repo/tirramind_structure.md` updated with:
> **The GNN is the join layer — but the real join is the feature space.**
> The actual core problem TirraMind is solving: maintaining a feature space that is simultaneously *temporally stable* and *informationally current*.
> EWC addresses parameter stability. It does not address representational alignment.
> Two distinct risks require distinct mitigations: Stability (EWC, replay, partial retraining) vs Alignment (downstream gradient from world-model likelihood or RL advantage).

Five gaps from external architecture review also recorded:
1. GNN embeddings not aligned with downstream objective → Phase 49
2. Convergence detection is a feature, not a control signal → Phase 49b
3. Regime-stratified GNN replay buffer (EWC alone insufficient for non-stationary regime shifts)
4. Trust score should be a latent variable in the world model, not a post-hoc filter
5. Reward shaping via world-model likelihood improvement

---

## 2. Current codebase state (2026-04-23, confirmed)

### Live DB metrics
```
entities:             1,087
entity_observations: 74,030
entity_links:           357

Observation type breakdown:
  68,089  instrument_daily       (91.97% — CRITICAL IMBALANCE)
   2,868  market_probability
   1,398  geopolitical_event
     445  instrument_volume
     445  instrument_volatility
     445  instrument_return
     164  insider_trade
      60  futures_positioning
      44  btc_transfer
      24  sovereign_yield
      20  tvl_change
      16  economic_activity
       8  pageview_spike
       3  cb_balance_sheet
       1  capital_flow
```

Modal imbalance status: **CRITICAL**. 93.8% (doc) / 91.97% (live re-check) are instrument_daily. The GNN cannot learn surveillance-domain structure from this distribution. Phase 47 backfill is the fix.

### Test suite
- Total confirmed passing: 9,676+ (from repo memory; includes 65 GNN tests + 13 EWC tests post-Phase-46)
- All tests passed clean on 2026-04-23 after Phase 46 completion

### DAG state
- 29 nodes in daily pipeline (22 production + 7 utility)
- Unwired tools: ~24 of 51 (47.1%) remain outside the DAG
- Daily schedule: 18:00→19:45 UTC (collection → gnn_inference → convergence → scoring → features → adversarial → world_model → rl → inference)

### Key model parameters
- OBSERVATION_TYPES: 46 (code-verified in `agent/models/gnn/graph_builder.py`)
  ⚠️ Repo memory says 32 — this is stale; canonical value is 46 from code
- ENRICHMENT_DIM: 41 (9 base stats + 32 obs_type_dist) — confirmed in code
- HetTGN: ~220K params, HGTConv + HeteroMemory GRU + Time2Vec
- EWC Fisher diagonal: 2 × 860 KB ≈ 1.7 MB overhead

---

## 3. File states (canonical, written this session)

| File | Status | Notes |
|---|---|---|
| `[[data_strategy_doctrine]]` | ✅ COMPLETE | 18 sections, governing Phase 47 doctrine, 15+ citations |
| `[[gnn_downstream_alignment]]` | ✅ stub filed | Phase 49 gate: post-Phase-40 only |
| `[[convergence_as_control]]` | ✅ stub filed | Phase 49b: can run during Phase 47 |
| `[[historical_backfill]]` | ✅ exists | 27 Group A + 24 Group B + 8 Group C tools catalogued |
| `[[historical_backfill_spec]]` | ✅ exists | Verified signatures, sliding-window design, 4-step plan |
| `[[quant_training_ground]]` | ✅ updated | Phase 49 + 49b added before Phase 48 |
| `memories/repo/tirramind_structure.md` | ✅ updated | New principle + 5 gaps + OBSERVATION_TYPES flag |
| `.vscode/mcp.json` | ✅ secured | Tavily key via `${input:}` prompt, never stored |
| `agent/models/gnn/ewc.py` | ✅ Phase 46 | EWCState, compute_fisher, ewc_penalty |
| `agent/pipeline/dags/gnn_inference.py` | ✅ Phase 46 | online_update wired |
| `scripts/backfill.py` | ❌ NOT YET | To be created in Phase 47 Step 47.1 |
| `scripts/density_audit.py` | ❌ NOT YET | To be created in Phase 47 Step 47.2 |

---

## 4. Immediate next actions (in exact order)

### Code prerequisite (~30 min, before any backfill)
Add `as_of_date` parameter to 4 capped tools (~5 lines each):
- `agent/tools/insider_filings.py` — current cap: 90d
- `agent/tools/form144.py` — current cap: 60d
- `agent/tools/finra_short_volume.py` — current cap: 20d
- `agent/tools/sanctions_monitor.py` — current cap: 365d

These tools enforce `days_back = max(1, min(days_back, N))`. The `as_of_date` param allows the backfill loop to call them with an offset: `observed_at = as_of_date - timedelta(days_back)`. Without this, there is no way to collect their pre-window history.

### Design decision to lock (before writing backfill.py)
Comtrade depth mode — lock into `[[historical_backfill_spec]]`:
- **Aggregate (recommended):** 300 calls (15y × 20 reporters), 3 hours. Gets bilateral flow totals.
- **Per-commodity:** 4,500 calls, 45 hours. Gets strategic commodity breakdowns. Defer to Phase 47c unless GNN diagnostics show commodity-level structure is needed.

### Phase 47 implementation steps
1. **47.0**: Lock Comtrade design in spec
2. **47.0b**: Add `as_of_date` to 4 capped tools + tests
3. **47.1**: Write `scripts/backfill.py` with BACKFILL_PLAN, checkpoint/resume, `--dry-run`
4. **47.2**: Write `scripts/density_audit.py` with FAIL/WARN/OK tiers, exit 1 on FAIL
5. **47.3**: Run `python scripts/backfill.py --dry-run` — verify BACKFILL_PLAN is correct
6. **47.4**: Run `python scripts/backfill.py --delay 1.5` — overnight, monitor with `tail -f backfill.log`
7. **47.5**: Run `python scripts/density_audit.py` — verify all 7 exit conditions from §11 of doctrine
8. **47.6**: If any entity type fails density audit → extend backfill window or wire additional tools
9. **47.7**: Mark Phase 47 COMPLETE in `[[quant_training_ground]]`

### After Phase 47
→ **Phase 49b** (convergence as control, can run immediately — no Phase 40 dependency)
→ **Phase 40** (Real GNN retrain on backfilled data — now unblocked by Phase 47)
→ **Phase 49** (GNN downstream alignment — gated on Phase 40 producing real embeddings)
→ **Phase 48** (Transformer World Model + Dreamer RL — gated on Phase 40 showing current stack ceiling)

---

## 5. Standing decisions and rules (invariants that do not change)

1. **No single observation category may exceed 30% of total observations** — structural constraint derived from heterogeneous GNN attention-collapse theory. Non-negotiable even after Phase 40 diagnostic.
2. **Raw observations are primary; prices are derivative** — collect vessel, grid, filings, GDELT first; prices extend farther in time but cannot substitute for raw drivers.
3. **Phase 40 is DATA-GATED, not code-gated** — do not run Phase 40 on a failed density audit.
4. **Phase 48 is DENSITY-GATED** — ≥500 obs/entity-type average AND no type below 100. If fails, extend Phase 47 (try `days_back=3650`) before Phase 48.
5. **No imputation in `entity_observations`** — NULL if not observed; imputation is a Layer 2 (`agent/quant/`) decision, not a storage decision.
6. **EWC is a stability tool, not an alignment tool** — representational alignment requires downstream gradient signal (Phase 49), not EWC.
7. **Convergence detection is already a control signal (Phase 49b)** — it only looks like a feature because it has not been wired yet.
8. **Model agnosticism** — current stack (HetTGN, pgmpy DAG, SAC, Kalman) is best-justified for now. After Phase 40, evaluate each layer by out-of-sample predictive edge. Replace anything that fails. No sunk-cost attachment.
9. **LLM is scaffolding, not the product** — every decision that reaches a prediction must go through the math layers (GNN → world model → Kalman → RL). The LLM synthesises and narrates; it does not decide.
10. **GNN-guided expansion** — do not wire new tools or upgrade to L2 based on coverage checklists. Wire based on GNN attention diagnostic output after Phase 40. Exception: Phase 47 backfill runs existing tools deeper (more history), which is always safe.
11. **Comtrade design decision** — aggregate calls (300 total) for Phase 47a; per-commodity deferred to Phase 47c pending GNN diagnostic evidence that commodity-level structure is needed.
12. **GDELT deferred to Phase 47b** — requires separate file-based runner (5,475 daily CSV files), not the REST call pattern. Not a Phase 40 blocker.
13. **Capped tool sliding-window design** — `as_of_date` param required before backfill; do not just accept the 90/60/20/365-day caps as the final depth for those tools.

---

## 6. Context graph / knowledge graph links

Research triad for Phase 47:
- Research: [[data_strategy_doctrine]] + [[historical_backfill]]
- Spec: [[historical_backfill_spec]]
- Task: [[quant_training_ground]] (Phase 47 entry)

Gated future phases:
- [[gnn_downstream_alignment]] (Phase 49 — gated on Phase 40)
- [[convergence_as_control]] (Phase 49b — NOT gated, can run now)
- [[transformer_world_model]] (Phase 48 — gated on Phase 40 + density)

Architecture memory:
- [[living_system_online_gnn]] + [[living_system_online_gnn_spec]] (Phase 46 — complete)
- [[tirramind_structure]] (canonical metrics, OBSERVATION_TYPES=46, DAG=29 nodes)

Completed phase checkpoints for reference:
- [[chat_checkpoint_2026-04-22_phase45_3_complete]] — last complete checkpoint before this one
- [[chat_checkpoint_2026-04-22_phase44_complete]] — Phase 44 batch DAG wiring
- [[chat_checkpoint_2026-04-22_strategy_transformer]] — Phase 48 Dreamer/transformer planning

---

## 7. Cold-start instructions for next session

1. Read this file.
2. Read `[[quant_training_ground]]` — find "Phase 47" entry for current status.
3. Read `[[historical_backfill_spec]]` — the implementation contract.
4. Check if `scripts/backfill.py` exists → if yes, run `--dry-run` to verify BACKFILL_PLAN.
5. Check if Comtrade design decision is recorded in spec → if not, lock it (aggregate vs per-commodity).
6. If the 4 capped tools don't have `as_of_date` yet → that is the first code task.
7. Do NOT read historical_backfill.md unless you need the tool-by-tool catalogue — the doctrine and spec are sufficient.
8. Do NOT re-run the Tavily research searches — all findings are in `[[data_strategy_doctrine]]`.

The immediate next piece of code: `as_of_date` param added to `insider_filings.py`, `form144.py`, `finra_short_volume.py`, `sanctions_monitor.py`. ~5 lines each + tests. That unblocks writing `scripts/backfill.py`.

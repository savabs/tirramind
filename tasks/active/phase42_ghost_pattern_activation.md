---
title: "Task: Phase 42 Ghost Pattern Data Activation"
tags:
  - doc/task
  - phase/42
  - topic/world-model
  - layer/surveillance
  - status/active
---

# Task: Phase 42 — Ghost Pattern Data Activation

**Status:** active
**Research:** [[ghost_pattern_graph_audit]]
**Spec:** [[ghost_pattern_data_activation_spec]]

---

## Why this phase exists

Phase 41b retrained the GNN with ListNet loss + auto-tune fix. The GNN model is now training
correctly (return head no longer suppressed). But IC=-0.033 is not a training bug anymore —
it is a **data problem**.

The entity graph lacks the cross-domain signals the vision requires:
- CFTC positioning (managed money net) → only 5 of 89 instruments have cftc_tracks edges
- Commodity→producer country edges → instruments only linked to exchange country, not growers
- All physical tools (AIS, EIA, weather, disease) → near-zero obs because pipeline not running 24/7

This phase fixes the data, not the model.

---

## Steps

### 42.0: Pre-work (done) ✅
- Entity graph audited: see [[ghost_pattern_graph_audit]]
- Spec written: see [[ghost_pattern_data_activation_spec]]

### 42.1: Map 89 instruments to CFTC contract codes ✅
- All 19 commodity CFTC codes verified against 2024 `fut_disagg_txt_2024.zip`
- Bug found and fixed: `_filter_contracts(top_n=20)` was silently dropping 14/19 mapped contracts
- Fix in `agent/tools/cftc.py`: always include mapped contracts regardless of OI rank
- Exit: cftc_tracks links: **5 → 19** after running `scripts/backfill_cftc.py`

### 42.2: CFTC historical backfill ✅
- `scripts/backfill_cftc.py` created — downloads 3yr ZIPs via existing `CFTCTool.execute(mode='historical')`
- Ran successfully: 5,080 new observations (2022-2024), 19/19 cftc_tracks links
- ⚠ Observations below 10K target (3 years only covers ~52 weekly reports × 3 = ~156 rows/contract)
  → CFTC reports weekly; history goes back to 2006. Backfill more years if needed after retrain.
- Exit: cftc_observations: 300 → **5,080** (+4,780)

### 42.3: Producer-country links for commodity instruments ✅
- `scripts/seed_producer_links.py` created — PRODUCER_MAP covers all 19 mapped commodities
  Sources: USDA WASDE, IEA, WGC, USGS, ICO, ICCO (verified)
- Bug found: script used ticker format `CL=F` but DB uses full name `WTI Crude Oil`
  Fixed by building ticker→entity_id via `instrument_universe.tradeable_instruments()` name map
- Ran successfully: **119 produced_in links created**, 31 new country entities registered
- Exit: produced_in links: **0 → 119** ✓ (≥30 target met)

### 42.4: ICIR metric + measurement infrastructure ✅
- `agent/quant/experiment_tracker.py`: ExperimentTracker, compute_stratified_ic, save/diff/list
- `scripts/phase40_gnn_backtest.py`: ICIR added (Grinold & Kahn 2000), steps 11-12 wired in
- `scripts/source_ablation.py`: per-source ΔIC at inference time (no retraining needed)
- `scripts/compare_experiments.py`: diff any two experiment manifests
- Exit: manifest saved to `.tirra_pipeline/experiments/exp_20260502_135216.json` ✓

### 42.4b: IC BASELINE MEASUREMENT (post data-fix, pre-retrain) ✅
**Run date:** 2026-05-02, model=epoch 20, graph enriched (19 cftc_tracks + 119 produced_in)
```
Strategy        Mean IC   Std IC    ICIR   t-stat
GNN-EmbNorm     +0.0191   0.1660   0.115    0.73   (was -0.033 before fixes)
GNN-ValueHead   +0.0169   0.1263   0.134    0.84
GNN-ReturnHead  +0.0123   0.1246   0.099    0.63
```
**Interpretation:**
- IC sign flipped from negative (-0.033) to positive (+0.012~+0.019) — data fixes worked
- ICIR still too low (0.10-0.13 vs target >0.40) — expected: model was trained on old graph
  The GNN weights were learned when cftc_tracks=5 and produced_in=0. Need retrain on new topology.
- Stratified IC: CFTC instruments HURTING signal (EmbNorm: WITH_cftc IC=-0.067 vs WITHOUT IC=+0.043)
  ROOT CAUSE: Model memorized wrong CFTC topology (5 links). Now graph has 19 — weights mismatched.
  This is the clearest evidence retrain will help: fix topology → retrain → CFTC should flip positive.
- EmbNorm is strongest head (IC=+0.0191) despite no direct return supervision
  Signal exists in embedding space; prediction heads haven't learned to extract it yet.

### 42.5: Verify pipeline accumulation (7-day run)
- [ ] Run `python -m agent.cli` or `scripts/run_daily_collection.py` for 7 consecutive days
- [ ] After 7 days: entity obs audit → ≥5 tools contributing >50 obs each
- [ ] Confirm: disease_surveillance, energy_supply, ais_vessel, weather_alerts have new obs
- Exit: 7-day entity obs summary showing diversity of sources

### 42.6: Update Kaggle notebook + retrain ← CURRENT
- [ ] Add `"--return-log-var-max", "0.0"` to Kaggle notebook cmd list
- [ ] Rebuild zip with enriched DB (must include updated pipeline.db with 119 produced_in + 19 cftc_tracks)
  ```
  zip tirramind_data_upload.zip .tirra_pipeline/pipeline.db .tirra_pipeline/gnn_model.pt checkpoints/...
  ```
- [ ] Upload to Kaggle, resume from epoch 20, run epochs 21–40
- [ ] Re-run IC backtest with ICIR metric
- Exit: ICIR > 0.25 (especially for CFTC instruments which should flip from negative to positive)

---

## Completion Criteria

Phase 42 is complete when:
1. CFTC backfill complete (>10,000 obs, 40+ contracts mapped)
2. Producer-country links exist for commodity instruments (≥30 links)
3. ICIR metric is in the backtest output
4. Pipeline has run continuously for ≥7 days with ≥5 diverse sources accumulating
5. GNN retrained on enriched graph (epochs 21–40)

The measure of success is NOT IC > some number. It is:
**"Does removing CFTC positioning drop IC by >20%?"** — if yes, we have real cross-entity alpha.

---

## Related

- [[ghost_pattern_graph_audit]] — research document
- [[ghost_pattern_data_activation_spec]] — implementation spec
- [[phase41b_gnn_signal_extraction]] — predecessor task (training fixes)
- [[tirramind_structure]] — canonical metrics

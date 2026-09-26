---
title: "Full defect audit — 2026-09-23"
tags:
  - doc/research
  - topic/architecture
  - status/current
---

# Full defect audit — 2026-09-23

Produced by an 8-lens parallel sweep (165 agents, 3,546 tool calls) with every
finding adversarially refuted by two independent verifiers before it counted.

**78 raw findings -> 70 confirmed, 8 refuted.**
Severity: 18 critical, 26 high, 20 medium, 6 low.

Two findings in this report were already fixed in HEAD while the audit ran
(the two CSRC defects, `LESSONS.md` F-15) — the audit found them independently,
which is useful corroboration.

One finding is a direct consequence of the F-14 fix committed earlier the same
day: gating edges on `created_at` is correct in principle, but `created_at` is
an *ingest* stamp, so it removes the graph instead of the leak. See P3.

---

# TirraMind — Sequenced Fix Plan

Derived from the confirmed defect set, re-verified against the working tree at `security/withdraw-paddle-exposure` (HEAD `efc2d72`) and the live DB on 2026-09-23.

---

## 0. Read this before planning any work

### 0.1 Two findings are already fixed in HEAD — do not re-fix them

Commit `3c25fb2` ("the anti-collapse loss was causing the collapse (F-15)") landed both CSRC fixes. Verified:

- `agent/models/gnn/trainer.py:1741-1750` now scatters through the sort permutation: `decile_assignments[sort_idx[start:end]] = d`. The `sorted_tgt` dead line is gone.
- `agent/models/gnn/trainer.py:1790-1796` now exponentiates first: `sim_exp = (sim - sim_max).exp()` then `(sim_exp * pos_mask).sum(dim=1)`.

The audit's line references for `trainer.py` (1736/1744/1778) predate that commit; **every `trainer.py` line number in the defect list is shifted by roughly +10 to +15**. Anchor on symbol names, not line numbers.

This matters for sequencing: the CSRC loss is now *correct*, which means it is now a real gradient path — and that makes the input-side defects (no edges, no memory, zero value features) the binding constraint rather than a shared blame.

### 0.2 The single fact that reorders everything

**The model has never been a graph network, and has never had memory.** Two independent, fully confirmed mechanisms:

| Mechanism | Evidence | Effect |
|---|---|---|
| `entity_links.created_at` is an *ingest* stamp | `graph_builder.py:139-140`; `min(created_at)` = 2026-04-18, `max` = 2026-08-27; train cutoff = 2025-07-16 | `_links_as_of` returns `[]` → `_build_edge_data` returns `{}` → `het_tgn.py:642 if edge_index_dict:` skips the whole HGT loop. **100% of train/val windows, 0 edges.** |
| observation dicts carry no `entity_type` | `het_tgn.py:798-802`; `store.py:2074-2084` returns only `entity_observations` columns | `update_memory_from_events` `continue`s on every one of 384,285 rows. **All 21 on-disk checkpoints have `memory.memory` nonzero-rows = 0.** |

Consequence: `HetTGN.forward` reduces to `relu(combiner(cat(type_projection(x), zeros)))`. Every `hgt_layers` parameter and every `memory.gru.*` / `memory.time_enc.*` parameter in every checkpoint is at random initialisation and has received zero gradient. Add the third input-side defect — node value features are hard zero for 99.8% of rows (`graph_builder.py:258-276`, `:388-401` probe six key names that no collector writes) — and the model is a per-node MLP over a type one-hot plus a count and a recency scalar.

**Therefore: no loss-weighting, split, or subsampling fix is worth measuring until the three input-side defects are fixed, and all three invalidate every existing checkpoint.** Do not tune anything before Phase 3.

### 0.3 Dependency graph

```
P0 instrumentation (make silence loud)
      │  no dependencies — do first, it validates everything after
      ▼
P1 write-path guards ──► P2 DB repair (needs backup + approval)
      │                        │
      ▼                        ▼
P3 model inputs: value keys ─┬─► P4 training loop ──► P5 retrain from scratch
   memory entity_type ───────┤        (split, EWC, upscale,
   edge time semantics ──────┤         subsample parity,
   reverse edges ────────────┘         collapse gate)
                                              │
                                              ▼
                             P6 layer 5-6 bootstrap + orchestration
P7 test integrity — parallel, no dependencies
```

Hard ordering constraints:

1. **P2 (timestamp purge) before P4 (split fix) before P5 (retrain).** The 912 future-dated `gov_contracts` rows and the count-based split are separate bugs that compound: purging the rows without fixing the split still leaves a count-based split; fixing the split without purging still admits 2027 rows into the test range.
2. **P3 before P5.** Fixing node features / memory / edges changes `in_channels` and the node ID space. Retraining before P3 burns compute on the same degenerate model.
3. **P1 before P2.** Purge the poisoned rows only after the guard that stops them being re-written exists, or the next collection run re-poisons the table.
4. **P0 before P5.** A retrain with no edge-count logging per window is unfalsifiable; you will not be able to tell whether P3 actually worked.

---

## (a) Safe to apply now — code only, no DB writes, no retrain required

### P0 — Instrumentation: convert every silence into a signal

These are the highest value-per-risk changes in the whole plan. They cost nothing, break nothing, and every later phase depends on them for verification. This codebase's documented signature failure is "green status, constant output"; P0 is what makes the later phases falsifiable.

**P0.1 — Log per-window edge and node counts in `build_from_cached`.**
`agent/models/gnn/graph_builder.py`, inside `build_from_cached` after `_build_edge_data`: log `len(links_kept)`, total edge count, and nodes-with-observations vs total nodes. `build()` already does this (`logs/smoke_retrain.log:105` shows `20 edge types, 17581 edges` — the only edge number that appears in any log, and it comes from the *unbounded* call).
*Test:* `test_build_from_cached_logs_edge_count` — build a window with `until` before `min(created_at)`, assert with `caplog` that an edge count of 0 was logged.

**P0.2 — Make an edgeless training window loud.**
In `Trainer.train()`, after pre-building window snapshots, count snapshots with `data.edge_types == []` and `log.error` (or raise, behind a config flag) when the fraction exceeds a small threshold. Replace the silent `if edge_index_dict:` no-op at `het_tgn.py:642` with an explicit branch that records "message passing skipped".
*Test:* build 10 snapshots with `until` in the past, assert the trainer raises or logs at ERROR.

**P0.3 — Memory non-zero assertion.**
After the first `update_memory_from_events` call in `train()`, assert `(model.memory.memory.abs().sum(dim=1) > 0).sum() > 0`. Add the same check in `save_model`: refuse to write a checkpoint whose memory buffer is entirely zero when events were consumed.
*Test:* `test_memory_nonzero_after_one_window` — feed real `_entity_obs_row_to_dict` output, assert nonzero rows > 0. **This test fails today**, which is the point.

**P0.4 — Fix the executor's status and zero-rows guard.**
`agent/pipeline/executor.py:607` sets `nr.status = "completed"` on any non-raising operator. Every layer 3-6 node is a `FunctionOperator` (`operators.py:138-146`) which returns its dict verbatim. Two changes:
- After a `FunctionOperator` returns, inspect the payload: if `status` is `"skipped"`/`"failed"` or `success is False`, set `nr.status` accordingly. Verified live: the latest completed `inference` run's stored envelope is literally `{"status": "skipped", "reason": "no_sac_model"}` and `emit_portfolio` is recorded `completed` with `started_at == finished_at` to six decimals.
- Replace the `nr.stored` proxy (`executor.py:215`, `:560`) with real per-destination-table row deltas. `scripts/run_chain.py:61-92` already has the `_WATCHED_TABLES` / `_print_deltas` machinery — reuse it. **`pipeline_data` must be excluded from the delta**, because writing the one-row summary envelope is exactly what makes the guard unable to fire.
- Call `_apply_zero_rows_guard` from `_reconcile_timeout` (`executor.py:558-562`).

*Test:* `test_skipped_payload_is_not_completed` — a `FunctionOperator` returning `{"status": "skipped"}` produces `nr.status == "skipped"` and a run that writes zero domain rows fails the guard. Verified precondition: 6 `completed` `adversarial_scan` runs, 6 `completed` `rl_training` runs, `adversarial_flags` / `rl_transitions` / `rl_policy_checkpoints` / `portfolio_weights` / `paper_trade_pnl` / `depth_evaluations` all `COUNT(*) = 0`.

**P0.5 — Stop `gnn_inference` swallowing a failed checkpoint save.**
`agent/pipeline/dags/gnn_inference.py:221-225` catches the save exception without even binding it, then returns `{"status": "completed", ..., "model_path": <path never written>}`.
*Fix:* bind the exception, log with `exc_info=True`, return `{"status": "failed", "reason": ...}`; at minimum assert `Path(versioned).exists()` and include `saved: bool`.
*Test:* patch `save_versioned` to raise, assert the returned status is not `"completed"`.

**P0.6 — Fix the F-01 collapse gate.**
`trainer.py:2763-2789` (verified at those lines today): `_emb_std = _ie.std(dim=0).mean().item()`, gate is `_emb_std < 0.05` at `:2773` and `:2775`. `_eff_rank` is computed at `:2768`, stored at `:2771`, and **never compared to anything**. The current checkpoint scores std = 6403 with `eff_rank` = 3.20/64 — the gate passes by a factor of 128,000 while 95% of dimensions carry no variance. `std` is scale-equivariant and the downstream losses L2-normalise, so embedding scale is unobservable to the objective and free to drift: **this gate can never fire.**
*Fix:* gate on scale-free quantities — `eff_rank / emb_dim < 0.25`, plus mean off-diagonal absolute cosine similarity of normalised embeddings > 0.9. Keep `emb_std` as a diagnostic only. Record `eff_rank` into the history arrays (see F-03: front-pad with NaN on resume). Move the bare `except Exception → log.debug` at `:2787-2789` so it wraps only the snapshot/forward call, and raise it to `log.warning(..., exc_info=True)`.
*Test:* `assert collapse_detected(torch.ones(64, 64) * 6403.0) is True` — that input is exactly the current checkpoint's profile and today's gate passes it. Pair with a full-rank negative case. **Zero tests in 281 test files currently touch this gate** (`grep -rn "effective_rank|emb_std|collapse_risk" tests/` → no hits).
*Also update `LESSONS.md` F-01:* the stated prevention rule `torch.std(emb, dim=0).mean() > 0.1` is unsound as written and is what licensed this gate.

**P0.7 — Fix `scripts/mp1_data_health.py`.**
`:84` grades on lifetime `COUNT(*)`; `latest_observed_at` is computed at `:85` and never read by `blockers` (`:127-131`) or `ready_for_chain_scan` (`:137`). It reports `ready: true` today on a pipeline that stopped 27 days ago. `:112` issues `SELECT name, last_status FROM dag_runs`; the columns are `dag_name, status` — verified `Error: no such column: name`, swallowed by the `except` at `:121` into a note that `main()` at `:179` never prints.
*Fix:* add a per-source recency term to `status` and to `blockers`; fix the query; let the `OperationalError` surface; non-zero exit on blockers.
*Test:* against a fixture DB whose newest row is 60 days old, assert `ready_for_chain_scan is False`; assert the DAG section renders non-empty.

### P1 — Write-path guards (must precede any DB repair)

**P1.1 — Reject implausible `observed_at` at the single write boundary.**
Verified: `agent/pipeline/store.py:1255-1291` `store_entity_observation` inserts `observed_at` verbatim with no range check, no comparison to `time.time()`, no rejection path. Live: 1,009 poisoned rows — `gov_contracts/contract_award` 928 rows spanning 1978-09-14 → 2030-01-30, `drug_regulatory/drug_approval` 80 rows from 1972-03-24, `gdelt/geopolitical_event` 1 row at 1920-01-01 (the only row in the table with `observed_at <= 0`).
*Fix:* raise (or quarantine to a `rejects` table) when `observed_at` is outside `[1990-01-01, now + 86400]`. Putting it at the shared boundary is the whole point — per-collector guards do not generalise, which is how the same class recurred three times.
*Test:* `store_entity_observation(observed_at=<2030>)` raises; `observed_at=<1920>` raises; `now + 3600` is accepted.

**P1.2 — Fix `gov_contracts` at source.**
`agent/tools/gov_contracts.py:147-158` sets `observed_at` from `award.get("start_date")`, which `:437` maps from USAspending's `"Start Date"` — the period-of-performance start, a *future* event — and `:343` sorts descending by that field, so the query preferentially returns contracts starting furthest ahead. The guard at `:151` is `if dt.year > now.year + 1`, which admits all of 2027 by construction, and when it *does* fire it substitutes `datetime.now().timestamp()` — converting a data error into a plausible-looking recent observation, which is worse than the error.
*Fix:* request and map the award action / last-modified date for `observed_at`, keep `start_date`/`end_date` in `value_json` as metadata; replace the year-granularity clamp with a skip. Also add `"Award Type"` to the `_FIELDS` list at `:73-74` (it is mapped at `:441` but never requested, so it is null in 100% of 1,060 rows).
*Test:* feed a fixture award with `Start Date` in 2030 and an action date in 2026; assert the stored `observed_at` is the 2026 value.

**P1.3 — Route country keys through `resolve_country_key`.**
`agent/pipeline/country_codes.py:17-22` states that every collector creating a country entity MUST route through `resolve_country_key`. Verified today: the only importers in `agent/` are `agent/quant/ghost_chains.py:21` (a *consumer*) plus `scripts/migrate_country_entities.py:39` and `tests/test_country_codes.py:17`. **Zero collectors import it.** `agent/tools/gdelt.py` still does raw `entity_id_from_key("country", ...)` at `:474`, `:476`, `:750`, `:790-791` with CAMEO alpha-3 actor codes.

This means the 214/215 connectivity figure is a property of a migrated *table*, not of the pipeline. The first GDELT run after the migration writes 92k+ observations to brand-new alpha-3-keyed entities and restores the original 0/215 failure, while `tests/test_country_codes.py` keeps passing because it exercises the module in isolation.
*Fix:* route at the point of entity creation in `gdelt.py` first, then the other ~19 collectors that call `entity_id_from_key("country", ...)` (`ais_vessel.py:1068,1131`, `central_bank_balance.py:952,984,1016,1048`, `consumer_sentiment.py`, `comtrade.py:745`, `capital_flows.py`, `disease_surveillance.py`, `global_pmi.py:551`, `food_security.py:422`, …).
*Test:* two tests, both needed. (i) A value test: for a sample of CAMEO alpha-3 codes, `entity_id_from_key("country", X)` after resolution equals the alpha-2-keyed id (`USA` → `f07b20bdd8b5f9bb`, not `65c0a83a115572ef`; `GBR` → `7e027bd2a4b8e561`; `DEU` → `6389d58d4f7fc8c0`). (ii) **An AST guard over `agent/tools/` in the same shape as the F-14 guard**, failing any `entity_id_from_key("country", X)` whose argument is not a `resolve_country_key(...)` call. A docstring instruction that nothing imports is not a fix; only the guard prevents the next collector reopening it.
*Also:* call `is_region_code` in the GDELT persistence loop (`gdelt.py:748-752`) so NATO / THE EUROPEAN UNION / AFRICA stop being registered as `country` (16 such entities hold 2,949 observations and 1,182 link endpoints; AFRICA alone has 1,526).

**P1.4 — Add the `entity_observations` uniqueness constraint.**
No UNIQUE index exists (`idx_entity_obs_lookup` on `(entity_id, source_tool, observed_at)` is non-unique); `store.py:1267-1282` is a plain `INSERT` with no `ON CONFLICT`. Live: 121,597 byte-identical redundant rows across 61,459 groups — **31.6% of the table** — of which `tvl_change` is 105,172. The ingest stamps prove a triple re-run of one backfill: 2026-05-11 16:13:09, 16:14:31, 16:14:58.
*Fix (code half — safe now):* `INSERT ... ON CONFLICT(entity_id, source_tool, observation_type, observed_at) DO UPDATE SET value_json=excluded.value_json, ingested_at=excluded.ingested_at`. Last-write-wins makes a re-fetch idempotent. The index half is P2.
*Test:* store the same observation twice; assert `COUNT(*) == 1` and that the second `value_json` won.

**P1.5 — Fix `EntityAnomalyScorer._extract_value`.**
`agent/fusion/entity_scorer.py:270` reads `obs.get("value_json")`, but `store.py:2077` does `d["value"] = json.loads(d.pop("value_json", "{}"))` — the key is popped. Every observation scores 0.0. Live confirmation: `select count(distinct cusum_statistic), count(distinct event_study_score), count(distinct memory_drift) from entity_alerts` → 1, 1, 1 across 4,852 rows, all exactly 0.0. CUSUM and event-study scoring have never produced a non-zero value in the life of the database.
*Fix:* `obs.get("value")` (keep a `str` branch for raw-row callers). Then make a key miss an *event*: count it and log once per `observation_type` rather than returning 0.0. Note the latent second instance — the candidate list at `:279-286` still omits `amount_usd`, the key `gov_contracts` actually writes, so those rows would still score 0.0 after the key fix. Resolve via the shared extractor in P3.1.
*Test:* `_extract_value` on real `_entity_obs_row_to_dict` output is non-zero; and a data test asserting `count(distinct cusum_statistic) > 1` after a scoring run.

**P1.6 — Wire memory into `EntityScorer.detect`.**
`entity_scorer.py:125` receives `events` and discards them; `:131` snapshots memory, `:135` calls `self._model(data, id_map)` — and `HetTGN.forward` (`het_tgn.py:592-595`) takes only `(data, id_map)` and never touches memory state. So `memory_after == memory_before` and `surprise.py:217-221` computes `||0||`.
*Fix:* call `update_memory_from_events(events, embeddings, id_map)` between the forward pass and the surprise extraction, mirroring `trainer.py:4063`. **Depends on P3.2** — without the `entity_type` fix this call is still a no-op.
*Test:* warn/raise when every entity in a scoring run has `memory_drift == 0.0`.

**P1.7 — `ConvergenceFeatureBuilder` must not assert confidence in a default.**
`agent/features/builders.py:198` queries `WHERE signal_name LIKE 'convergence.%'`; all 92 rows in `signals` are `gnn_alignment_signal.*`, so it always takes `_zero_features` (`:233-259`) which sets `value=0.0, quality=1.0` — maximum confidence. `agent/pipeline/dags/world_model_update.py:57-59` maps these to Kalman observation indices 3, 4, 5, and `grep -c quality world_model_update.py` → 0. The Kalman filter is being told with certainty that there is never any stress; `beliefs` shows the result as degenerate one-hot posteriors with `confidence=1.0`.
*Fix:* `quality=0.0` in `_zero_features` (or delete the branch and let `_missing_features` handle absence); have `world_model_update` scale each observation's measurement-noise R by `1/quality` or drop observations below a quality floor. Give `convergence_detection`'s two zero-paths (`dags/convergence_detection.py:152`) distinct `reason` keys — they currently return byte-identical dicts for "no evidence in store" and "ran and found nothing".
*Test:* a store with no convergence signals yields `quality == 0.0`; a `world_model_update` step with `quality=0.0` leaves the latent state unchanged.

**P1.8 — `adversarial_scan` and `rl_training` must fail rather than lie.**
- `agent/pipeline/dags/adversarial_scan.py:42-49` passes `signal_returns={}`, `market_returns=np.array([])`, `clusters=[]`, … with inline comments "populated from PipelineStore in production". `store` is opened at `:38` and used only at `:58` to write flags. Layer 6 is structurally incapable of producing a flag and has reported `completed` 6 times. *Interim fix:* `if not signal_returns and not clusters: raise RuntimeError(...)`.
- `agent/pipeline/dags/rl_training.py:106-107` sets `result["sac_trained"] = True` unconditionally, even when `_train_sac` returned at `:265` with `{"status": "insufficient_data", "buffer_size": 0}`. *Fix:* `result["sac_trained"] = sac_metrics.get("status") != "insufficient_data"`, and surface `buffer_size`.

*Test:* both DAG nodes with empty inputs produce a non-`completed` run status (uses P0.4).

### P7 — Test integrity (parallel, no dependencies, do continuously)

**P7.1 — Remove four tautological assertions.** `tests/test_comtrade_edge.py:159` (`result is not None or result is None`), `tests/test_transport_throughput_edge.py:97` (`r.success or True`), `tests/test_surprise.py:538-540` (`... or True` — the *sole* assertion in `test_neighborhood_is_avg_of_neighbors`), `tests/test_pattern_extractor.py:212`. Drop the `or True` and fix whichever side actually fails; if `test_surprise` genuinely fails, it is an `xfail` with a reason, not a disarmed assert.

**P7.2 — Four clamp tests with no assertion at all.** `tests/test_transport_throughput_edge.py:99-120` — `test_months_back_clamped_high`, `test_limit_clamped_low`, `test_limit_clamped_high`, `test_extra_kwargs_ignored` each call `execute(...)`, bind `r`, and end. Deleting the clamp at `agent/tools/transport_throughput.py:221-222` keeps all four green. *Fix:* the sibling pattern at `tests/test_bankruptcy_court_edge.py:316-360` — `m.assert_called_once(); assert m.call_args.kwargs["months_back"] == 60`.

**P7.3 — Preflight fail-open is pinned by a test.** `agent/preflight.py:351-353`, `:376-378`, `:396-398` — all three data-readiness checks `return PreflightResult.passed()` on any exception, and `tests/test_preflight.py:299-303` asserts `r.ok is True` for a raising `_get_conn`. Anyone tightening this has to delete a passing test. *Fix:* add a third `CHECK_FAILED` state that is not `ok` for automated runs, narrow the catch to `sqlite3.OperationalError`, and replace the test assertion with `assert r.ok is False and r.reason == "CHECK_FAILED"`. Add a test that a *typo'd* table name is detected.

**P7.4 — The country regression guards never run in CI.** `tests/test_country_codes.py:114` skips `TestLiveGraphCoverage` when `.tirra_pipeline/` is absent, and that directory is gitignored — so the only guard against a collector re-splitting the country graph is skipped in every CI run. *Fix:* snapshot the live distinct country codes into `tests/fixtures/live_country_codes.txt` and run the classification test unconditionally against it. **Separately and urgently:** `tests/test_ghost_chains_edge.py:118` and `:134` open the live 150MB production DB **read-write** (`sqlite3.connect(str(db))`) during a normal `pytest tests/` run. Change both to `sqlite3.connect(f"file:{db}?mode=ro", uri=True)`. Also strengthen its `len(eids) >= 2` to an exact set — the count passes even if both codes resolve to the same wrong entity.

**P7.5 — `make quality-gate` runs the live-network tests CI deselects.** `.github/workflows/ci.yml:87` runs `-m "not live and not slow"`; `scripts/quality_gate.py:39-47` and `Makefile:93` run `pytest tests/` with no marker filter, so `tests/test_power_grid_edge.py` `TestLiveNetwork` (whose fixture checks only that the MIS *directory* responds, then asserts `total_peak_mw > 0`) turns the gate red on any day NYISO has no data. These are the 2 failures in the 10,997/2 run. *Fix:* add `-m "not live and not slow"` to `quality_gate.py:41`, and make the fixture skip on empty payload rather than on host reachability. The assertion worth keeping is on *parsing* a captured CSV fixture, not availability.

**P7.6 — F-02 guard covers only argparse.** `TrainerConfig.use_concat_head: bool = False` is confirmed at `trainer.py:1034` today. The bypass branch is live at `trainer.py:~2520` (`model.return_raw_head(_raw_feat_t)`). The only guard is `tests/test_retrain_gnn_cli.py:22-36`, an argparse check on one script. Any caller constructing `TrainerConfig()` directly — Kaggle notebooks, `tests/test_trainer.py`, agent code — silently reintroduces F-02 and the suite stays green. *Fix:* a **runtime-branch** test, not another flag test: build a Trainer with default `TrainerConfig()`, run one epoch, zero the GNN embedding tensor and assert the return prediction changes — `assert not torch.allclose(pred_with_emb, pred_with_zeroed_emb)`. It fails today with the default config, which is the point. Then flip the default to `True` (this is also a P4 item, see P4.1).

**P7.7 — The subsample test cannot distinguish 5% from 95%.** `tests/test_obs_subsample.py:46-47` asserts only `0 < gdelt_kept < 100` for 100 rows; the function is seeded and deterministic (`obs_subsample.py:54`). The default no-thinning branch (`:59-63`), taken by every bare `Trainer(store, TrainerConfig())`, has zero coverage and returns `observations` unsorted despite the docstring promising sorted output. *Fix:* assert exact deterministic counts plus monotonicity `kept(0.05) < kept(0.5) < kept(0.95)` — the monotonicity assertion is what catches an inverted comparison. Add a default-branch test asserting the sort contract.

**P7.8 — Federal Register tests are `skipif(True)`.** `tests/test_regulatory_gazette_edge.py:1248-1277` — four tests that no flag, marker or env var can enable, and the class carries no `live` marker so `-m live` does not reach them. Convert the parsing half to offline tests against a captured response, or delete them.

---

## (b) Mutates the live DB — needs backup + explicit owner approval

> **Preconditions for the whole of P2:** P1.1 and P1.2 merged (or the rows come back on the next run), and a verified backup. A pre-merge backup already exists at `.tirra_pipeline/pipeline.db.bak_20260923_premerge` — take a fresh one, verify it opens and row counts match, before any statement below.

**P2.1 — Purge the 1,009 out-of-range observations.**

```sql
-- inspect first
SELECT source_tool, observation_type, COUNT(*),
       date(MIN(observed_at),'unixepoch'), date(MAX(observed_at),'unixepoch')
FROM entity_observations
WHERE observed_at < strftime('%s','1990-01-01')
   OR observed_at > strftime('%s','now') + 86400
GROUP BY 1,2;
-- expected: gov_contracts/contract_award 928 (1978-09-14 .. 2030-01-30),
--           drug_regulatory/drug_approval 80 (1972-03-24 .. 1999-12-21),
--           gdelt/geopolitical_event 1 (1920-01-01)
```

Note the audit's internal disagreement, which the owner should settle before deleting: one finding calls the 1972 `drug_regulatory` dates *genuine* openFDA `latest_submission_date` values (`drug_regulatory.py:184-188`) and the 2000-2014 `dividend_data` dates genuine historical ex-dates (`dividend_data.py:75`). **Recommendation: delete the 912 future-dated `gov_contracts` rows and the single 1920 GDELT row unconditionally; quarantine rather than delete the pre-2000 `drug_regulatory` rows** — they may be real, and the fix for them is a floor on the split, not a purge. The re-ingest after P1.2 restores `gov_contracts` with correct dates.

*Verification:* `SELECT MAX(observed_at) FROM entity_observations` ≤ now + 86400; re-run the split computation and assert `test_end` ≤ today.

**P2.2 — Collapse the 121,597 duplicate rows and add the unique index.**

```sql
-- keep the newest ingested_at per group
DELETE FROM entity_observations WHERE id NOT IN (
  SELECT id FROM (
    SELECT id, ROW_NUMBER() OVER (
      PARTITION BY entity_id, source_tool, observation_type, observed_at
      ORDER BY ingested_at DESC, id DESC) rn
    FROM entity_observations) WHERE rn = 1);
CREATE UNIQUE INDEX idx_entity_obs_unique
  ON entity_observations(entity_id, source_tool, observation_type, observed_at);
```

**This changes the split boundaries** (verified: 70/15/15 moves from train→2025-07-16, val→2026-03-25 to train→2025-06-17, val→2026-04-07) and changes the `count` node feature at `graph_builder.py:673`. Do it *before* P5, never between a train and its evaluation.
*Test:* `COUNT(*)` drops by 121,597 to 262,688; the index creation succeeding is itself the proof no duplicates remain; re-inserting a duplicate through `store_entity_observation` updates rather than appends.

**P2.3 — Backfill link effective dates.** See P3.3 — this is the DB half of the edge-time fix. Add `entity_links.effective_from`, backfill from the evidence that justified each link (the GDELT event date for `event_involves`, the filing date for `works_for`, the market creation date for `topic_relates_to_instrument`, the tx time for `transacts_with`), for all 17,581 rows.

**P2.4 — Repair `convergence_clusters` orphans.** 6 of 42 rows reference 185 distinct entity ids that the country merge deleted; all 185 were `entity_type='country'` in the pre-merge backup. They live in `member_entity_ids_json`, which is why no FK caught them (`PRAGMA foreign_key_check` is empty; every other entity reference in the DB is clean).
*Fix:* rewrite the JSON through the same alpha-3 → alpha-2 map in `country_codes.py`, dropping ids with no successor. Add a repo-wide integrity check that scans JSON-encoded entity-id columns, not just declared foreign keys.

**P2.5 — Delete the 145 provably-bogus topic edges.** Of 1,657 `repair_topic_links` edges, 145 have no whole-word producer: `'uk'` inside `ukraine` → GBPUSD=X/EWU, `'dow'` inside `down` (58 edges) → YM=F/DIA, `'nato'` inside the tennis player `cecchinato` → VIXY/GC=F/CL=F, `'corn'` inside `john cornyn`, `'eth'` inside `pete hegseth`, `'war'` inside `warditv championship` → XLE/BZ=F/USO. Identify them by re-running the matcher at `repair_topic_links.py:183` with `re.search(r'(?<![a-z])'+kw+r'(?![a-z])')` and deleting edges no anchored rule reproduces. Fix the substring match in both `repair_topic_links.py:183` and `agent/tools/polymarket.py` so they do not come back.
The remaining 2,607 are an owner decision — see (c).

---

## (c) Decisions only the owner can make

**C1 — What `entity_links.created_at` should become.** The correct fix is an `effective_from` column derived from the justifying evidence (P2.3), but the backfill quality differs per relation and some relations may have no defensible effective date at all. The owner decides whether to (i) backfill properly, (ii) refuse to train on windows before `MIN(created_at)` as an interim, or (iii) treat the graph as static topology and move time-varying information onto node features. **Nothing downstream can be measured until this is decided** — it is the single largest blocker in the plan.

**C2 — The 2,607 remaining topic→instrument edges.** They come from a 7-entry category→ticker lookup (`repair_topic_links.py:28-36` and the identical `polymarket.py:78-86`), so all 342 "politics" topics are byte-identical neighbours of SPY and mean-aggregate to one near-constant vector. Target concentration: VIXY 256, TLT 236, ZN=F 234, SPY 220, ES=F 189 of 1,657; only 38 of 93 instruments receive any topic edge. Yet these 2,752 edges are 66% of everything that can reach an instrument at hop 1. Options: replace the category constant with per-topic text-similarity confidence; drop the relation and feed Polymarket price/volume as a topic *node* feature; or keep as-is. Keep the 27 `bitcoin` / 17 `ethereum` whole-word edges either way.

**C3 — Undirected edges vs explicit inverse edges.** 11,295 of 17,581 edges (64.2%) point *out* of instruments (`instrument|produced_in|country` 119, `located_in` 67, `exchange_country` 20, `fx_base/quote_country` 15 each) and HGT flow is source→target, so all 217 countries, 1,156 persons, 1,066 companies, 60 organizations, 54 protocols and 20 domains contribute zero gradient to the instrument objective at any depth. `ToUndirected()` is one line and roughly doubles per-relation parameter count (requiring a `hidden_dim` re-tune); explicit inverse edges at write time are more surgical. **The country-code merge fixed a connectivity metric, not the model** — whichever option is chosen, re-check reachability afterwards (should rise from 6,286 to ~17.5k).

**C4 — Breaking the RL cold-start deadlock.** Verified closed loop: no SAC checkpoint → `_sac_inference` returns `{"status":"skipped","reason":"no_sac_model"}` (`inference.py:231`) → `_emit_portfolio` never calls `complete_pending_transition` → `rl_transitions` stays 0 → `_train_sac` returns `insufficient_data` before `store_rl_checkpoint` (`rl_training.py:265` vs `:291`) → no checkpoint. `scripts/run_chain.py:18` documents "rl_transitions only materialises on the SECOND consecutive inference run", which is not reachable. Options: write a random-init bootstrap checkpoint tagged `is_best=False`, or have `_emit_portfolio` store a pending transition from an equal-weight fallback. Note separately that `_build_surprise_returns` (`rl_training.py:157-160`) fits the weight learner on a target derived from its own inputs — that is a modelling decision, not a bug to patch silently.

**C5 — Scheduling, and the deliberate layer 5-6 exclusion.** 69/69 `dag_runs` are `trigger='manual'`; zero scheduled runs have ever occurred. No crontab on this host; the systemd timers are Linux-only and `docs/README_SYSTEM.md:331-334` says they are not installed. Collectors are 27-147 days stale. Separately, `deploy/systemd/tirra-chain.service` *deliberately* excludes `gnn_inference,entity_scoring,inference,rl_training` (1 vCPU / 1.9 GB, 84% RSS on 2026-08-27) — so layers 5-6 are structurally unreachable from the scheduled path even on a correct deploy, and fixing the scheduler will not fill them. Owner decides: a second low-frequency memory-capped timer, a bigger box, or accepting that layers 5-6 are manual-only (in which case record it where the chain is defined, not in a service-file comment).

**C6 — Cost discipline check.** Nothing in this plan adds an external API or paid service. A full retrain (P5) on local compute is the only material compute cost; call it out before starting it per CLAUDE.md §7.

**C7 — Dead subsystems: wire or delete.** `agent/pipeline/depth_eval.py` (278 lines, `run_depth_evaluation` — the intended producer of the 0-row `depth_evaluations`) and `agent/pipeline/cross_entity.py` (740 lines) have no non-test callers anywhere. `edge_attr` (`graph_builder.py:838-842`) is built for every edge and read by no model — `het_tgn.py:102,107` is HGTConv's own learned `p_rel`, not the data tensor. `ToolRoutingBandit` is never passed to `build_daily_collection_dag` (`dags/__init__.py:53` calls it with no arguments). Each is "green tests, zero production effect"; the owner decides per module. Note `cross_entity.py:257-283` already resolves vessel ports to FIPS codes, so a `vessel→country` relation is one change away — and all 502 vessels are currently isolated nodes.

---

## Phases that require a retrain

### P3 — Model inputs (all three invalidate every checkpoint)

**P3.1 — Per-observation-type value extractor.** One shared module-level mapping replacing the identical dead 6-key probe in **four** places: `graph_builder.py:258-276` (`_compute_obs_stats`), `graph_builder.py:388-401` (`_compute_distributional_features`), `trainer.py:1653-1662` (`_compute_targets`), `fusion/surprise.py:309-326` and `fusion/entity_scorer.py:279-286`. All five probe `('usd_amount','btc_amount','value','estimated_value','goldstein_scale','num_articles')`; live SQL shows **736 of 384,285 rows (0.19%)** contain any of them, all from `petroleum_inventory` (640) and `economic_activity` (96), spanning 8 distinct entities. Every name is a near-miss: real keys are `goldstein`/`num_mentions` (GDELT, 92,211 rows), `tvl_usd` (162,251), `close`/`log_return` (77,014), `value_btc` (4,612), `yes_price` (8,730), `amount_usd` (`gov_contracts.py:166`), `amount`, `shares`/`price`.

Two consequences, both load-bearing: five node feature dimensions are identically 0.0 for 6,102 of 6,110 entities (constant dimensions cannot contribute rank — this is a direct mechanical contributor to `eff_rank` 3.20/64), and the value head is trained to emit 0 on 99.8% of examples while seeing raw $1B+ magnitudes on the other 0.19% — which is exactly the documented `val_loss = 1,094,629` spike.

*Fix:* the shared mapping, **per-type standardisation before the feature tensor** (raw magnitudes span 0.1 to 1.5e9), and in `_compute_targets` **skip** examples with no extractable value rather than emitting 0.0.
*Test:* every `observation_type` present in the DB resolves to a non-None numeric value, so a new collector with a new key name fails loudly; plus a coverage assertion > 90% against the live schema.

**P3.2 — Populate `entity_type` on observation dicts.** The filter is in *three* memory-update paths: `het_tgn.py:798`, `cde_encoder.py:191`, `mamba_encoder.py:258`. *Fix at the source:* set `d["entity_type"]` in `store.py:_entity_obs_row_to_dict` via a join, or map through the trainer's existing `self._eid_to_type_cache`. Runtime proof of the fix: 200 real observations currently produce 0 nonzero memory rows; with `entity_type` injected, 67.
*Test:* P0.3's assertion, which fails today.

**P3.3 — Edge time semantics** — see C1/P2.3.

**P3.4 — Reverse edges** — see C3.

**P3.5 — Time-bound the node set.** `graph_builder.py:978-982` `prepare_static()` calls `query_all_entities()` with no time argument and that `cached_id_map` is reused for every window and every backtest fold (`trainer.py:2065,2183`, `phase40_gnn_backtest.py:1128`, and 6 other scripts). 6,110 entities in every window; only 542 have any observation by the train cutoff; **all 6,110 have `created_at` in 2026-04 or later**, so `entities.created_at` is useless as a filter for the same reason `entity_links.created_at` is. 91% of node rows in any training window are identical constant vectors over which the cross-sectional losses, VICReg, and the F-01 gate are all computed.
*Fix:* materialise `entities.first_observed_at` from `MIN(observed_at) GROUP BY entity_id`, give `prepare_static()` a **required** `until` kwarg with F-14's discipline, and extend the existing `TestEveryCallSitePassesUntil` AST walk in `tests/test_graph_builder.py` to cover it. Log nodes-with-observations vs total per window.

**P3.6 — Also guard `ts2vec.fit_and_encode` now, while it is dormant.** `ts2vec_encoder.py:168-169` calls `query_all_entities()` / `query_all_observations()` with no bounds and encodes each entity's *whole history* to one vector, injected unchanged into every window (`trainer.py:2188,2984,4012`; `graph_builder.py:1054-1064`, no `current_time` guard at all, unlike `_compute_price_features` at `:472`). This is strictly worse than F-14 — it leaks a learned summary of the future price path into the node feature vector — and it is one CLI flag from being armed (`use_ts2vec: bool = False`, `trainer.py:855`). Add the required `until` kwarg and the AST guard **before** anyone turns it on, and add `use_ts2vec`/`use_signatures` to the `LOSS_MODE:` audit line so a contaminated run is distinguishable in the log header.

**P3.7 — Checkpoint identity.** Before any of P3 lands, fix the positional memory-restore assumption. The country merge already changed the global node ID of **5,101 of 5,969 surviving entities (85.5%)**, because IDs come from `ORDER BY created_at` iteration order (`store.py:1572`) and removing 203 rows compacts everything after. `trainer.py:1883-1898` restores memory by position with only a row-count check — and here the count *shrank* (6172→6110), so the warning fires with a benign zero-pad message while rows are wholesale misaligned. This is latent only because P3.2 keeps the buffer at zero; **fix P3.2 without fixing this and the next resume silently scrambles every node's history.**
*Fix:* persist the ordered entity-id list beside the memory buffer in `save_model` and in the per-epoch payload, remap on load, zero rows for unknown entities. Make the ordering stable: `ORDER BY created_at, entity_id` (31 entities currently share one `created_at` value, so even an unchanged entity set does not guarantee a reproducible id_map). Also add `observation_types` and `entity_types` to the checkpoint dict — 10 checkpoints on disk were trained against a 45- or 46-type vocabulary against today's 52, their `obs_type_head` is silently re-randomised on load, and **nothing on disk records which vocabulary was used**, so the class-index shift cannot be reconstructed after the fact.

### P4 — Training loop (after P3, before P5)

**P4.1 — `use_concat_head` default.** Flip `trainer.py:1034` to `True`, matching the CLI, and land P7.6's runtime-branch test. Beware the interaction with `--freeze-backbone`: `trainer.py:1527-1535` unfreezes only `return_raw_head`, but the selection chain at `~2499-2512` prefers `return_concat_head` when it exists — so `--freeze-backbone --use-concat-head` yields `total.requires_grad == False`, the entire backward/step block is skipped, and epochs, loss curves and per-epoch `.pt` checkpoints are still emitted. With auto-tune on it is worse: only the log-variance parameters move, producing a plausibly drifting weight table over a frozen model. *Fix:* unfreeze whichever head the runtime path will use, and assert `total.requires_grad` on the first window of every epoch.

**P4.2 — Calendar-time split with a floor and a purge gap.** Replace the count-based `int(n*0.70)` in both `_split_observations` and the duplicated copy in `evaluate()`. Use `t_hi = min(max(observed_at), now)`, an explicit `t_lo` floor, strict inequalities so a shared timestamp cannot straddle a cut (227 rows share exactly 2025-07-16 00:00:00; 239 share 2026-03-25 00:00:00), and a purge gap of `forward_return_horizon` between train and val. Add a sanity assertion that each split's min/max fall inside a plausible range — a test window ending in 2030 should *fail*, not print itself in the `TRAINING_AUDIT` line.
Also fix `_make_windows`: it buckets by `int(t // window_size)` and iterates only non-empty buckets, so the 1920→1972 and 2026→2030 jumps become single "next window" supervision pairs with `time_delta` targets of `log1p(1.6e9) = 21.2` clamped to the 20.0 ceiling. Emit empty buckets, or refuse to pair buckets more than one window apart.
*Test:* against a fixture with a 4-year gap, assert no window pair spans more than one window width, and assert `test_end <= now`.

**P4.3 — Wire EWC into `train()`.** Verified today: `ewc_penalty` is imported at `trainer.py:63` and called **exactly once**, at `:3566`, inside `online_update()`. It appears nowhere in `train()`. Meanwhile `trainer.py:~1989` logs "EWC regularisation active from epoch 1". Every multi-block resumed run has had zero forgetting protection while the log asserted otherwise, and the Fisher computation plus sidecar write are pure cost.
*Fix:* add `total = total + ewc_penalty(model, self._ewc_state)` once per window before the backward branch, and include it in the PCGrad task dict. **Until it lands, change the log line** so it says loaded-but-not-applied.
Also fix `_loss_from_window` (`~3330-3439`), the Fisher closure: it builds `total` from obs_type + time_delta + value + contrastive only, with no return term at all, so Fisher diagonals for every return-head parameter are 0 — EWC would protect everything *except* the ranking signal. Its docstring claims "Identical loss formulation to the training loop", which is false. Factor the return-loss block out of `train()` and call it from both sites.
*Test:* `set(compute_fisher(...).keys()) >= {n for n,_ in model.named_parameters() if "return" in n}` with non-zero values; and a loss-value test that EWC changes `total` when `_ewc_state` is set.

**P4.4 — Return-loss upscale.** `trainer.py:2118-2121` derives `68.65x` from 6,110 registered entities / 89 return-labelled — but the actual per-window gradient batch averages 110.4 distinct entities vs 61.0 instrument entities, a true ratio of **1.81**. Both numbers in the justifying comment (53 and 2,145) are wrong. Since `clip_grad_norm_` renormalises the *sum* of task gradients to a fixed budget, obs_type/time_delta/value receive ~1/38 of their intended share of every update — which is the real mechanism behind the "obs_type spike" the code has accumulated clamps for. Also, `_scaled_task_loss` is applied *after* the multiply, so `--use-log-loss` means two different things by path (with ListNet the log cancels the upscale almost entirely; with Huber it preserves it) — a ~60x swing in effective return weight from an unrelated flag.
*Fix:* compute the ratio from actual per-window term counts, or set `return_weight` explicitly. Apply `_scaled_task_loss` **before** any scale multiplier, and log the resolved effective multiplier per path at startup.

**P4.5 — Subsample parity.** `apply_training_obs_subsample` is applied only to the prefetched *feature* stream (`trainer.py:2077-2081`), never to `windows`, so at `--gdelt-frac 0.05` the model is asked to predict ~87,600 GDELT events whose evidence was deleted from its input — irreducible label noise in the dominant loss, and an obs_type floor that is an artifact of the flag. Meanwhile `evaluate()` applies **no** subsample at all (`trainer.py:3994`, `:4011`; `grep obs_subsample trainer.py` → only the import inside `train()`), so val/test inputs carry ~20x the GDELT density training saw — a covariate shift the run deliberately introduced, measured as generalisation.
*Fix:* apply the subsample **once**, before `_split_observations`, and derive both windows and the feature stream from the same filtered list; pass the identical config and seed through `evaluate()`.
*Test:* assert the window observations are a subset of the prefetched list by identity.
While here, fix the banner: `scripts/retrain_gnn.py:779-781` hardcodes `901704` and prints ~45,085 GDELT rows kept when the true figure is ~4,611 — GDELT is 92,211 rows (24% of the DB), not 901K (the TrainerConfig docstring at `~698` repeats the claim). `tvl_change` is the actual dominant modality at 162,251 rows (42%), and `--defi-frac` defaults to 1.0 — **the aggressive cap is on the wrong source.**

**P4.6 — Wire `validate_schema_against_store` into the training path.** It has exactly one real caller, `scripts/run_chain.py:158`, where it is non-blocking: the `try/except` sets `schema_ok = False`, control falls through to `scheduler.trigger(name)` at `:164`, and because the guard is gated on `if name in _MODEL_DEPENDENT and schema_ok` the remaining model DAGs are never checked at all. `LESSONS.md:502` claims "Called before anything trains or scores." *Fix:* call it at the top of `scripts/retrain_gnn.py` and in `Trainer.build_model()`; make the `run_chain` failure `continue` rather than fall through, and re-check per DAG.
Also raise `CheckpointSchemaDriftError` — currently defined at `trainer.py:1173` and **raised nowhere** — when any `type_projections` key is skipped, with an explicit `allow_partial=True` escape. And add the live-vs-checkpoint `in_channels` comparison that `LESSONS.md:531` (F-12 rule 3) specifies and `load_model` does not implement: `_describe_checkpoint_schema_drift` compares the model built *from* the checkpoint against the checkpoint itself, so it can only catch self-contradiction, never disagreement with the live feature schema.

### P5 — Retrain from scratch

After P3 and P4. **From scratch, not resumed** — P3.1 changes `in_channels`, P3.5 changes the node set, P3.7 changes the ID space, P2.2 changes the `count` feature and the split boundaries. Per CLAUDE.md §5, document in `docs/memory/checkpoint_<date>_post_audit.md` why the old weights do not transfer, and treat every existing `.pt` as historical.

Quarantine `.tirra_pipeline/checkpoints/gnn_model_20260825T230121.pt` first: it records `in_channels` `{'country':15,'instrument':50, …}` while its stored tensors are `(64,14)` and `(64,49)` — **all 12 projections contradict their own recorded widths.** Loading it yields a model whose every input projection is randomly initialised while the log says only "Skipped N keys due to shape mismatch".

Acceptance criteria for the retrain, all of which are currently unmeasurable:
- median per-window edge count > 0 across train windows (P0.1)
- memory nonzero-rows > 0 at every checkpoint write (P0.3)
- `eff_rank / emb_dim` > 0.25 (P0.6)
- value-head target coverage > 90% (P3.1)
- test split max date ≤ today (P4.2)

### P6 — Layers 5-6 and orchestration (after P5, gated on C4/C5)

**P6.1 — `instrument_daily` has no scheduled producer and never has had one.** Confirmed: the only DAG price node is `fetch_instruments` → `run_instrument_ingest` → `ingest_daily_prices`, which writes `instrument_return`, `instrument_volume`, `instrument_volatility` — **not** `instrument_daily`. The only writers of `instrument_daily` are `instrument_universe.py:1207` (`backfill_historical_prices`) and `:1331` (`backfill_recent_readout_prices`), and neither is called from any DAG — the latter's sole caller is `scripts/ghost_pattern_daily.py:45`, for `MP1_READOUT_TICKERS` only. The ingest-date histogram shows three manual backfills and nothing else. `instrument_daily` is the label source for the GNN (`trainer.py:~2114`, `graph_builder.py:470`) and for forward returns and microstructure. **The 77,014-row backfill restored history but not the pipeline: the next trading day starts a fresh stall, and freshness signals stay green because `fetch_instruments` does run.** This is the 106-day stall's actual root cause, not the scheduler.
*Fix:* write `instrument_daily` from `ingest_daily_prices`, one row per bar date keyed on the bar's own date (not `as_of`).
*Test:* a `daily_collection` run advances `MAX(observed_at)` for `observation_type='instrument_daily'`.

**P6.2 — CHAIN must cover the registry.** `whale_tracking` declares `schedule="*/15 * * * *"` (`whale_tracking.py:546`) and has **zero** rows in `dag_runs`, ever — it is absent from `run_chain.py:43-53`'s CHAIN, and `run_chain.py` is now the only execution path. *Test:* `set(run_chain.CHAIN) == set(registry.list_names()) - EXPLICIT_OPT_OUT`, so the next DAG added to `get_default_dags` cannot silently vanish.

**P6.3 — `db_path` is baked into six nodes at build time.** `daily_collection.py:349, 634, 668, 675, 682, 705` freeze the default `.tirra_pipeline/pipeline.db` into node params, and `dags/__init__.py:53` calls the builder with no arguments — so `run_chain.py --db-path /other.db` sends ~50 tool nodes to the new DB and those six function nodes to the live one, with no error. Any staging or test-DB run silently contaminates production. *Fix:* thread `db_path` through `get_default_dags` / `DAGRegistry.load_defaults`, or have function operators take the store from the executor context.

**P6.4 — `cftc_derived` has no DAG node at all** (only `scripts/add_cftc_derived_features.py:169`); promote it to a `daily_collection` child of `fetch_cftc`.

---

## What I would NOT fix, and why

| Item | Why not |
|---|---|
| **The 1972 `drug_regulatory` and 2000-2014 `dividend_data` dates** | Two audit findings disagree on whether these are corrupt. The evidence that they are *genuine* upstream dates (`drug_regulatory.py:184-188` `latest_submission_date`; `dividend_data.py:75` historical ex-dates) is specific and cited; the evidence that they are corrupt is only that they are old. Deleting real history to make a split look tidy is the wrong fix — P4.2's explicit `t_lo` floor handles them correctly without destroying data. |
| **`event_involves` re-architecture** (92,211 GDELT events collapsed into 9,487 static country-pair edges by `UNIQUE(a,b,link_type)`) | Correctly diagnosed — it encodes only "these two countries co-occurred at least once", with no time series, which is the one thing GDELT exists to measure. But it is a redesign, not a fix, and it is **downstream of C1 and C3**: until edges reach training at all and country→instrument flow exists, re-architecting this relation is unmeasurable. Revisit after the P5 retrain produces a baseline. |
| **Single-destination and tiny relations** (`trades_instrument` 1,384 edges → 1 destination; `tracks_protocol` 2; `domain_owned_by` 2; 6 relations under 20 edges) | The analysis is right that these cannot discriminate, but the fix (a min-edge / min-destination threshold in `_build_edge_data`) **changes the metadata tuple and therefore the checkpoint parameter shape**. Bundle it into the P5 retrain or not at all — never as a standalone change. The `trades_instrument` root cause (whale_alert resolving every transfer to BTC-USD) is worth fixing on its own merits, as a collector fix, not a graph-builder filter. |
| **`edge_attr` wiring** | It is dead (built at `graph_builder.py:838-842`, read by no model), *and* `confidence` is a per-source constant on 18 of 20 relations — a relabelling of `source`, not a belief. Wiring `edge_attr` while confidence is constant buys nothing; varying confidence while `edge_attr` is unread buys nothing. **Do both or neither**, and that is C2's decision. The `num_heads=2` coincidence that hid the shape mismatch is worth a comment either way. |
| **`entity_type_registry` 9 vs 12** | Real (`instrument`, `cftc_contract`, `maritime_area` missing; `is_valid_type("instrument")` returns `False` for the only type the return head is supervised on) but **entirely inert**: both consumers, `validate_entity_type` and `get_connected_types`, have no production caller. Fixing a registry nobody consults is motion. Fold into C7 — wire it or delete it, do not repair it in place. |
| **The 1,660 isolated entities** (all 502 vessels, 917 of 1,517 topics, 3 instruments) | Mostly a symptom of C2 and C7, not an independent defect. The 3 orphan *instruments* are worth handling in P5 — exclude them from the supervised batch so they are not scored against graph-informed peers — but the vessels need a linking rule (a design decision) and the sports topics need filtering before entity creation, not after linking. |
| **`tvl_change` null delta fields** (162,251 rows, 42% of the table, both `change_1d_pct` and `change_7d_pct` hardcoded `None` at `defi_flows.py:466-473`) | Worth fixing, but only *after* P3.1 — until the value extractor reads `tvl_usd`, even the level never reaches a node feature, so computing deltas adds fields nothing consumes. Do fix the `or 0.0` at the same line now (it collapses a missing field and a genuine zero; 36 rows currently show `tvl_usd = 0`), since that one is a P1-class write-path bug. |
| **Anything in `docs/`, `wiki/`, or `LESSONS.md` beyond the two corrections named above** (F-01's unsound std rule, F-12's "called before anything trains" claim) | Documentation drift is real here but it is not blocking, and rewriting the fuckup log mid-audit risks losing the record of what actually happened. Add new entries; correct only claims that actively mislead a future fix. |

---

## Checkpoint invalidation summary

Every existing `.pt` becomes unusable at P3. Specifically:

| Change | What it breaks |
|---|---|
| P3.1 per-type value extractor | `in_channels` for every entity type → all `type_projections` |
| P3.5 time-bounded node set | `num_nodes`, memory buffer size |
| P3.7 stable ID ordering | global ID ↔ memory row mapping (already broken by the country merge for 85.5% of entities) |
| P2.2 dedup | `count` node feature, split boundaries |
| C3 reverse edges | metadata tuple → per-relation parameter shapes |
| Relation thresholding (deferred) | metadata tuple, same reason |

Nothing in **P0, P1, P2.1, P2.4, P2.5, P6, or P7** invalidates a checkpoint. That is deliberate — it is the entire safe-to-start-today surface, it is where the highest-confidence defects live, and it is what makes the P5 retrain verifiable rather than another green-looking, empty run.
---

## Appendix — all 70 confirmed findings

| # | sev | lens | finding |
| --- | --- | --- | --- |
| 1 | critical | data-integrity | Node value features (mean_value, variance, min, max, iqr) are hard-zero for 99.8% of observations — the probe key list matches no collector's schema |
| 2 | critical | data-integrity | entity_observations has no dedup or unique constraint — 121,597 byte-identical duplicate rows, 31.6% of the table |
| 3 | critical | data-integrity | The country alpha-3/alpha-2 merge was a one-off DB migration only — no collector routes through resolve_country_key, so the next GDELT run re-splits the graph |
| 4 | critical | graph-semantics | The graph is strictly directed with no reverse edges, so 64% of all edges — including every one of GDELT's 9,487 country edges — can never reach the instrument embeddings the loss supervises |
| 5 | critical | graph-semantics | Every edge's created_at is an ingestion timestamp in 2026-04..08, so the F-14 future-blindness filter leaves 86.9% of training windows with a completely edgeless graph |
| 6 | critical | leakage | The F-14 `until` gate is keyed on ingest wall-clock, so 100% of training and validation windows are built with ZERO edges and the HGT stack is silently skipped |
| 7 | critical | leakage | HeteroMemory is never updated — `update_memory_from_events` reads an `entity_type` key that observation dicts do not have, so the memory block is a constant zero everywhere |
| 8 | critical | pipeline-orchestration | No DAG node ever writes `instrument_daily` — ingest_daily_prices() writes three other observation_types, so the 106-day stall will recur the moment the backfill ages out |
| 9 | critical | pipeline-orchestration | Zero scheduled DAG runs have ever occurred: 69/69 dag_runs are trigger='manual', and every collector is 26-147 days stale |
| 10 | critical | schema-checkpoint | HeteroMemory is never written: every observation event is dropped because store observation dicts have no 'entity_type' key — all 21 checkpoints on disk carry an all-zero TGN memory |
| 11 | critical | silent-failure | EntityAnomalyScorer._extract_value reads a dict key that PipelineStore renamed away — every observation scores 0.0, killing 2 of the 8 alert components across all 4,852 alerts |
| 12 | critical | silent-failure | A DAG node that returns {"status": "skipped"} is recorded as "completed", and the zero-rows guard that exists to catch this counts nodes rather than rows — verified on 5 live inference runs that wrote 0 rows to all three of their destination tables |
| 13 | critical | silent-failure | adversarial_scan calls the scanner with hardcoded empty inputs — layer 6 is structurally incapable of producing a flag and has reported "completed" 6 times with 0 rows |
| 14 | critical | silent-failure | The RL loop is a closed cold-start deadlock, and rl_training sets sac_trained=True even when _train_sac returns {"status": "insufficient_data"} |
| 15 | critical | test-integrity | The F-01 embedding-collapse gate has no test, gates on the wrong statistic, and swallows its own failure |
| 16 | critical | training-correctness | CSRC loss assigns return deciles by array position, not by return rank — the only loss tying the GNN backbone to returns is trained on random labels |
| 17 | critical | training-correctness | CSRC InfoNCE masks before exponentiating, so every masked-out pair contributes exp(0)=1 — the loss is a near-constant that ignores the embeddings |
| 18 | critical | training-correctness | value-head target is exactly 0.0 for 99.81% of observations — the key list in _compute_targets matches no field name any collector actually writes |
| 19 | high | data-integrity | gov_contracts writes contract period-of-performance START dates as observed_at, so 86% of its rows are future-dated; the in-code guard still admits 15 months of future |
| 20 | high | data-integrity | memory_drift is identically 0.0 in all 4,852 entity_alerts — EntityScorer snapshots memory but never runs the memory update |
| 21 | high | graph-semantics | edge_attr (confidence, age_days) is built for every edge and never read by any model — the careful reference_time leak fix guards a tensor nothing consumes |
| 22 | high | graph-semantics | All 2,752 topic->instrument edges come from a 7-entry category lookup plus unanchored substring matching; 145 edges are provably produced by substring false positives alone |
| 23 | high | graph-semantics | Five relations have exactly one destination node, so they carry zero cross-node signal while consuming 1,654 edges and full per-relation HGT parameter blocks |
| 24 | high | leakage | The node set is not time-bounded: `prepare_static()` puts all 6,110 present-day entities into every historical window, of which only 542 have any observation by the train cutoff |
| 25 | high | pipeline-orchestration | The executor's zero-rows guard counts writes to the generic `pipeline_data` blob table, so DAGs that populate nothing in their domain tables report status='completed' indefinitely |
| 26 | high | pipeline-orchestration | scripts/mp1_data_health.py — the only data-health check in the repo — grades on cumulative row counts and ignores recency entirely, and its DAG-status query references columns that do not exist |
| 27 | high | pipeline-orchestration | store_entity_observation applies no sanity bound to observed_at — the single write boundary every collector passes through accepts 1920 and 2030 dates |
| 28 | high | schema-checkpoint | The country merge changed the global node ID of 85.5% of surviving entities; the checkpoint resume path re-applies memory rows by position with only a row-count check |
| 29 | high | schema-checkpoint | validate_schema_against_store() is never called on the training path, and at its single call site it is non-blocking — LESSONS.md F-12 claims it is 'called before anything trains or scores' |
| 30 | high | schema-checkpoint | An archived checkpoint records in_channels that contradict its own weights for all 12 entity types, and load_model only warns — CheckpointSchemaDriftError is defined but raised nowhere |
| 31 | high | silent-failure | gov_contracts stamps observed_at with the contract start date and only rejects dates more than one year ahead — 912 future-dated observations (max 2030-01-30) are what push the trainer's test window to 2030 |
| 32 | high | silent-failure | ConvergenceFeatureBuilder emits constant 0.0 with quality=1.0 because no signal named 'convergence.*' has ever existed — and the only consumer ignores the quality field entirely |
| 33 | high | silent-failure | gnn_inference swallows a failed model checkpoint save and still returns status "completed" |
| 34 | high | silent-failure | The F-01 embedding-collapse gate thresholds unnormalized std, so it cannot fire during collapse — and the whole diagnostic is swallowed at log.debug if it throws |
| 35 | high | test-integrity | Four assertions in the suite are logical tautologies — `or True` / `x is not None or x is None` |
| 36 | high | test-integrity | The repo's own pre-completion gate runs the live-network tests that CI deselects, so `make quality-gate` is red whenever NYISO has no data |
| 37 | high | test-integrity | The F-02 guard only covers argparse; the TrainerConfig default is still the GNN-bypassing branch and no test exercises the runtime path |
| 38 | high | training-correctness | Train/val/test split is by row COUNT, not calendar time, over a timestamp range polluted by 912 future-dated gov_contracts rows; split boundaries also cut through a shared timestamp |
| 39 | high | training-correctness | EWC is computed, checkpointed, resumed, and logged as active — but ewc_penalty is never added to the training loss |
| 40 | high | training-correctness | _loss_from_window omits the return loss entirely, so the EWC Fisher carries zero information about the return heads and online_update never trains them |
| 41 | high | training-correctness | gdelt_frac/defi_frac thin the graph-feature stream but not the supervision windows, so the model is asked to predict 95% of GDELT events after their evidence was deleted from its input |
| 42 | high | training-correctness | evaluate() builds its graph snapshots from the UNSUBSAMPLED observation stream, so val/test inputs carry ~20x the GDELT density the model was trained on |
| 43 | high | training-correctness | The 68.65x return-loss upscale is derived from registered-entity count rather than the per-window gradient batch; the measured ratio it claims to correct is 1.81x |
| 44 | high | training-correctness | The F-01 collapse gate tests an unbounded scale statistic, so it passes while effective rank collapses — and the rank metric that would catch it is computed on the adjacent line and discarded |
| 45 | medium | data-integrity | tvl_change carries both of its delta fields hardcoded to None across 162,251 rows — 42% of the observation table is a level with no signal |
| 46 | medium | data-integrity | 16 CAMEO region and organization codes (NATO, the EU, AFRICA, ARAB LEAGUE) are registered as `country` entities holding 2,949 observations |
| 47 | medium | graph-semantics | Confidence is a per-source constant on 18 of 20 relations, so even if it were wired into the model it would be a dead feature |
| 48 | medium | graph-semantics | The UNIQUE(a,b,link_type) constraint collapses 92,211 GDELT events into 9,487 static country-pair edges, discarding all intensity, timing and tone, and duplicating 4,041 pairs as reciprocals |
| 49 | medium | graph-semantics | Six link types have under 20 edges each (62 edges, 0.35% of the graph) yet each receives its own full HGT relation parameter block |
| 50 | medium | leakage | The 912 future-dated rows do not enter historical windows, but they stretch the count-based split so the test range runs to 2030 — and the final eval windows are the only ones that receive the complete present-day link set |
| 51 | medium | pipeline-orchestration | The only deployed schedule excludes gnn_inference, entity_scoring, inference and rl_training by design, so layers 5-6 can never populate even on a correct deploy |
| 52 | medium | pipeline-orchestration | whale_tracking DAG is registered with a */15 cron schedule but is absent from run_chain.py's CHAIN and has never executed once |
| 53 | medium | pipeline-orchestration | db_path is frozen into six daily_collection node params at build time, so --db-path splits writes across two databases |
| 54 | medium | pipeline-orchestration | agent/pipeline/depth_eval.py and agent/pipeline/cross_entity.py have no non-test callers anywhere in the codebase |
| 55 | medium | pipeline-orchestration | The Change-12 tool-routing bandit is unreachable: no caller ever passes tool_router to build_daily_collection_dag |
| 56 | medium | schema-checkpoint | Checkpoints never record OBSERVATION_TYPES, so obs_type_head class indices are unrecoverable; 10 checkpoints were trained against a 45- or 46-type vocabulary and their heads are silently dropped on load |
| 57 | medium | schema-checkpoint | load_model never compares the checkpoint's in_channels against live GraphBuilder output — F-12 prevention rule 3 is unimplemented |
| 58 | medium | schema-checkpoint | convergence_clusters still references 185 country entities that the merge deleted |
| 59 | medium | test-integrity | Four parameter-clamp tests in test_transport_throughput_edge.py contain no assertion — deleting the clamp keeps them green |
| 60 | medium | test-integrity | Every preflight data-readiness check returns "passed" on any exception, and a test locks that behaviour in as correct |
| 61 | medium | test-integrity | The country-split regression guards only run where the live DB exists, so CI never runs them; one opens the 150MB production DB read-write |
| 62 | medium | test-integrity | The GDELT subsample test cannot distinguish 5% from 95%, and the default no-thinning branch has no test at all |
| 63 | medium | training-correctness | --freeze-backbone combined with the CLI's default --use-concat-head produces a run where no parameter in the loss graph is trainable, yet epochs, loss curves and checkpoints are still emitted |
| 64 | medium | training-correctness | The training banner reports GDELT rows kept from a hardcoded 901,704 that is ~10x the true count |
| 65 | low | data-integrity | contract_award records carry a constant country and an always-null award_type — the UK branch has never written a single row |
| 66 | low | graph-semantics | 1,660 entities (26% of all nodes) have no edge at all, including every vessel and 60% of topics |
| 67 | low | leakage | TS2Vec pretraining embeddings are computed over the entire observation history and injected into every historical window's node features (dormant, one CLI flag from being armed) |
| 68 | low | schema-checkpoint | entity_type_registry holds 9 of the 12 entity types in use because OntologyRegistry never reads the types actually present in `entities`, contradicting its own docstring |
| 69 | low | schema-checkpoint | Global node IDs derive from an ORDER BY with no tiebreak; 31 entities share one created_at value, so their IDs are not reproducible across runs |
| 70 | low | test-integrity | The Federal Register live tests are permanently disabled with `skipif(True)` — four tests that can never run |

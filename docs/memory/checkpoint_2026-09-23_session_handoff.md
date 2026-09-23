---
title: "Checkpoint 2026-09-23 — full session record and handoff"
tags:
  - doc/checkpoint
  - topic/architecture
  - status/current
---

# Checkpoint 2026-09-23 — session record and handoff

Session `ad1d8cd1-d5a2-4943-b4d1-f0e3e4afb377` · branch `security/withdraw-paddle-exposure`

```
cd ~/Projects/tirramind && claude --resume ad1d8cd1-d5a2-4943-b4d1-f0e3e4afb377
```

> **The working tree is RED and holds unreviewed agent work. Read §6 first.**

---

## 1. The finding that reframes everything

**The model has never been a graph network, and has never had memory.** Three
input-side defects, independently confirmed against code and the live DB:

| # | mechanism | evidence | effect |
|---|---|---|---|
| 1 | `entity_links.created_at` is an **ingest** stamp | live range 2026-04-19 → 2026-08-27; 86.9% of observations predate it | `_links_as_of` → `[]` → `het_tgn.py:642 if edge_index_dict:` skips the entire HGT loop |
| 2 | observation dicts carry no `entity_type` | `het_tgn.py:798-802` vs `store.py:2074-2084` | all 384,285 events skipped; **all 21 checkpoints: `memory.memory` nonzero-rows = 0** |
| 3 | value features probe 6 key names no collector writes | `graph_builder.py:258-276`, `:388-401`; 736/384,285 rows match | `mean_value` = 0.0 for 6,102 of 6,110 entities |
| 4 | graph is strictly directed, no reverse edges | HGT passes src→dst only | 64% of edges, incl. **all 9,487 GDELT country edges**, never propagate |

→ `HetTGN.forward` reduced to `relu(combiner(cat(type_projection(x), zeros)))`:
a per-node MLP over a type one-hot, a count, a recency scalar. **Every
`hgt_layers` and `memory.gru.*` parameter in every checkpoint is at random init
with zero gradient.**

Explains IC 0.047, `eff_rank` 1.2/64, `pred_std == raw_std`.

**The graph hypothesis is UNTESTED, not refuted.** `cot_null_result.md` (0/51)
covered 5,488 of 375,657 observations — 1.5%, at ~10% power, on a spec its own
§7 says the literature doesn't endorse.

---

## 2. Committed & pushed (8 commits)

| commit | subject |
|---|---|
| `30ad872` | docs: the audit + this handoff |
| `efc2d72` | country merge silently emptied the GDELT ghost chains |
| `3c25fb2` | **F-15** anti-collapse loss was causing the collapse |
| `e02e859` | country merge applied to live DB |
| `0b554f7` | **F-14** `build_from_cached` leaked every window |
| `4a80c6a` | `agent/pipeline/country_codes.py` + 41 tests |
| `a73700a` | 2026-09-06/07 research docs + frontmatter |

`docs/studio/` is deliberately **untracked** (named prospect lists; repo is public).

**F-14** — `_links_as_of` was always correct; `until` defaulted to `None` and all
13 call sites omitted it. Now a required kwarg (omission = `TypeError`) + an AST
guard test. Verified the guard catches an injected regression.

**F-15** — two defects in `_cross_sectional_ranking_contrastive`:
(a) `sorted_tgt, _ = tgt.sort()` discarded the permutation and was never read
again, so deciles partitioned *arrival order*; (b) `(sim*mask).exp()` sends
excluded cells to `exp(0)=1`, pinning the loss (`0.2291` identical across
epochs). 13 tests, 6 fail on old code. My first masking tests *passed* against
the bug at n=8 — replaced with a reference-implementation comparison.

---

## 3. Live DB — mutated, both states backed up

```
entities 6,110 · observations 384,285 · links 17,581 · instrument_daily 77,014 (→2026-09-22)
.tirra_pipeline/pipeline.db.bak_20260923_premerge
.tirra_pipeline/pipeline.db.bak_20260923_prebackfill
```

**Country merge** — GDELT keyed CAMEO alpha-3 (`USA`), instruments ISO alpha-2
(`US`); `entity_id = sha256("country:<key>")` so they were different entities.
The US existed twice; the CAMEO record was named **"ALASKA"** (from
`Actor1Name`), Japan **"TOYOTA"**, Canada **"SASKATCHEWAN"**.

```
entities 6172→6110 (203 merged, 141 created) · countries 279→217
observations & links unchanged · 0 self-links · 0 collisions
GDELT countries reaching an instrument ≤2 hops:  0/215 → 214/215
```
New path: `United States → Qatar --produced_in--> Natural Gas`.
16 CAMEO regional blocs (EUR/MEA/SEA…) deliberately **not** retyped — a new
entity type would be the F-12 drift class.

**instrument_daily** was stalled **106 days**; backfilled 89/89 tickers, +8,859
rows. Last labelable date moved ~mid-May → **~2026-08-23**, i.e. past the Aug-26
checkpoint cutoff — a genuine out-of-sample window now exists.
231 duplicates collapsed **keeping the richest payload** (195 pairs disagreed:
`{close,log_return,volume,realized_vol_20d}` vs `{close}`). **3 groups left
untouched — they genuinely disagree on `close`.**

---

## 4. Audit — 70 confirmed defects

`docs/research/full_audit_2026-09-23.md` · 8 lenses, 165 agents, 3,546 tool
calls, every finding refuted by 2 independent verifiers · 78 raw → **70
confirmed / 8 refuted** · 18 critical, 26 high, 20 medium, 6 low.

Criticals beyond §1:
- **121,597 byte-identical duplicate rows (31.6%)**
- **Zero scheduled DAG runs ever**: 69/69 `trigger='manual'`; collectors 26–147d stale
- **No DAG node writes `instrument_daily`** — the stall recurs by construction
- Country merge was one-off; **~19 collectors still call `entity_id_from_key("country", raw)`**
  (`ais_vessel.py:1068,1131`, `central_bank_balance.py:952,984,1016,1048`, `comtrade.py:7xx`, …)
- value-head target 0.0 for 99.81% of observations
- EWC computed/checkpointed/logged active — **`ewc_penalty` never added to the loss**
- F-01 gate thresholds unnormalised std → cannot fire during collapse; swallows its own failure
- Split is by **row count**, not calendar time: `train[1920-01-01→2025-07-16] test[2026-03-25→2030-01-30]`
- `gov_contracts` stamps `observed_at` with period-of-performance **start** date → 912 future-dated rows
- `adversarial_scan` calls the scanner with **hardcoded empty literals** — layer 6 structurally cannot flag

---

## 5. ⚠️ My own F-14 fix has a bad side effect

`created_at` = when the row was **inserted**, not when the fact became true.
Making the `until` gate mandatory blanks the graph for 86.9% of windows.
**Do not revert** — that restores the leak. Fix the *time basis*: structural
relations were true before ingestion; event relations are genuinely time-bound.
Proper fix = `entity_links.effective_from` column + backfill (audit P2.3 / C1,
**owner decision**).

---

## 6. Working tree state — RED, do not bulk-commit

7-agent workflow, one agent per file. **Zero scope violations.** Reviewers
(default-reject) rejected **4 of 7**:

| group | own tests | review | blocking objection |
|---|---|---|---|
| graph-builder | 67 ✓ | **REJECT** | classified `awarded_by`(108)/`sanctioned_under`(9) as structural → reintroduces a leak; took owner-only decision C1 unilaterally |
| het-tgn | 18 ✓ | accept | — |
| store-writepath | 25 ✓ | **REJECT** | `observed_at` floor turns **121 tests red** |
| collectors | 23 ✓ | accept | — |
| orchestration | 32 ✓ | **REJECT** | domain-table guard unconditional → false failures; `test_adversarial_scan_dag.py` fails now |
| test-integrity | 759 ✓ | accept | — |
| trainer | 20 ✓ | **REJECT** | `train()` now **raises** where it returned empty history — undisclosed contract change; 2 tests break unreported |

**Root cause of the 121 failures:** `trainer.py:499` `SyntheticGraphGenerator`
sets `t_start=0.0` and writes **relative seconds** as `observed_at` (e.g.
`2517.7`), which the new floor rejects. It was in no agent's scope. Fix that one
line and most of the red clears. The dedup guard caused **zero** regressions.

**Tests that now fail BY DESIGN (update them, don't revert the fix):**
- `test_graph_builder.py` — `TestComputeObsStats::{test_with_observations,test_goldstein_value_extraction,test_btc_amount_extraction}`, `TestBuildNodeFeatures::test_obs_stats_populated` (assert the dead 6-key probe); `TestLinkFutureBlindness::test_edge_age_uses_window_clock_not_wallclock` (unpacks 1 edge_attr, now 2); `TestBuildEdgeData::test_multiple_edge_types` (2→4); `TestGraphBuilder::test_full_build` (3→6); `::test_duplicate_links_same_count` (3→6)
- `TestBuildFromCachedIsTimeGated::test_future_link_is_absent_from_a_historical_snapshot` uses `works_for`, now structural → **re-point at an event relation; do NOT move `works_for` into EVENT_RELATIONS**
- `test_cross_domain_links.py:305` total 3→6
- `test_het_tgn.py:481-489` asserts events are skipped — that *was* the bug
- `test_gdelt_l2.py::test_name_fallback_to_country_code` asserts `"IR"`, now `"Iran"`
- `test_gov_contracts_edge.py::test_recent_sorted_by_date` asserts sort `"Start Date"` → `"Base Obligation Date"`
- `test_trainer.py::test_split_sizes` asserts the split is a partition; purge gap breaks that

**Snapshot (gitignored):** `.wip-snapshots/20260923_handoff.tracked.patch` (21
files) + `.untracked.tgz` (11 paths) · `git apply` to restore.
Workflow script: `~/.claude/projects/-Users-becmachlean-Projects-tirramind/ad1d8cd1-.../workflows/scripts/tirramind-fix-confirmed-defects-wf_54e7b92a-2a7.js`

**Recommended:** land `collectors` + `het-tgn` (both accepted) alone first, then
triage the four rejections one at a time. Do not try to land all seven.

---

## 7. 🚨 The audit's own P2.2 migration is WRONG and would destroy data

The store agent verified this read-only and it is the single most dangerous item
here. The audit prescribes dedup on
`(entity_id, source_tool, observation_type, observed_at)` — but that key has
**198,181 collisions**, far more than the 121,597 byte-identical duplicates.
Running the audit's `DELETE … ROW_NUMBER() OVER (PARTITION BY …)` would delete
**legitimate distinct rows**.

Also: the UNIQUE index cannot go in `_SCHEMA_SQL` — `_init_schema()` runs on
every `PipelineStore` construction and would raise against the live DB. It
belongs in the P2 migration, **after** dedup.

**Verify the key and the counts yourself before any DELETE.**

---

## 8. Next steps — dependency order

```
P0 instrumentation ──► P1 write-path guards ──► P2 DB repair (owner approval)
   (edge count/window)         │                        │
                               └──► P3 model inputs ──► P4 training loop ──► P5 retrain
P7 test integrity — parallel, no deps
```
Hard constraints: **P1 before P2** (else next run re-poisons) · **P2 before P4
before P5** · **P0 before P5** (a retrain with no per-window edge logging is
unfalsifiable).

**The experiment that answers the real question:** a single retrain proves
nothing. Run a controlled A/B — same code, same leak-free harness, same seed and
windows — **pre-merge backup vs post-merge live DB**. The delta is the merge's
effect. Only possible because the backup exists.

Retrain must be **from scratch, not `--resume`**: the merge changed the node id
space, so per-node TGN memory no longer corresponds.

---

## 9. Owner-only — cannot be done from this machine

- **Archive the 4 live Paddle prices** (task `withdraw_paddle_exposure` 1.3) —
  the only server-side control; IDs are in public git history.
  `pri_01m115xetf6w75k6c5zcwvzzcy` (data) · `pri_01m115xgaz1jf9mwm1dxcdxnyh`
  (entity) · `pri_01m115xhnrk809xef82cd31d29` (scheduler) ·
  `pri_01m115xk4r22eee3qf6m2ptdk3` (brief).
  Local `.env` is **sandbox-only** (`pdl_sdbx…`, `test_…`) and `update_price()`
  has no `status` field — not scriptable from here.
- **1.4** delete retained Cloudflare Pages deployments containing `pricing.html`
  (permanent `<hash>.tirramind.pages.dev` URLs)
- `support@tirramind.com` referenced 5× in terms/refunds/privacy; **tirramind.com
  has no MX records**
- The 3 `instrument_daily` rows where sources disagree on `close`
- Whether `gdelt_frac=0.05` still holds now the merge connected those rows
  (discards 87,658 of 92,211). Note `retrain_gnn.py:779-781` prints
  `int(901704 * gdelt_frac)` ≈ 45,085 — **the banner is wrong**, true value ~4,611
- `entity_links.effective_from` column + backfill (audit C1)

---

## 10. Standing context

- tirramind.com serves the **grid interconnection queue** project; `api.tirramind.com`
  is dead; no live checkout reachable. Nothing is being sold.
- Positioning: **subscription infrastructure — "shovels", not predictions.**
  Honest given §1.
- **Pre-commit will block you:** `TIRRA_WORKFLOW_TASK=tasks/active/<x>.md` is
  required on every commit; the obsidian lint blocks on FM01 (missing
  frontmatter) **vault-wide**, not just staged files; `ruff format` rewrites
  files mid-commit so re-`git add` and retry.
- CI runs `-m "not live and not slow"` with **no fixed seed and no `-p no:randomly`**;
  `make test` still runs unfiltered and hits live endpoints.
- Full suite ≈ 17 min, ~11,000 tests. Known-environmental failures:
  `test_power_grid_edge.py::TestLiveNetwork::{test_live_demand,test_live_fuel_mix}`.

### The lesson of this session, seen three times

F-14 (a correct fix nobody called), F-15 (the anti-collapse loss causing
collapse), ghost_chains (a merge that silently emptied a chain) — all the same
shape: **the mechanism was right, the wiring was wrong, and everything reported
green.** Before believing a fix works, prove a caller reaches it.

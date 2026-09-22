---
title: "Checkpoint 2026-09-23 — full session record and handoff"
tags:
  - doc/checkpoint
  - topic/architecture
  - status/current
---

# Checkpoint 2026-09-23 — full session record and handoff

Desktop session: `ad1d8cd1-d5a2-4943-b4d1-f0e3e4afb377`
Branch: `security/withdraw-paddle-exposure`

Resume this exact conversation in a terminal:

```
cd ~/Projects/tirramind && claude --resume ad1d8cd1-d5a2-4943-b4d1-f0e3e4afb377
```

**Read section 6 before touching the working tree — it is currently RED and
contains unreviewed, partially-rejected agent work.**

---

## 1. The one finding that reframes the project

**The model has never been a graph network, and has never had memory.**

Three independent input-side defects, each separately confirmed against code and
the live DB:

| # | mechanism | evidence | effect |
| --- | --- | --- | --- |
| 1 | `entity_links.created_at` is an *ingest* stamp | min 2026-04-19, max 2026-08-27; train cutoff 2025-07-16 | `_links_as_of` returns `[]`, `het_tgn` skips the whole HGT loop on an empty `edge_index_dict`. **~87-100% of train/val windows, zero edges** |
| 2 | observation dicts carry no `entity_type` key | `het_tgn.py:798-802` vs `store.py:2074-2084` | `update_memory_from_events` skipped all 384,285 rows. **All 21 checkpoints have `memory.memory` nonzero-rows = 0** |
| 3 | node value features probe six key names no collector writes | `graph_builder.py:258-276`, `:388-401` | `mean_value` identically 0.0 for 6,102 of 6,110 entities |

Consequently `HetTGN.forward` reduced to
`relu(combiner(cat(type_projection(x), zeros)))` — a per-node MLP over a type
one-hot, a count and a recency scalar. **Every `hgt_layers` and `memory.gru.*`
parameter in every checkpoint is at random initialisation and has received zero
gradient.**

This explains the weak IC (0.047), `eff_rank=1.2` of 64, and `pred_std == raw_std`.

### What this does and does not mean

- It does **not** mean the project has no edge.
- It means **no test for edge has ever actually run.** Every green training run
  measured a model that was not receiving the data.
- `docs/publications/cot_null_result.md` (0/51 hypotheses) tested **only** CFTC
  COT positioning — 5,488 of 375,657 observations, **1.5% of the corpus**, at
  ~10% power, on a specification its own Section 7 says the literature does not
  endorse. It says nothing about the entity graph.

**The cross-domain graph hypothesis is UNTESTED, not refuted.**

---

## 2. Committed and pushed (origin/security/withdraw-paddle-exposure)

| commit | what |
| --- | --- |
| `efc2d72` | the country merge silently emptied the GDELT ghost chains — fixed via deterministic entity key |
| `3c25fb2` | the anti-collapse loss was causing the collapse (LESSONS F-15) |
| `e02e859` | country merge applied to the live DB |
| `0b554f7` | F-14: `build_from_cached` leaked every window (LESSONS F-14) |
| `4a80c6a` | `agent/pipeline/country_codes.py` shared resolver + 41 tests |
| `a73700a` | 2026-09-06/07 research docs, frontmatter added |

`docs/studio/` is deliberately **untracked and unpushed** — it holds named
prospect lists and this repository is public.

### 2.1 F-14 — a correct fix nobody called

`_links_as_of` was correct and unit-tested from the day F-04 was recorded.
`build_from_cached` wired it in correctly too. But `until` carried a default of
`None` and **all 13 production call sites omitted it**, so every historical
snapshot back to 2023 received the complete present-day link set. The existing
tests exercised the pure function, where it always behaved.

Fixed: `until` is now a required keyword-only argument (omitting it is a
`TypeError`), all 13 call sites pass their window end, plus an integration test
and an AST guard that fails CI if a new call site omits it. Verified the guard
catches an injected regression.

### 2.2 F-15 — the anti-collapse loss was causing the collapse

Two defects in `_cross_sectional_ranking_contrastive`, the loss written
specifically to prevent F-01:

1. `sorted_tgt, _ = tgt.sort()` discarded the permutation and was **never read
   again**; `decile_assignments[start:end] = d` then partitioned the ORIGINAL row
   order. "Same return decile" actually meant "arrived near each other in the
   batch" — the F-01 no-true-negatives condition, reintroduced.
2. `(sim * mask).exp()` sends every excluded cell to `exp(0) = 1`, adding a
   spurious unit to both numerator and denominator. Across ~89 instruments that
   constant pinned the loss — hence `contrastive: 0.2291` byte-identical across
   epochs.

Fixed, with 13 tests (6 fail on the old code). Note: my first two masking tests
PASSED against the bug (at n=8 the constant is negligible), so they were replaced
with a comparison against an independent reference implementation.

### 2.3 The country merge

GDELT keyed countries on CAMEO alpha-3 (`USA`), the instrument universe on ISO
alpha-2 (`US`). `entity_id = sha256("country:<key>")`, so the same country was
two unrelated entities and the two halves of the graph never touched.

The US existed twice; the CAMEO record (canonical_name **"ALASKA"**, because
`gdelt.py` stored `Actor1Name`) carried 385 links and 18,980 observations and
touched nothing tradeable. Japan's record was named **"TOYOTA"**, Canada's
**"SASKATCHEWAN"**.

Applied to the live DB after a dry run:

```
entities      6172 -> 6110    (203 merged away, 141 targets created)
countries      279 ->  217
observations         375,657 unchanged
links                 17,581 unchanged    0 self-links, 0 collisions
GDELT countries reaching an instrument within 2 hops:  0/215 -> 214/215 (100%)
```

Paths that did not exist before now do:
`United States -> Qatar --produced_in--> Natural Gas`.

Deviation from spec INV-3, deliberate: the 16 CAMEO regional blocs (EUR, MEA,
SEA…) were NOT retyped. A new entity type would be the F-12 schema-drift class.

---

## 3. Live database state

| item | value |
| --- | --- |
| entities | 6,110 |
| entity_observations | 384,285 |
| entity_links | 17,581 |
| instrument_daily | 77,014 rows, latest **2026-09-22** |
| backups | `pipeline.db.bak_20260923_premerge`, `pipeline.db.bak_20260923_prebackfill` |

`instrument_daily` had been **stalled 106 days** (last row 2026-06-09) while every
other collector kept running, and nothing alerted. Backfilled all 89 tickers
(0 failed, +8,859 rows). Last labelable feature date moved from ~mid-May to
**~2026-08-23**, which is after the Aug-26 checkpoint cutoff — so a genuine
out-of-sample window exists for the first time.

231 duplicate `(entity, timestamp)` rows collapsed **keeping the richest
payload** — 195 of the pairs disagreed, one row holding
`{close, log_return, volume, realized_vol_20d}` and the other only `{close}`.
Lowest-id dedup would have silently discarded features while keeping labels.

**3 groups remain that genuinely disagree on `close`** for the same instrument
and timestamp. Left untouched deliberately — two sources contradict each other
and that needs a real answer, not a heuristic.

---

## 4. The audit — 70 confirmed defects

`docs/research/full_audit_2026-09-23.md`. 8-lens parallel sweep, 165 agents,
3,546 tool calls; every finding adversarially refuted by two independent
verifiers before it counted. 78 raw -> **70 confirmed, 8 refuted**.
18 critical, 26 high, 20 medium, 6 low.

Beyond the three in section 1, the criticals include:

- **121,597 byte-identical duplicate observation rows — 31.6% of the table.**
- **Zero scheduled DAG runs have ever occurred**: 69/69 `dag_runs` are
  `trigger='manual'`, every collector 26–147 days stale.
- **No DAG node writes `instrument_daily` at all** — the stall will recur.
- The country merge was a one-off migration; **no collector routes through
  `resolve_country_key`**, so GDELT re-splits the graph on its next run.
- The graph is strictly directed with no reverse edges: 64% of edges, including
  all 9,487 GDELT country edges, can never propagate.
- value-head target is 0.0 for 99.81% of observations.
- EWC is computed, checkpointed, resumed and logged as active — but
  `ewc_penalty` is never added to the loss.
- The F-01 collapse gate thresholds an unnormalised std, so it cannot fire during
  the collapse it exists to detect, and swallows its own failure.
- Train/val/test split is by row COUNT, not calendar time, over a range polluted
  by 912 future-dated `gov_contracts` rows (they stamp `observed_at` with the
  period-of-performance START date). Current split:
  `train[1920-01-01 -> 2025-07-16] test[2026-03-25 -> 2030-01-30]`.

---

## 5. A fix of mine with a bad side effect — READ THIS

The F-14 fix is correct in principle and wrong in its time basis.

`created_at` records when a row was INSERTED into our database, not when the
relationship became true. Earliest edge is 2026-04-19; **86.9% of observations
predate it**. So making the `until` gate mandatory means historical windows build
with **zero edges**. I closed the leak and deleted the graph.

**Do not fix this by reverting** — that restores the F-14 leak. The fix is the
time basis: structural relations (`produced_in`, `exchange_country`, `works_for`…)
describe near-static facts and were true before we ingested them; event relations
(`event_involves`, `transacts_with`…) are genuinely time-bound.

Properly this needs an `entity_links.effective_from` column plus a backfill —
a DB migration, which is an owner decision (audit C1).

---

## 6. The working tree is RED — do not commit it blindly

A 7-agent fix workflow ran with strict one-agent-per-file ownership.
**Zero scope violations** — the ownership discipline held. But reviewers, who
default to rejecting, flagged **4 of 7 patches unsound**:

| group | tests | review | blocking objection |
| --- | --- | --- | --- |
| graph-builder | 67 passed | **UNSOUND** | classified `awarded_by`/`sanctioned_under` as structural — reintroduces a leak; and took an owner-only decision (audit C1) unilaterally |
| het-tgn | 18 passed | sound | — |
| store-writepath | 25 passed | **UNSOUND** | the `observed_at` floor turns **~121 green tests red**; root cause is `SyntheticGraphGenerator` (trainer.py:499) writing relative seconds, which was in no agent's scope |
| collectors | 23 passed | sound | — |
| orchestration | 32 passed | **UNSOUND** | domain-table guard is unconditional → false failures; `test_adversarial_scan_dag.py` currently FAILS in the tree |
| test-integrity | 759 passed | sound | — |
| trainer | 20 passed | **UNSOUND** | `train()` now RAISES where it returned empty history (undisclosed contract change); 2 existing tests break, reported in neither `not_fixed` nor `out_of_scope_needed` |

Also: the graph-builder agent disclosed that **6+ existing tests in
`tests/test_graph_builder.py` now fail because they pinned the defective
behaviour** and need updating by that file's owner. In particular
`TestBuildFromCachedIsTimeGated::test_future_link_is_absent_from_a_historical_snapshot`
uses `works_for`, which is now classified structural — it must be re-pointed at
an event relation, **not** "fixed" by moving `works_for` into EVENT_RELATIONS.

Snapshot of the tree taken before handoff (gitignored):

```
.wip-snapshots/20260923_handoff.tracked.patch    21 modified files
.wip-snapshots/20260923_handoff.untracked.tgz    11 untracked paths
git apply .wip-snapshots/20260923_handoff.tracked.patch
```

Workflow script, re-runnable:
`~/.claude/projects/-Users-becmachlean-Projects-tirramind/ad1d8cd1-d5a2-4943-b4d1-f0e3e4afb377/workflows/scripts/tirramind-fix-confirmed-defects-wf_54e7b92a-2a7.js`

**Recommended first action in the new session:** triage these four rejections
one at a time, smallest blast radius first (collectors and het-tgn are already
sound and could be committed alone), rather than trying to land all seven.

---

## 7. Next steps, in dependency order

1. **P0 instrumentation** — log edge count per window. Without it a retrain is
   unfalsifiable; you cannot tell whether the input fixes worked.
2. **P1 write-path guards** — must land BEFORE any DB repair, or the next
   collection run re-poisons the table.
3. **P2 DB repair** — purge 912 future-dated rows and 121,597 duplicates.
   **Needs owner approval + backup.**
4. **P3 model inputs** — the three defects in section 1. All invalidate every
   checkpoint.
5. **P4 training loop** — calendar split, EWC into the loss, F-01 gate on
   effective rank, subsample parity.
6. **P5 retrain from scratch** — not `--resume`; the merge changed the node id
   space, so per-node TGN memory no longer corresponds.

Hard constraint: **P1 before P2 before P4 before P5**, and **P0 before P5**.

### The experiment that answers the real question

A single retrain proves nothing about whether the merge helped. Run a controlled
A/B: same code, same leak-free harness, same seed and windows, two graphs —
the **pre-merge backup** vs the **post-merge live DB**. The delta is the merge's
effect. This is only possible because the backup exists.

---

## 8. Owner-only — cannot be done from this machine

- **Archive the four live Paddle prices** (task `withdraw_paddle_exposure` step
  1.3). This is the only server-side control; the IDs are in public git history,
  so removing them from HEAD removed nothing from the internet.
  `pri_01m115xetf6w75k6c5zcwvzzcy` (data), `pri_01m115xgaz1jf9mwm1dxcdxnyh`
  (entity), `pri_01m115xhnrk809xef82cd31d29` (scheduler),
  `pri_01m115xk4r22eee3qf6m2ptdk3` (brief).
  Local `.env` is **sandbox-only** (`pdl_sdbx…`), so this cannot be scripted from
  here — and `update_price()` has no `status` field anyway.
- **Step 1.4** — delete retained Cloudflare Pages deployments containing
  `pricing.html`; each sits at a permanent `<hash>.tirramind.pages.dev` URL.
- `support@tirramind.com` is referenced 5× across terms/refunds/privacy while
  **tirramind.com has no MX records at all**.
- The 3 `instrument_daily` rows where two sources disagree on `close`.
- Whether `gdelt_frac=0.05` still makes sense now that the merge connected those
  92,211 observations (it currently discards 87,658 of them).

---

## 9. Standing context

- tirramind.com currently serves the **grid interconnection queue** project, not
  the old storefront. `api.tirramind.com` is dead. No live checkout is reachable.
- Nothing is being sold. The owner's positioning is **subscription
  infrastructure — "shovels", not predictions** — which is the honest framing
  given section 1.
- Pre-commit hooks that will block you: `TIRRA_WORKFLOW_TASK=<task file>` is
  required on every commit; the obsidian lint blocks on FM01 (missing
  frontmatter) **vault-wide**, not just on staged files.
- This session's recurring lesson, seen three times (F-14, F-15, ghost_chains):
  **the mechanism was right and the wiring was wrong, while everything reported
  green.** Check that a fix is actually reached before believing it works.

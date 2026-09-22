---
title: "Task: Reconnect the world-events graph to the instrument graph"
tags:
  - doc/task
  - status/active
  - topic/world-model
  - layer/world-model
---

# Task: Reconnect the world-events graph to the instrument graph

Status: active — **blocked on owner decisions (see Decisions Needed)**
Research: `docs/research/graph_connectivity_failure.md`
Spec: `docs/specs/graph_connectivity_repair_spec.md`

## Goal

Make a GDELT geopolitical event able to reach a tradeable instrument along
graph edges. Today 0 of 215 GDELT countries can.

## What was found (2026-09-23)

Three specialists plus direct graph analysis. Four independent defects, all
pointing the same way: **the project's core thesis has never actually been
tested, because the apparatus that was supposed to test it was broken.**

1. **The graph is two disconnected islands.** GDELT keys countries on CAMEO
   alpha-3 (`USA`), the instrument universe on ISO alpha-2 (`US`). Different
   hash, different entity. The US exists twice — the CAMEO record carries 385
   links and 18,980 observations and touches no instrument. 92,211
   `geopolitical_event` observations (24.5% of the corpus) were orphaned.
2. **F-04 leakage is live and unpatched.** `build_from_cached()`
   (`graph_builder.py:1031`) accepts `until` but defaults it to `None`, and
   **none of its 11 call sites pass it** — so every historical snapshot back to
   2023 sees the full present-day link set. The fix exists and is wired into the
   slow `build()` path only.
3. **F-02: the deployed checkpoint bypasses the GNN.** `gnn_model.pt` was saved
   2026-08-26 **15:58:36**; the commit (`7c79c02`) that made
   `--use-concat-head` default landed **16:16:38** — 18 minutes later. The
   checkpoint's own config says `use_concat_head=False`, with
   `return_concat_head.*` absent and `return_raw_head.*` present. The shipped
   model is a linear head on raw price features.
4. **F-12: the only IC number measures a deleted feature space.** The June
   `ic_results.json` used `gnn_model_phase50.pt` — 9 entity types / 46
   observation types / 23 instrument dims, vs 12 / 52 / 50 today. It is not a
   valid baseline. And `GNN-ConcatReturnHead`, the declared
   `primary_ic_strategy`, has **never been evaluated on any checkpoint that has
   ever existed** — no checkpoint contains those weights.

Also noted: the F-01 diversity gate (`std > 0.1`) **passes while collapse is
happening** — current checkpoint scores `std = 6403` but `effective_rank = 3.20`
of 64 dims (phase50: `1.71` of 128). The gate is satisfied by raw magnitude, not
by directional diversity. That is a gate asserting the absence of the bug it was
written to catch.

## Progress

- [x] Research written: `docs/research/graph_connectivity_failure.md`
- [x] Spec written: `docs/specs/graph_connectivity_repair_spec.md`
- [x] **1.1** `agent/pipeline/country_codes.py` — shared resolver.
      Complete ISO 3166-1 alpha-3→alpha-2 (249), CAMEO region blocs, legacy
      codes, `resolve_country_key()`, `country_name()`.
- [x] **1.2** `tests/test_country_codes.py` — **41 passed**, including a live-DB
      coverage test asserting all 216 codes in the graph classify.
- [x] **Dry-run simulation of the merge** (read-only, no DB mutation):

      BEFORE: 215 GDELT countries | 1-hop: 0 | within 2 hops:   0  (0%)
      AFTER : 199 GDELT countries | 1-hop: 52 | within 2 hops: 198 (99%)

      203 country entities merge away, 16 regional codes reclassified,
      **94,441 country observations join the main component.**
      This clears the spec's >=90% acceptance bar (step 4.1).

- [x] **0.1-0.2** F-04 leakage patch — `until` now required, 13 call sites fixed,
      integration test + AST guard added. LESSONS.md F-14. Full suite 10,997 passed
      (2 pre-existing TestLiveNetwork env failures).
- [ ] **0.3** Retrain after the patch — owner decision, see below
- [ ] **2.1** Route `gdelt.py` through the resolver
- [ ] **2.2** Delete the three partial private maps in `agent/tools/`
- [x] **3.1-3.3** `scripts/migrate_country_entities.py` written and **APPLIED to the
      live DB** 2026-09-23. Backup: `.tirra_pipeline/pipeline.db.bak_20260923_premerge`.
      entities 6172->6110, countries 279->217, observations 375,657 unchanged,
      links 17,581 unchanged, 0 self-links, 0 duplicate collisions.
      **Connectivity 0/215 -> 214/215 (100%) within 2 hops.**
      Deviation from INV-3: the 16 CAMEO regional blocs were NOT retyped. A new
      entity type would be the F-12 schema-drift class. They are left inert and
      flagged for a separate decision. `EU` also left alone (not an ISO country).
- [ ] **4.1** `tests/test_graph_connectivity.py` acceptance test
- [ ] **5.x** Quarantine `repair_topic_links` (1,657 edges, conf 0.6)
- [ ] **6.x** Fresh, leak-audited, BH-corrected IC measurement

## Decisions Needed From Owner

1. ~~Apply the country merge~~ — **DONE 2026-09-23**, verified, backup retained.
2. **Retrain?** Per CLAUDE.md §13 this is ask-first. It is unavoidable if any IC
   number is to mean anything: current weights were learned with leaked topology
   AND with the GNN outside the return path. Re-scoring them cannot separate
   signal from leakage.
3. **`instrument_daily` has been stalled 106 days** (last row 2026-06-09) while
   every other collector kept running. Labels need 21 trading days forward, so
   the last labelable date is ~mid-May 2026 — *before* the current checkpoint's
   training cutoff. **There is no clean out-of-sample window today.** Restarting
   that collector is a prerequisite for an honest holdout.

## Note on the null result

`docs/publications/cot_null_result.md` is not evidence against this work. It
tested CFTC COT positioning only — 1.5% of the corpus — at ~10% power, on a
specification its own Section 7 says the literature does not endorse. The
cross-domain graph hypothesis is **untested**, not refuted. That distinction is
the entire basis of this task, and the honesty standard set by that publication
applies to whatever this work eventually measures.

## Related

- [[graph_connectivity_failure]] — research
- [[graph_connectivity_repair_spec]] — spec
- [[withdraw_paddle_exposure]] — the other open task on this branch

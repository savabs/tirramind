---
title: "Spec: Reconnect the world-events graph to the instrument graph"
tags:
  - doc/spec
  - topic/world-model
  - layer/world-model
date: 2026-09-23
status: active
research: docs/research/graph_connectivity_failure.md
---

# Spec: Reconnect the world-events graph to the instrument graph

Research: `docs/research/graph_connectivity_failure.md`

## Goal

Make it possible for a GDELT geopolitical event to reach a tradeable instrument
along graph edges. Today **0 of 215** GDELT countries reach an instrument in one
hop, because GDELT keys country entities on CAMEO alpha-3 (`USA`) and the
instrument side keys them on ISO alpha-2 (`US`).

## Non-goals

- No model architecture change. No trainer change. No retrain in this spec.
- No claim about predictive edge. This spec makes the hypothesis *testable*;
  it does not test it.
- No change to `products/`, payments, or anything customer-facing.

## Layer

Layer 1 → Layer 3 boundary (entity resolution). The resolver belongs in
`agent/pipeline/`, not in any individual tool — three partial copies of this
mapping already exist in `agent/tools/{comtrade,migration_flows,global_pmi}.py`
and that duplication is the proximate cause.

## Invariants

- **INV-1**: The migration must be reversible. A backup of `pipeline.db` is
  taken before any write, and the migration records every merge it performs.
- **INV-2**: No observation or link may be dropped. Counts before and after must
  match exactly, modulo intentional de-duplication of links that become
  identical after the merge (those must be counted and reported, not silently
  swallowed — see `LESSONS.md` on silent failure).
- **INV-3**: CAMEO *regional* codes are not countries and must not be merged
  into one. They get their own entity type.
- **INV-4**: `entity_id` remains `sha256(f"{type}:{key}")[:16]`. The fix
  normalizes the *key*, it does not change the hashing scheme.

## Steps

### 0. PREREQUISITE — patch the F-04 graph-leakage bug first

**This step blocks every other step.** Reconnecting the graph without this makes
the leak strictly worse, because the merge hands the leaking code path thousands
of additional edges to leak backwards in time.

`agent/models/gnn/graph_builder.py:1031` — `build_from_cached()` calls
`_links_as_of(links, until)`, but `until` defaults to `None`, which means
"live/current, everything in scope". **None of the 11 call sites pass `until=`:**
`trainer.py:2182, 2977, 4005`; `scripts/phase40_gnn_backtest.py:144, 244, 373,
482, 734`; plus `consumer_view.py`, `phase41b_propagation_diagnostic.py`,
`honest_baseline_audit.py`, `gnn_eval_diagnostics.py`, `source_ablation.py`.

Every historical snapshot — back to 2023 — receives the complete present-day
link set. The fix (`_links_as_of`) already exists and is correctly wired into the
slow path (`build()`, `graph_builder.py:900-907`), whose own docstring states
this failure "voids the eval entirely (LESSONS.md F-04)". It was simply never
wired into the fast path that training and all IC strategies actually use.

- [ ] **0.1** Make `until` a **required** argument of `build_from_cached()`, so
      omitting it is a `TypeError` rather than a silent full-graph leak. Update
      all 11 call sites to pass the fold/snapshot timestamp.
      Verification: `grep -n "build_from_cached(" -r agent/ scripts/` shows
      `until=` at every site; a test asserting `build_from_cached()` without
      `until` raises.
- [ ] **0.2** Regression test: build a snapshot at `t0` on a fixture store whose
      links include one created after `t0`, and assert that edge is absent.
      This is the test `F-04` should have had.
- [ ] **0.3** Owner decision required: the current embeddings were **learned**
      with leaked topology, so re-scoring existing weights cannot separate real
      signal from leakage. A retrain after 0.1 is needed before any IC number
      means anything. Per CLAUDE.md §13, retraining is an ask-first action.

**Consequence for the baseline:** combined with `F-12` (the June eval used a
discarded 9-type/46-obs/23-dim feature space) and the fact that
`use_concat_head=False` on *every* checkpoint that has ever existed — so
`GNN-ConcatReturnHead`, the designated `primary_ic_strategy`, has **never been
evaluated once** — there is no usable historical IC baseline for this project.
The pre-migration baseline must be measured fresh, after 0.1.

### 1. Shared country resolver

- [ ] **1.1** Create `agent/pipeline/country_codes.py` with:
  - `ISO_ALPHA3_TO_ALPHA2`: complete ISO 3166-1 table (249 entries).
  - `CAMEO_REGION_CODES`: the non-country bloc codes observed in the live graph
    (`AFR ASA CAS CRB EAF EUR LAM MEA NMR PGS SAF SAM SAS SEA WAF WST`).
  - `LEGACY_ALPHA3`: superseded codes still emitted by GDELT (`TMP` → `TL`).
  - `resolve_country_key(code) -> str | None`: returns the canonical ISO alpha-2
    key, or `None` for regional/unknown codes. Accepts alpha-2 (idempotent),
    alpha-3, and legacy codes.
  - `country_name(alpha2) -> str`: proper display name, to replace the
    actor-name garbage (`ALASKA`, `TOYOTA`).
  - Verification: `resolve_country_key` is idempotent
    (`resolve(resolve(x)) == resolve(x)`) and covers **all 216** codes present in
    `entity_aliases` — either mapping them or classifying them as regional.

- [ ] **1.2** Unit tests in `tests/test_country_codes.py` covering: alpha-3 →
  alpha-2, alpha-2 passthrough, regional codes → `None`, legacy `TMP` → `TL`,
  unknown → `None`, and the full 216-code live-graph coverage assertion.
  Verification: `pytest tests/test_country_codes.py` green, and the coverage
  test fails loudly if a new unmapped code appears.

### 2. Stop the bleeding at the writer

- [ ] **2.1** In `agent/tools/gdelt.py` (~line 750), route the country key
      through `resolve_country_key()` before `entity_id_from_key`, and set
      `canonical_name` from `country_name()` rather than `Actor1Name`.
      Events whose actor country resolves to `None` (regional blocs) must not be
      written as `country` entities.
      Verification: a fixture batch containing a `USA` actor and an `EUR` actor
      produces one `country:US` entity and zero `country:EUR` entities.

- [ ] **2.2** Point `agent/tools/{comtrade,migration_flows,global_pmi}.py` and
      `agent/quant/n1_geo.py` at the shared resolver, deleting their partial
      private maps.
      Verification: `grep -rn '"USA": "US"' agent/` returns only
      `country_codes.py`.

### 3. Migrate the existing graph

- [ ] **3.1** Write `scripts/migrate_country_entities.py`, **`--dry-run` by
      default**, which for each `country` entity: reads its alpha-3 alias,
      resolves to alpha-2, finds-or-creates the alpha-2 entity, and repoints
      `entity_links` (both endpoints), `entity_observations`, `entity_aliases`
      and `entity_alerts`. Regional-code entities are retyped, not merged.
      It must print a before/after table and refuse to write unless `--apply`
      is passed **and** a backup path is given.
      Verification: dry-run reports non-zero merges and zero row loss.

- [ ] **3.2** Run the dry run and record the output in the task file. **Report
      the projected connectivity gain to the owner and get approval before any
      `--apply`.** The production graph is 150 MB of accumulated collection and
      is not to be mutated unattended.

- [ ] **3.3** Apply with backup. Verification below.

### 4. Acceptance test — the number that matters

- [ ] **4.1** Add `tests/test_graph_connectivity.py` asserting, against the live
      store, that **>= 90% of GDELT-referenced countries reach an instrument
      within 2 hops**. Current value: **0 of 215**. This test is the whole point
      of the spec and must fail on today's database.

- [ ] **4.2** Re-measure and record: countries reaching an instrument, total
      cross-domain (world→tradeable) link count, and the size of the largest
      connected component, before vs after.

### 5. Quarantine the noise bridge

- [ ] **5.1** `repair_topic_links` contributes 1,657 of 2,752
      `topic_relates_to_instrument` edges at confidence 0.6, and the sample in
      the research doc shows them to be close to arbitrary (a James Bond casting
      topic linked to 10-Year T-Note Futures). Mark these edges with a
      `provenance` flag so they can be included/excluded as an experimental
      variable.
      Verification: the graph can be built with and without them via one flag,
      and both link counts are reported.

- [ ] **5.2** Do **not** delete them yet. Once the graph is reconnected, whether
      they help or hurt is an empirical question for the evaluator.

### 6. Only then: measure

- [ ] **6.1** Hand off to `quant-evaluator` for a leak-audited (`F-04`),
      multiple-testing-corrected IC evaluation on the reconnected graph, using
      `gnn_model.pt` (valid against current schema) with
      `GNN-ConcatReturnHead` actually enabled per `F-02`.
      Note per `F-12`: the June `ic_results.json` is **not** a valid baseline —
      it measures a 9-entity-type / 46-obs-type / 23-instrument-dim feature
      space that no longer exists. A fresh pre-migration measurement on
      `gnn_model.pt` is required as the true baseline.

- [ ] **6.2** Report the result honestly in either direction, with power stated,
      following the standard set by `docs/publications/cot_null_result.md`.

## Definition of done

The acceptance test in 4.1 passes, a documented before/after connectivity
measurement exists, and a fresh IC number on the reconnected graph is recorded
with its power and correction — whatever that number turns out to be.

## Related

- [[graph_connectivity_failure]] — the research this spec implements
- [[graph_connectivity_repair]] — task tracking it

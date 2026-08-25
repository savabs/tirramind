---
title: "Checkpoint: 2026-04-17 Logout — Phase 35 + Repo State"
tags:
  - doc/checkpoint
  - phase/35
  - topic/gnn
  - topic/quant
  - layer/world-model
---

# Logout Checkpoint — Phase 35 + Current Repo State

Date: 2026-04-17
Task: [[quant_training_ground]]
Research: [[phase35_gnn_retrain_expanded_graph]]
Spec: [[phase35_gnn_retrain_expanded_graph_spec]]
Supersedes: [[chat_checkpoint_2026-04-18_phase35_complete]]

## Executive State

Phase 35 is functionally implemented in the codebase and marked complete in [[quant_training_ground]], but the current test surface is **not fully stable**.

What is true right now:
- `SyntheticGraphGenerator` in `agent/models/gnn/trainer.py` has been expanded to the full Phase 35 schema.
- The Phase 35 task entry is marked complete in [[quant_training_ground]].
- The current targeted verification status is:
  - `tests/test_trainer.py`: **27/27 pass**
  - `tests/test_phase34_commodity_links.py`: **34/34 pass**
  - `tests/test_phase35_gnn_retrain.py`: **32/33 pass**
- The current failing test is:
  - `TestExpandedTraining.test_obs_type_accuracy_above_random`
  - observed value: `0.0185185185`
  - threshold in test: `> 0.02`

Interpretation: Phase 35 is mostly in place, but the training/eval test is **flaky / borderline** under current conditions. Do not assume the Phase 35 suite is stably green without re-running it.

## Canonical Files To Read First Next Session

1. [[quant_training_ground]]
2. [[phase35_gnn_retrain_expanded_graph]]
3. [[phase35_gnn_retrain_expanded_graph_spec]]
4. `agent/models/gnn/trainer.py`
5. `tests/test_phase35_gnn_retrain.py`
6. `agent/models/gnn/graph_diagnostics.py`
7. `agent/tools/instrument_universe.py`
8. `tests/test_phase34_commodity_links.py`

If you only read one memory artifact, read this file and then open the four files above in that order.

## What Phase 35 Actually Changed

### 1. Synthetic graph generator expanded

`agent/models/gnn/trainer.py` now supports all 11 current entity types through `SyntheticGraphGenerator`:
- `company`
- `country`
- `vessel`
- `wallet`
- `instrument`
- `person`
- `cftc_contract`
- `organization`
- `protocol`
- `topic`
- `domain`

Important implementation detail:
- new entity count parameters were added with defaults of `0` for the newly introduced types
- `self.num_entities` filters out zero-count types entirely
- this is the backward-compat mechanism that keeps the original 4-type tests working

### 2. Observation mapping expanded

`SyntheticGraphGenerator._obs_types_for()` now maps all known entity types to plausible observation families aligned to the current graph schema.

Current mapping highlights:
- `company`: insider, legal, financial stress, pharma, lobbying, short-interest signals
- `country`: macro, geopolitical, food, internet, migration, trade, disease, campaign-finance, grid-demand signals
- `instrument`: return, volatility, volume, price movement
- `person`: insider_trade, sell_intent, campaign_finance
- `protocol`: tvl_change
- `domain`: cert_issued, dns_change
- `topic`: pageview_spike, market_probability, research_velocity, price_movement

### 3. Link generation expanded

The generator now creates the following link types in practice:
- `headquartered_in`
- `operates_in`
- `market_authorized_in`
- `lobbies_for`
- `debtor_of`
- `awarded_by`
- `works_for`
- `port_call_to`
- `exchange_based_in`
- `transacts_with`
- `trades_instrument`
- `tracks_issuer`
- `located_in`
- `fx_base_country`
- `fx_quote_country`
- `exchange_country`
- `tracks_protocol`
- `cftc_tracks`
- `sanctioned_under`

Important discrepancy:
- multiple docs/comments still say “18 link types”
- the current implementation and current test expectation set contain **19 named link types**
- the mismatch is documentation/spec drift, not a hidden extra edge in the tests

### 4. Cross-domain pattern tests exist

`tests/test_phase35_gnn_retrain.py` defines six injected patterns that exercise the new cross-type graph:
1. `person.insider_trade -> company.sell_intent via works_for`
2. `country.sanctions_listing -> company.creditor_filing via headquartered_in`
3. `vessel.port_call -> country.trade_flow via port_call_to`
4. `wallet.btc_transfer -> instrument.price_movement via trades_instrument`
5. `cftc_contract.futures_positioning -> instrument.instrument_volatility via cftc_tracks`
6. `country.pathogen_level -> country.economic_activity via sanctioned_under`

Important discrepancy versus spec:
- the spec text mentions `via operates_in` for the sanctions pattern
- the implemented test fixture currently uses `via headquartered_in`
- if Phase 35 is revisited, decide which one is canonical and align research/spec/test/code

## Verification Done In This Logout Pass

### Passing now

- `tests/test_trainer.py` -> 27 passed
- `tests/test_phase34_commodity_links.py` -> 34 passed

### Failing now

- `tests/test_phase35_gnn_retrain.py` -> 32 passed, 1 failed

Current failure details:
- test: `TestExpandedTraining.test_obs_type_accuracy_above_random`
- code path: the fixture trains for 5 epochs with `hidden_dim=32`, `memory_dim=32`, `message_dim=32`, `time_dim=8`, `num_layers=1`
- assertion requires `metrics["obs_type_acc_top1"] > 0.02`
- observed in this run: `0.018518518518518517`

Assessment:
- this is not a structural code break
- it looks like a stochastic / marginal-threshold failure in the synthetic training regime
- the Phase 35 suite should be treated as **needing stabilization**, not as fully locked down

## Most Likely Root Cause Of The Current Phase 35 Failure

The failing assertion is on downstream eval accuracy, not on graph construction or loss explosions.

That suggests one or more of:
- PyTorch randomness not fully pinned for deterministic accuracy outcomes
- the 5-epoch lightweight training fixture is too weak for a stable accuracy threshold
- synthetic class balance is broad enough that top-1 accuracy oscillates around the threshold
- the threshold was chosen from a stronger earlier run and is now too tight for repeated CI-like execution

The rest of the training checks still pass:
- total loss decreases
- losses remain finite
- embeddings are produced with correct shape

So the next session should treat this as a **test-stability problem first**, not a broken graph-schema problem.

## Recommended First Move Next Session

The next atomic step should be:

**Stabilize Phase 35 training verification before starting Phase 36.**

Preferred order:
1. Reproduce the failing accuracy test repeatedly to measure variance.
2. Decide whether to stabilize by:
   - setting explicit Torch seeds in the training fixture
   - slightly increasing epochs/model capacity in the fixture
   - relaxing the threshold to a justified margin above random
3. Only after that, move to Phase 36 design.

## Phase 36 Candidate Direction

Based on the earlier Phase 35 analysis, the strongest next technical direction is still:
- connect currently disconnected `domain` and `topic` nodes into the graph
- investigate starved country-facing edges with near-zero attention

Likely Phase 36 themes:
- add explicit graph links for `domain` and `topic`
- review whether `exchange_country`, `located_in`, `market_authorized_in`, `sanctioned_under`, `exchange_based_in` need denser evidence or better synthetic coverage
- possibly normalize / rescale `time_delta` training so that raw-seconds MSE stops dominating the total loss magnitude

## Task File State

`[[quant_training_ground]]` currently says:
- Latest completed phase: **Phase 35 — GNN Retrain on Expanded Entity Graph**
- Next queued phase: **Phase 36 — TBD**
- Latest checkpoint: **[[chat_checkpoint_2026-04-18_phase35_complete]]**

Important note:
- that linked checkpoint file is future-dated relative to the current date in this session
- this file should replace it as the canonical logout handoff

## Other Important Worktree Changes Present Right Now

These are in the repo diff and should not be forgotten when resuming, even though they were not the main focus of this logout request.

### A. Autonomous memory / lesson-promotion pipeline work exists

Files involved:
- `agent/memory/candidates.py` (new)
- `agent/memory/store.py`
- `agent/core/autonomous.py`
- `agent/config/settings.py`
- `tests/test_candidates.py` (new)

What this work does:
- stages raw lessons as candidates
- promotes only after cross-run evidence thresholds
- adds `validated` and `run_id` fields to `LearningEntry`
- adds episodic decay with archive behavior
- adds new config knobs:
  - `lesson_min_support`
  - `lesson_min_runs`
  - `episode_ttl_days`

This is significant and separate from Phase 35.

### B. Phase 30–34 surveillance/L2 persistence work is also in the working tree

Major areas present in diff:
- Phase 30 crypto linking:
  - `agent/tools/instrument_universe.py`
  - `agent/tools/whale_alert.py`
  - `tests/test_phase30_crypto_links.py`
  - `tests/test_phase30_diagnostic.py`
- Phase 31 country-signal persistence:
  - `consumer_sentiment.py`
  - `food_security.py`
  - `internet_outages.py`
  - `migration_flows.py`
  - `tests/test_phase31_country_signal_l2.py`
  - `tests/test_phase31_diagnostic.py`
- Phase 32 persistence:
  - `comtrade.py`
  - `disease_surveillance.py`
  - `political_risk.py`
  - `transport_throughput.py`
  - `tests/test_phase32_l2.py`
- Phase 33 persistence:
  - `regulatory_gazette.py`
  - `electricity_monitor.py`
  - `tests/test_phase33_l2.py`
- Phase 34 diagnostics and commodity exchange-country links:
  - `agent/models/gnn/graph_diagnostics.py`
  - `agent/tools/instrument_universe.py`
  - `tests/test_phase34_commodity_links.py`

### C. Graph builder schema is already expanded

`agent/models/gnn/graph_builder.py` in the current diff includes:
- expanded `OBSERVATION_TYPES`
- `ENRICHMENT_DIM = 54`

This is the schema that Phase 35 is targeting.

## Important Mismatches / Drift To Remember

These are the exact places where the next session can waste time if forgotten.

1. The task file points to a future-dated checkpoint.
2. Phase 35 docs/comments often say “18 link types,” but the implemented/tested set currently names 19.
3. The spec text and the Phase 35 test differ on the sanctions pattern edge (`operates_in` vs `headquartered_in`).
4. The todo list from the interrupted session is stale; actual code state is further along than that todo list implies.
5. Current targeted verification is not fully green because the Phase 35 accuracy test is flaky.

## If You Need To Resume Fast

Use this exact flow:
1. Open [[quant_training_ground]].
2. Open `tests/test_phase35_gnn_retrain.py` and inspect `TestExpandedTraining`.
3. Re-run only the failing Phase 35 test or the whole file.
4. Decide whether to stabilize the test or retrain fixture before any Phase 36 work.
5. After the suite is stable, design Phase 36 around reconnecting `domain` and `topic` plus low-attention edges.

## Reality Check On What Is Safe To Assume

Safe to assume:
- Phase 35 code landed in `trainer.py`
- Phase 34 graph diagnostics utility exists
- instrument metadata in `instrument_universe.py` includes exchange-country and protocol support
- backward-compat trainer tests still pass

Not safe to assume:
- Phase 35 suite is consistently green
- spec/research/task/comments are perfectly aligned on counts and pattern edges
- all broader worktree diffs outside Phase 35 have been revalidated together

## Related

- [[quant_training_ground]]
- [[phase35_gnn_retrain_expanded_graph]]
- [[phase35_gnn_retrain_expanded_graph_spec]]
- [[chat_checkpoint_2026-04-17_phase34_complete]]
- [[chat_checkpoint_2026-04-18_phase35_complete]]

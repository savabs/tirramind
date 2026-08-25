---
title: "Checkpoint: 2026-04-15 — Full Picture Logout Handoff"
tags:
  - doc/checkpoint
  - phase/27
  - topic/entity-linking
  - topic/graph-connectivity
  - topic/fx
  - topic/central-bank
  - topic/quant
  - layer/surveillance
  - layer/world-model
---

# Checkpoint: 2026-04-15 — Full Picture Logout Handoff

## Executive Summary

This session did **not** start implementation of the next model phase. Instead, it did the planning and prioritization work needed to avoid random expansion and to make the next stretch of graph/L2 work coherent.

At the end of the session, the project is in a good planning state:

- [[phase25_cross_domain_entity_linking]] is complete and its bookkeeping is closed.
- [[phase27_fx_country_monetary_linking]] is the active next implementation task.
- A full starved-class audit now exists in [[starved_class_audit]].
- A multi-phase remediation roadmap now exists in [[l2_expansion_roadmap]].
- [[quant_training_ground]] now reflects the new ordered phase plan beyond Phase 27.
- No code for Phase 27 has been written yet in this session.

The practical result is that the next session should begin directly with **Phase 27 implementation**, not more discovery, unless priorities change.

---

## What Happened Across The Recent Sessions

There were four major arcs leading into this checkpoint.

### 1. Phase 25 was completed

Phase 25 connected instrument nodes into the broader entity graph so instruments were no longer isolated. That work established:

- instrument → company links
- instrument → country links
- cftc_contract → instrument links
- topic and wallet integration from Polymarket and whale-related flows
- broader graph builder coverage for cross-domain paths

The important architectural consequence of Phase 25 is that the graph now has the **shape** needed for message passing across asset and non-asset entities, but not yet the **density of country-level causal signal** needed for FX and country-sensitive instruments.

### 2. The next-step question was resolved

After Phase 25, there were effectively three candidate paths:

- continue the MCP/tooling path
- continue core graph/L2 expansion
- do bookkeeping and create the next proper triad

The chosen path was:

1. close Phase 25 bookkeeping
2. create the next research/spec/task triad
3. then implement the next high-value graph/L2 phase
4. defer Phase 26.4 unless it proves materially useful

That decision matters because it explicitly prioritized **model substrate quality** over developer convenience.

### 3. Phase 27 triad was created

The next core-model task was formalized as:

- [[phase27_fx_country_monetary_linking]]
- [[phase27_fx_country_monetary_linking_spec]]
- [[chat_checkpoint_2026-04-15_phase27_preflight]]

The core design choice in that triad was:

- do **not** introduce a new `central_bank` entity type yet
- instead, reuse existing `country` nodes and attach monetary-state observations there
- enrich FX instruments with explicit two-country structure so both sides of the pair are graph-visible

This was chosen because the current graph builder still privileges seeded type encodings, so adding a brand-new entity type now would create unnecessary type-system debt.

### 4. A full starved-class audit and expansion roadmap were created

This session then moved beyond only Phase 27 and answered the broader question: **what exactly is still starved, in what order should we fix it, and what should be deliberately excluded?**

That work produced two durable documents:

- [[starved_class_audit]]
- [[l2_expansion_roadmap]]

This is the main new value from the session.

---

## Current Strategic Position

The project is currently in a transition from:

- broad surveillance surface creation
- to targeted entity-graph densification
- to improved downstream learnability for the GNN/world-model stack

The key insight from the audit is that **the next bottleneck is not the number of tools; it is where observations land in the graph**.

The most important gap is:

- country nodes exist structurally
- many things link into them
- but they receive almost no direct observations

This means the graph already has many routes through country nodes, but those nodes are mostly dark. That makes them weak relay hubs rather than informative causal anchors.

So the current direction is correct:

- enrich the graph where it is already structurally central
- prefer country-level persistence before inventing more entity types
- repair the most starved instrument classes in order of leverage, not novelty

---

## The Core Finding: What Is Actually Starved

The audit in [[starved_class_audit]] established the current ranking.

### Most starved entity class

**Country** is the most starved entity type in the graph.

Reason:

- many tools link to countries
- many instruments link to countries
- but direct observations on country nodes are extremely sparse
- at audit time, country effectively had only GDELT geopolitical observations as direct signal

This is the central structural deficiency in the graph.

### Most starved instrument classes

The prioritized order of starved instrument classes is:

1. crypto
2. FX
3. commodity futures
4. equity index futures

Important nuance:

- **Crypto** is the most isolated by raw graph degree, but only two instruments are affected.
- **FX** is the most important immediate model target because it is both materially under-connected and macro-causal.
- **Commodity futures** are under-connected, but some of that is architecturally legitimate because not every commodity should be forced into country ownership.

### Why FX was chosen before crypto

FX was chosen as the current next phase because it offers the strongest leverage per unit of work:

- 15 instruments affected immediately
- strong direct link to monetary state
- country nodes are already the right substrate
- central-bank and macro-country signals have causal relevance for FX

Crypto remains important, but its remediation is less globally amplifying than making country nodes useful.

---

## Current Graph State In Plain Terms

The graph currently has 11 seeded entity types and 27 registered observation types in the audited state.

The important practical picture is:

### Stronger entity types right now

These entity types already receive multiple direct observation families:

- company
- person
- instrument
- topic

These are not perfect, but they are no longer the darkest parts of the graph.

### Weak but functioning entity types

These have some direct observations, but limited breadth:

- wallet
- vessel
- domain
- cftc_contract
- protocol

### Structurally important but under-observed

These are the problem classes:

- country
- organization

Country is the major issue. Organization is the secondary one.

### Feature geometry

The graph builder currently uses:

- `BASE_FEAT_DIM = 14`
- `ENRICHMENT_DIM = 36`
- total node feature width = 50

This matters because every new observation type increases the observation-type distribution slice and changes feature width expectations in graph tests and integrations.

So every observation-type addition must be treated as a graph-builder change, not just a tool change.

---

## What Was Added This Session

### 1. Starved class audit

Created:

- [[starved_class_audit]]

This document does four important things:

- identifies the starved entity and instrument classes
- maps direct observation coverage by entity type
- categorizes non-L2 tools into upgrade-worthy vs aggregate-only vs utility vs ephemeral-event classes
- provides a ranked, practical prioritization instead of a vague list of ideas

### 2. Multi-phase roadmap

Created:

- [[l2_expansion_roadmap]]

This is the main long-term planning artifact from this session.

It turns the audit into a proper ordered remediation program across Phases 27–34.

### 3. Umbrella task updated

Updated:

- [[quant_training_ground]]

It now contains the expanded upcoming sequence:

- Phase 27: FX Country Wiring + Central Bank L2
- Phase 28: Country Node Enrichment (Macro)
- Phase 29: Company + Investigative L2
- Phase 30: Crypto Islands + Cross-Domain Linking
- Phase 31: Remaining Country Signals
- Phase 32: Trade + Disease + Political L2
- Phase 33: Organization + Grid Enrichment
- Phase 34: Commodity Country Links + Full Diagnostic Sweep

This means the umbrella plan is no longer only “do Phase 27 next.” It now has the broader ordered picture.

---

## The Roadmap: Whole Picture

This is the current whole-picture sequence.

### Phase 27 — FX Country Wiring + Central Bank L2

Target:

- fix FX pair under-linking
- begin making country nodes informative

Planned work:

- add explicit second-country metadata for FX pairs
- persist both country links for FX instruments
- upgrade `central_bank_balance` to persist compact country-level observations
- register new monetary observation types in the graph builder
- rerun diagnostics

Expected effect:

- country nodes go from 1 to 3 observation types
- FX pairs go from one visible country side to two

This is the current active task, but no implementation was started in this session.

### Phase 28 — Country Node Enrichment (Macro)

Target:

- deepen country nodes with high-signal macro observations

Planned tools:

- `sovereign_debt`
- `capital_flows`
- `global_pmi`

Expected new observation families:

- `sovereign_yield`
- `capital_flow`
- `economic_activity`

Expected effect:

- country nodes move from 3 to 6 direct observation types

This is the second major country-node buildout.

### Phase 29 — Company + Investigative L2

Target:

- deepen company/person signals with high-alpha entity-level sources

Planned tools:

- `bankruptcy_court`
- `foia_requests`
- `academic_preprints`

Expected new observation families:

- `bankruptcy_status`
- `investigation_signal`
- `research_velocity`

Expected effect:

- company nodes gain materially more causal and investigative depth
- person nodes also gain from FOIA-related persistence where appropriate

This phase is important because it focuses on uniquely asymmetric signal, not generic macro conditioning.

### Phase 30 — Crypto Islands + Cross-Domain Linking

Target:

- fix BTC and ETH being graph islands

Planned work:

- add crypto → protocol links
- add wallet → instrument links in whale-related paths
- ensure protocol naming is consistent with `defi_flows`

Expected effect:

- BTC and ETH stop being isolated price-only nodes
- wallet activity can influence crypto instruments through real graph paths

### Phase 31 — Remaining Country Signals

Target:

- add remaining lower-priority but still meaningful country observations

Planned tools:

- `consumer_sentiment`
- `food_security`
- `internet_outages`
- `migration_flows`

Expected new observation families:

- `consumer_confidence`
- `food_security_index`
- `internet_health`
- `migration_flow`

Expected effect:

- country nodes become one of the richest entity classes in the graph

### Phase 32 — Trade + Disease + Political L2

Target:

- medium-effort, high-signal cross-domain upgrades

Planned tools:

- `comtrade`
- `transport_throughput`
- `disease_surveillance`
- `political_risk`

Expected new observation families:

- `trade_flow`
- `border_throughput`
- `pathogen_level`
- `campaign_finance`

This is the first phase in the roadmap that deliberately steps into a more complex multi-domain entity representation problem.

### Phase 33 — Organization + Grid Enrichment

Target:

- repair the weak organization entity type
- decide whether region should become explicit or remain encoded through existing types

Planned tools:

- `regulatory_gazette`
- `electricity_monitor`

Expected new observation families:

- `regulatory_velocity`
- `grid_demand`

This phase is where organization may stop being mostly a passive link target.

### Phase 34 — Commodity Country Links + Full Diagnostic Sweep

Target:

- clean up remaining commodity under-linking where justified
- retrain and benchmark the GNN after the whole expansion program

Planned work:

- add `primary_exchange_country` for domestically anchored commodities
- leave truly global commodities structurally global
- run a full diagnostic sweep
- establish post-expansion performance baseline vs Phase 25 baseline

This is effectively the consolidation phase.

---

## What Is Explicitly Excluded For Now

The roadmap also made clear what is *not* worth doing right now.

### Deliberately excluded as aggregate-only or not worth L2 persistence yet

These are not currently planned for L2 graph persistence:

- `energy_supply`
- `treasury_receipts`
- `building_permits`
- `labor_disruptions`
- `macro_data` as a wrapper
- `market_data` as a separate L2 persistence target
- `job_postings`

Reason patterns:

- too aggregate
- better represented through the underlying more specific tool
- insufficient entity granularity for current graph goals

### Deliberately excluded as event overlays rather than persistent entity observations

These are valuable, but not as core persistent graph entities right now:

- `earthquake_proximity`
- `satellite_activity`
- `weather_alerts`

Reason:

- event identity is too ephemeral
- better used as triggers or overlays than as persistent entity observations

### Deferred architectural changes

These were explicitly deferred:

- a dedicated `central_bank` entity type
- Phase 26.4 custom TirraMind MCP server, unless it becomes a clear bottleneck reducer
- sector-entity ontology expansion for `job_postings`
- deeper ASN-level internet structure unless later diagnostics justify it

This is good discipline. The current plan avoids mixing graph densification with type-system refactors.

---

## Exact State Of Phase 27 Right Now

The task exists, the research exists, and the spec exists.

Files:

- [[phase27_fx_country_monetary_linking]]
- [[phase27_fx_country_monetary_linking_spec]]
- [[chat_checkpoint_2026-04-15_phase27_preflight]]

### Phase 27 status

Current status:

- active
- preflight complete
- implementation not started

### Phase 27 steps still all open

- 27.1 add deterministic two-country metadata for FX pairs
- 27.2 persist FX links to both country nodes
- 27.3 upgrade `central_bank_balance` to persist compact monetary observations
- 27.4 update graph registries/integration tests
- 27.5 rerun diagnostics
- 27.6 regression and checkpoint

### Immediate next action for the next session

Start implementation at **27.1**.

That means:

1. extend `InstrumentDef` in `agent/tools/instrument_universe.py`
2. add explicit second-country metadata for all deterministic FX pairs
3. update link persistence logic to create both country links idempotently
4. add the step-local tests before moving on to central-bank persistence

There is no planning blocker left before coding.

---

## Repository Status Notes

Current active task context is coherent, but the worktree is not fully clean.

### Important note about local changes already present

There are unrelated or pre-existing working-tree changes visible in the repository, including new scripts and tests such as:

- `scripts/extract_patterns.py`
- `scripts/quality_gate.py`
- `scripts/rotate_checkpoints.py`
- `scripts/session_checkpoint.py`
- multiple test files related to earlier phases and learned-architecture work

These were **not** modified as part of the starved-class planning work in this session.

For next-session safety:

- do not assume the worktree is clean
- do not revert these files blindly
- inspect them only if they become relevant to the next task

### Current planning artifacts created/updated in this session

Created:

- [[starved_class_audit]]
- [[l2_expansion_roadmap]]

Updated:

- [[quant_training_ground]]

Already existing and still governing next implementation:

- [[phase27_fx_country_monetary_linking]]
- [[phase27_fx_country_monetary_linking_spec]]

---

## Current Knowledge Graph / Documentation State

At the end of the session, the main docs relevant to the next work are:

### Core task and roadmap docs

- [[quant_training_ground]]
- [[phase27_fx_country_monetary_linking]]
- [[phase27_fx_country_monetary_linking_spec]]
- [[starved_class_audit]]
- [[l2_expansion_roadmap]]

### Historical baseline docs still relevant

- [[phase25_cross_domain_entity_linking]]
- [[phase25_gnn_diagnostic]]
- [[7b-Z_central_bank_balance_sheets]]
- [[e2e_global_integration]]

### Earlier checkpoint relevant to the current branch point

- [[chat_checkpoint_2026-04-15_phase27_preflight]]

This is enough for a cold restart without reopening broad codebase exploration.

---

## Important Architectural Decisions Confirmed This Session

### 1. Reuse country entities before adding central-bank entities

This is the most important design decision for the next phase.

Reason:

- current graph builder still has seeded-type assumptions
- country nodes are already central in the graph
- central-bank state can be usefully projected onto country nodes now
- this delivers signal faster with less schema debt

### 2. Prefer densifying central hubs over adding more disconnected tools

The current frontier is not “more tools at any cost.”

It is:

- move strong existing L1/L2 data into the graph
- land those observations on the right nodes
- improve the causal substrate for GNN/world-model learning

### 3. FX before crypto, despite crypto being more isolated by degree

This was intentional.

Reason:

- FX touches more instruments and more macro structure
- country-node enrichment benefits many non-FX instruments too
- crypto-link repair is still important, but more local in its effect

### 4. Explicit exclusions are part of the plan, not omissions

The roadmap is intentionally not trying to upgrade every tool.

That is a good sign. It means the project now has a principled filter for what is worth graph persistence.

---

## Risks And Cautions For The Next Session

### 1. Feature dimension drift

Any new observation type added in Phase 27 and later will require careful updates to:

- graph builder observation type registry
- derived feature dimensions
- graph tests expecting specific shapes or observation counts

This is a recurring failure mode.

### 2. Country code consistency

Country-node persistence will only be clean if country-code usage remains deterministic and normalized.

Particular care:

- ECB maps to `EU`
- FX pairs involving USD must explicitly include `US`
- do not guess mappings for ambiguous or basket-like instruments

### 3. Mixed publication cadences for central-bank data

Central-bank balance data and policy-rate data may not share cadence.

The next implementation should preserve source timestamps rather than force artificial alignment.

### 4. Dirty worktree risk

Because the repository already has other active changes, next-session edits should stay strictly scoped to the Phase 27 files and tests unless intentionally broadening scope.

---

## Suggested Next Session Opening Move

If resuming cleanly, the next session should start with:

1. open [[phase27_fx_country_monetary_linking]]
2. open [[phase27_fx_country_monetary_linking_spec]]
3. implement 27.1 and 27.2 together as one small batch
4. add/extend edge-case tests immediately for FX metadata and dual-country linking
5. only then move to `central_bank_balance`

That sequence is better than trying to do all of Phase 27 in one blind pass.

Why:

- it isolates the instrument-registry change from the persistence/tooling change
- it gives a quick graph-structure win early
- it reduces the number of moving pieces when tests fail

---

## Final Status Snapshot

### Completed in the project before this session

- Phases 0–25 broadly complete as recorded in [[quant_training_ground]]
- Phase 25 specifically complete and closed out

### Completed in this session

- full starved-class research pass
- full L2 expansion phase map across Phases 27–34
- umbrella task updated with the new roadmap
- next-step ambiguity removed

### Not completed in this session

- no Phase 27 code implementation
- no Phase 27 tests added
- no new runtime verification performed for Phase 27

### Most important next implementation target

- [[phase27_fx_country_monetary_linking]]

---

## Minimal Cold-Start Prompt For The Next Session

If starting a new chat, the shortest accurate resume prompt is:

> Resume from [[chat_checkpoint_2026-04-15_full_picture_logout]]. Phase 25 is done. Planning artifacts [[starved_class_audit]] and [[l2_expansion_roadmap]] are complete. Start implementing [[phase27_fx_country_monetary_linking]] from spec step 27.1, then carry into 27.2 with tests.

---

## Related

- [[quant_training_ground]]
- [[phase27_fx_country_monetary_linking]]
- [[phase27_fx_country_monetary_linking_spec]]
- [[starved_class_audit]]
- [[l2_expansion_roadmap]]
- [[chat_checkpoint_2026-04-15_phase27_preflight]]
- [[phase25_cross_domain_entity_linking]]
- [[phase25_gnn_diagnostic]]
- [[7b-Z_central_bank_balance_sheets]]

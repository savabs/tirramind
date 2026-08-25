---
title: "Research: Phase 36 — Connect Disconnected Entity Types"
tags:
  - doc/research
  - phase/36
  - topic/entity-linking
  - topic/gnn
  - layer/world-model
  - layer/surveillance
---

# Research: Phase 36 — Connect Disconnected Entity Types

## Context

Phase 35 retrained the GNN on the full 11-type / 45-obs / 19-link schema and
revealed two structurally disconnected entity types:

| Entity Type | Mean Degree | Producers | Obs Types | Links Created |
|---|---|---|---|---|
| **domain** | 0 | cert_transparency, dns_monitor | cert_issued, dns_change | **0** |
| **topic** | 0 | polymarket, academic_preprints | market_probability, research_velocity, pageview_spike | **0** |

Both types receive observations but participate in zero entity links.
The GNN cannot propagate signal to or from them.  They are functionally
dead nodes — observations exist but cannot influence any other entity's
prediction through message-passing.

### Secondary Finding: Starved Country Edges

Five country-facing link types have zero GNN attention:
`market_authorized_in`, `sanctioned_under`, `exchange_country`, `located_in`,
`exchange_based_in`.  Investigation shows these links **are** produced by
their tools, but almost all point to the single country node "US" — giving
the GNN zero discriminative signal.  This is a data-diversity issue (US-centric
instrument universe), not a missing-link issue.  It cannot be fixed by adding
more link code; it requires adding non-US instruments to the universe.
**Out of scope for Phase 36.**

---

## Problem Analysis

### Domain entities

**Producers:**
- `cert_transparency._persist_entities_inner()` (L478–L510): registers
  `entity_type="domain"`, stores `cert_issued` observations.  The domain
  string (e.g. `stripe.com`) is the entity key.  No `link_entities()` call.
- `dns_monitor._persist_entities_inner()` (L955–L983): registers
  `entity_type="domain"`, stores `dns_change` observations.  Same key.
  No `link_entities()` call.

**Available linking data:**
- The domain string itself carries implicit company ownership (e.g.
  `apple.com` → Apple, `stripe.com` → Stripe).
- The tool's *input parameter* is typically a company domain — the user/pipeline
  already knows which company owns the domain.
- No external lookup is needed: we can extract the base domain, strip TLD,
  normalize, and match against existing company entity canonical names.

**Proposed link type:** `domain_owned_by` — domain → company.
- Source: `cert_transparency`, `dns_monitor`.
- Confidence: 0.8 (heuristic match, not verified ownership).
- Implementation: after persisting the domain entity, attempt to find a
  matching company entity by normalizing the domain name and comparing to
  `normalize_company_name()` output of known companies.  Use the
  `INSTRUMENTS` issuer list as the match corpus (deterministic, no DB query).

### Topic entities

**Producers:**
- `polymarket._persist_entities_inner()` (L344–L394): registers
  `entity_type="topic"` keyed by market `slug`.  Stores `market_probability`
  obs.  Metadata includes `category` (politics, crypto, finance, etc.) and
  the market `question` text.  No links.
- `academic_preprints._persist_entities_inner()` (L504–L530): registers
  `entity_type="topic"` keyed by arXiv category (e.g. `cs.AI`, `q-fin`).
  Stores `research_velocity` obs.  No links.

**Available linking data for polymarket topics:**
- `category` field maps to normalized categories via `_TAG_CATEGORIES`:
  `crypto`, `finance`, `politics`, `geopolitics`, `tech`, `science`, `sports`.
- The `question` text often names specific instruments/assets (e.g.
  "Will Bitcoin hit $100k?", "Will the S&P 500 close above 5000?").
- Polymarket Gamma API event `tags` include slugs like `bitcoin`, `ethereum`,
  `stocks`, `interest-rates` — direct asset-class matches.

**Available linking data for academic_preprints topics:**
- arXiv category string (e.g. `q-fin`, `cs.AI`).  Too coarse for instrument
  linking.  But `trials` mode already creates company entities from sponsor
  names — a `topic_studied_by` link from arXiv category → company is very
  low signal.  **Skip academic_preprints topic linking for now** — the arXiv
  category granularity is too coarse to produce meaningful graph edges.

**Proposed link types:**

1. `topic_relates_to_instrument` — topic → instrument.
   - Source: `polymarket`.
   - Matching: static keyword map from Polymarket tag slugs and question
     keywords to INSTRUMENTS tickers.  e.g. `bitcoin` → `BTC-USD`,
     `ethereum` → `ETH-USD`, `stocks` → `SPY`, `interest-rates` → `ZN=F`.
   - Confidence: 0.7 (keyword heuristic).
   - This is the highest-value link because it directly connects prediction
     market signal to tradeable instruments.

2. `wallet_bets_on` — wallet → topic.
   - Source: `polymarket_whales`.
   - The whale tool already has per-wallet trade data including the market
     the wallet traded on.  This naturally produces wallet→topic links.
   - Requires the polymarket_whales tool to know the market slug for each
     trade so it can resolve the topic entity.
   - **Problem:** the current `_persist_wallet_entities()` only stores
     aggregate wallet stats, not per-trade market slugs.  It does not have
     the slug/condition_id in its persisted data.  Adding per-trade linking
     would require structural changes to the persistence layer.
   - **Decision: defer wallet→topic for Phase 36.**  Focus on the simpler,
     higher-value links first.  The wallet entity type already has graph
     connectivity through `transacts_with` and `trades_instrument` edges.

### SyntheticGraphGenerator gap

The synthetic generator defaults to `num_topics=0, num_domains=0` and has
**no link generation code** for these types.  Phase 36 must add:
- domain → company (`domain_owned_by`) link generation.
- topic → instrument (`topic_relates_to_instrument`) link generation.
- Default `num_topics` and `num_domains` > 0 so these types appear in
  training data.

---

## Implementation Plan (high-level)

1. Add `domain_owned_by` links to `cert_transparency._persist_entities_inner()`.
2. Add `domain_owned_by` links to `dns_monitor._persist_entities_inner()`.
3. Add `topic_relates_to_instrument` links to `polymarket._persist_entities_inner()`.
4. Update SyntheticGraphGenerator: add domain/topic link generation code,
   set `num_topics=3, num_domains=3` as defaults.
5. Add new injected patterns for domain→company and topic→instrument causal chains.
6. Edge case tests for all new linking code.
7. Regression run.

### What we are NOT doing

- **wallet→topic linking**: deferred, requires structural changes to whale persistence.
- **academic_preprints topic linking**: arXiv categories too coarse.
- **Fixing starved country edges**: data-diversity issue, needs non-US instruments.
- **Adding new observation types**: this is a linking-only phase.

---

## Risks

1. **Domain→company heuristic accuracy**: `stripe.com` → `stripe` is easy.
   `msn.com` → `microsoft` is not.  We accept imperfect recall; false
   positives are worse than missed matches.  Use a conservative whitelist
   approach with the INSTRUMENTS issuer names as the match corpus.
2. **Topic→instrument keyword ambiguity**: "Will Tesla stock go up?" matches
   both a company and an instrument.  We link to the instrument (the
   tradeable entity), not the company, to maximize GNN utility.
3. **Polymarket tag coverage**: not all markets have tags that map to our
   instruments.  Only matched markets get links; unmatched ones remain
   isolated.  This is acceptable — partial connectivity is better than none.

---

## Related

- [[phase35_gnn_retrain_expanded_graph]]
- [[phase35_gnn_retrain_expanded_graph_spec]]
- [[quant_training_ground]]
- [[l2_expansion_roadmap]]
- [[tool_priority_ranking]]

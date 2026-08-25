---
title: "Spec: Entity Linking Layer"
tags:
  - doc/spec
  - phase/17
  - topic/surveillance
  - topic/world-model
  - layer/surveillance
  - layer/world-model
---

# Spec: Entity Linking Layer

## Goal

Add `link_entities()` calls to 9 of 12 existing L2 tools. After this phase, the GNN will have typed edges between entities — currently it has zero. No new tools, no new entity types, no new observation types. Pure wiring.

## Guardrail

This phase may hard-code only **explicit factual edges present in source data**. It may not hard-code learned behavior.

In scope:
- Deterministic extraction of relationships explicitly stated by the source record
- Deterministic entity creation needed to materialize those relationships
- Confidence values that reflect source certainty, not predictive strength

Out of scope:
- Hand-coded predictive scores
- Hand-coded market-direction logic
- Hand-coded inferred/latent relationships not explicit in the source
- Any rule that substitutes for GNN learning, pattern extraction, or supervision

## PipelineStore API (existing)

```python
store.link_entities(
    entity_id_a=str, entity_id_b=str, link_type=str,
    source=str, confidence=float, metadata=dict|None
)
```

Entity IDs are produced by `entity_id_from_key(entity_type, key)` → SHA-256[:16].

Interpretation rule: links created here are structural inputs to the graph, not model outputs and not trading signals.

## Files Affected

| File | Change |
|------|--------|
| `agent/tools/insider_filings.py` | Add `works_for` link (person→company) |
| `agent/tools/form144.py` | Add `works_for` link (person→company) |
| `agent/tools/whale_alert.py` | Add `transacts_with` link (wallet→wallet) |
| `agent/tools/gdelt.py` | Add `event_involves` link (country→country) |
| `agent/tools/ais_vessel.py` | Add `port_call_to` link (vessel→country); create country entity |
| `agent/tools/lobbying.py` | Add `lobbies_for` link (company→company); create client company entity |
| `agent/tools/cert_transparency.py` | Add `cert_for` link (domain→company); create company entity from cert org field |
| `agent/tools/interconnection_queue.py` | Add `located_in` link (company→country); create US country entity |
| `agent/tools/patent_filings.py` | Add `patents_in` link (company→country); create US country entity |
| `tests/test_entity_linking.py` | New test file — comprehensive edge case suite |

### Files NOT changed (no natural cross-entity link data)

- `agent/tools/dns_monitor.py` — cloud provider is a string, not an entity
- `agent/tools/wikipedia_pageviews.py` — topic entities are standalone
- `agent/tools/defi_flows.py` — "chain" is not a current entity type

## Implementation Steps

### 17a: Tier A — person→company links (insider_filings + form144)

**17a.1: insider_filings — add `works_for` link**

In `_persist_entities_inner()`, after both `register_entity()` calls (company and person), add:

```python
if issuer_cik and reporter_cik:
    store.link_entities(
        entity_id_a=insider_eid,
        entity_id_b=company_eid,
        link_type="works_for",
        source="insider_filings",
        confidence=1.0,
        metadata={"relationship": txn.get("role", "")},
    )
```

**Insertion point:** After the `store_entity_observation()` call (inside the `if reporter_cik:` block), since both `insider_eid` and `company_eid` are available there.

**17a.2: form144 — add `works_for` link**

Same pattern as insider_filings:

```python
if issuer_cik and reporter_cik:
    store.link_entities(
        entity_id_a=insider_eid,
        entity_id_b=company_eid,
        link_type="works_for",
        source="form144",
        confidence=1.0,
        metadata={"relationship": f.get("relationship", "")},
    )
```

### 17b: Tier B.1 — same-tool intra-entity links

**17b.1: whale_alert — add `transacts_with` link (wallet→wallet)**

Whale alert creates wallet entities for each address. The transaction naturally links sender→receiver. In `_persist_entities_inner()`, after persisting both input and output wallet entities:

```python
for sender_eid in sender_eids:
    for receiver_eid in receiver_eids:
        if sender_eid != receiver_eid:
            store.link_entities(
                entity_id_a=sender_eid,
                entity_id_b=receiver_eid,
                link_type="transacts_with",
                source="whale_alert",
                confidence=1.0,
                metadata={"tx_hash": tx.get("hash", ""), "btc": tx.get("value_btc", 0)},
            )
```

**17b.2: gdelt — add `event_involves` link (country→country)**

GDELT events have actor1.country and actor2.country. When both exist and differ, create a bilateral link:

```python
if country_a_eid and country_b_eid and country_a_eid != country_b_eid:
    store.link_entities(
        entity_id_a=country_a_eid,
        entity_id_b=country_b_eid,
        link_type="event_involves",
        source="gdelt",
        confidence=0.9,
        metadata={"event_code": event.get("event_code", "")},
    )
```

Note: GDELT already creates country entities for actor1.country. Need to also create entity for actor2.country if not already done.

### 17c: Tier B.2 — cross-entity links requiring new entity creation

**17c.1: ais_vessel — add `port_call_to` link (vessel→country)**

AIS vessel data includes `destination` field. Extract country from destination string (e.g., "ROTTERDAM" → "NL"). Create country entity if not exists, then link.

```python
if destination_country:
    country_eid = entity_id_from_key("country", destination_country)
    store.register_entity(
        entity_type="country",
        canonical_name=destination_country,
        entity_id=country_eid,
    )
    store.link_entities(
        entity_id_a=vessel_eid,
        entity_id_b=country_eid,
        link_type="port_call_to",
        source="ais_vessel",
        confidence=0.8,
    )
```

Will need a small destination→country mapping dict (20-30 major ports).

**17c.2: lobbying — add `lobbies_for` link (company→company)**

Lobbying data has registrant (already persisted) and client info. Create client entity if different from registrant:

```python
if client_name and client_name != registrant_name:
    client_eid = entity_id_from_key("company", normalize_company_name(client_name))
    store.register_entity(
        entity_type="company",
        canonical_name=normalize_company_name(client_name),
        entity_id=client_eid,
    )
    store.link_entities(
        entity_id_a=registrant_eid,
        entity_id_b=client_eid,
        link_type="lobbies_for",
        source="lobbying",
        confidence=0.9,
    )
```

**17c.3: cert_transparency — add `cert_for` link (domain→company)**

SSL certificates contain an organization field. When present and non-empty, create a company entity and link:

```python
if org_name:
    company_eid = entity_id_from_key("company", normalize_company_name(org_name))
    store.register_entity(
        entity_type="company",
        canonical_name=normalize_company_name(org_name),
        entity_id=company_eid,
    )
    store.link_entities(
        entity_id_a=domain_eid,
        entity_id_b=company_eid,
        link_type="cert_for",
        source="cert_transparency",
        confidence=0.7,
    )
```

Lower confidence (0.7) because cert org field is self-reported and often abbreviated.

### 17d: Tier C — fixed-country links

**17d.1: interconnection_queue — add `located_in` link (company→country)**

All generators in EIA data are US-based:

```python
us_country_eid = entity_id_from_key("country", "US")
store.register_entity(entity_type="country", canonical_name="United States", entity_id=us_country_eid)
store.link_entities(
    entity_id_a=company_eid,
    entity_id_b=us_country_eid,
    link_type="located_in",
    source="interconnection_queue",
    confidence=1.0,
    metadata={"state": record.get("state", "")},
)
```

**17d.2: patent_filings — add `patents_in` link (company→country)**

Same US-only pattern:

```python
us_country_eid = entity_id_from_key("country", "US")
store.register_entity(entity_type="country", canonical_name="United States", entity_id=us_country_eid)
store.link_entities(
    entity_id_a=company_eid,
    entity_id_b=us_country_eid,
    link_type="patents_in",
    source="patent_filings",
    confidence=1.0,
)
```

### 17e: Edge case test suite

**17e.1: Create `tests/test_entity_linking.py`**

Test categories:

1. **Link creation tests per tool** (9 tools × 2-3 tests each):
   - Normal case: both entities exist, link created
   - Missing data: entity field is empty/None → link gracefully skipped
   - Dedup: same link called twice → no error (INSERT OR IGNORE)

2. **Cross-tool link consistency** (4-5 tests):
   - Same company created by insider_filings and lobbying → same entity_id
   - Same country created by gdelt and ais_vessel → same entity_id
   - Links from different tools targeting same entity are both queryable

3. **Edge cases** (8-10 tests):
   - Self-link prevention (entity_id_a == entity_id_b) → ValueError
   - Empty entity_id fields → link skipped gracefully
   - Very long metadata → doesn't crash
   - Concurrent persistence from multiple tools → no database corruption
   - PipelineStore not available (no pipeline kwarg) → persistence silently skipped
   - normalize_company_name fails → fallback to raw name

4. **Graph builder integration** (3-4 tests):
   - Links created by tools → graph builder builds correct HeteroData edge_index
   - Multiple link types → separate edge_type triplets
   - Bidirectional query works (query by either entity)

## Edge Cases

- **Entity not yet registered when link is attempted:** Always `register_entity()` before `link_entities()`. The register call is idempotent.
- **Self-links:** `link_entities()` raises `ValueError` for `entity_id_a == entity_id_b`. Guard with `if eid_a != eid_b:` before calling.
- **Empty/None fields:** If the data field needed for the link is missing, skip the link silently. Don't error.
- **Entity ID consistency across tools:** All tools use `entity_id_from_key(type, key)` which is SHA-256-deterministic. The key must be the same string for the same real-world entity. For companies, all tools use `normalize_company_name()` → consistent.
- **Country keys:** Need to standardize on ISO 3166-1 alpha-2 (e.g., "US", "NL", "JP") or FIPS codes (used by GDELT). Check what GDELT uses and match.

## Testing Plan

- All tests use in-memory SQLite (`:memory:`) via PipelineStore
- Mock HTTP calls — no live API access during tests
- Verify link count, link types, entity IDs, confidence values
- Verify graph builder produces correct HeteroData from persisted links
- Verify no predictive fields or directional scoring are added as part of link persistence
- Target: 60-80 tests across all categories

## Related

- [[entity_linking_layer]] — research for this spec
- [[tool_priority_ranking]] — diagnostic analysis that identified this as highest priority
- [[l2_tool_expansion]] — Phase 13 L2 expansion (tools being modified)
- [[l2_tool_expansion_spec]] — Phase 13 spec (original L2 persistence pattern)

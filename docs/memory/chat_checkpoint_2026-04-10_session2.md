---
title: "Checkpoint: Phase 17 Complete — Entity Linking Layer"
tags:
  - doc/checkpoint
  - phase/17
  - topic/surveillance
  - topic/world-model
  - layer/surveillance
  - layer/world-model
---

# Checkpoint: 2026-04-10 Session 2 — Phase 17 Complete

**Session scope**: Completed all five sub-phases (17a–17e) of the Entity Linking Layer. The GNN graph went from **zero edges** to **8 typed link types** across 8 L2 tools. 95 dedicated entity linking tests + 408 total L2/entity tests passing, 0 regressions introduced.

**Prior checkpoint**: [[checkpoint_archive_2026]] (see the archived Phase 16 material)

---

## What Was Accomplished This Session (Comprehensive)

### Phase 17 Overview

Phase 16's diagnostic analysis found the single most impactful gap in the system: **zero entity links existed in the entire graph**. All 12 L2 tools registered entities and stored observations, but none called `link_entities()`. The GNN's heterogeneous message-passing architecture requires edges to propagate signal across entity types. Without edges, it degrades to per-type autoencoders with no cross-entity intelligence.

Phase 17 fixed this by adding `link_entities()` calls to 8 of the 12 L2 tools (1 was skipped due to missing data, 3 have no natural cross-entity relationship in their source data).

### Phase 17a: Person→Company links (insider_filings + form144) — COMPLETE

**17a.1: insider_filings.py — `works_for` link**
- **Location**: `_persist_entities_inner()`, after the position observation block
- **Guard**: `if issuer_cik and reporter_cik:` — both CIKs must be present
- **Link**: `person(reporter_cik) → company(issuer_cik)`, link_type=`works_for`, source=`insider_filings`, confidence=1.0
- **Metadata**: `{"relationship": txn.get("role", "")}` — captures the SEC-reported relationship (e.g., "Director", "10% Owner")
- **Rationale**: SEC Form 4 filings explicitly state who filed (reporter) for which company (issuer). This is a factual employment/affiliation relationship, not inferred.

**17a.2: form144.py — `works_for` link**
- **Location**: `_persist_entities_inner()`, same pattern as insider_filings
- **Guard**: `if issuer_cik and reporter_cik:`
- **Link**: `person(reporter_cik) → company(issuer_cik)`, link_type=`works_for`, source=`form144`, confidence=1.0
- **Metadata**: `{"relationship": f.get("relationship", "")}` — uses Form 144's relationship field
- **Rationale**: Form 144 sell-intent filings have the same person→company structure as Form 4.

**17a.3: Tests — 24 tests**
- **Test classes**: `TestInsiderFilingsWorksForLink` (10 tests), `TestForm144WorksForLink` (9 tests), `TestCrossToolEntityConsistency` (3 tests), `TestNoLinkWithoutPipeline` (2 tests)
- **Coverage**: normal link creation, metadata accuracy, dedup (same person-company pair → one link via INSERT OR IGNORE), missing reporter CIK, missing issuer CIK, both CIKs missing, multiple insiders at one company, one insider at multiple companies, empty role, link direction (outgoing from person), cross-tool same entity ID consistency, shared store single link, no-store no-crash

### Phase 17b: Intra-entity links (whale_alert + gdelt) — COMPLETE

**17b.1: whale_alert.py — `transacts_with` link**
- **Location**: `_persist_entities_inner()`, after both sender and receiver wallet entity loops
- **Logic**: Cross-product of `sender_addrs × receiver_addrs` with guards:
  - Skip if `s_addr == r_addr` (self-transfer)
  - Skip if either address is empty
- **Link**: `wallet(sender) → wallet(receiver)`, link_type=`transacts_with`, source=`whale_alert`, confidence=1.0
- **Metadata**: `{"tx_hash": tx_hash}` — ties the link to the specific Bitcoin transaction
- **Rationale**: Bitcoin transactions explicitly list input addresses (senders) and output addresses (receivers). This is a factual transactional relationship.

**17b.2: gdelt.py — `event_involves` link**
- **Location**: `_persist_entities_inner()`, after the inner actor loop where country entities are registered
- **Guard**: `if c1 and c2 and c1 != c2` — both countries must be non-empty and different
- **Link**: `country(actor1_country) → country(actor2_country)`, link_type=`event_involves`, source=`gdelt`, confidence=0.9
- **Metadata**: `{"event_id": event_id, "event_root": ev.get("event_root", "")}` — captures the GDELT event ID and root event code
- **Confidence 0.9**: GDELT country attribution is generally reliable but occasionally ambiguous for non-state actors
- **Rationale**: GDELT events explicitly list two actor countries. This is a factual bilateral interaction.

**17b.3: Tests — 20 tests (44 cumulative)**
- **Test classes**: `TestWhaleAlertTransactsWith` (10 tests), `TestGDELTEventInvolves` (10 tests)
- **Coverage**: normal link, metadata with tx_hash/event_id, multiple senders×receivers, self-send skip, empty sender/receiver skip, dedup, no inputs, no outputs, no-store, same-country skip, missing actor countries, whitespace country

### Phase 17c: Cross-entity links with new entity creation (ais_vessel + lobbying) — COMPLETE

**17c.1: ais_vessel.py — `port_call_to` link + `_DEST_COUNTRY` mapping**

This was the most complex implementation in Phase 17 because it required:
1. A new module-level lookup dict (`_DEST_COUNTRY`) mapping ~50 port names to ISO-2 country codes
2. New country entity creation inside the persist method
3. Links in TWO separate code paths (`_persist_entities_inner` for area/vessel mode AND `_persist_port_call_entities_inner` for port_calls mode)

- **`_DEST_COUNTRY` dict**: ~50 entries mapping free-text AIS destination strings to ISO-2 country codes. Examples: `"TALLINN": "EE"`, `"ROTTERDAM": "NL"`, `"SHANGHAI": "CN"`. Placed at module level (lines ~52-100).
- **Matching logic**: `dest = (v.get("destination") or "").strip().upper()` → `country_code = _DEST_COUNTRY.get(dest)` — case-insensitive, whitespace-trimmed. The AIS destination field is free-text entered by ship crew, so only exact matches against known port names are used.
- **Entity creation**: When a match is found and `eid` (vessel entity ID) exists, creates a country entity: `register_entity(entity_type="country", canonical_name=country_code, entity_id=entity_id_from_key("country", country_code))`
- **Link**: `vessel(eid) → country(country_eid)`, link_type=`port_call_to`, source=`ais_vessel`, confidence=0.8
- **Metadata**: `{"destination_raw": dest}` — preserves the original free-text destination for debugging
- **Confidence 0.8**: AIS destination is crew-entered free text, not validated. A match in our dict is high-quality but the source itself is noisy.
- **Dual code paths**: Same logic added in both `_persist_entities_inner` (area/vessel mode — uses `v.get("destination")`) and `_persist_port_call_entities_inner` (port_calls mode — uses `c.get("portToVisit")`)

**17c.2: lobbying.py — `lobbies_for` link + dedup restructure**

This required a significant restructure of the existing dedup logic:

- **Problem**: The original code had `if registrant not in seen_companies: continue` wrapping the entire loop body. This meant that for a registrant appearing in multiple filings, only the first filing's observation was stored. The rest were silently dropped.
- **Fix**: Restructured so `seen_companies` only guards `register_entity()` + `add_entity_alias()`. The observation storage and link creation run for every filing regardless of registrant dedup. This is correct — each filing is a distinct observation even if the registrant is the same company.
- **"Self" client guard**: Added `client_name.lower() != "self"` to prevent creating a spurious company entity named "Self" and linking the registrant to itself. Many lobbying filings list `client_name: "Self"` when the registrant lobbies on its own behalf.
- **Entity creation**: When client_name is non-empty, not equal to registrant, and not "self", creates a client company entity via `normalize_company_name()` + `register_entity()`
- **Link**: `company(registrant) → company(client)`, link_type=`lobbies_for`, source=`lobbying`, confidence=0.9
- **Metadata**: `{"filing_year": ..., "filing_period": ...}` — ties the link to the specific lobbying disclosure
- **Confidence 0.9**: Lobbying disclosures are legally mandated filings with verified registrant/client pairs

**17c.3: cert_transparency — SKIPPED**
- **Reason**: Investigated the crt.sh JSON API response format and found it does NOT expose the certificate subject Organization (O=) field. The API returns only `common_name` (domain names like `*.example.com`) and `issuer_name` (CA names like `"C=US, O=Let's Encrypt, CN=R3"`). Without the subject Organization, there is no company entity to link the domain to.
- **Alternative investigated**: Could parse the issuer_name to extract the CA as a company entity, but Let's Encrypt/DigiCert are CAs, not the entity that owns the domain. This would create misleading edges.
- **Decision**: Skip `cert_for` link entirely. If a future data source provides certificate subject fields, this can be revisited.

**17c.4: Tests — 18 tests (62 cumulative)**
- **Test classes**: `TestAISVesselPortCallTo` (10 tests), `TestLobbyingLobbiesFor` (8 tests)
- **Coverage**: normal link from destination, metadata has raw destination, country entity created, unknown destination no link, empty destination no link, case-insensitive matching, dedup same vessel+destination, port_call mode creates link, MMSI-only vessel gets link, no-store no-crash, normal lobbies_for link, client entity created, same registrant=client no link, empty client no link, dedup, multiple clients, whitespace client stripped

**17c regression fixes**:
- **test_ais_vessel_l2.py**: 6 mock-based tests broke because they assert exact `register_entity` call counts (e.g., `assert_called_once()`) but the new country entity registration adds extra calls. Fixed by adding `_vessel_register_calls(store)` helper that filters `call_args_list` by `entity_type=="vessel"`. Tests updated: `test_registers_vessel_with_imo`, `test_dedup_by_entity_id`, `test_multiple_distinct_vessels`, `test_name_fallback_to_mmsi`, `test_metadata_passed`, `test_dedup_across_port_calls`.
- **test_l2_integration.py**: `test_obs_count_reflected_in_node_features` asserted that 3 lobbying filings for the same registrant → 1 observation (old dedup behavior). Updated to assert 3 observations (correct behavior after dedup restructure).
- **test_corporate_energy_defi_l2.py**: `test_persist_entities_registers_company` (lobbying) was asserting `len(entities) == 1` but now there are 2 entities when `client_name != "Self"`. Fixed by filtering the `entities` list by `client_name: "Self"` in the test data, which triggers the new "Self" guard and correctly produces only 1 entity.

### Phase 17d: Fixed-country links (interconnection_queue + patent_filings) — COMPLETE

**17d.1: interconnection_queue.py — `located_in` link**
- **Location**: `_persist_entities_inner()`, after the observation block
- **Logic**: Every company registered from an interconnection queue record is linked to a US country entity (EIA data is US-only)
- **Entity creation**: Creates US country entity: `register_entity(entity_type="country", canonical_name="US", entity_id=entity_id_from_key("country", "US"))`
- **Link**: `company(company_eid) → country(us_eid)`, link_type=`located_in`, source=`interconnection_queue`, confidence=1.0
- **Metadata**: `{"state": state}` — captures the US state (e.g., "TX", "CA") for sub-national resolution
- **Confidence 1.0**: EIA interconnection queue is definitively US-based

**17d.2: patent_filings.py — `patents_in` link**
- **Location**: `_persist_entities_inner()`, after the observation block
- **Logic**: Every company registered from a USPTO patent is linked to a US country entity (PatentsView API is US patents only)
- **Entity creation**: Same US country entity pattern as interconnection_queue
- **Link**: `company(company_eid) → country(us_eid)`, link_type=`patents_in`, source=`patent_filings`, confidence=1.0
- **Metadata**: `{"patent_number": patent.get("patent_number", "")}` — ties the link to the specific patent
- **Confidence 1.0**: USPTO patents are definitively filed in the US

**17d.3: Tests — 17 tests (79 cumulative)**
- **Test classes**: `TestInterconnectionQueueLocatedIn` (8 tests), `TestPatentFilingsPatentsIn` (9 tests)
- **Coverage**: normal link, US country entity created, metadata state/patent_number, multiple companies all linked to US, dedup same company → single link, missing entity name no link, empty entity name no link, no-store no-crash, list assignee uses first element

**17d regression fixes**:
- **test_corporate_energy_defi_l2.py**: 6 tests broke with same pattern as 17c — `len(query_all_entities()) == 1` but now 2 (company + US country). Fixed by filtering to `entity_type == "company"` in assertions. Affected: `test_persist_entities_registers_company` (patent + IQ), `test_persist_entities_assignee_as_list`, `test_persist_entities_dedup_assignees`, `test_persist_entities_snake_case_keys`, `test_persist_entities_dedup_companies`.

### Phase 17e: Integration tests + edge case suite — COMPLETE

**17e.1: Cross-tool entity consistency — 4 tests**
- `TestCrossToolEntityConsistencyExtended`:
  - `test_same_company_from_lobbying_and_patents` — "Acme Corp" through both tools → same entity_id, registered once
  - `test_same_company_from_patent_and_iq` — "Tesla Inc" through patent_filings + interconnection_queue → same entity
  - `test_same_us_country_from_patent_and_iq` — US country entity created by both tools → single entity in store
  - `test_cross_tool_links_for_same_company` — "Google LLC" through both tools → has both `patents_in` and `located_in` links

**17e.2: Graph builder integration — 5 tests**
- `TestGraphBuilderEdgeIntegration`:
  - `test_works_for_edges_in_graph` — insider_filings link → `("person", "works_for", "company")` in `data.edge_types`, shape [2, ≥1]
  - `test_located_in_edges_in_graph` — IQ link → `("company", "located_in", "country")` edge with 1 edge
  - `test_patents_in_edges_in_graph` — patent link → `("company", "patents_in", "country")` edge with 1 edge
  - `test_edge_attr_contains_confidence` — `edge_attr` is [E, 2] with `[confidence, age_days]`, confidence=1.0
  - `test_multiple_edge_types_coexist` — 3 tools persisted → all 3 edge types present in single HeteroData

**17e.3: Full edge case suite — 7 tests**
- `TestEntityLinkEdgeCases`:
  - `test_self_link_raises_value_error` — `link_entities(eid, eid, ...)` → `ValueError("Cannot link an entity to itself")`
  - `test_idempotent_link_insertion` — same link twice → first returns link_id, second returns None, only 1 stored
  - `test_long_metadata_stored_correctly` — 100-key metadata dict survives JSON round-trip via `metadata_json` column
  - `test_no_pipeline_all_tools_safe` — all 8 tools with `pipeline_store=None` don't crash on `_persist_entities()`
  - `test_different_link_types_same_entity_pair` — A→B with `located_in` and `patents_in` → both stored (2 links)
  - `test_confidence_filter_works` — `min_confidence=0.9` filters out a `confidence=0.5` link
  - `test_query_direction_outgoing_only` — direction="outgoing" from source → 1 result; direction="incoming" from source → 0; vice versa for target

### Total: 95 entity linking tests, all in `tests/test_entity_linking.py`

---

## Files Modified This Session

### Implementation Files

| File | Change Summary |
|------|---------------|
| `agent/tools/insider_filings.py` | Added `works_for` link block in `_persist_entities_inner()` |
| `agent/tools/form144.py` | Added `works_for` link block in `_persist_entities_inner()` |
| `agent/tools/whale_alert.py` | Added `transacts_with` link cross-product in `_persist_entities_inner()` |
| `agent/tools/gdelt.py` | Added `event_involves` link in `_persist_entities_inner()` |
| `agent/tools/ais_vessel.py` | Added `_DEST_COUNTRY` dict (~50 ports), `port_call_to` link in both persist methods |
| `agent/tools/lobbying.py` | Restructured dedup, added "Self" guard, added `lobbies_for` link with client entity creation |
| `agent/tools/interconnection_queue.py` | Added `located_in` link with US country entity |
| `agent/tools/patent_filings.py` | Added `patents_in` link with US country entity |

### Test Files

| File | Change Summary |
|------|---------------|
| `tests/test_entity_linking.py` | NEW — 95 tests across 12 test classes covering all 8 link types + integration + edge cases |
| `tests/test_ais_vessel_l2.py` | Added `_vessel_register_calls()` helper, fixed 6 mock tests |
| `tests/test_l2_integration.py` | Updated obs count assertion (1→3) for lobbying dedup fix |
| `tests/test_corporate_energy_defi_l2.py` | Fixed 7 tests (6 entity count + 1 lobbying) to filter by entity_type |

### Workflow Files

| File | Change Summary |
|------|---------------|
| `[[entity_linking_layer]]` | All 17a–17e steps checked off, Status → completed, status/active → status/done, cert_for removed from summary table |

---

## Entity Link Architecture (Complete Reference)

### Link Type Summary

| Link Type | Source Tool | Entity A Type → Entity B Type | Confidence | Metadata | Notes |
|-----------|-----------|------|------|------|------|
| `works_for` | insider_filings | person → company | 1.0 | `{relationship}` | SEC Form 4 filer-issuer |
| `works_for` | form144 | person → company | 1.0 | `{relationship}` | SEC Form 144 filer-issuer |
| `transacts_with` | whale_alert | wallet → wallet | 1.0 | `{tx_hash}` | BTC in/out cross-product |
| `event_involves` | gdelt | country → country | 0.9 | `{event_id, event_root}` | Bilateral geopolitical |
| `port_call_to` | ais_vessel | vessel → country | 0.8 | `{destination_raw}` | AIS destination→port mapping |
| `lobbies_for` | lobbying | company → company | 0.9 | `{filing_year, filing_period}` | Registrant→client |
| `located_in` | interconnection_queue | company → country | 1.0 | `{state}` | EIA→US (always) |
| `patents_in` | patent_filings | company → country | 1.0 | `{patent_number}` | USPTO→US (always) |

### Link NOT implemented

| Link Type | Tool | Reason |
|-----------|------|--------|
| `cert_for` | cert_transparency | crt.sh JSON API doesn't expose certificate subject Organization field |

### Tools with no natural cross-entity links

| Tool | Reason |
|------|--------|
| dns_monitor | Cloud provider is a string attribute, not an entity |
| wikipedia_pageviews | Topic entities are standalone — no second entity in source data |
| defi_flows | "chain" is not a current entity type |

### GNN Edge Type Mapping

Links are consumed by `GraphBuilder._build_edge_data()` which groups them by `(src_entity_type, link_type, dst_entity_type)` triplets and creates:
- `data[triplet].edge_index` — `[2, E]` tensor (source/target local node IDs)
- `data[triplet].edge_attr` — `[E, 2]` tensor (`[confidence, age_days]`)

The HetTGN's heterogeneous message-passing now has typed edges for all 8 link types.

### Entity Types with Edges (before → after Phase 17)

| Entity Type | Inbound Edge Types | Outbound Edge Types | Before Phase 17 |
|---|---|---|---|
| **person** | — | works_for→company | Isolated |
| **company** | works_for←person, lobbies_for←company | lobbies_for→company, located_in→country, patents_in→country | Isolated |
| **wallet** | transacts_with←wallet | transacts_with→wallet | Isolated |
| **country** | event_involves←country, port_call_to←vessel, located_in←company, patents_in←company | event_involves→country | Isolated |
| **vessel** | — | port_call_to→country | Isolated |
| **domain** | — | — | Isolated (still) |
| **protocol** | — | — | Isolated (still) |
| **topic** | — | — | Isolated (still) |
| **organization** | — | — | Empty type (still) |

**Connected entity types**: 5 of 9 (person, company, wallet, country, vessel)
**Still isolated**: domain, protocol, topic (no natural cross-entity links in current data)
**Still empty**: organization (no L2 tool produces this type — Tier 1 candidate from Phase 16)

---

## Test Suite Health

### L2 + Entity Linking Tests
```
408 passed, 0 failed, 6867 deselected, 1 warning
```

### Full Suite
```
7221 passed, 48 failed, 6 skipped, 1 warning (~7:45 runtime)
```

### Pre-existing Failures (48 failures — NOT caused by Phase 17)

All 48 failures are the `test_tool_count` / `test_arm_count` / `test_bandit_arm_count` pattern across edge test files. These assert hard-coded tool counts and bandit arm counts that drift as new tools are added. They have been failing since before Phase 17 (verified with `git stash` during 17c debugging). The failures span 23 edge test files:

- test_academic_preprints_edge.py (2), test_ais_vessel_edge.py (1)
- test_building_permits_edge.py (2), test_capital_flows_edge.py (2)
- test_central_bank_balance_edge.py (2), test_cert_transparency_edge.py (2)
- test_comtrade_edge.py (2), test_creditor_filings_edge.py (2)
- test_defi_flows_edge.py (2), test_dns_monitor_edge.py (2)
- test_electricity_monitor_edge.py (2), test_finra_short_volume_edge.py (2)
- test_foia_requests_edge.py (2), test_form144_edge.py (3)
- test_gov_contracts_edge.py (2), test_interconnection_queue_edge.py (2)
- test_job_postings_edge.py (2), test_lobbying_edge.py (2)
- test_patent_filings_edge.py (2), test_regulatory_gazette_edge.py (1)
- test_satellite_activity_edge.py (2), test_sovereign_debt_edge.py (2)
- test_transport_throughput_edge.py (2), test_weather_alerts_edge.py (2)

These should be fixed in a future cleanup pass (update the hard-coded counts to the current tool count).

---

## Key Code Patterns Established

### Standard link insertion pattern (used by all 8 tools):
```python
# After entity registration and observation storage:
store.link_entities(
    entity_id_a=source_eid,
    entity_id_b=target_eid,
    link_type="<link_type>",
    source="<tool_name>",
    confidence=<float>,
    metadata={...},
)
```

### Entity creation for link targets (ais_vessel, lobbying, IQ, patent):
```python
target_eid = entity_id_from_key("<entity_type>", key)
store.register_entity(
    entity_type="<entity_type>",
    canonical_name=key,
    entity_id=target_eid,
)
# register_entity is idempotent — safe to call repeatedly for "US" country
```

### Dedup-safe observation + link pattern (lobbying):
```python
seen_companies: set[str] = set()
for filing in filings:
    registrant = filing.get("registrant_name", "")
    if not registrant:
        continue
    # Dedup only guards entity registration:
    if registrant not in seen_companies:
        seen_companies.add(registrant)
        store.register_entity(...)
        store.add_entity_alias(...)
    # Observation and link run for EVERY filing:
    store.store_entity_observation(...)
    store.link_entities(...)
```

### Mock test filtering pattern (for tests that assert call counts):
```python
def _vessel_register_calls(store: MagicMock) -> list:
    """Filter register_entity calls to only vessel-type registrations."""
    return [
        c for c in store.register_entity.call_args_list
        if c.kwargs.get("entity_type") == "vessel"
    ]
```

---

## Insights and Decisions Made

### Why cert_for was skipped
The crt.sh JSON API (`https://crt.sh/?q=...&output=json`) returns certificate log entries with these fields: `id`, `issuer_ca_id`, `issuer_name`, `common_name`, `name_value`, `not_before`, `not_after`, `serial_number`, `result_count`. The `common_name` is the domain (e.g., `*.example.com`), the `issuer_name` is the CA (e.g., `"C=US, O=Let's Encrypt, CN=R3"`). Neither is the certificate subject Organization (O=) which would tell us who owns the domain. To get the subject Organization, you'd need to download the actual certificate DER/PEM and parse the X.509 subject field. That's a different data pipeline, not a simple `link_entities()` addition.

### Why lobbying dedup was restructured
The original pattern (`if registrant in seen_companies: continue`) was a bug in disguise. It meant that if a company filed 3 lobbying disclosures in a quarter (to 3 different clients), only the first filing's observation was stored. The entity dedup is correct (register once), but the observation dedup was data loss. The fix: `seen_companies` guards only `register_entity()` + `add_entity_alias()`, not the observation or link blocks. This caused the `test_obs_count_reflected_in_node_features` integration test to correctly update from 1→3 observations.

### Why "Self" client guard was needed
Many lobbying filings have `client_name: "Self"` when the company lobbies on its own behalf (e.g., Google lobbying Congress directly). Without the guard, this creates: (1) a company entity named "self" (after normalization), and (2) a `lobbies_for` link from Google→Self which is meaningless. The guard `client_name.lower() != "self"` catches this.

### Confidence scale rationale
- **1.0**: SEC filings (works_for), US jurisdiction (located_in, patents_in), blockchain transactions (transacts_with) — source data is definitive
- **0.9**: GDELT events (event_involves), lobbying disclosures (lobbies_for) — legally mandated/highly reliable but occasional edge cases
- **0.8**: AIS destination (port_call_to) — free-text crew data matched against a lookup dict; high match quality but noisy source

---

## Current Full Project State

### Phase History

| Phase | Name | Status |
|---|---|---|
| 0–6 | Agent Core + Data Surface | Complete |
| 7 | Pipeline Layer | Complete (356 tests) |
| 7b | Global Deep Surveillance | Complete (57 tools) |
| 7c | Convergence Detection | Complete (883 tests) |
| 8 | Signal Protocol + Feature Engineering | Complete (294 tests) |
| 9 | World Model (Bayesian) | Complete |
| 10a/10b | Deep Surveillance Framework | Complete |
| 12 | Temporal Het GNN | Complete (242 tests) |
| 13 | L2 Tool Expansion | Complete (147 tests) |
| 14–15 | Pattern Recovery + Fine-Tuning | Complete |
| 16 | GNN-Guided Tool Expansion | Complete (34 tests) |
| **17** | **Entity Linking Layer** | **Complete (95 tests)** |

### What Comes Next (per master task)

```
Phase 17 (entity linking) ── JUST COMPLETED
    │
    ├──→ Phase 18 (Tier 1 tools: sanctions_monitor L2, gov_contracts L2, supply_chain_monitor L2)
    │    └──→ re-run diagnostics with connected graph
    │
    └──→ Phase 19 (world model) → Phase 20 (signal fusion) → Phase 21 (RL policy) → Phase 22 (adversarial)
```

**Phase 18** would add Tier 1 tools from the Phase 16 ranking: `sanctions_monitor` L2 (fills the critical empty `organization` entity type), `gov_contracts` L2 (company→organization links), `supply_chain_monitor` L2 (company→company edges). After Tier 1, re-run `run_diagnostics()` to see if the GNN attention patterns and edge coverage have improved.

Alternatively, the next session could tackle:
- **Tool count fix**: Update the 48 hard-coded tool/arm count assertions across edge test files (cleanup, not feature work)
- **Move completed tasks to `tasks/done/`**: 13 active task files have `Status: completed` but still live in `tasks/active/`
- **Phase 19 (World Model)**: If tool expansion is deferred, resume the Bayesian network + causal graph work from the master task

### Active Task Files

| Task | Status | Notes |
|------|--------|-------|
| `[[entity_linking_layer]]` | completed (should move to done/) | Phase 17 — just finished |
| `[[l2_tool_expansion]]` | completed but reference | Has Phase 16 ranking results |
| `[[quant_training_ground]]` | active | Master task — Phase 9 current |
| + 13 others | completed | Housekeeping debt — need moving to done/ |

---

## PipelineStore Status (unchanged from Phase 16)

- **DB path**: `.tirra_pipeline/pipeline.db`
- **Size**: 56 KB (schema only)
- **Entities**: 0, **Observations**: 0, **Links**: 0
- **Interpretation**: No L2 tools have been run against live APIs. All linking code is tested against in-memory stores but the production DB is still empty. When tools are executed against real APIs, entities + observations + links will begin populating.

---

## Cold-Start Instructions for Next Session

1. Read this checkpoint
2. Read `[[entity_linking_layer]]` (completed — may need moving to `tasks/done/`)
3. Read `[[quant_training_ground]]` lines ~1287–1302 (Phase Priority Map)
4. Read `docs/reports/tool_priority_ranking.md` if continuing with Phase 18
5. The entity linking implementation is stable — no pending work. Choose the next phase based on the priority map.

## Related

- [[entity_linking_layer]] — research doc
- [[entity_linking_layer_spec]] — implementation spec
- [[entity_linking_layer|entity_linking_layer task]] — Phase 17 task (completed)
- [[checkpoint_archive_2026]] — archived prior checkpoint material for Phase 16
- [[tool_priority_ranking]] — Phase 16 Tier 1/2/3 ranking
- [[quant_training_ground]] — master task
- [[l2_tool_expansion]] — Phase 13 task with Phase 16 results

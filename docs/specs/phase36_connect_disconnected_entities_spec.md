---
title: "Spec: Phase 36 — Connect Disconnected Entity Types"
tags:
  - doc/spec
  - phase/36
  - topic/entity-linking
  - topic/gnn
  - layer/world-model
  - layer/surveillance
---

# Spec: Phase 36 — Connect Disconnected Entity Types

## Goal

Connect the two structurally disconnected entity types (domain, topic) to the
rest of the entity graph by adding `link_entities()` calls to the tools that
produce them.  Update the SyntheticGraphGenerator to include these types and
their links in training data.

After this phase, the GNN will be able to propagate signal through
domain→company and topic→instrument edges.

## Files Affected

| File | Change |
|------|--------|
| `agent/tools/cert_transparency.py` | Add `domain_owned_by` link in `_persist_entities_inner()` |
| `agent/tools/dns_monitor.py` | Add `domain_owned_by` link in `_persist_entities_inner()` |
| `agent/tools/polymarket.py` | Add `topic_relates_to_instrument` link in `_persist_entities_inner()` + keyword map |
| `agent/models/gnn/trainer.py` | Add domain/topic link generation + bump defaults to `num_topics=3, num_domains=3` |
| `tests/test_phase36_entity_linking.py` | New: comprehensive edge case tests |

## Implementation Steps

### 36.1: Add `_TOPIC_INSTRUMENT_MAP` to polymarket.py

Add a static dict mapping Polymarket tag category strings to a list of
instrument tickers.  This is the lookup table for topic→instrument links.

```python
_TOPIC_INSTRUMENT_MAP: dict[str, list[str]] = {
    "crypto": ["BTC-USD", "ETH-USD"],
    "finance": ["ES=F", "SPY", "ZN=F"],
    "stocks": ["ES=F", "SPY", "NQ=F", "QQQ"],
    "interest-rates": ["ZN=F", "ZB=F", "TLT"],
}
```

Only categories with clear instrument mappings are included.  `politics`,
`geopolitics`, `tech`, `science`, `sports` do NOT map to specific instruments
and are deliberately excluded.

### 36.2: Add `topic_relates_to_instrument` links to polymarket `_persist_entities_inner()`

After registering the topic entity and storing its observation, add:

```python
# Link topic → instruments based on category
if category:
    tickers = _TOPIC_INSTRUMENT_MAP.get(category, [])
    for ticker in tickers:
        inst_eid = entity_id_from_key("instrument", ticker)
        store.link_entities(
            entity_id_a=topic_eid,
            entity_id_b=inst_eid,
            link_type="topic_relates_to_instrument",
            source="polymarket",
            confidence=0.7,
        )
```

The `category` is already available in the `mkt` dict (computed from event
tags by `_categorize_event()`).  Use `entity_id_from_key("instrument", ticker)`
which matches the instrument entity ID scheme in `instrument_universe.py`.

### 36.3: Add `_DOMAIN_COMPANY_KEYWORDS` and domain→company linking helper to `instrument_universe.py`

Add a public helper `build_domain_company_map()` that builds a dict mapping
lowercase domain substrings to `(company_canonical_name, company_entity_id)`.
Source: INSTRUMENTS issuer field, normalized via `normalize_company_name()`.

This is placed in `instrument_universe.py` because it depends on `INSTRUMENTS`,
not in the domain tools.

```python
def build_domain_company_map() -> dict[str, tuple[str, str]]:
    """Build lowercase keyword → (company_canonical_name, entity_id) map.

    Derived from INSTRUMENTS issuer names.  Used by domain tools to attempt
    domain_owned_by linking.
    """
    ...
```

Also export a simpler flat dict `_DOMAIN_KEYWORDS` that maps common domain
roots to company canonical names, for cases where the domain name doesn't
match the issuer name exactly (e.g. `blackrock.com` → `blackrock`).

### 36.4: Add `domain_owned_by` links to `cert_transparency._persist_entities_inner()`

After registering the domain entity and storing cert observations, attempt
to match the domain's base name (strip TLD, subdomains) against the company
keyword map.  If matched, create a `domain_owned_by` link.

```python
from agent.tools.instrument_universe import build_domain_company_map

# Attempt domain → company link
base = domain.split(".")[-2] if "." in domain else domain
base = base.lower()
company_map = build_domain_company_map()
if base in company_map:
    canon, company_eid = company_map[base]
    store.link_entities(
        entity_id_a=domain_eid,
        entity_id_b=company_eid,
        link_type="domain_owned_by",
        source="cert_transparency",
        confidence=0.8,
    )
```

### 36.5: Add `domain_owned_by` links to `dns_monitor._persist_entities_inner()`

Same logic as 36.4 but sourced from `dns_monitor`.

### 36.6: Update SyntheticGraphGenerator

1. Change defaults: `num_topics=3`, `num_domains=3`.
2. Add link generation code after existing link block:
   - `domain → company` (`domain_owned_by`): each domain links to a random company.
   - `topic → instrument` (`topic_relates_to_instrument`): each topic links to 1-2 random instruments.
3. Update docstring to reflect 21 link types (was 19).

### 36.7: Write edge case tests

Create `tests/test_phase36_entity_linking.py` covering:

**For polymarket topic→instrument:**
- Market with `category="crypto"` → links to BTC-USD, ETH-USD
- Market with `category="finance"` → links to ES=F, SPY, ZN=F
- Market with `category="politics"` → NO links created
- Market with empty category → NO links created
- Market with no slug → no entity registered, no links
- Multiple markets with same slug → dedup, links created once
- Correct entity IDs match `entity_id_from_key("instrument", ticker)`

**For cert_transparency domain→company:**
- Domain `stripe.com` with matching company → link created
- Domain `unknown.com` with no matching company → no link
- Domain with no dots → handled gracefully
- Domain with subdomain (`api.stripe.com`) → base extraction works
- Empty domain → no crash

**For dns_monitor domain→company:**
- Same cases as cert_transparency

**For SyntheticGraphGenerator:**
- Generator with `num_topics=3, num_domains=3` produces:
  - `domain_owned_by` links
  - `topic_relates_to_instrument` links
- Total link type count is 21

### 36.8: Run full regression

```bash
make test
```

Verify 9000+ tests pass, no regressions.

## Edge Cases

1. Domain name does not match any known company → no link, no error
2. Multiple certs for same domain → entity already exists (idempotent),
   link is idempotent (INSERT OR IGNORE)
3. Polymarket market with unmapped category → no instrument links
4. Polymarket market with multiple categories → only first tag match is used
   (existing `_categorize_event` behavior)
5. `build_domain_company_map()` called multiple times → returns consistent
   results (pure function over static INSTRUMENTS)

## Testing Plan

1. Unit tests for each new link path (mocked store)
2. Integration test: SyntheticGraphGenerator produces connected domain/topic nodes
3. Regression: full `make test` passes

## Related

- [[phase36_connect_disconnected_entities]]
- [[quant_training_ground]]
- [[phase35_gnn_retrain_expanded_graph_spec]]

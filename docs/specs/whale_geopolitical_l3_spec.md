---
title: "Spec: Whale Crypto × Geopolitical L3 Pattern"
tags:
  - doc/spec
  - phase/11
  - topic/surveillance
  - topic/convergence
  - layer/feature-engineering
---

# Spec: Whale Crypto × Geopolitical L3 Pattern (Phase 11d)

## Goal

Implement the third L3 cross-entity pattern in `cross_entity.py`:
detect temporal co-occurrences between large BTC transfers and
high-impact geopolitical events, using exchange-based wallet→country
linking.

## Files Affected

| File | Change |
|------|--------|
| `agent/pipeline/cross_entity.py` | Add constants, resolve_wallet_exchange(), seed_whale_country_links(), detect_whale_geopolitical() |
| `tests/test_cross_entity.py` | Add Phase 11d test classes |

No new files.

## Implementation Steps

### 11d.1: Constants + resolve_wallet_exchange()

Add to `cross_entity.py`:

- `WHALE_WINDOW_SECONDS = 24 * 3600`
- `WHALE_GOLDSTEIN_THRESHOLD = -5.0`
- `WHALE_VALUE_SCALE = 100.0` (BTC normalization for scoring)
- `KNOWN_EXCHANGE_WALLETS: dict[str, tuple[str, str]]` — empty default dict
  with docstring explaining production population
- `resolve_wallet_exchange(address, exchange_wallets=None) -> tuple[str, str] | None`
  — returns `(exchange_name, country_fips)` if address is in the dict

### 11d.2: seed_whale_country_links()

```python
def seed_whale_country_links(
    store,
    exchange_wallets: dict[str, tuple[str, str]] | None = None,
) -> int:
```

Logic:
1. Query all wallet entities
2. For each: get `btc_address` alias
3. Look up in exchange_wallets dict (or KNOWN_EXCHANGE_WALLETS)
4. If found: create `exchange_based_in` link wallet → country
5. Register country entity if needed
6. Return count of new links

### 11d.3: detect_whale_geopolitical()

Add method to `CrossEntityDetector`:

```python
def detect_whale_geopolitical(
    self,
    wallet_entity_id: str,
    *,
    window_seconds: float = WHALE_WINDOW_SECONDS,
    goldstein_threshold: float = WHALE_GOLDSTEIN_THRESHOLD,
    value_scale: float = WHALE_VALUE_SCALE,
    since: float | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
```

Logic:
1. Follow `exchange_based_in` links → country entities
2. Query co-occurrences: whale_alert btc_transfer vs gdelt geopolitical_event
3. Filter: Goldstein ≤ threshold (default -5.0)
4. Score: `min(value_btc / value_scale, 1.0) × (|G| / 10) × proximity`
5. Return pattern dicts with `pattern_type="whale_x_geopolitical"`

### 11d.4: Edge case test suite

Test classes to add:
- `TestResolveWalletExchange` (~5 tests): hit, miss, None, empty, custom dict
- `TestSeedWhaleCountryLinks` (~6 tests): basic seeding, no wallets, idempotent,
  unknown address, custom exchange dict parameter, link type verification
- `TestDetectWhaleGeopolitical` (~8 tests): basic hit, no links, no obs,
  Goldstein filter (+ve filtered, -3 filtered at -5 threshold), low-value
  scoring, high-value scoring, since filter, multiple co-occurrences
- `TestWhaleGeopoliticalIntegration` (~2 tests): full pipeline, value_btc
  in pattern dict verification

## Edge Cases

- Wallet with no `btc_address` alias → skip silently
- Empty exchange_wallets dict → seeder creates 0 links (no crash)
- btc_transfer observation missing `value_btc` → treat as 0, skip or score 0
- Goldstein field missing → skip co-occurrence
- Same wallet linked to multiple exchange countries → one link per unique country

## Testing Plan

- Unit tests per function/method as specified in 11d.4
- Regression: all existing cross_entity tests must still pass
- Run: `pytest tests/test_cross_entity.py tests/test_entity_links.py tests/test_pipeline_store.py -v`

## Related

- [[whale_geopolitical_l3]]
- [[whale_geopolitical_l3|task]]
- [[cross_entity_l3]]
- [[vessel_sanctions_l3_spec]]

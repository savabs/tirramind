---
title: "Research: Whale Crypto × Geopolitical L3 Pattern"
tags:
  - doc/research
  - phase/11
  - topic/surveillance
  - topic/convergence
  - layer/feature-engineering
---

# Research: Whale Crypto × Geopolitical L3 Pattern (Phase 11d)

## Goal

Third L3 cross-entity pattern: detect temporal co-occurrences between large
BTC whale transfers and high-impact geopolitical events, revealing capital
flight signals invisible from either data source alone.

---

## Current Data Model

### whale_alert (L2 entities)

- **Entity type:** `wallet` (keyed by BTC address)
- **Observation type:** `btc_transfer`
- **Observation value:**
  ```python
  {
      "tx_hash": "abc123...",
      "value_btc": 500.0,
      "direction": "out" | "in",
      "counterparty_count": 3,
      "confirmed": True,
      "block_height": 800000,
  }
  ```
- **Source tool:** `"whale_alert"`
- **Depth level:** 2
- **Entity registration:** Each unique BTC address gets a wallet entity + `btc_address` alias.

### GDELT (L2 entities)

- **Entity type:** `country` (keyed by FIPS code)
- **Observation type:** `geopolitical_event`
- **Source tool:** `"gdelt"`

---

## Wallet → Country Linking Strategy

### The Challenge

Raw blockchain data has no geographic attribution. BTC addresses don't carry
country information. The bridge is *exchange attribution*: if a wallet
is a known exchange wallet (or transacts with one), we can infer the
jurisdiction.

### Approach: Known Exchange Wallet Matching

1. **Direct identification:** Maintain a `KNOWN_EXCHANGE_WALLETS` dict mapping
   BTC addresses to `(exchange_name, country_fips)`. When a wallet entity's
   address matches, create an `exchange_based_in` link.

2. **Parameterized seeder:** `seed_whale_country_links(store, exchange_wallets=None)`
   accepts an explicit wallet dict (for testing and config injection). Falls
   back to module-level dict if None.

3. **Production population:** The default dict is intentionally small (empty
   or minimal). Production deployments populate from blockchain analytics
   services or a `data/exchange_wallets.json` config file.

### Why Direct-Only for MVP

Counterparty-based linking (wallet X transacted with known exchange →
infer wallet X's jurisdiction) requires cross-observation tx_hash matching.
This is a more complex graph traversal best deferred. The direct approach
is clean, testable, and captures the highest-value wallets (the exchanges
themselves are the biggest whales).

### Exchange → Country FIPS Mapping

| Exchange   | Jurisdiction  | FIPS |
|-----------|--------------|------|
| Coinbase  | United States | US   |
| Kraken    | United States | US   |
| Binance   | Cayman Islands| CJ   |
| Bitfinex  | Br. Virgin Is.| VQ   |
| Bitstamp  | Luxembourg    | LU   |
| Huobi/HTX | Seychelles    | SE   |
| OKX       | Seychelles    | SE   |

---

## Detector Design

### Temporal Window: 24 hours

Rationale: BTC transfers are near-instant (mempool: seconds; confirmation:
~10min). GDELT events lag by ~15min. No filing disclosure delay like SEC (T+2).
A 24h window captures event-driven capital flight while limiting
false co-occurrences.

Comparison with other patterns:
| Pattern              | Window | Reason |
|---------------------|--------|--------|
| Insider × GDELT     | 72h    | SEC T+2 filing lag |
| Vessel × Sanctions  | 48h    | AIS is T0, ship movements are slow |
| **Whale × Geopolitical** | **24h** | **Crypto is near-instant** |

### Goldstein Threshold: -5.0 (stricter)

Standard threshold for other patterns is -2.0. For whale×geopolitical we use
-5.0 because:
- Crypto flows respond to *high-impact* events, not minor diplomatic friction
- Goldstein -5 corresponds to "serious conflict / major sanctions"
- Filtering at -5 reduces false co-occurrences from the volume of GDELT events
- The L3 research doc says: "Goldstein ≤ -5" for this pattern specifically

### No CAMEO Root Code Filter

Unlike vessel×sanctions (which filters CAMEO 16/17), whale×geopolitical
responds to ANY high-impact event: wars, sanctions, coups, disasters.
Capital flight doesn't discriminate by event type. Only filter is Goldstein ≤ -5.

### Scoring Formula

```
score = value_weight × severity × proximity

where:
  value_weight = min(value_btc / 100.0, 1.0)   # transfers ≥100 BTC get full weight
  severity     = |Goldstein| / 10.0
  proximity    = max(0, 1 − |Δt| / window_hours)
```

**Why include value_btc in scoring?** The other two patterns don't weight by
event magnitude (insider trade size isn't in the score, vessel position doesn't
have magnitude). But whale transfers have a natural magnitude — value_btc.
A 500 BTC move coinciding with instability is far more meaningful than a 5 BTC
move. The `min(..., 1.0)` cap keeps the score in [0, 1].

The 100 BTC threshold was chosen because whale_alert typically filters at
≥10 BTC. At 100 BTC (~$6M at $60K/BTC), the transfer is unambiguously
a whale move.

### Pattern Dict Keys

```python
{
    "pattern_type": "whale_x_geopolitical",
    "entity_a": wallet_entity_id,
    "entity_b": country_entity_id,
    "whale_event": obs_a_value,   # {tx_hash, value_btc, direction, ...}
    "gdelt_event": obs_b_value,   # {goldstein, event_root_code, ...}
    "value_btc": float,           # extracted for convenience
    "direction": str,             # "in" or "out"
    "time_delta_hours": float,
    "goldstein": float,
    "score": float,
    "obs_a_id": int,
    "obs_b_id": int,
}
```

---

## Risks

1. **Low coverage:** Without comprehensive exchange-address lists, few wallets
   will have country links. Mitigated by parameterized seeder + future
   counterparty linking.
2. **Spurious correlation:** BTC volume is inherently noisy. The -5.0 Goldstein
   threshold and value_btc weighting help, but permutation testing (future)
   is needed for statistical validation.
3. **Attribution accuracy:** Exchange addresses change. The curated dict may
   go stale. Mitigated by loading from external config at runtime.

## Related

- [[cross_entity_l3]]
- [[whale_geopolitical_l3_spec]]
- [[whale_geopolitical_l3|task]]
- [[whale_alert_l2]]
- [[gdelt_l2]]
- [[vessel_sanctions_l3]]

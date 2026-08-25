---
title: "Research: Crypto Islands + Cross-Domain Linking"
tags:
  - doc/research
  - phase/30
  - topic/entity-linking
  - topic/crypto
  - layer/surveillance
  - layer/world-model
---

# Research: Crypto Islands + Cross-Domain Linking

## Problem Statement

BTC-USD and ETH-USD are graph islands: they have instrument nodes with price observations (`instrument_return`, `instrument_volatility`, `instrument_volume`) but **zero entity links**. The HetTGN cannot propagate attention from on-chain activity (wallets, protocol TVL) into crypto instrument predictions because no edges connect those subgraphs.

Meanwhile, `whale_alert` creates wallet entities with `btc_transfer` observations, and `defi_flows` creates protocol entities with `tvl_change` observations — but neither tool links those entities to crypto instruments. The entities exist; the wiring is missing.

## Current Architecture

### InstrumentDef (agent/tools/instrument_universe.py)

```python
InstrumentDef("BTC-USD", "Bitcoin", "crypto", "Global")
InstrumentDef("ETH-USD", "Ethereum", "crypto", "Global")
```

No `protocol` field. No `issuer`, no `country`, no `cftc_code` — all `None`. This is correct: crypto has no corporate issuer and no single country. But the lack of a protocol field means `_persist_instrument_links` skips them entirely.

### Entity ID Patterns

- Instrument: `_entity_id(ticker)` → `hashlib.sha256(f"instrument:{ticker}").hexdigest()[:16]`
- Protocol: `entity_id_from_key("protocol", name.lower())` — e.g., `"protocol:bitcoin"`, `"protocol:ethereum"`
- Wallet: `entity_id_from_key("wallet", addr)` — BTC addresses from whale_alert

### Existing Links (none for crypto)

`_persist_instrument_links` creates:
- `tracks_issuer` (instrument → company) — skipped when `issuer is None`
- `located_in` (instrument → country) — skipped when `country is None`
- `fx_base_country` / `fx_quote_country` — FX only

### Protocol Entities from defi_flows

`defi_flows` tool registers protocol entities with:
- `entity_type="protocol"`, `canonical_name=name` (display case, e.g., "Uniswap")
- `entity_id = entity_id_from_key("protocol", name.lower())` — lowercased key
- Alias: `("protocol_name", name)`
- Observation: `tvl_change` (depth_level=2)

Protocols include blockchain-level names like "bitcoin", "ethereum" (from DeFiLlama chains) as well as dApp-level protocols like "Uniswap", "Aave".

### whale_alert Wallet Entities

Registers wallet entities per BTC address:
- `entity_type="wallet"`, `canonical_name=addr`
- `entity_id = entity_id_from_key("wallet", addr)`
- Observation: `btc_transfer` (depth_level=2, direction in/out)
- **No links created** — wallets are isolated

### Graph Builder (agent/models/gnn/graph_builder.py)

- `protocol` is a canonical entity type (index 7)
- `instrument` is a canonical entity type (index 4)
- `wallet` is a canonical entity type (index 10)
- Edge types derived from entity_links table dynamically — any valid (type_a, link_type, type_b) triplet works
- No code changes needed in graph_builder for new link types

## Design

### New InstrumentDef Field

Add `protocol: str | None = None` to `InstrumentDef`. Only used for crypto:

```python
InstrumentDef("BTC-USD", "Bitcoin", "crypto", "Global", protocol="bitcoin")
InstrumentDef("ETH-USD", "Ethereum", "crypto", "Global", protocol="ethereum")
```

### New Link Types

| Link Type | From → To | Source | Created By |
|---|---|---|---|
| `tracks_protocol` | instrument → protocol | `instrument_universe` | `_persist_instrument_links` |
| `trades_instrument` | wallet → instrument | `whale_alert` | `_persist_entities_inner` |

### Protocol Naming Consistency

DeFiLlama (used by `defi_flows`) uses lowercased chain names: `"bitcoin"`, `"ethereum"`. Our protocol entity IDs are `entity_id_from_key("protocol", name.lower())`. So:

- BTC-USD `protocol="bitcoin"` → `entity_id_from_key("protocol", "bitcoin")` — matches defi_flows for the Bitcoin chain
- ETH-USD `protocol="ethereum"` → `entity_id_from_key("protocol", "ethereum")` — matches defi_flows for the Ethereum chain

The entity IDs will be identical. `register_entity` is idempotent (INSERT OR IGNORE), so even if both `instrument_universe` and `defi_flows` register the same protocol entity, no conflict occurs.

### whale_alert → Instrument Linking

When `whale_alert` processes BTC transactions, it knows all transactions are BTC. Add `trades_instrument` links from each wallet to BTC-USD:

```
wallet:{addr} --trades_instrument--> instrument:BTC-USD
```

This is deterministic: the tool only handles BTC on-chain transactions from blockchain.com. Every wallet that appears in a whale BTC transaction trades BTC by definition.

### Expected Graph Effect

Before Phase 30:
```
[BTC-USD] (isolated)    [ETH-USD] (isolated)
[wallet:abc] (isolated) [protocol:bitcoin] (isolated)
```

After Phase 30:
```
[wallet:abc] --trades_instrument--> [BTC-USD] --tracks_protocol--> [protocol:bitcoin]
                                    [ETH-USD] --tracks_protocol--> [protocol:ethereum]
```

Wallet → instrument → protocol paths enable the GNN to propagate attention from on-chain whale activity through to crypto price predictions.

## Risks

1. **Protocol entity naming drift**: If DeFiLlama changes chain names, our hardcoded `protocol="bitcoin"` won't match. Mitigation: we control the `protocol` field in InstrumentDef, and defi_flows lowercases its input — both sides are deterministic.

2. **whale_alert is BTC-only**: Currently only blockchain.com (BTC). ETH whale tracking isn't implemented (Etherscan requires API key). So `trades_instrument` links will only exist for BTC-USD wallets initially. ETH-USD gets linked via `tracks_protocol` only.

3. **Large wallet count**: whale_alert can register many wallets per run. Each gets a `trades_instrument` link. `link_entities` is idempotent (INSERT OR IGNORE), so duplicates are free. The link table can handle this.

## Data Requirements

No new data sources needed. All entities already exist in the pipeline store from existing L2 tools.

## Math/Algorithm Survey

No new algorithms. This is a graph wiring change — adding edges between existing entity nodes.

## Depth Roadmap

- **L1**: Price observations on crypto instruments (already done)
- **L2**: Protocol entities (defi_flows) + wallet entities (whale_alert) — already done
- **L3 (this phase)**: Cross-domain links between wallet → instrument → protocol

## Related

- [[crypto_islands_cross_domain_linking_spec]]
- [[phase30_crypto_islands]]
- [[starved_class_audit]]
- [[l2_expansion_roadmap]]
- [[phase25_cross_domain_entity_linking]]
- [[whale_alert]]
- [[7b-L_defi_flows|defi_flows]]

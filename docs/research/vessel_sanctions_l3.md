---
title: "Research: Vessel × Sanctions L3 Pattern (Phase 11c)"
tags:
  - doc/research
  - phase/11
  - topic/surveillance
  - topic/convergence
  - layer/feature-engineering
---

# Research: Vessel × Sanctions L3 Pattern (Phase 11c)

## Goal

Implement the second L3 cross-entity pattern: detect temporal co-occurrences
between **vessel port visits/destinations** and **GDELT sanctions/coercion events**
targeting the same country. The signal: when ships stop going to a country
that just got sanctioned (compliance) or keep going (evasion), that's T0
intelligence before news or price impact.

---

## Existing Infrastructure

### Available from Phase 11a/11b

| Component | Location | Notes |
|-----------|----------|-------|
| `entity_links` table | `store.py` | DDL + CRUD (link_entities, query_entity_links) |
| `query_co_occurrences()` | `store.py` | Temporal cross-join across entity_observations |
| `CrossEntityDetector` | `cross_entity.py` | Base class + store_l3_observations method |
| `seed_company_country_links()` | `cross_entity.py` | Pattern for seeder functions |

### AIS Vessel L2 Data Model

Entity type: `vessel` (IMO-first, MMSI-fallback key via `_vessel_entity_id()`).

**Position observations** (`observation_type="vessel_position"`):
```python
value = {"lat": float, "lon": float, "sog": float, "cog": float,
         "heading": float, "nav_status": str}
```
Vessel metadata includes `destination` field — UN LOCODE format (e.g., "RU LED" = Russia, St. Petersburg).

**Port call observations** (`observation_type="port_call"`):
```python
value = {"port": str, "prev_port": str, "next_port": str,
         "arrival_with_cargo": bool}
```
Port call data is Finnish-only (Digitraffic API), but `prev_port` and `next_port` reference international ports.

### GDELT L2 Data Model

Entity type: `country` (FIPS code key).

**Geopolitical event observations** (`observation_type="geopolitical_event"`):
```python
value = {"goldstein": float, "event_code": str, "event_root_code": str,
         "quad_class": int, "num_articles": int, "avg_tone": float, ...}
```

### CAMEO Codes Relevant to Sanctions

| Root Code | Label | Relevance |
|-----------|-------|-----------|
| 16 | Reduce Relations | Diplomatic sanctions |
| 17 | Coerce | Economic sanctions, embargoes, trade restrictions |
| 13 | Threaten | Pre-sanctions warnings |
| 12 | Reject | Refusing trade/cooperation |

Quad class 4 = Material Conflict (covers enforcement actions).

---

## Vessel → Country Linking Strategy

### UN LOCODE Mapping (Primary)

The `destination` field in AIS vessel metadata uses UN LOCODE format where the
first 2 characters are the ISO country code. Example formats observed in
destination_flow mode's `strategic` groups:

| Destination String | Country | ISO | FIPS |
|-------------------|---------|-----|------|
| RU LED, RULED, SPB, ST PETERSBURG | Russia | RU | RS |
| EGPSD, EG PSD, PORT SAID | Egypt | EG | EG |
| NLRTM, NL RTM, ROTTERDAM | Netherlands | NL | NL |
| BEANR, BE ANR, ANTWERP | Belgium | BE | BE |

**Strategy:** Extract the 2-letter ISO country prefix from UN LOCODE-style
destinations, then convert ISO → FIPS for matching GDELT entities.

### Port Name Mapping (Fallback for port_call data)

Finnish API port names (portToVisit, prevPort, nextPort) may use local names.
For Baltic coverage, map major port names to FIPS codes:

| Port Name Patterns | FIPS |
|-------------------|------|
| HELSINKI, KOTKA, HAMINA, TURKU, RAUMA, PORI, OULU | FI |
| STOCKHOLM, GOTHENBURG, LULEA, MALMO | SW |
| TALLINN, MUUGA | EN |
| RIGA, VENTSPILS, LIEPAJA | LG |
| KLAIPEDA | LH |
| GDANSK, GDYNIA, SZCZECIN | PL |
| COPENHAGEN | DA |
| HAMBURG, ROSTOCK, LUBECK, BREMERHAVEN | GM |
| ST PETERSBURG, UST-LUGA, PRIMORSK, VYSOTSK, KALININGRAD | RS |

### ISO → FIPS Conversion

GDELT uses FIPS 10-4 which differs from ISO 3166-1 alpha-2 for several countries:

| ISO | FIPS | Country |
|-----|------|---------|
| RU | RS | Russia |
| DE | GM | Germany |
| SE | SW | Sweden |
| EE | EN | Estonia |
| LV | LG | Latvia |
| LT | LH | Lithuania |
| DK | DA | Denmark |
| GB | UK | United Kingdom |
| AT | AU | Austria |
| CH | SZ | Switzerland |

For most countries, ISO == FIPS.

---

## Pattern Design

### Link Type: `port_call_to`

```
vessel (IMO/MMSI) --[port_call_to]--> country (FIPS)
```

Created by scanning existing `port_call` observations (port, prev_port, next_port)
and vessel position metadata (destination field). Confidence = 1.0 for directly
observed port calls.

### Co-occurrence Definition

For a linked (vessel, country) pair:
- **Side A:** vessel's `port_call` or `vessel_position` observations
- **Side B:** country's `geopolitical_event` observations filtered to
  sanctions-relevant CAMEO codes (root codes 16, 17; or quad_class 4)

Window: **48 hours** (tighter than Insider×GDELT because AIS is T0 real-time,
not T+2 filing lag like SEC).

### Scoring

`score = event_severity × temporal_proximity`

Where:
- `event_severity = abs(goldstein) / 10.0` (normalized 0→1)
- `temporal_proximity = max(0, 1 - abs(delta_h) / window_h)` (1 at exact match, 0 at window edge)

Same formula as Insider×GDELT for consistency.

### Sanctions-Special Filter

In addition to Goldstein threshold, filter GDELT events to sanctions-relevant
CAMEO root codes: {16, 17}. This is stricter than the Insider×GDELT detector
(which uses any negative-Goldstein event) because we specifically want trade
sanctions, not general conflict.

---

## Risks

1. **Port name ambiguity** — Same port name in different countries. Mitigate by
   focusing on UN LOCODE prefix (2-letter ISO) which is unambiguous.
2. **Sparse vessel→country links** — Only vessels that have been observed with
   destination or port_call data will be linked. Coverage depends on API usage.
3. **Finnish API bias** — Port calls are Finnish-only, so direct port_call
   observations are FI-bound. But prevPort/nextPort + destination give international coverage.
4. **False co-occurrences** — A vessel heading to Russia during routine trade
   is not the same as one going there during sanctions. Need min_score threshold.

## Related

- [[cross_entity_l3]]
- [[cross_entity_l3_spec]]
- [[vessel_sanctions_l3_spec]]
- [[ais_vessel_l2]]

---
title: "Research: ais_vessel L2 Upgrade"
tags:
  - doc/research
  - phase/10
  - topic/surveillance
  - layer/surveillance
---

# Research: ais_vessel L2 Upgrade

## Goal

Upgrade `AISVesselTool` from L1 (aggregate vessel counts and position snapshots) to L2 (entity-resolved vessel observations in the PipelineStore entity registry). AIS is the strongest L0 physics signal in the surveillance stack — a 300m tanker can't fake its position. Entity-resolved vessel tracking enables L3 cross-domain patterns like sanctions evasion detection (vessel rerouting × insider selling × OFAC additions).

---

## Current Architecture

### ais_vessel.py (~742 lines)

**Data source:** Finnish Digitraffic AIS API — free, no auth, real-time positions for 18K+ vessels in the Baltic Sea and Northern European waters.

**Four modes:**

| Mode | What It Does | Returns |
|------|-------------|---------|
| `area` | Vessels in a bounding box or named area, optional ship_type filter | Per-vessel: mmsi, lat, lon, sog, cog, heading, nav_status, name, destination, ship_type, imo |
| `vessel` | Single vessel lookup by MMSI | Full metadata + position: imo, call_sign, destination, ship_type, draught |
| `port_calls` | Finnish port arrivals/departures | Per-call: vesselName, portToVisit, prevPort, nextPort, arrivalWithCargo |
| `destination_flow` | Aggregate destination distribution by ship_type | dest_counts, strategic destinations (suez, russia, rotterdam, antwerp) |

**Constructor:** `__init__(self, cache: DataCache | None = None)` — no PipelineStore.

**Key data structures:**

1. **Locations** (from `/api/ais/v1/locations`): GeoJSON FeatureCollection. Each feature has:
   - `geometry.coordinates` → [lon, lat]
   - `properties` → {mmsi, sog, cog, heading, navStat, timestamp}
   - Root-level `mmsi` also available

2. **Metadata** (from `/api/ais/v1/vessels` or `/api/ais/v1/vessels/{mmsi}`):
   - `mmsi` (int) — Maritime Mobile Service Identity, 9-digit. Assigned per radio installation. Can change when vessel changes flag state.
   - `imo` (int | None) — International Maritime Organization number, 7-digit. Permanent hull identifier. Follows vessel through ownership/flag changes. **This is the stable entity key.**
   - `name` (str) — vessel name. Can change with ownership.
   - `callSign` (str) — radio call sign.
   - `destination` (str) — AIS-reported destination (free text, often poorly formatted).
   - `shipType` (int) — AIS ship type code (60-69=passenger, 70-79=cargo, 80-89=tanker, etc.)
   - `draught` (int) — current draught in 1/10 m.

3. **Port calls** (from `/api/port-call/v1/port-calls`):
   - `imoLloyds` (int) — IMO number (different field name than metadata!)
   - `vesselName` (str)
   - `mmsi` (int)
   - `portToVisit` (str) — locode or name
   - `prevPort` / `nextPort` (str)
   - `arrivalWithCargo` (bool)
   - `eta`, `ata`, `atd` timestamps

### Identity Model: MMSI vs IMO

This is the critical nuance for entity resolution:

| Property | MMSI | IMO |
|----------|------|-----|
| Digits | 9 | 7 |
| Scope | Radio station (flag-assigned) | Hull (permanent) |
| Changes? | Yes — on reflagging | No — through entire vessel life |
| Coverage | All AIS-equipped vessels | Only SOLAS vessels (≥300GT international, ≥500GT domestic) |
| In locations API? | Yes (primary key) | No |
| In metadata API? | Yes | Yes (may be null for small vessels) |
| In port calls? | Yes | Yes (as `imoLloyds`) |

**Decision:** Use MMSI as the primary alias (it's always present), but register IMO as a second alias when available. The entity_id should be derived from IMO when available (stable), falling back to MMSI (present for all vessels).

```
entity_id = entity_id_from_key("vessel", str(imo)) if imo else entity_id_from_key("vessel", f"mmsi:{mmsi}")
```

This means: two records with the same IMO but different MMSI (vessel reflagged) will resolve to the same entity. Two records with same MMSI but no IMO will be treated as one entity (no ambiguity within a session).

### Where Entity Data Appears Per Mode

| Mode | MMSI | IMO | Name | Destination | Ship Type | Additional |
|------|------|-----|------|-------------|-----------|------------|
| `area` | Always | When metadata fetched | When metadata fetched | When metadata fetched | When metadata fetched | lat, lon, sog, cog, nav_status |
| `vessel` | Always | From metadata | From metadata | From metadata | From metadata | + call_sign, draught |
| `port_calls` | In call data | As `imoLloyds` | As `vesselName` | N/A | N/A | prev/nextPort, cargo flag |
| `destination_flow` | In metadata | In metadata | In metadata | Primary data | In metadata | Aggregate counts — no persistence needed |

### Which Modes Should Persist Entities?

**`area` mode:** Yes — each matched vessel has MMSI and (when metadata loaded) IMO, name, position. This is the primary L2 hook. Observation type: `vessel_position`.

**`vessel` mode:** Yes — single vessel lookup gives richest metadata. Observation type: `vessel_position` (same as area, but with more fields).

**`port_calls` mode:** Yes — each port call has IMO + MMSI + port names. Observation type: `port_call`.

**`destination_flow` mode:** No — this is an aggregate view. L2 entity work happens in the other modes.

---

## Differences from Prior L2 Tools

| Aspect | insider_filings / form144 / whale_alert | ais_vessel |
|--------|----------------------------------------|------------|
| Entity type | person/company/wallet | vessel |
| Primary key | CIK / BTC address | IMO (stable) or MMSI (always present) |
| Dual identity? | No (CIK is canonical) | Yes — MMSI + IMO are different identifiers for the same hull |
| Alias sources | sec_cik, ticker, btc_address | mmsi, imo |
| Number of persist hooks | 1 (in execute) | 3 (area, vessel, port_calls) |
| Observation types | 1 per tool | 2 (vessel_position, port_call) |
| Position data? | No | Yes — lat/lon/sog/cog/heading |
| Temporal behavior? | Filing dates | Real-time positions + port timing |

### Key Design Decisions

1. **Entity ID derivation:** IMO-first, MMSI-fallback:
   ```python
   entity_id = entity_id_from_key("vessel", str(imo)) if imo else entity_id_from_key("vessel", f"mmsi:{mmsi}")
   ```

2. **Dual aliases:** Always register MMSI alias. Register IMO alias when available. This means `resolve_entity("imo", "1234567")` and `resolve_entity("mmsi", "123456789")` both return the same entity_id.

3. **Three persist hooks:** `_mode_area`, `_mode_vessel`, and `_mode_port_calls` each call `_persist_entities()` with their mode-specific vessel records.

4. **Observation value schemas differ by mode:**
   - `vessel_position`: `{lat, lon, sog, cog, heading, nav_status, destination, ship_type}`
   - `port_call`: `{port, prev_port, next_port, cargo, eta, ata, atd}`

5. **`destination_flow` is L1 only** — it's an aggregate view by design. No entity persistence.

6. **Metadata may not be loaded in `area` mode.** When `ship_type == "all"`, the tool skips the metadata fetch for performance. In that case, IMO/name/destination are unavailable. We still persist with MMSI-only identity — the entity will gain IMO alias when the vessel appears in `vessel` or `port_calls` mode later. This is intentional incremental enrichment.

---

## Risks

1. **Volume in `area` mode.** The `full_baltic` area can return 18K+ features. Persisting all of them would be expensive in a single call. Mitigate: only persist the filtered/matched vessels (after bbox + ship_type filter), which is typically 50-500.
2. **Metadata not always available.** In `area` mode with `ship_type=all`, metadata is not fetched. Persist with MMSI-only identity; IMO alias added on later enrichment.
3. **Port call IMO format.** The port call API uses `imoLloyds` (integer) while metadata uses `imo` (integer). Same value, different field name — handle in parsing.
4. **Destination text is noisy.** AIS destination is free-text, entered by crew. "PORT SAID", "EGPSD" and "EG PSD" are all the same port. This is an L3 concern (entity linking destinations to port entities); L2 just stores the raw text.

---

## Observation Value Schemas

### vessel_position (area + vessel modes)

```python
{
    "mmsi": 123456789,
    "lat": 59.43,
    "lon": 24.75,
    "sog": 12.5,
    "cog": 180.0,
    "heading": 178,
    "nav_status": "under_way_engine",
    "destination": "NLRTM",
    "ship_type": "tanker",
    "ship_type_code": 80,
    "name": "NORDIC SPIRIT",
}
```

### port_call (port_calls mode)

```python
{
    "port": "FIHEL",
    "prev_port": "DEHAM",
    "next_port": "RULED",
    "cargo": True,
    "vessel_name": "NORDIC SPIRIT",
}
```

---

## Step-Local References

- **L2 pattern template:** [[deep_surveillance_10b|insider_filings L2]], [[deep_surveillance_10b2|form144 L2]], whale_alert L2
- **Entity utilities:** `entity_id_from_key("vessel", key)` in `agent/pipeline/entity.py`
- **PipelineStore API:** `register_entity()`, `add_entity_alias()`, `store_entity_observation()` in `agent/pipeline/store.py`
- **Depth evaluation:** `agent/pipeline/depth_eval.py` for MI measurement integration test
- **MMSI/IMO identity:** ITU-R M.585 (MMSI format), IMO resolution A.1078(28) (IMO number scheme)
- **Digitraffic API:** `https://meri.digitraffic.fi/api/ais/v1/` — positions, metadata, port calls

---

## Related

- [[deep_surveillance_tools]]
- [[ais_vessel_l2_spec]]
- [[l2_tool_expansion]]
- [[adsb_jet_tracking]]
- [[deep_surveillance_10b]]
- [[deep_surveillance_10b2]]
- [[project_memory]]

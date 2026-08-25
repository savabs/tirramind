---
title: "Feature: Sanctions Monitor (7b-AF)"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/sanctions
---

# Feature: Sanctions Monitor (7b-AF)

## Current Architecture
- Layer 1 (Surveillance Surface): `agent/tools/`
- Pattern: `Tool` ABC → `execute(**kwargs)` → `ToolResult(success, output, data)`
- Registration: `cli.py` → `build_tool_registry()` → `DataCache` shared
- Bandit: `GoalArm(name, description, tools, examples)` in `DEFAULT_ARMS`
- Reference tool: `earthquake_proximity.py` (newest, multi-mode, infra-overlay)

## Data Sources Probed

### OFAC SDN (US Treasury) ✅ PRIMARY
- **URL**: `https://www.treasury.gov/ofac/downloads/sdn.csv`
- **Status**: 200, 5.5MB, ~18,708 entries
- **Format**: CSV, no header row, `-0-` for null fields
- **Fields** (12): uid, SDN_Name, SDN_Type, Program, Title, Call_Sign, Vess_type, Tonnage, GRT, Vess_flag, Vess_owner, Remarks
- **Types**: `"individual"` or `-0-` (entity)
- **Programs**: CUBA, SDGT, IRAN, UKRAINE-EO13662, etc.
- **Remarks**: Contains DOB, nationality, passport #, sanctions info
- **Companion files**: `add.csv` (addresses), `alt.csv` (aliases) — not needed for v1
- **Update frequency**: A few times per month
- **No per-entry listing date** — only file-level Publish_Date
- **No change tracking API** — sdnew.csv, sdnew.xml, sdn_changes.csv all return 404
- **No search API** — must download full list, search in-memory

### UN Security Council Consolidated List ✅ PRIMARY
- **URL**: `https://scsanctions.un.org/resources/xml/en/consolidated.xml`
- **Status**: 200, 2MB, ~900 entries
- **Format**: XML with `<CONSOLIDATED_LIST>` → `<INDIVIDUALS>` + `<ENTITIES>`
- **Key fields per INDIVIDUAL**: DATAID, FIRST_NAME, SECOND_NAME, UN_LIST_TYPE, REFERENCE_NUMBER, LISTED_ON, GENDER, COMMENTS1, NATIONALITY, LAST_DAY_UPDATED, INDIVIDUAL_ALIAS, INDIVIDUAL_ADDRESS
- **Has LISTED_ON (YYYY-MM-DD)** — ideal for "recent additions" mode
- **Has LAST_DAY_UPDATED** — tracks modifications
- **UN_LIST_TYPE values**: DRC, CAR, ISIL, YEM, SOL, etc. (maps to country/regime)

### EU Sanctions XML — DEFERRED (v2)
- **URL**: `https://webgate.ec.europa.eu/fsd/fsf/public/files/xmlFullSanctionsList/content?token=dG9rZW4tMjAxNw`
- **Status**: 200, 24MB
- **Structure**: `<sanctionEntity>` with `<nameAlias>`, `<regulation>`, `<subjectType>`
- **Reason to defer**: 24MB is heavy; OFAC + UN already cover the critical signal

### ITA Consolidated Screening List — NOT WORKING
- Returns 301 redirect to HTML page. API v2 broken.

### OpenSanctions — NOT WORKING (requires API key)
### BIS Entity List — NOT WORKING (HTML page, not data file)

## Signal Theory
- **New entity addition** = policy escalation → commodity/sector price moves
- **New sanctions program** = geopolitical regime change → broad market impact
- **Program entity count growth** = conflict intensification
- **Cross-reference**: Entity on SDN + operating in energy → oil supply disruption
- **Alpha**: Sanctions changes are announced via OFAC press releases but the structured data appears hours later. Monitoring the SDN list directly provides T0 data.

## Risks
- OFAC CSV is 5.5MB — manageable with 6-hour cache TTL
- UN XML is 2MB — very lightweight
- OFAC CSV parsing: multi-line records possible (remarks can wrap). Python csv module handles this.
- OFAC has no per-entry dates → "recent" mode for OFAC requires tracking file changes (Publish_Date header or file timestamp). For v1, UN-only for recent.
- No rate limiting documented, but be respectful: single download per cache TTL.
- XML parsing: use defusedxml.ElementTree (or xml.etree) for security

## Data Requirements
- `sdn.csv` downloaded + cached as parsed records
- `consolidated.xml` downloaded + cached as parsed records
- Normalized record format: source, entity_id, name, type, programs, listed_date, nationality, aliases, remarks

## Architecture
- Download full lists → cache aggressively (TTL 6h, list changes at most weekly)
- Parse into normalized dicts in-memory
- Search/filter/aggregate on parsed records
- Never stream the raw file per-query — always parse-then-cache

---

## Related

- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]

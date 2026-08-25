---
title: "Research: Government Contract Awards (7b-G)"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
---

# Research: Government Contract Awards (7b-G)

**Date:** 2026-03-28 (retroactive documentation)
**Tool:** `agent/tools/gov_contracts.py` → `GovContractsTool`
**Status:** IMPLEMENTED, TESTED — **US-ONLY, NEEDS GLOBALIZATION**

## APIs Probed

### USASpending.gov ✅ SELECTED (US)
- **URL:** `https://api.usaspending.gov/api/v2/search/spending_by_award/` (POST)
- **URL:** `https://api.usaspending.gov/api/v2/autocomplete/awarding_agency/` (GET, defined but unused)
- **Method:** POST (awards search)
- **Auth:** None
- **Format:** JSON
- **Rate limits:** Undocumented, generous
- **Coverage:** **US federal only** — all contract types (A/B/C/D), excludes grants/loans

### International Sources — NOT YET PROBED
These need to be researched and tested before globalization:

| Source | Country | URL | Status |
|--------|---------|-----|--------|
| TED (Tenders Electronic Daily) | EU (27 countries) | `https://ted.europa.eu/api/` | **NOT PROBED** — free, structured, all EU public procurement >€139K |
| Contracts Finder | UK | `https://www.contractsfinder.service.gov.uk/api` | **NOT PROBED** — free REST API |
| e-Procurement | Japan | `https://www.e-gov.go.jp` | **NOT PROBED** — may be Japanese-only |
| AusTender | Australia | `https://www.tenders.gov.au/` | **NOT PROBED** — free, structured |
| GeM | India | `https://gem.gov.in/` | **NOT PROBED** — Government e-Marketplace |
| BuyAndSell.gc.ca | Canada | `https://buyandsell.gc.ca/` | **NOT PROBED** — Public Works Canada |
| KONEPS | South Korea | `https://www.g2b.go.kr/` | **NOT PROBED** — Korean Government Procurement |
| ComprasNet | Brazil | `https://compras.gov.br/` | **NOT PROBED** |

## Geographic Coverage
- Currently: **US only** — `[G:US-ONLY]` `[G:NEEDS-EXPANSION]`
- Target: Global (at minimum US + EU + UK + Japan + Australia)

## Modes Implemented
1. `recent` — latest contract awards, sorted by date
2. `top` — largest contracts by dollar amount
3. `agency` — filter by awarding agency name
4. `search` — keyword search in recipient names

## Signal Value
- Defense contract surges = geopolitical escalation (when seen across US + EU + Japan = coordinated)
- Healthcare contract spikes = pandemic preparation (cross-country)
- Tech contracts = government AI/cyber investment direction
- Infrastructure awards = fiscal stimulus pipeline
- Cross-country simultaneous awards to same contractor = multinational government coordination

## Globalization Priority: HIGH
- TED (EU) is the single most impactful addition — covers 27 countries with structured API
- UK Contracts Finder is second priority — well-documented free API
- Cross-country analysis (same sector spiking in US + EU + UK) is the unique edge

## Risks
- USASpending API: no versioning guarantee, occasionally slow
- International APIs may have different data models — normalization needed
- Currency conversion needed for cross-country comparison

---

## Related

- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]

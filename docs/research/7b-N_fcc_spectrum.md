---
title: "Research: 7b-N — FCC / Spectrum & Telecom Filings"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
---

# Research: 7b-N — FCC / Spectrum & Telecom Filings

**Date reviewed:** 2026-04-02
**Status:** SKIP — all APIs dead or blocked

## Endpoints Probed

| Endpoint | URL | Status | Notes |
|---|---|---|---|
| FCC ECFS (filings) | publicapi.fcc.gov/ecfs/filings | **403 Forbidden** | Even with browser UA |
| FCC ECFS (proceedings search) | publicapi.fcc.gov/ecfs/search/proceedings | **403 Forbidden** | |
| FCC ULS (licensing) | data.fcc.gov/api/license-view | **Timeout** | With follow_redirects |
| FCC Equipment Authorization | apps.fcc.gov/oetcf/eas | **Timeout** | |
| FCC Spectrum Auction Data | auctiondata.fcc.gov | **DNS failure** | Domain doesn't resolve |
| FCC Daily Digest RSS | fcc.gov/news-events/daily-digest/rss | **Timeout** | |
| FCC Open Data (Socrata) | opendata.fcc.gov | **200** | Stale: broadband 2016-2021, pirate radio, consumer complaints |
| FCC API Catalog | publicapi.fcc.gov/ | **404** | |
| ITU Radio Regulations | itu.int | **404** | |
| Ofcom UK | ofcom.org.uk/api | **404** | HTML only |

## FCC Socrata Open Data Detail

The only working endpoint. 30+ datasets found but all are stale broadband deployment data (June 2016 – June 2021) or consumer complaint aggregates. No spectrum auction results, no equipment authorization, no filing data.

## Decision: SKIP

Same pattern as 7b-F (ADS-B) — insufficient API coverage for the surveillance use case. All primary endpoints (ECFS, ULS, Equipment Auth, Auction Data) are either:
- 403 blocked (ECFS API appears decommissioned or geo-restricted)
- Timeout (ULS, Equipment Auth — likely firewall or deprecated)
- DNS failure (Auction Data subdomain no longer exists)

International alternatives (ITU, Ofcom, ETSI) also returned 404 or HTML-only pages.

No free, programmatic access to spectrum auction results, equipment authorization filings, or telecom regulatory proceedings exists as of 2026-04-02.

---

## Related

- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]

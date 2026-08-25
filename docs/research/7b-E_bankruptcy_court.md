---
title: "Feature: 7b-E — Bankruptcy, Court Filings & Regulatory Actions (Global)"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/bankruptcy
---

# Feature: 7b-E — Bankruptcy, Court Filings & Regulatory Actions (Global)

## Signal Theory

Legal filings with irreversible consequences. Bankruptcy = terminal event. Enforcement = regulatory
regime shift. Both leak directional signal before market digests implications.

- Chapter 11 filings in PACER appear hours before wire services pick up
- SEC enforcement (Admin Proceedings) = direct action against companies/individuals
- 8-K Item 1.03 (Bankruptcy) = companies self-reporting insolvency events to SEC
- UK Gazette insolvency = official publication of winding-up petitions, administrations
- Cross-jurisdiction clustering = systemic stress signal (multiple filings in same sector across countries)

## API Probe Results (6 batches, 30+ endpoints tested)

### Tier 1 — Structured, Real-Time, Free, No Auth

| Source | URL Pattern | Format | Notes |
|--------|-------------|--------|-------|
| **PACER RSS** | `https://ecf.{court}.uscourts.gov/cgi-bin/rss_outside.pl` | XML RSS | Real-time. 6 courts verified. Case number, chapter, trustee, docket type. Volumes: SDTX 391KB, CDCA 904KB, NDIL 551KB, DEL 115KB, NJ 356KB, SDNY 74KB |
| **SEC Admin Proceedings RSS** | `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=ADMIN&dateb=&owner=include&count=40&search_text=&action=getcompany&output=atom` | Atom XML | Enforcement actions with titles, dates, links to PDFs, release numbers |
| **SEC EFTS (8-K Item 1.03)** | `https://efts.sec.gov/LATEST/search-index?q="1.03"&forms=8-K&_source=form,file_date,display_names,items` | JSON | Companies self-reporting bankruptcy/receivership. ~255 results for a 30-day window |
| **UK Gazette Atom Feed** | `https://www.thegazette.co.uk/insolvency/data.feed` | Atom XML | Official UK insolvency notices. Paginated (209K+ pages). Winding-up petitions, administrations |
| **GOV.UK Search API (SFO)** | `https://www.gov.uk/api/search.json?filter_organisations=serious-fraud-office&count=N&order=-public_timestamp` | JSON | UK Serious Fraud Office investigations/prosecutions. Structured: title, date, link, format |

### Tier 2 — Semi-structured / Limited

| Source | Status | Notes |
|--------|--------|-------|
| SEC EFTS full-text | Works | Can search all filings for bankruptcy-related terms, but noisy (65KB results) |
| GOV.UK FCA search | Works but sparse | Only 15 total results, mostly press releases not enforcement |
| EUR-Lex CELLAR SPARQL | Works but empty | SPARQL endpoint responds but query optimization needed |

### Dead / Blocked (Tested & Eliminated)

| Source | Status | Reason |
|--------|--------|--------|
| DOJ Press Releases API | 404 | Old API deprecated |
| FTC Enforcement API | 404 | Old API deprecated |
| UK Companies House API | 401 | Requires API key (free but needs registration) |
| FCA Direct API | 403/404 | Cloudflare blocked / endpoint doesn't exist |
| FCA Final Notices RSS | 404 | |
| UK SFO RSS | HTML | Returns GOV.UK HTML, not RSS feed |
| ESMA Enforcement | 404 | |
| BaFin Meldungen RSS | 404 | |
| AMF (France) RSS | 404 | |
| CONSOB (Italy) RSS | 404 | |
| JFSA (Japan) RSS | 404 | |
| ASIC (Australia) RSS | 404 (both URLs) | |
| SEBI (India) Orders | 404 | |
| WTO Disputes API | 401 | Requires subscription key |
| WTO HTML | HTML only | Not structured |
| Canada OSB | Login required | Auth wall |
| HK Gazette | HTML only | No API |

## PACER Courts Verified (6 major bankruptcy courts)

All handle ~90% of major corporate Chapter 11 filings:

| Court | Domain | Coverage |
|-------|--------|----------|
| S.D. New York | `ecf.nysb.uscourts.gov` | Wall Street, major financial Ch.11 |
| Delaware | `ecf.deb.uscourts.gov` | ~50% of large corporate Ch.11 |
| S.D. Texas | `ecf.txsb.uscourts.gov` | Houston, oil/gas/energy Ch.11 |
| C.D. California | `ecf.cacb.uscourts.gov` | LA/entertainment/tech Ch.11 |
| N.D. Illinois | `ecf.ilnb.uscourts.gov` | Chicago, industrial/retail Ch.11 |
| D. New Jersey | `ecf.njb.uscourts.gov` | Pharma, big corporate Ch.11 |

RSS URL pattern: `https://ecf.{domain}/cgi-bin/rss_outside.pl`

## Architecture

### Tool Design

**4 modes:**

1. **`us_bankruptcy`** — PACER RSS across 6 major courts. Parse XML, extract case number, chapter (7/11/13), debtor name, docket type. Filter by court, chapter. Default: all 6 courts, last entries.

2. **`sec_enforcement`** — SEC Admin Proceedings RSS. Parse Atom XML for enforcement actions. Extract respondent names, release numbers, dates, PDF links.

3. **`sec_bankruptcy`** — SEC EFTS for 8-K Item 1.03 filings. Companies self-reporting bankruptcy or receivership. Extract company names, filing dates, CIK.

4. **`uk_insolvency`** — UK Gazette Atom feed for insolvency notices. Parse pagination. Optional SFO enforcement via GOV.UK API.

### Parsing

- PACER RSS: `xml.etree.ElementTree` — `<item><title>`, `<link>`, `<description>`, `<pubDate>`
  - Title format: `{case_number} {debtor_name}` — parse with regex
  - Description contains chapter type, trustee, docket action
- SEC Admin Proceedings: Atom XML — `<entry><title>`, `<link>`, `<updated>`
- SEC EFTS: JSON — `hits.hits[]._source.{display_names, items, file_date}`
- UK Gazette: Atom XML — standard Atom parsing

### Caching Strategy

- PACER RSS: 10 minute TTL (real-time priority, feeds are small)
- SEC Admin: 30 minute TTL (lower frequency)
- SEC EFTS: 1 hour TTL (batch data, not real-time)
- UK Gazette: 1 hour TTL

### Cross-Sector Clustering (Signal Extraction)

When multiple bankruptcies appear in the same SIC code or sector across jurisdictions within a short window, that's a systemic stress signal. The tool output should include enough data for the world model to detect this:
- Company names (for ticker lookup)
- Chapter type (11 = restructuring, 7 = liquidation — very different signals)
- Filing date/time (for temporal clustering)
- Court/jurisdiction (for geographic clustering)

## Risks

- PACER RSS can be slow (3-5s per court) — parallel fetch recommended
- PACER format varies slightly by court — need robust regex for title parsing
- UK Gazette pagination may change — handle gracefully
- SEC EFTS can be slow (4-5s) — already observed
- No international coverage beyond UK (all EU/Asia regulators blocked) — document this limitation

## Data Requirements

- No API keys needed for any source
- httpx for HTTP (consistent with all other tools)
- xml.etree.ElementTree for RSS/Atom parsing (stdlib, no dependency)
- re for PACER title parsing

## Files Affected

- CREATE: `agent/tools/bankruptcy_court.py`
- MODIFY: `agent/cli.py` (import + register)
- CREATE: `tests/test_bankruptcy_court_edge.py`
- MODIFY: `[[quant_training_ground]]` (mark steps done)

---

## Related

- [[7b-E_bankruptcy_court_spec|Spec: 7B-E Bankruptcy Court]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]

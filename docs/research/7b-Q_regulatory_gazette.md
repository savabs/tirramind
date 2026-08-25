---
title: "Feature: 7b-Q Regulatory Gazette / Official Journal"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
---

# Feature: 7b-Q Regulatory Gazette / Official Journal

## API Findings

### US Federal Register — WINNER
- **URL:** `https://www.federalregister.gov/api/v1/documents.json`
- **Auth:** None. Free. No key.
- **Response format:** JSON
- **Document types:** RULE, PRORULE (proposed rule), NOTICE, PRESDOCU (presidential document)
- **Fields available:** title, type, abstract, document_number, publication_date, agencies (with slug/id/name), action, dates, comment_url, comments_close_on, effective_on, docket_ids, topics, significant, regulation_id_number_info, cfr_references, page_length, subtype
- **Filtering:** type[], agencies[] (by slug), term (keyword search), publication_date[gte/lte] (date range)
- **Pagination:** per_page (max 1000), page, next_page_url
- **Volume:** 132 proposed rules in March 2026 alone; 752 results for "semiconductor"; 1780 SEC rules total
- **470 agencies** with slugs for filtering
- **Speed:** <1s per request

### UK legislation.gov.uk — SECONDARY
- **URL:** `https://www.legislation.gov.uk/new/data.feed?page=N`
- **Auth:** None. Free.
- **Response format:** Atom XML
- **Content:** All new UK legislation (Statutory Instruments, Acts, etc.)
- **Pagination:** 20 items/page, `morePages` attribute
- **Fields:** title, updated, link (to full text), category
- **Issue:** Some entries have missing titles. Categories inconsistent. No direct keyword search.
- **Volume:** 374 UK SIs in 2026 YTD

### EUR-Lex / CELLAR SPARQL — SKIP
- CELLAR SPARQL endpoint exists but CDM ontology queries return 0 results (data in named graphs, joins fail)
- All EUR-Lex HTML endpoints return 202 (SPA rendered client-side)
- ELI API also returns 202
- Too fragile for production tool. Note for future expansion.

### Others (not probed deeply)
- Brazil DOU: Connection refused
- Australia: SPA, no JSON API surfaced
- Japan e-Gov: Would need Japanese parsing

## Architecture

### Modes
1. **recent** — Latest rules/proposed rules with optional agency/keyword/type filter. Primary scan mode.
2. **search** — Keyword search across Federal Register. For targeted investigation.
3. **agency** — Rules by specific agency (SEC, FDA, FERC, etc.). Market-sector regulatory tracking.
4. **upcoming** — Documents with open comment periods (comments_close_on > today). The future regulatory pipeline.

### Market-Relevant Agencies (sorted by impact)
- SEC (466): securities-and-exchange-commission
- Federal Reserve (188): federal-reserve-system
- CFTC (77): commodity-futures-trading-commission
- FTC (192): federal-trade-commission
- EPA (145): environmental-protection-agency
- FDA (199): food-and-drug-administration
- FCC (161): federal-communications-commission
- FERC (167): federal-energy-regulatory-commission
- Treasury (497): treasury-department
- DOJ (268): justice-department
- DOD (103): defense-department
- Commerce (54): commerce-department
- Energy (136): energy-department
- Transportation (492): transportation-department
- Agriculture (12): agriculture-department
- CFPB (573): consumer-financial-protection-bureau
- NRC (383): nuclear-regulatory-commission
- Interior (253): interior-department

### Signal Value
- **Proposed rules with open comment periods** = regulatory change forming, 30-90 day lead time
- **Significant rules** = economically significant (>$100M impact)
- **Agency clustering** = if SEC + CFTC + FTC all propose rules on same topic within 30 days = coordinated regulatory wave
- **Keyword surge** = new term appearing in regulations = emerging policy focus

### Caching
- recent/search results: 2hr TTL (new rules publish daily)
- agency list: 24hr TTL (rarely changes)

## Risks
- Federal Register is US-only (but it's the richest, most structured regulatory API in the world)
- UK Atom feed has missing titles for some entries
- EUR-Lex unreliable — future expansion
- Rate limits undocumented (but generous, no issues in testing)

---

## Related

- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]

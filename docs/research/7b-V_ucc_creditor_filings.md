---
title: "Feature: 7b-V — UCC / Secured Creditor Filings"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
---

# Feature: 7b-V — UCC / Secured Creditor Filings

## Signal Theory
When creditors file security claims, they're getting nervous about getting paid.
Surge in secured creditor filings precedes bankruptcies by months. New liens,
unsatisfied charges, and material credit facility changes are leading indicators
of financial distress.

## Data Sources

### SEC EDGAR 8-K (US — PRIMARY, free, no auth)
- EFTS full-text search: `https://efts.sec.gov/LATEST/search-index`
- Already used in form144.py, insider_filings.py, bankruptcy_court.py
- Search keywords: "security interest", "pledge", "lien", "collateral"
- 8-K Item 1.01 (material agreements), 2.04 (events of default)
- Rate limit: 10 req/sec (0.15s delay between requests)
- Returns: company name, CIK, filing date, item types, form type

### UK Companies House (UK — SECONDARY, free API key required)
- Base URL: `https://api.company-information.service.gov.uk`
- Auth: HTTP Basic (free API key as username, empty password)
- Key endpoints:
  - `GET /search/companies?q={name}` — company name search
  - `GET /company/{number}/charges` — list secured charges
- Returns: charge type, creditor, status (outstanding/satisfied), dates
- Registration: https://developer.company-information.service.gov.uk/

## Implementation Plan
3 modes:
1. **sec_credit_events** — Search EDGAR 8-K for credit/security language.
   Detect material credit facility changes, security pledges, defaults.
2. **uk_charges** — Companies House charges for a company. Show outstanding
   vs satisfied charges, new charge surge detection.
3. **stress_scan** — Broad scan: recent 8-K credit events + UK charge flags.
   Entity-level and sector-level stress signals.

---

## Related

- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]

---
title: "Research: Insider Filings (Phase 1)"
tags:
  - doc/research
  - topic/insider-filings
---

# Research: Insider Filings (Phase 1)

**Date:** 2026-03-28 (retroactive documentation)
**Tool:** `agent/tools/insider_filings.py` → `InsiderFilingsTool`
**Status:** IMPLEMENTED, TESTED — **US-ONLY, INHERENT**

## APIs Used

### SEC EDGAR Full-Text Search (EFTS) ✅
- **URL:** `https://efts.sec.gov/LATEST/search-index`
- **Method:** GET
- **Auth:** None (User-Agent required: `TirraMind/1.0`)
- **Format:** JSON (search results), XML (Form 4 filing documents)
- **Rate limits:** 10 req/sec (SEC-enforced). Tool uses 0.15s delay (6.7 req/s effective). 429 retry with 2s backoff.
- **Coverage:** **US only** — SEC Form 4 (insider buying/selling by officers/directors/10% owners of US-listed companies)

### SEC EDGAR Archives ✅
- **URL:** `https://www.sec.gov/Archives/edgar/data`
- **Method:** GET
- **Format:** XML (Form 4 schema X0407/X0508)
- **Features:** Filing XML with transaction details, ownership type, shares, price

## Geographic Coverage
- SEC Form 4 is US securities law requirement — inherently US-only
- Covers all US-listed companies (NYSE, NASDAQ, OTC)
- **Verdict:** `[G:INHERENT]` — region-locked by data source nature

## Implementation Details
- Single mode (no mode parameter)
- Parameters: `days_back` (1-90), `ticker` (optional filter), `min_cluster_size` (default 3)
- Cluster detection: 3+ distinct insiders buying within 14-day window
- Conviction scoring: C-suite presence + cluster size
- Transaction type: open-market purchases only (`transactionCode == "P"`)
- Capped at 500 filings per scan

## Signal Value
- Insider buying clusters = strongest known legal insider signal
- C-suite buying more informative than board member buying
- Cluster (3+ insiders) in same window = coordinated conviction
- Form 144 (sell intent) covered by separate form144.py tool

## Global Expansion (separate tools needed)
| Jurisdiction | Registry | Status |
|---|---|---|
| UK | Companies House (Directors' Dealings) | Free API |
| India | SEBI (Insider Trading Disclosures) | Public |
| Japan | EDINET (Insider Reports) | Free API |
| Hong Kong | HKEX (Connected Transactions) | Public |
| Germany | Bundesanzeiger (Directors' Dealings) | Public |
| Australia | ASX (Director Interest Notices) | Public |
| Canada | SEDAR+ (Insider Reports / SEDI) | Free |

These would be NEW tools, not modifications to this tool.

## Related

- [[project_memory]]

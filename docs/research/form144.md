---
title: "Feature: SEC Form 144 — Intent to Sell"
tags:
  - doc/research
  - topic/form144
---

# Feature: SEC Form 144 — Intent to Sell

## What Is Form 144?

SEC Rule 144 requires insiders (officers, directors, 10%+ holders) to file Form 144 **before or concurrently with** placing a sell order for restricted or control securities when the sale exceeds 5,000 shares or $50,000 in a 3-month period.

**Key distinction from Form 4:**
- Form 144 is filed at T+0 or earlier (before/with sell order)
- Form 4 is filed at T+2 (within 2 business days after execution)
- Form 144 captures **intent**; Form 4 captures **execution**

This 2+ day timing advantage is the primary signal value.

## Current Architecture

### EDGAR EFTS API (reusable)

Same infrastructure as InsiderFilingsTool (Phase 5c):
- Search endpoint: `https://efts.sec.gov/LATEST/search-index?forms=144`
- XML archives: `https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}`
- Rate limit: 10 req/s, User-Agent required, 0.15s delay
- EFTS response structure identical to Form 4 searches
- Ticker available in EFTS `display_names` field (not in XML itself)

### Volume (live-tested 2026-03-25)

| Period | Count |
|--------|-------|
| 30 days | ~4,440 |
| 5 days | ~325 |
| Per day avg | ~148 |
| Form 144/A (amendments, 3 months) | 71 |

Manageable volume. Pagination cap at 500 per scan is sufficient.

## Form 144 XML Schema (live-tested)

```
edgarSubmission
  headerData
    submissionType: "144"
    filerInfo/filer/filerCredentials/cik
  formData
    issuerInfo
      issuerCik                     → company CIK
      issuerName                    → "ROSS STORES, INC."
      secFileNumber
      issuerAddress
      issuerContactPhone
      nameOfPersonForWhose...       → insider name (long element name)
      relationshipsToIssuer
        relationshipToIssuer        → "Officer" | "Director" | "10% Stockholder"
    securitiesInformation           → THE SIGNAL DATA
      securitiesClassTitle          → "Common" | "Class B" | "Ordinary Shares"
      brokerOrMarketmakerDetails    → broker name + address
        name                        → "Morgan Stanley Smith Barney LLC" etc.
      noOfUnitsSold                 → 4154 (shares planned to sell)
      aggregateMarketValue          → 884428.56 (dollar value)
      noOfUnitsOutstanding          → 323444928 (total outstanding)
      approxSaleDate                → "03/24/2026" (MM/DD/YYYY — planned date)
      securitiesExchangeName        → "NASDAQ" | "NYSE"
    securitiesToBeSold[]            → acquisition history (multiple entries possible)
      securitiesClassTitle
      acquiredDate                  → "03/20/2026" (MM/DD/YYYY)
      natureOfAcquisitionTransaction → "Performance Stock Units" | "Restricted Stock Units" | "Private Placement" | "Open Market Purchase"
      nameOfPersonfromWhomAcquired  → "Issuer" (typical)
      isGiftTransaction             → "Y" | "N"
      amountOfSecuritiesAcquired
      paymentDate
      natureOfPayment
    nothingToReportFlagOnSecuritiesSoldInPast3Months → "Y" | "N"
    securitiesSoldInPast3Months (optional)
      sellerDetails/name, address
      saleDate, amountOfSecuritiesSold, grossProceeds
    noticeSignature
      noticeDate                    → "03/24/2026" (signing date)
      planAdoptionDates (optional)  → 10b5-1 plan dates
      signature
```

### Namespace Variations (live-tested)

- Filing 1: `xmlns:ns2="http://www.sec.gov/edgar/common"` → `ns2:street1`
- Filing 2: `xmlns:com="http://www.sec.gov/edgar/common"` → `com:street1`
- Root namespace: `xmlns="http://www.sec.gov/edgar/ownership"` (consistent)

Must parse dynamically — same approach as Form 4 parser (`root.tag.split("}")[0] + "}"`).

### No Ticker in XML

Unlike Form 4's `issuerTradingSymbol`, Form 144 has no ticker element.
Ticker extracted from EFTS `display_names`: `"ROSS STORES, INC.  (ROST)  (CIK 0000745732)"`.
Reuse existing `_extract_ticker()` regex from insider_filings.py.

## Signal Theory

### Why Form 144 Is Uniquely Valuable

1. **Timing advantage**: T+0 or earlier vs Form 4's T+2. Two days lead time on insider sell activity.
2. **Crowd gap**: Nearly everyone watches Form 4. Very few systematically process Form 144. Less efficient = more alpha.
3. **Intent vs execution**: Form 144 captures intention. Some 144s are filed but trades never execute (intent changed — also informative). The gap between 144 volume and subsequent Form 4 volume is itself a signal.
4. **Entity filers reveal sophisticated exits**: 10%+ holders (PE funds, VC, activist investors) with board seats file 144 when reducing stakes. These are the most informed non-management sellers.

### Signal Taxonomy by Acquisition Type

| Acquisition Type | Example | Signal Strength | Why |
|-----------------|---------|-----------------|-----|
| Open Market Purchase | Bought shares voluntarily | VERY HIGH | Deliberately bought, now deliberately selling |
| Private Placement | PE/VC investment | HIGH | Sophisticated investor exiting |
| RSU/PSU vesting | Compensation vesting | LOW-MODERATE | Often automatic tax withholding |
| Gift | Gifted shares | NOISE | Non-economic motivation |

### Urgency Signal

`filing_date - approxSaleDate` gap matters:
- **Same day**: Urgent, already selling → strong signal
- **1-7 days**: Near-term planned → moderate signal  
- **>7 days**: Pre-planned, possibly 10b5-1 schedule → weaker signal (planned ≠ informed)

### Dollar Significance

- `aggregateMarketValue / (noOfUnitsOutstanding × est_price)` ≈ % of float planned for sale
- Threshold: >0.1% of outstanding in a single filing = noteworthy, >1% = major

## Risks

1. **RSU/PSU noise dominates**: Most Form 144s are routine vesting sells. Must filter aggressively.
2. **No ticker in XML**: Depends on EFTS display_name parsing. If EFTS changes format, extraction breaks.
3. **Date format MM/DD/YYYY**: Unlike Form 4's YYYY-MM-DD. Must parse both formats.
4. **Entity filer names may not match insider names**: "Yorktown Energy Partners IX, L.P." is a fund, not a person. Different dedup logic needed.
5. **Form 144/A amendments**: 71 in 3 months — low volume but exist. Amendments may supersede original filings.
6. **planAdoptionDates**: 10b5-1 pre-arranged trading plans reduce signal strength (selling is pre-committed, not responsive to current info).

## Data Requirements

- EDGAR EFTS API (same as Phase 5c, no additional auth)
- httpx, xml.etree.ElementTree (already in project)
- DataCache integration (same pattern)
- No new dependencies

## Algorithm: Sell Intent Cluster Detection

Group by company (via ticker from EFTS), sliding 14-day window, rank by:
`conviction_score = insider_count × aggregate_dollar_value × acquisition_signal_weight`

Where `acquisition_signal_weight`:
- Open Market Purchase: 3.0
- Private Placement: 2.0  
- RSU/PSU vesting: 0.5
- Gift: 0.0 (excluded)

## Cross-Signal Potential (Future Phases)

| Cross-Reference | Meaning | Phase |
|----------------|---------|-------|
| Form 144 cluster + Form 4 buy cluster (same company) | Internal disagreement — anomaly | Phase 8 (World Model) |
| Form 144 filed → no subsequent Form 4 | Intent abandoned — possibly positive | Phase 7 (Pipeline) |
| Form 144 cluster + dark pool anomaly | Institutional distribution | Phase 6e + 8 |
| Form 144 10%+ holder + declining Polymarket odds | Smart money exiting ahead of catalyst | Phase 8 |

---

## Related

- [[form144_spec|Spec: Form144]]

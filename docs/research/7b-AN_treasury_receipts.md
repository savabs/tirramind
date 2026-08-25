---
title: "Feature: 7b-AN — Government Tax Receipts (Treasury Daily Treasury Statement)"
tags:
  - doc/research
  - layer/surveillance
  - phase/7b
  - topic/treasury
---

# Feature: 7b-AN — Government Tax Receipts (Treasury Daily Treasury Statement)

## Current Architecture
- **Layer:** 1 (Surveillance Surface) — `agent/tools/`
- **Pattern:** Tool inherits from `Tool` base class, uses `httpx` for HTTP, `DataCache` for caching
- **Existing overlap:** `sovereign_debt.py` fetches from US Treasury XML yield data; this tool targets a different Treasury API (Fiscal Data)
- **Registration:** `agent/cli.py` (import + `registry.register()`), `agent/learning/bandit.py` (GoalArm)

## Data Source: US Treasury Fiscal Data API

- **API:** `https://api.fiscaldata.treasury.gov/services/api/fiscal_service/`
- **Auth:** None required. Open, public, unrestricted.
- **License:** "free, without restriction, available to copy, adapt, redistribute, or otherwise use for non-commercial or commercial purposes" — explicitly commercial-safe.
- **Format:** JSON (default), CSV, XML
- **Rate limits:** Not documented, but fair-use expected. Pagination: 100 records default, configurable `page[size]`.
- **Data range:** 10/03/2005 to present
- **Update frequency:** Daily (M-F), next business day release

### Key Endpoints (DTS = Daily Treasury Statement)

1. **Operating Cash Balance:** `/v1/accounting/dts/operating_cash_balance`
   - Fields: `record_date`, `account_type`, `open_today_bal`, `open_month_bal`, `open_fiscal_year_bal`
   - Signal: Treasury General Account (TGA) balance — sudden drawdowns = fiscal stress; rebuilds after debt ceiling = liquidity drain from markets

2. **Deposits & Withdrawals:** `/v1/accounting/dts/deposits_withdrawals_operating_cash`
   - Fields: `record_date`, `transaction_type` (deposit/withdrawal), `transaction_today_amt`, `transaction_mtd_amt`, `transaction_fytd_amt`
   - Categories: Individual income tax withheld, corporate tax deposits, customs duties, FHFA, SBA, etc.
   - Signal: Withheld income tax = real-time employment proxy (T+0 vs BLS T+30). Corporate tax deposits = real-time earnings proxy. Customs duties = real-time import volume proxy.

3. **Public Debt Transactions:** `/v1/accounting/dts/public_debt_transactions`
   - Fields: `record_date`, `transaction_type`, `transaction_today_amt`, `transaction_mtd_amt`, `transaction_fytd_amt`
   - Signal: T-bill issuance pace, net borrowing, debt ceiling dynamics

4. **Inter-Agency Tax Transfers:** `/v1/accounting/dts/inter_agency_tax_transfers`

### Query Parameters
- `fields=` — comma-separated field list
- `filter=record_date:gte:2026-01-01` — SQL-like filtering
- `sort=-record_date` — sort descending
- `page[number]=1&page[size]=100` — pagination
- `format=json` (default)

## Signal Theory

Tax receipts are among the **highest-value free signals** in existence:
- **Withheld income taxes** are a T+0 proxy for aggregate employment × wages — faster than ADP (T+2), BLS (T+30), or UI claims (T+7)
- **Corporate tax deposits** correlate with real-time corporate earnings — faster than quarterly earnings reports
- **Customs duties** = real-time import volumes, trade balance proxy
- **TGA balance** movements drive overnight repo rates and money market liquidity — the Fed watches this
- **Public debt transactions** reveal Treasury issuance patterns that move bond markets

This is the kind of data TirraMind was built for: free, public domain, daily, and contains predictive signal that most market participants ignore because it requires parsing a government API.

## Observations
- The API is RESTful, well-documented, and returns clean JSON
- No authentication barrier whatsoever
- Historical data back to 2005 enables backtesting
- Daily granularity with business-day updates

## Risks
- Government shutdown → API might go offline temporarily
- Schema changes (field names) are possible but unlikely for established tables
- Weekend/holiday gaps in data
- Large response sizes for historical queries (use pagination)

## Data Requirements
- Operating cash balance (TGA levels)
- Deposits/withdrawals by category (withheld income, corporate, customs)
- Public debt transactions (issuance, redemptions)
- MTD and FYTD aggregates for trend analysis

## Math/Algorithm Survey
- Z-score of daily deposits vs trailing 52-week mean (surprise detection)
- YoY growth rate of withheld income tax (employment momentum)
- TGA balance rate of change (liquidity signal)
- Customs duty momentum (trade flow proxy)
- All computed internally, no external math libs needed beyond basic stats

## OSS/External Research
- No existing Python library wraps this specific API
- Treasury Fiscal Data API documentation is the authoritative source
- No license conflicts — public domain US government data

---

## Related

- [[7b-AN_treasury_receipts_spec|Spec: 7B-An Treasury Receipts]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]

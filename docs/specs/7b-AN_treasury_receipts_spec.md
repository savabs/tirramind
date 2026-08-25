---
title: "Spec: 7b-AN — Treasury Receipts Tool"
tags:
  - doc/spec
  - layer/surveillance
  - phase/7b
  - topic/treasury
---

# Spec: 7b-AN — Treasury Receipts Tool

## Goal
Fetch and parse US Treasury Daily Treasury Statement (DTS) data via the Treasury Fiscal Data API. Provide operating cash balance (TGA), tax receipt breakdowns (deposits/withdrawals by category), and public debt transactions. Compute quant signals: YoY income tax growth, TGA rate of change, customs duty momentum.

## Files Affected
- **Create:** `agent/tools/treasury_receipts.py`
- **Modify:** `agent/cli.py` — add import + registration
- **Modify:** `agent/learning/bandit.py` — add GoalArm
- **Create:** `tests/test_treasury_receipts_edge.py`

## Implementation Steps

### 2.1: Create `agent/tools/treasury_receipts.py`
- Class `TreasuryReceiptsTool(Tool)`
- `name = "treasury_receipts"`
- 3 modes: `cash_balance`, `deposits_withdrawals`, `public_debt`
- Parameters: `mode`, `start_date` (YYYY-MM-DD), `end_date` (YYYY-MM-DD), `category_filter` (for deposits/withdrawals), `top_n`
- `__init__(self, cache: DataCache | None = None)`
- Use `httpx.Client` with `timeout=20`, User-Agent header
- Cache with `DataCache` (key by mode + date range)
- Parse JSON response, extract relevant fields
- Compute signals: TGA delta, YoY growth (if sufficient historical data in response), category rankings
- Return `ToolResult` with formatted output + structured `data`

### 2.2: API Implementation Details
- Base URL: `https://api.fiscaldata.treasury.gov/services/api/fiscal_service`
- Endpoints:
  - `/v1/accounting/dts/operating_cash_balance` (cash_balance mode)
  - `/v1/accounting/dts/deposits_withdrawals_operating_cash` (deposits_withdrawals mode)
  - `/v1/accounting/dts/public_debt_transactions` (public_debt mode)
- Query params: `fields=`, `filter=record_date:gte:{start},record_date:lte:{end}`, `sort=-record_date`, `page[size]=500`
- Response shape: `{"data": [...], "meta": {...}, "links": {...}}`

### 2.3: Register in `agent/cli.py`
- Import `TreasuryReceiptsTool` from `agent.tools.treasury_receipts`
- Add `registry.register(TreasuryReceiptsTool(cache=cache))` after energy_supply registration

### 2.4: Add GoalArm in `agent/learning/bandit.py`
- `name="treasury_receipt_monitor"`
- `tools=["treasury_receipts", "web_search"]`
- Examples: TGA balance check, withheld income tax trend, customs duties, public debt issuance

### 2.5: Write edge-case tests
- Cover: all 3 modes, invalid mode, invalid dates, empty response, HTTP errors, malformed JSON, cache hit/miss, large response pagination, category filtering, signal computation, tool schema validation

## Edge Cases
- API returns empty `data: []` for future dates or weekends
- Pagination: responses > 500 records need page handling (or accept truncation with top_n)
- Government shutdown: API offline → HTTP error → graceful failure
- Missing fields in JSON response (some tables have sparse fields)
- Date filter edge: `start_date > end_date` → validation error

## Testing Plan
- Mock `httpx.Client.get` responses with synthetic JSON matching API schema
- Test each mode independently
- Test signal computation with known values
- Test error paths: 404, 500, timeout, invalid JSON, network error
- Validate `ToolResult` structure and `data` dict format
- Test cache integration (hit + miss paths)

---

## Related

- [[7b-AN_treasury_receipts|Research: 7B-An Treasury Receipts]]
- [[convergence_detection]]
- [[tier1_signal_expansion]]
- [[tier2_signal_expansion]]
- [[observational_surface]]

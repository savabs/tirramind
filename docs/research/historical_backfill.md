---
title: "Research: Historical Backfill Runner"
tags:
  - doc/research
  - phase/47
  - topic/backfill
  - topic/training-data
  - layer/surveillance
  - status/active
---

# Research: Historical Backfill Runner (Phase 47)

## Goal

Run all 51 tools in backfill mode across 2–5 years of history before the first real GNN training run (Phase 40). This replaces the 6-week live accumulation wait with a 1–2 day data collection job.

**Why this matters:** The GNN learns from temporal event sequences — it needs to have seen market cycles, crises, supply shocks, sanctions waves, and cross-domain causation before it can generalise. 5 years of history beats 6 weeks of patience every time.

## All 51 Tools — Backfill Status

### Group A: Full historical API (confirmed backfillable, 27 tools)

| Tool | API source | History available | Backfill param |
|---|---|---|---|
| `academic_preprints` | arXiv / bioRxiv | 5+ years | `days_back` |
| `ais_vessel` | OpenSky / Marine | 1–2 years free | `days_back` |
| `backtest` | internal | n/a — engine only | skip |
| `bankruptcy_court` | PACER/EDGAR | 5+ years | `days_back` |
| `building_permits` | Census.gov | 5+ years | `days_back` |
| `capital_flows` | IMF BOP | 5+ years | `days_back` |
| `central_bank_balance` | Fed/ECB/BOJ | 5+ years | `days_back` |
| `cftc` | CFTC.gov COT | 10+ years | `days_back` |
| `comtrade` | UN Comtrade | 5+ years | `days_back` |
| `consumer_sentiment` | BLS/Eurostat | 5+ years | `days_back` |
| `creditor_filings` | PACER | 5+ years | `days_back` |
| `earthquake_proximity` | USGS | 5+ years | `days_back` |
| `food_security` | FAO/FEWS | 5+ years | `days_back` |
| `form144` | SEC EDGAR | 5+ years | `days_back` |
| `gdelt` | GDELT Project | 5+ years | `days_back` |
| `gov_contracts` | USASpending.gov | 5+ years | `days_back` |
| `insider_filings` | SEC EDGAR | 5+ years | `days_back` |
| `job_postings` | BLS | 3+ years | `days_back` |
| `labor_disruptions` | BLS | 5+ years | `days_back` |
| `liquidity_regime` | internal (yfinance+FRED) | 10+ years | `lookback_years` |
| `macro_data` | FRED | 20+ years | `days_back` |
| `market_data` | yfinance | 10+ years | `period` |
| `migration_flows` | UN/UNHCR | 5+ years | `days_back` |
| `pipeline_query` | internal DB | n/a — query only | skip |
| `satellite_activity` | Copernicus/NASA | 3+ years | `days_back` |
| `supply_chain_monitor` | BLS/FRED | 5+ years | `days_back` |
| `transport_throughput` | BTS/Eurostat | 5+ years | `days_back` |

### Group B: Partial historical — verify per tool (24 tools)

| Tool | Likely history | Action |
|---|---|---|
| `cert_transparency` | Live only (crt.sh searches current) | Skip backfill — live-only |
| `internet_outages` | Live only (RIPE/Cloudflare real-time) | Skip backfill — live-only |
| `defi_flows` | The Graph / on-chain — historical available | Verify endpoint |
| `disease_surveillance` | WHO/CDC — 3+ years | Verify endpoint |
| `dns_monitor` | Live bulk-resolve only | Skip backfill — live-only |
| `drug_regulatory` | FDA Drugs@FDA — 5+ years | Verify endpoint |
| `electricity_monitor` | EIA/Entso-E — 2+ years | Verify endpoint |
| `energy_supply` | EIA — 5+ years | Verify endpoint |
| `finra_short_volume` | FINRA — 2+ years | Verify endpoint |
| `foia_requests` | MuckRock/FOIA.gov — 3+ years | Verify endpoint |
| `global_pmi` | Markit/ISM — 5+ years | Verify endpoint |
| `interconnection_queue` | FERC — 3+ years | Verify endpoint |
| `internet_infrastructure` | BGP/RIPE — historical dumps | Verify endpoint |
| `lobbying` | OpenSecrets/senate.gov — 5+ years | Verify endpoint |
| `patent_filings` | USPTO/EPO — 5+ years | Verify endpoint |
| `political_risk` | FEC campaign finance — 5+ years | Verify endpoint |
| `polymarket` | Gamma API — 2+ years market history | Verify endpoint |
| `polymarket_whales` | Gamma API — 2+ years | Verify endpoint |
| `power_grid` | EIA/RTO — 2+ years | Verify endpoint |
| `regulatory_gazette` | Federal Register — 5+ years | Verify endpoint |
| `sanctions_monitor` | OFAC/UN — 5+ years | Verify endpoint |
| `sovereign_debt` | IMF/OECD — 5+ years | Verify endpoint |
| `treasury_receipts` | Treasury.gov — 5+ years | Verify endpoint |
| `weather_alerts` | NOAA archive — 5+ years | Verify endpoint |
| `whale_alert` | Whale Alert API — 1–2 years | Verify endpoint |
| `wikipedia_pageviews` | Wikimedia API — 5+ years | Verify endpoint |

### Group C: Internal utility tools — no backfill needed

| Tool | Reason |
|---|---|
| `backtest` | Engine, not data source |
| `code_executor` | Utility |
| `file_manager` | Utility |
| `instrument_universe` | Reference data, not time-series |
| `pipeline_query` | Internal query |
| `shell_runner` | Utility |
| `web_browse` | Live only |
| `web_search` | Live only |

## Architecture

### Simple design — one script, one pass

```
scripts/backfill.py
  - For each tool in BACKFILL_TOOLS list:
      call tool with days_back=1825 (5 years)
      tool writes to DB with correct timestamps
      log: tool name, obs count written, elapsed time
  - Rate-limit between tools (1–2 sec sleep) to be polite to free APIs
  - Resume from checkpoint: skip tools already completed
  - Total time estimate: 2–4 hours for all Group A tools
```

No new infrastructure. No new DB tables. Every tool already writes to `entity_observations` with timestamps. The backfill is just calling the existing tools with a longer time window.

### Rate limiting strategy

Free APIs have limits. The right approach:
- FRED: 120 calls/min — no issue
- SEC EDGAR: 10 calls/sec — no issue
- CFTC: no stated limit — no issue
- yfinance: no stated limit, self-throttle at 1 call/sec
- GDELT: no stated limit — no issue
- UN Comtrade: 100 calls/hour — batch by country pair, sleep between

Total estimated API calls for full 5-year backfill across Group A: ~500–2,000 calls. Well within free tier limits for every source.

## What the GNN gains from this

Current training data:
- 1,087 entities, 74,030 observations, 357 links, 3 days

After backfill:
- Same entity types, but each entity gets years of observation history
- Estimated: 500K–2M observations
- GNN sees: COVID shock (2020), supply chain crisis (2021–22), rate cycle (2022–23), sanctions waves, geopolitical shifts
- Cross-domain causation becomes detectable: vessel → commodity → price, policy → capital flows → FX, disease → supply → production

This is the difference between a model that can generalise and one that memorised 3 days.

## Risks

- **API availability**: Free APIs go down. Backfill runner must be resumable — checkpoint which tools completed.
- **Timestamp correctness**: Must verify each tool writes `observed_at` correctly for historical records, not `now()`.
- **Deduplication**: If daily collection already ran and wrote some observations, backfill must not double-write. Use upsert logic or date-range checks.
- **Group B verification**: 24 tools need endpoint-by-endpoint check before assuming backfill works. Do not assume — test.

## References

- FRED API docs: https://fred.stlouisfed.org/docs/api/fred/
- SEC EDGAR full-text search: https://efts.sec.gov/LATEST/search-index?q=
- CFTC COT historical: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
- GDELT: https://www.gdeltproject.org/data.html
- UN Comtrade API: https://comtradeplus.un.org/
- yfinance: https://github.com/ranaroussi/yfinance

## Related

- [[data_strategy_doctrine]] — governing doctrine (depth/density/coverage/diversity framework, sample complexity bounds, modal balance targets)
- [[historical_backfill_spec]]
- [[quant_training_ground]]
- [[living_system_online_gnn]]
